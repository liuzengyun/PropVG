"""Lightweight GrabVG-style text-conditioned graph binding.

This module keeps proposal boxes fixed and only refines proposal features,
which makes it a drop-in addition to the existing PropVG REFHead.
"""
import torch
from torch import nn
import torch.nn.functional as F


class GrabGraphBinding(nn.Module):
    def __init__(self, hidden_channels=256, num_heads=8, k=6, layers=4, anchors=48):
        super().__init__()
        self.k = k
        self.text_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_channels)
        self.filter_head = nn.Linear(hidden_channels, 1)
        self.anchor_query = nn.Parameter(torch.randn(anchors, hidden_channels) * 0.02)
        self.offset_head = nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, anchors * 2))
        self.visual_proj = nn.Linear(hidden_channels, hidden_channels)
        self.geo = nn.Sequential(nn.Linear(7, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.q_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.e_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, hidden_channels)) for _ in range(layers)])
        self.gate = nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, queries, boxes, text, text_mask=None, visual=None):
        # text_mask follows BEiT-3 convention: 1 means padding.
        key_padding = text_mask.bool() if text_mask is not None else None
        txt, _ = self.text_attn(queries, text, text, key_padding_mask=key_padding)
        txt = self.text_norm(queries + txt)
        filt = self.filter_head(txt).squeeze(-1)
        q = queries
        centers = boxes[..., :2]
        wh = boxes[..., 2:].clamp(min=1e-5)
        delta = centers[:, :, None, :] - centers[:, None, :, :]
        dist = torch.sqrt((delta ** 2).sum(-1, keepdim=True) + 1e-8)
        direction = delta / (dist + 1e-6)
        logscale = torch.log(wh[:, :, None, :] / wh[:, None, :, :].clamp(min=1e-5))
        geo = torch.cat([delta, dist, direction, logscale], dim=-1)
        if visual is not None:
            # Language-conditioned 48-point appearance binding around each box.
            b, n, _ = q.shape
            anchors = torch.tanh(self.offset_head(torch.cat([q, txt], dim=-1))).view(b, n, -1, 2)
            points = centers[:, :, None, :] + anchors * wh[:, :, None, :] * 0.5
            grid = points.mul(2.0).sub(1.0).view(b, n * anchors.shape[2], 1, 2)
            sampled = F.grid_sample(visual, grid, mode="bilinear", align_corners=True)
            sampled = sampled.squeeze(-1).transpose(1, 2).view(b, n, anchors.shape[2], -1).mean(2)
            q = q + self.visual_proj(sampled)
        # Detached graph construction prevents unstable topology gradients.
        knn = dist.squeeze(-1).detach().topk(min(self.k, boxes.shape[1]), dim=-1, largest=False).indices
        for l, (qp, kp, vp, ep, out) in enumerate(zip(self.q_proj, self.k_proj, self.v_proj, self.e_proj, self.out)):
            qh, kh, vh = qp(q), kp(q), vp(q)
            neigh_k = kh[:, None].expand(-1, boxes.shape[1], -1, -1).gather(2, knn[..., None].expand(-1, -1, -1, kh.shape[-1]))
            neigh_v = vh[:, None].expand(-1, boxes.shape[1], -1, -1).gather(2, knn[..., None].expand(-1, -1, -1, vh.shape[-1]))
            edge = ep(self.geo(geo.gather(2, knn[..., None].expand(-1, -1, -1, 7))))
            logits = (qh[:, :, None, :] * neigh_k).sum(-1) / (q.shape[-1] ** 0.5)
            alpha = F.softmax(logits, dim=-1)
            msg = (alpha[..., None] * (neigh_v + edge)).sum(2)
            q = q + out(msg)
        q = q + torch.sigmoid(self.gate(torch.cat([q, txt], dim=-1))) * txt
        return q, filt
