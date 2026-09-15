"""GrabVG-style hypothesis filtering and graph-attentive feature binding."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def _gather_nodes(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class GraphBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, message):
        x = self.norm1(x + message)
        return self.norm2(x + self.ffn(x))


class GrabGraphBinding(nn.Module):
    def __init__(self, hidden_channels=256, num_heads=8, k=6, layers=4, anchors=48, keep_ratio=0.5):
        super().__init__()
        self.k = k
        self.keep_ratio = keep_ratio
        self.num_anchors = anchors

        # Text-aware hypothesis filtering.
        self.filter_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.filter_norm1 = nn.LayerNorm(hidden_channels)
        self.filter_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4), nn.GELU(), nn.Linear(hidden_channels * 4, hidden_channels)
        )
        self.filter_norm2 = nn.LayerNorm(hidden_channels)
        self.filter_head = nn.Linear(hidden_channels, 1)

        # Language-guided box-relative appearance sampling.
        self.anchor_query = nn.Parameter(torch.randn(anchors, hidden_channels) * 0.02)
        side_y, side_x = 6, 8
        yy, xx = torch.meshgrid(torch.linspace(-0.5, 0.5, side_y), torch.linspace(-0.5, 0.5, side_x), indexing="ij")
        self.register_buffer("base_points", torch.stack((xx, yy), dim=-1).reshape(anchors, 2), persistent=False)
        self.anchor_text_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, 2)
        )
        self.level_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(4)])
        self.appearance_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.appearance_norm = nn.LayerNorm(hidden_channels)

        # Language-conditioned edge-aware graph attention.
        self.geo_mlp = nn.Sequential(nn.Linear(7, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.text_edge = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.Sigmoid())
        self.q_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.edge_k = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.edge_v = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.edge_bias = nn.ModuleList([nn.Linear(hidden_channels, 1) for _ in range(layers)])
        self.blocks = nn.ModuleList([GraphBlock(hidden_channels) for _ in range(layers)])
        self.final_gate = nn.Linear(hidden_channels * 2, hidden_channels)

    @staticmethod
    def _sentence_pool(text, text_mask):
        if text_mask is None:
            return text.mean(1)
        valid = (~text_mask.bool()).to(text.dtype)[..., None]
        return (text * valid).sum(1) / valid.sum(1).clamp(min=1.0)

    def _filter(self, queries, boxes, text, text_mask):
        padding = text_mask.bool() if text_mask is not None else None
        context, _ = self.filter_attn(queries, text, text, key_padding_mask=padding)
        fused = self.filter_norm1(queries + context)
        fused = self.filter_norm2(fused + self.filter_ffn(fused))
        logits = self.filter_head(fused).squeeze(-1)
        keep = max(1, int(round(queries.shape[1] * self.keep_ratio)))
        indices = logits.topk(keep, dim=1).indices
        # Per paper, the original decoder features—not filter features—continue.
        return _gather_nodes(queries, indices), _gather_nodes(boxes, indices), logits, indices

    def _appearance(self, queries, boxes, text, text_mask, visuals, visual_masks=None):
        b, n, c = queries.shape
        padding = text_mask.bool() if text_mask is not None else None
        anchor = self.anchor_query[None].expand(b, -1, -1)
        anchor_text, _ = self.anchor_text_attn(anchor, text, text, key_padding_mask=padding)
        anchor = anchor[:, None].expand(-1, n, -1, -1)
        anchor_text = anchor_text[:, None].expand(-1, n, -1, -1)
        proposal = queries[:, :, None].expand(-1, -1, self.num_anchors, -1)
        offsets = torch.tanh(self.offset_head(torch.cat((proposal, anchor, anchor_text), dim=-1))) * 0.5
        centers, sizes = boxes[..., :2], boxes[..., 2:].clamp(min=1e-5)
        points = centers[:, :, None] + (self.base_points[None, None] + offsets) * sizes[:, :, None]
        points = points.clamp(0.0, 1.0)

        sampled_levels = []
        for level, visual in enumerate(visuals):
            # Proposal boxes are normalized against the valid, unpadded image,
            # whereas grid_sample coordinates span the complete padded feature
            # map. Convert into each level's padded coordinate system using
            # the same validity masks as the deformable DETR decoder.
            level_points = points
            if visual_masks is not None:
                mask = visual_masks[level]
                valid_h = (~mask[:, :, 0]).sum(1).to(points.dtype) / mask.shape[1]
                valid_w = (~mask[:, 0, :]).sum(1).to(points.dtype) / mask.shape[2]
                valid_ratio = torch.stack((valid_w, valid_h), dim=-1)
                level_points = points * valid_ratio[:, None, None, :]
            grid = level_points.mul(2.0).sub(1.0).reshape(b, n * self.num_anchors, 1, 2)
            sampled = F.grid_sample(visual, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, n, self.num_anchors, c)
            sampled_levels.append(self.level_proj[level](sampled))
        sampled = torch.stack(sampled_levels, dim=0).mean(0)
        q = queries.reshape(b * n, 1, c)
        memory = torch.cat((q, sampled.reshape(b * n, self.num_anchors, c)), dim=1)
        update, _ = self.appearance_attn(q, memory, memory)
        return self.appearance_norm(q + update).reshape(b, n, c)

    def _topology(self, appearance, boxes, sentence):
        b, n, c = appearance.shape
        centers, sizes = boxes[..., :2], boxes[..., 2:].clamp(min=1e-5)
        delta = centers[:, :, None] - centers[:, None, :]
        distance = torch.sqrt((delta ** 2).sum(-1) + 1e-8)
        direction = delta / (distance[..., None] + 1e-6)
        logscale = torch.log(sizes[:, :, None] / sizes[:, None, :])
        geometry = torch.cat((delta, distance[..., None], direction, logscale), dim=-1)

        # Symmetrized kNN plus self-loops. Dense masks preserve every reverse
        # edge, unlike a second fixed-size top-k truncation.
        detached_dist = distance.detach()
        eye = torch.eye(n, dtype=torch.bool, device=boxes.device)[None]
        nearest = detached_dist.masked_fill(eye, float("inf")).topk(min(self.k, max(n - 1, 1)), largest=False).indices
        adjacency = torch.zeros((b, n, n), dtype=torch.bool, device=boxes.device)
        adjacency.scatter_(2, nearest, True)
        adjacency = adjacency | adjacency.transpose(1, 2) | eye

        edge = self.geo_mlp(geometry) * self.text_edge(sentence)[:, None, None]
        z = appearance
        for qp, kp, vp, ek, ev, eb, block in zip(
            self.q_proj, self.k_proj, self.v_proj, self.edge_k, self.edge_v, self.edge_bias, self.blocks
        ):
            q, key, value = qp(z), kp(z), vp(z)
            logits = (q[:, :, None] * (key[:, None] + ek(edge))).sum(-1) / math.sqrt(c)
            logits = logits + eb(edge).squeeze(-1)
            logits = logits.masked_fill(~adjacency, -1e4)
            attention = F.softmax(logits, dim=-1)
            message = (attention[..., None] * (value[:, None] + ev(edge))).sum(2)
            z = block(z, message)
        gate = torch.sigmoid(self.final_gate(torch.cat((z, appearance), dim=-1)))
        return gate * z + (1.0 - gate) * appearance

    def forward(self, queries, boxes, text, text_mask=None, visuals=None, visual_masks=None):
        queries, boxes, filter_logits, selected = self._filter(queries, boxes, text, text_mask)
        appearance = self._appearance(queries, boxes, text, text_mask, visuals, visual_masks)
        graph = self._topology(appearance, boxes, self._sentence_pool(text, text_mask))
        return graph, boxes, filter_logits, selected
