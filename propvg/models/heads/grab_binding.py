"""GrabVG binding structure from the detailed paper, Sections 3.2--3.3/A.2.

Losses and teacher supervision are deliberately owned by the existing caller.
The initialization-only legacy layout preserves disabled-Grab experiments.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def _gather_nodes(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class GraphBlock(nn.Module):
    def __init__(self, dim, project_message=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.out_proj = nn.Linear(dim, dim) if project_message else nn.Identity()

    def forward(self, x, message):
        x = self.norm1(x + self.out_proj(message))
        return self.norm2(x + self.ffn(x))


class GrabGraphBinding(nn.Module):
    def __init__(self, hidden_channels=256, num_heads=8, k=6, layers=4, anchors=48, keep_ratio=0.5,
                 legacy_layout=False):
        super().__init__()
        # Only the disabled branch's meta descriptor uses this layout. Its
        # parameter shapes, names, order and RNG draws must remain historical.
        self._legacy_layout = legacy_layout
        self.k = k
        self.keep_ratio = keep_ratio
        self.num_anchors = anchors
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.edge_channels = hidden_channels if legacy_layout else 64

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
        if legacy_layout:
            yy, xx = torch.meshgrid(torch.linspace(-0.5, 0.5, 6), torch.linspace(-0.5, 0.5, 8), indexing="ij")
            base_points = torch.stack((xx, yy), dim=-1).reshape(anchors, 2)
        else:
            base_points = self._square_ring_points(anchors)
        self.register_buffer("base_points", base_points, persistent=False)
        self.anchor_text_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        if not legacy_layout:
            self.anchor_text_norm1 = nn.LayerNorm(hidden_channels)
            self.anchor_text_ffn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 4), nn.GELU(), nn.Linear(hidden_channels * 4, hidden_channels)
            )
            self.anchor_text_norm2 = nn.LayerNorm(hidden_channels)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, 2)
        )
        if legacy_layout:
            self.level_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(4)])
        else:
            self.level_fusion = nn.Linear(hidden_channels * 4, hidden_channels)
        self.appearance_attn = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.appearance_norm = nn.LayerNorm(hidden_channels)
        if not legacy_layout:
            self.appearance_ffn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 4), nn.GELU(), nn.Linear(hidden_channels * 4, hidden_channels)
            )
            self.appearance_norm2 = nn.LayerNorm(hidden_channels)

        # Language-conditioned edge-aware graph attention.
        geometry_channels = 7 if legacy_layout else 8
        self.geo_mlp = nn.Sequential(
            nn.Linear(geometry_channels, self.edge_channels), nn.ReLU(), nn.Linear(self.edge_channels, self.edge_channels)
        )
        self.text_edge = nn.Sequential(nn.Linear(hidden_channels, self.edge_channels), nn.Sigmoid())
        self.q_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(layers)])
        self.edge_k = nn.ModuleList([nn.Linear(self.edge_channels, hidden_channels) for _ in range(layers)])
        self.edge_v = nn.ModuleList([nn.Linear(self.edge_channels, hidden_channels) for _ in range(layers)])
        self.edge_bias = nn.ModuleList([nn.Linear(self.edge_channels, 1 if legacy_layout else num_heads) for _ in range(layers)])
        self.blocks = nn.ModuleList([GraphBlock(hidden_channels, project_message=not legacy_layout) for _ in range(layers)])
        if legacy_layout:
            self.final_gate = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.final_gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, hidden_channels)
            )

    @staticmethod
    def _square_ring_points(anchors):
        """Equal-perimeter samples on inner, boundary and 1.2x outer squares.

        The paper specifies the three scales but not point ordering; use an
        equal allocation (16 points per ring for its 48-anchor setting).
        """
        if anchors <= 0 or anchors % 12:
            raise ValueError("Grab anchors must be a positive multiple of 12 for three square rings")
        steps = torch.arange(anchors // 12, dtype=torch.float32) / (anchors // 12)
        half = torch.full_like(steps, 0.5)
        ring = torch.cat((
            torch.stack((steps - 0.5, -half), dim=-1),
            torch.stack((half, steps - 0.5), dim=-1),
            torch.stack((0.5 - steps, half), dim=-1),
            torch.stack((-half, 0.5 - steps), dim=-1),
        ))
        return torch.cat([ring * scale for scale in (0.5, 1.0, 1.2)])

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
        anchor_text = self.anchor_text_norm1(anchor + anchor_text)
        anchor_text = self.anchor_text_norm2(anchor_text + self.anchor_text_ffn(anchor_text))
        anchor = anchor[:, None].expand(-1, n, -1, -1)
        anchor_text = anchor_text[:, None].expand(-1, n, -1, -1)
        proposal = queries[:, :, None].expand(-1, -1, self.num_anchors, -1)
        offsets = self.offset_head(torch.cat((proposal, anchor, anchor_text), dim=-1))
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
            sampled_levels.append(sampled)
        sampled = self.level_fusion(torch.cat(sampled_levels, dim=-1))
        q = queries.reshape(b * n, 1, c)
        memory = torch.cat((q, sampled.reshape(b * n, self.num_anchors, c)), dim=1)
        update, _ = self.appearance_attn(q, memory, memory)
        appearance = self.appearance_norm(q + update)
        appearance = self.appearance_norm2(appearance + self.appearance_ffn(appearance))
        return appearance.reshape(b, n, c)

    @staticmethod
    def _geometry(boxes):
        """r[j, i] describes source i relative to receiver j (paper Eq. 5--6)."""
        centers, sizes = boxes[..., :2], boxes[..., 2:].clamp(min=1e-5)
        delta = centers[:, None, :] - centers[:, :, None]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / (distance[..., None] + 1e-6)
        logscale = torch.log(sizes[:, None, :] / sizes[:, :, None])
        geometry = torch.cat((delta, distance[..., None], torch.log(distance[..., None] + 1e-6),
                              direction, logscale), dim=-1)
        return geometry, distance

    def _adjacency(self, distance):
        """Symmetrized kNN, including every reverse edge and each self-loop."""
        b, n, _ = distance.shape
        eye = torch.eye(n, dtype=torch.bool, device=distance.device)[None]
        nearest = distance.detach().masked_fill(eye, float("inf")).topk(min(self.k, n - 1), largest=False).indices
        adjacency = torch.zeros((b, n, n), dtype=torch.bool, device=distance.device)
        adjacency.scatter_(2, nearest, True)
        return adjacency | adjacency.transpose(1, 2) | eye

    def _topology(self, appearance, boxes, sentence):
        b, n, c = appearance.shape
        geometry, distance = self._geometry(boxes)
        adjacency = self._adjacency(distance)

        edge = self.geo_mlp(geometry) * self.text_edge(sentence)[:, None, None]
        z = appearance
        for qp, kp, vp, ek, ev, eb, block in zip(
            self.q_proj, self.k_proj, self.v_proj, self.edge_k, self.edge_v, self.edge_bias, self.blocks
        ):
            q, key, value = [projection(z).reshape(b, n, self.num_heads, self.head_dim)
                             for projection in (qp, kp, vp)]
            edge_key = ek(edge).reshape(b, n, n, self.num_heads, self.head_dim)
            edge_value = ev(edge).reshape(b, n, n, self.num_heads, self.head_dim)
            # Each head independently attends over source neighbors i for j.
            logits = (torch.einsum("bjhd,bihd->bhji", q, key)
                      + torch.einsum("bjhd,bjihd->bhji", q, edge_key)) / math.sqrt(self.head_dim)
            logits = logits + eb(edge).permute(0, 3, 1, 2)
            logits = logits.masked_fill(~adjacency[:, None], float("-inf"))
            attention = F.softmax(logits, dim=-1)
            message = (torch.einsum("bhji,bihd->bjhd", attention, value)
                       + torch.einsum("bhji,bjihd->bjhd", attention, edge_value))
            z = block(z, message.reshape(b, n, c))
        gate = torch.sigmoid(self.final_gate(torch.cat((appearance, z), dim=-1)))
        return appearance + gate * (z - appearance)

    def forward(self, queries, boxes, text, text_mask=None, visuals=None, visual_masks=None):
        if self._legacy_layout:
            raise RuntimeError("legacy_layout is an initialization-only descriptor, not an active Grab model")
        queries, boxes, filter_logits, selected = self._filter(queries, boxes, text, text_mask)
        appearance = self._appearance(queries, boxes, text, text_mask, visuals, visual_masks)
        graph = self._topology(appearance, boxes, self._sentence_pool(text, text_mask))
        return graph, boxes, filter_logits, selected
