"""Sector/radial context, ordered transitions and gated residual enhancement.

Geometry uses the valid image and detached normalized cxcywh boxes. Each ray
extends from the candidate boundary to its own image-edge intersection. Region
pooling is midpoint quadrature, not a dense segmentation or pure-background mask.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class BackgroundContextEnhancement(nn.Module):
    """Eight sectors, four radial regions and three recurrent transitions each.

    Original text reads all valid evidence. Evidence relevance is grouped by
    sector before sparsemax; a zero-valued null option permits no background.
    The candidate remains a residual identity, never a concatenation-MLP output.
    """

    direction_names = ("up", "down", "left", "right", "upper_left",
                       "upper_right", "lower_left", "lower_right")
    num_directions = 8
    position_dim = 9

    def __init__(self, hidden_channels=256, context_dim=128, num_heads=4,
                 radial_edges=(0., .1, .25, .5, 1.), angular_samples=5,
                 radial_samples=2, min_valid_points=5, context_strength=.1,
                 chunk_size=32, use_position=True, use_transitions=True,
                 use_gru=True, use_text_attention=True, use_sparse_selection=True,
                 learnable_gate=True, checkpoint_chunks=True):
        super().__init__()
        for name, value in (("hidden_channels", hidden_channels), ("context_dim", context_dim),
                            ("num_heads", num_heads), ("angular_samples", angular_samples),
                            ("radial_samples", radial_samples), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if context_dim % num_heads:
            raise ValueError("context_dim must be divisible by num_heads")
        edges = tuple(float(value) for value in radial_edges)
        if (len(edges) < 3 or not all(math.isfinite(x) for x in edges)
                or edges[0] != 0. or edges[-1] != 1.
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError("radial_edges must strictly increase from 0 to 1 with at least two regions")
        if (isinstance(min_valid_points, bool) or not isinstance(min_valid_points, int)
                or not 1 <= min_valid_points <= angular_samples * radial_samples):
            raise ValueError("min_valid_points must fit inside each radial region's sampling grid")
        if not math.isfinite(context_strength) or context_strength < 0:
            raise ValueError("context_strength must be finite and nonnegative")
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_regions = len(edges) - 1
        self.num_nodes = 8 * self.num_regions
        self.use_position = bool(use_position)
        self.use_transitions = bool(use_transitions)
        self.use_gru = bool(use_gru and use_transitions)
        self.num_transitions = 8 * (self.num_regions - 1) if self.use_transitions else 0
        self.evidence_per_sector = self.num_regions + (self.num_regions - 1 if self.use_transitions else 0)
        self.angular_samples = angular_samples
        # Samples per radial region, rather than per complete sector.
        self.radial_samples = radial_samples
        self.num_points = angular_samples * radial_samples
        self.min_valid_points = min_valid_points
        self.context_strength = float(context_strength)
        self.chunk_size = chunk_size
        self.use_text_attention = bool(use_text_attention)
        self.use_sparse_selection = bool(use_sparse_selection)
        self.learnable_gate = bool(learnable_gate)
        self.checkpoint_chunks = bool(checkpoint_chunks)

        angles = torch.tensor([-math.pi / 2, math.pi / 2, math.pi, 0,
                               -3 * math.pi / 4, -math.pi / 4,
                               3 * math.pi / 4, math.pi / 4], dtype=torch.float32)
        offsets = ((torch.arange(angular_samples) + .5) / angular_samples - .5) * (math.pi / 4)
        sub_angles = angles[:, None] + offsets[None]
        self.register_buffer("sector_rays", torch.stack((sub_angles.cos(), sub_angles.sin()), -1), persistent=False)
        edge_tensor = torch.tensor(edges, dtype=torch.float32)
        self.register_buffer("radial_edges", edge_tensor, persistent=False)
        fractions = edge_tensor[:-1, None] + edge_tensor.diff()[:, None] * (torch.arange(radial_samples) + .5) / radial_samples
        self.register_buffer("radial_fractions", fractions, persistent=False)
        self.visual_projection = nn.Conv2d(hidden_channels, context_dim, 1, bias=False)
        self.visual_norm = nn.LayerNorm(context_dim)
        self.position_mlp = self._mlp(self.position_dim, context_dim) if self.use_position else None
        self.region_fusion = self._mlp(context_dim * (1 + int(self.use_position)), context_dim)
        self.node_type = nn.Parameter(torch.randn(context_dim) * .02)
        if self.use_transitions:
            self.pair_position = self._mlp(2 * self.position_dim + 1, context_dim) if self.use_position else None
            self.transition_mlp = self._mlp(context_dim * (3 + int(self.use_position)), context_dim)
            self.transition_gru = nn.GRUCell(context_dim, context_dim) if self.use_gru else None
            self.transition_type = nn.Parameter(torch.randn(context_dim) * .02)
        else:
            self.pair_position = self.transition_mlp = self.transition_gru = None
            self.register_parameter("transition_type", None)
        self.text_projection = nn.Linear(hidden_channels, context_dim, bias=False) if self.use_text_attention else None
        self.evidence_key = nn.Linear(context_dim, context_dim, bias=False) if self.use_text_attention else None
        self.evidence_value = nn.Linear(context_dim, context_dim, bias=False)
        self.null_key = nn.Parameter(torch.randn(context_dim) * .02) if self.use_text_attention else None
        self.reader_output = nn.Linear(context_dim, context_dim, bias=False)
        self.environment_projection = nn.Linear(context_dim, hidden_channels, bias=False)
        # Same inputs as the first version: candidate, context, original sentence.
        self.gate = self._mlp(2 * hidden_channels + context_dim, 1) if self.learnable_gate else None

    def _mlp(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, self.context_dim), nn.GELU(),
                             nn.Linear(self.context_dim, output_dim))

    @staticmethod
    def _image_sizes(img_metas, visual):
        if img_metas is None or len(img_metas) != visual.shape[0]:
            raise ValueError("BCEM needs img_shape/pad_shape for every image")
        valid = visual.new_tensor([meta["img_shape"][:2] for meta in img_metas], dtype=torch.float32)
        padded = visual.new_tensor([meta["pad_shape"][:2] for meta in img_metas], dtype=torch.float32)
        if not torch.isfinite(valid).all() or not torch.isfinite(padded).all():
            raise ValueError("BCEM image sizes must be finite")
        if (valid <= 0).any() or (padded < valid).any():
            raise ValueError("BCEM image sizes must be positive and fit inside pad_shape")
        return valid, padded

    @staticmethod
    def _feature_mask(visual, image_sizes, pad_sizes):
        h, w = visual.shape[-2:]
        valid_h = torch.ceil(image_sizes[:, 0] * h / pad_sizes[:, 0])
        valid_w = torch.ceil(image_sizes[:, 1] * w / pad_sizes[:, 1])
        yy = torch.arange(h, device=visual.device)[None, :, None]
        xx = torch.arange(w, device=visual.device)[None, None, :]
        return (yy >= valid_h[:, None, None]) | (xx >= valid_w[:, None, None])

    def _sampling_geometry(self, boxes, image_sizes):
        boxes = boxes.detach().float()
        image_xy = image_sizes.flip(-1)[:, None]
        centers = boxes[..., :2] * image_xy
        sizes = boxes[..., 2:].clamp(min=0) * image_xy
        bounds = torch.cat((centers - sizes / 2, centers + sizes / 2), -1)
        rays = self.sector_rays.float()
        exit_distance = ((sizes / 2)[:, :, None, None] / rays.abs().clamp(min=1e-7)[None, None]).min(-1).values
        axis_distance = torch.where(rays[None, None] > 0,
                                    image_xy[:, :, None, None] - centers[:, :, None, None],
                                    centers[:, :, None, None])
        image_exit = torch.where(
            rays.abs()[None, None] > 1e-7,
            axis_distance / rays.abs().clamp(min=1e-7)[None, None],
            torch.full_like(axis_distance, float("inf"))).min(-1).values
        span = (image_exit - exit_distance).clamp(min=0)
        widths = self.radial_edges.diff().float()
        distance = exit_distance[..., None, None] + span[..., None, None] * self.radial_fractions.float()
        points = centers[:, :, None, None, None, None] + distance[..., None] * rays[None, None, :, :, None, None]
        # r dr dtheta: retain both ray length and each region's fractional width.
        region_width = span[..., None, None] * widths[:, None]
        area_weights = distance * region_width
        b, n = boxes.shape[:2]
        shape = (b, n, 8, self.num_regions, self.num_points)
        points = points.permute(0, 1, 2, 4, 3, 5, 6).reshape(*shape, 2)
        area_weights = area_weights.permute(0, 1, 2, 4, 3, 5).reshape(shape)
        region_width = region_width.expand_as(distance).permute(0, 1, 2, 4, 3, 5).reshape(shape)
        return points, area_weights, centers, sizes, bounds, region_width

    def _sample_sectors(self, visual, boxes, image_sizes, pad_sizes, visual_mask):
        points, area_weights, centers, sizes, bounds, region_width = self._sampling_geometry(boxes, image_sizes)
        b, n = boxes.shape[:2]
        image_xy = image_sizes.flip(-1)
        inside_image = ((points >= 0) & (points < image_xy[:, None, None, None, None])).all(-1)
        inside_box = ((points >= bounds[:, :, None, None, None, :2]) &
                      (points <= bounds[:, :, None, None, None, 2:])).all(-1)
        grid = 2 * points / pad_sizes.flip(-1)[:, None, None, None, None] - 1
        grid = grid.reshape(b, n * self.num_nodes * self.num_points, 1, 2)
        # float32 sampling also supports CPU autocast; cast pooled features back.
        sampled = F.grid_sample(visual.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        support = F.grid_sample((~visual_mask[:, None]).float(), grid,
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, n, 8, self.num_regions, self.num_points, self.context_dim)
        support = support.reshape(b, n, 8, self.num_regions, self.num_points)
        center_valid = ((centers >= 0) & (centers <= image_xy[:, None])).all(-1)
        point_valid = inside_image & ~inside_box & (support >= 1 - 1e-6) & (area_weights > 0)
        point_valid &= ((sizes > 0).all(-1) & center_valid)[:, :, None, None, None]
        counts = point_valid.sum(-1)
        valid = counts >= self.min_valid_points
        weights = area_weights.masked_fill(~point_valid, 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = (sampled.masked_fill(~point_valid[..., None], 0) * weights[..., None]).sum(-2)
        features = self.visual_norm(pooled.to(visual.dtype)).masked_fill(~valid[..., None], 0)
        centroid = (points * weights[..., None]).sum(-2)
        short = image_sizes.min(-1).values[:, None, None, None, None]
        relative = (centroid - centers[:, :, None, None]) / short
        # Actual Euclidean distance to the rectangle, not just a radial index.
        delta = (points - centers[:, :, None, None, None]).abs() - sizes[:, :, None, None, None] / 2
        boundary_distance = (delta.clamp(min=0).norm(dim=-1) * weights).sum(-1, keepdim=True) / short
        physical_width = (region_width * weights).sum(-1, keepdim=True) / short
        absolute = centroid / image_xy[:, None, None, None]
        box_scale = (sizes / image_xy[:, None])[:, :, None, None].expand(-1, -1, 8, self.num_regions, -1)
        coverage = counts.float() / self.num_points
        # dx/S, dy/S, distance/S, width/S, x/W, y/H, box_w/W, box_h/H, coverage.
        position = torch.cat((relative, boundary_distance, physical_width, absolute,
                              box_scale, coverage[..., None]), -1)
        return features, position.masked_fill(~valid[..., None], 0).to(features.dtype), valid, coverage

    def _encode_evidence(self, features, position, node_valid):
        pieces = [features]
        if self.use_position:
            pieces.append(self.position_mlp(position))
        nodes = F.layer_norm(self.region_fusion(torch.cat(pieces, -1)), (self.context_dim,)) + self.node_type
        nodes = nodes.masked_fill(~node_valid[..., None], 0)
        if not self.use_transitions:
            return nodes, node_valid, node_valid[..., :0]
        near, far = features[..., :-1, :], features[..., 1:, :]
        inputs = [near, far, far - near]
        pair_position = None
        if self.use_position:
            gap = position[..., 1:, 2:3] - position[..., :-1, 2:3]
            pair_position = self.pair_position(torch.cat((position[..., :-1, :], position[..., 1:, :], gap), -1))
            inputs.append(pair_position)
        changes = self.transition_mlp(torch.cat(inputs, -1))
        transition_valid = node_valid[..., :-1] & node_valid[..., 1:]
        if self.use_gru:
            state = changes.new_zeros((*changes.shape[:3], self.context_dim)).reshape(-1, self.context_dim)
            outputs = []
            for step in range(self.num_regions - 1):
                state = self.transition_gru(changes[..., step, :].reshape(-1, self.context_dim), state)
                # A missing adjacent observation breaks the radial chain.
                state = state.masked_fill(~transition_valid[..., step, None].reshape(-1, 1), 0)
                outputs.append(state.reshape(*changes.shape[:3], self.context_dim))
            changes = torch.stack(outputs, -2)
        if pair_position is not None:
            changes = changes + pair_position
        changes = (changes + self.transition_type).masked_fill(~transition_valid[..., None], 0)
        # Keep raw regions AND all recurrent states; never only the last state.
        return torch.cat((nodes, changes), -2), torch.cat((node_valid, transition_valid), -1), transition_valid

    @staticmethod
    def _sparse_sector_weights(weights, selectable):
        """Sparsemax(2 * relevance) over sectors AND the null option.

        Returns exact zeros for excluded entries and all-invalid rows. No top-k.
        The factor two is a fixed starting setting, not a semantic guarantee.
        """
        scores = (2 * weights).masked_fill(~selectable, -1e4)
        scores = scores - scores.max(-1, keepdim=True).values
        scores = scores.masked_fill(~selectable, -1e4)
        ordered = scores.sort(-1, descending=True).values
        ranks = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
        cumulative = ordered.cumsum(-1)
        count = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True).clamp(min=1)
        threshold = (cumulative.gather(-1, count - 1) - 1) / count
        return (scores - threshold).clamp(min=0).masked_fill(~selectable, 0)

    def _read_evidence(self, evidence, evidence_valid, text_queries, text_valid):
        b, n, directions, length, dim = evidence.shape
        total = directions * length
        head_dim = dim // self.num_heads
        sector_valid = evidence_valid.any(-1)
        clean = evidence.masked_fill(~evidence_valid[..., None], 0)
        value = self.evidence_value(clean).reshape(b, n, total, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
        allowed = torch.cat((evidence_valid.reshape(b, n, total), sector_valid.new_ones(b, n, 1)), -1)
        active = allowed[:, :, None, None] & text_valid[:, None, None, :, None]
        if self.use_text_attention:
            key = self.evidence_key(clean).reshape(b, n, total, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)
            null = self.null_key.reshape(1, 1, self.num_heads, 1, head_dim).expand(b, n, -1, -1, -1)
            key = torch.cat((key, null.to(key.dtype)), -2)
            query = text_queries.reshape(b, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            logits = torch.einsum('bhlc,bnhec->bnhle', query.float(), key.float()) / math.sqrt(head_dim)
            attention = logits.masked_fill(~active, -torch.finfo(logits.dtype).max).softmax(-1)
            attention = attention.masked_fill(~active, 0)
        else:
            # The uniform ablation balances sectors, despite unequal evidence counts.
            local = evidence_valid.float() / evidence_valid.sum(-1, keepdim=True).clamp(min=1)
            local = local / sector_valid.sum(-1)[..., None, None].clamp(min=1)
            null = (~sector_valid.any(-1)).float()[..., None]
            uniform = torch.cat((local.reshape(b, n, total), null), -1)
            attention = uniform[:, :, None, None].expand(-1, -1, self.num_heads, text_valid.shape[1], -1)
            attention = attention.masked_fill(~active, 0)
        mean = attention.sum(-2) / text_valid.sum(-1)[:, None, None, None].clamp(min=1)
        grouped = mean[..., :-1].reshape(b, n, self.num_heads, directions, length)
        dense = torch.cat((grouped.sum(-1).mean(2), mean[..., -1:].mean(2)), -1)
        selectable = torch.cat((sector_valid, sector_valid.new_ones(b, n, 1)), -1)
        selectable &= text_valid.any(-1)[:, None, None]
        selected = (self._sparse_sector_weights(dense, selectable)
                    if self.use_text_attention and self.use_sparse_selection else dense)
        # Reweight each sector's ORIGINAL attention contribution once. With sparse
        # selection disabled this reproduces the ordinary multi-head text read.
        sector_value = value.reshape(b, n, self.num_heads, directions, length, head_dim)
        contributions = (grouped[..., None] * sector_value.float()).sum(-2)
        contributions = contributions.permute(0, 1, 3, 2, 4).reshape(b, n, directions, dim)
        sector_read = self.reader_output(contributions.to(evidence.dtype))
        conditional = sector_read.float() / dense[..., :-1, None].clamp(min=1e-12)
        context = (conditional * selected[..., :-1, None]).sum(-2).to(evidence.dtype)
        token_sector = attention[..., :-1].reshape(b, n, self.num_heads, text_valid.shape[1], directions, length).sum(-1).mean(2)
        token_null = attention[..., -1:].mean(2)
        return context, selected[..., :-1], dense[..., :-1], token_sector, selected[..., -1:], dense[..., -1:], token_null

    def _forward_chunk(self, visual, boxes, text_queries, text_valid, image_sizes, pad_sizes, visual_mask):
        features, position, valid, coverage = self._sample_sectors(visual, boxes, image_sizes, pad_sizes, visual_mask)
        evidence, evidence_valid, transition_valid = self._encode_evidence(features, position, valid)
        return (*self._read_evidence(evidence, evidence_valid, text_queries, text_valid), valid, transition_valid, coverage)

    def _fuse_context(self, context, queries, sentence, can_use):
        if self.learnable_gate:
            sentence = sentence[:, None].expand(-1, queries.shape[1], -1)
            gate = self.gate(torch.cat((queries, context, sentence), -1)).sigmoid().to(queries.dtype)
        else:
            gate = queries.new_ones((*queries.shape[:2], 1))
        gate = gate.masked_fill(~can_use[..., None], 0)
        correction = self.context_strength * gate * self.environment_projection(context)
        enhanced = queries + correction
        return torch.where(can_use[..., None], enhanced, queries), gate

    def forward(self, visual, boxes, queries, text, text_mask, img_metas, visual_mask=None):
        b, n, channels = queries.shape
        if channels != self.hidden_channels or boxes.shape != (b, n, 4):
            raise ValueError("BCEM queries and normalized cxcywh boxes must share candidate indices")
        if visual.shape[:2] != (b, channels) or text.shape[0] != b or text.shape[-1] != channels:
            raise ValueError("BCEM visual/text/query channels and batch sizes must agree")
        if text.shape[1] == 0 or (text_mask is not None and text_mask.shape != text.shape[:2]):
            raise ValueError("BCEM needs a nonempty token axis and a matching text mask")
        image_sizes, pad_sizes = self._image_sizes(img_metas, visual)
        if visual_mask is None:
            visual_mask = self._feature_mask(visual, image_sizes, pad_sizes)
        if visual_mask.shape != (b, *visual.shape[-2:]):
            raise ValueError("BCEM visual_mask must match the selected feature map")
        visual_mask = visual_mask.bool()
        projected = self.visual_projection(visual.masked_fill(visual_mask[:, None], 0))
        projected = projected.masked_fill(visual_mask[:, None], 0)
        text_valid = (torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
                      if text_mask is None else ~text_mask.bool())
        clean_text = text.masked_fill(~text_valid[..., None], 0)
        sentence = clean_text.sum(1) / text_valid.sum(-1)[:, None].clamp(min=1).to(text.dtype)
        text_queries = self.text_projection(clean_text) if self.use_text_attention else clean_text
        chunks = []
        for start in range(0, n, self.chunk_size):
            args = (projected, boxes[:, start:start + self.chunk_size].detach(), text_queries,
                    text_valid, image_sizes, pad_sizes, visual_mask)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._forward_chunk, *args, use_reentrant=False))
            else:
                chunks.append(self._forward_chunk(*args))
        if chunks:
            context, weights, dense, token_attention, null, null_dense, token_null, regions, transitions, coverage = [
                torch.cat(parts, 1) for parts in zip(*chunks)]
        else:
            context = queries.new_empty(b, 0, self.context_dim)
            weights = dense = queries.new_empty(b, 0, 8)
            token_attention = queries.new_empty(b, 0, text.shape[1], 8)
            null = null_dense = queries.new_empty(b, 0, 1)
            token_null = queries.new_empty(b, 0, text.shape[1], 1)
            regions = torch.empty(b, 0, 8, self.num_regions, dtype=torch.bool, device=queries.device)
            transitions = regions[..., :self.num_regions - 1] if self.use_transitions else regions[..., :0]
            coverage = queries.new_empty(b, 0, 8, self.num_regions)
        valid_sectors = regions.any(-1)
        can_use = valid_sectors.any(-1) & text_valid.any(-1)[:, None] & (weights.sum(-1) > 0)
        enhanced, gate = self._fuse_context(context, queries, sentence, can_use)
        correction = enhanced - queries
        ratio = correction.float().norm(dim=-1, keepdim=True) / queries.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        selected = weights > 0
        return enhanced, {
            "bcem_context": context.detach(),
            "bcem_direction_weights": weights.detach(),
            "bcem_direction_weights_dense": dense.detach(),
            "bcem_null_weight": null.detach(),
            "bcem_null_weight_dense": null_dense.detach(),
            "bcem_selected_sectors": selected.detach(),
            "bcem_selected_sector_count": selected.sum(-1).detach(),
            "bcem_text_sector_attention": token_attention.detach(),
            "bcem_text_null_attention": token_null.detach(),
            "bcem_background_weight": gate.detach(),
            "bcem_valid_sectors": valid_sectors.detach(),
            "bcem_valid_regions": regions.detach(),
            "bcem_valid_transitions": transitions.detach(),
            "bcem_region_coverage": coverage.detach(),
            "bcem_sector_coverage": coverage.mean(-1).detach(),
            "bcem_residual_ratio": ratio.detach(),
        }
