"""Original text selects visual sectors for candidate/background MLP fusion.

Boxes use normalized cxcywh against the valid image. Each exterior sector spans
from the box to the valid image edge. Detached geometry uses input pixels;
grid_sample spans the padded feature canvas.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BackgroundContextEnhancement(nn.Module):
    """Let original text select visual sectors, then concatenate and fuse.

    Explicit sector position is disabled by default; it is an optional ablation.
    """

    direction_names = ("up", "down", "left", "right", "upper_left",
                       "upper_right", "lower_left", "lower_right")
    position_dim = 4
    num_directions = 8

    def __init__(self, hidden_channels=256, context_dim=128,
                 angular_samples=5, radial_samples=5,
                 min_valid_points=5, fusion_init=0.1, chunk_size=32,
                 use_position=False, use_text_attention=True, use_learnable_fusion=True,
                 use_sparse_selection=True):
        super().__init__()
        if hidden_channels < 1 or context_dim < 1:
            raise ValueError("BCEM channel dimensions must be positive")
        for name, value in (("angular_samples", angular_samples),
                            ("radial_samples", radial_samples), ("chunk_size", chunk_size)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not isinstance(min_valid_points, int) or not 1 <= min_valid_points <= angular_samples * radial_samples:
            raise ValueError("min_valid_points must fit inside the sector sampling grid")
        if not math.isfinite(fusion_init) or not 0 < fusion_init < 1:
            raise ValueError("fusion_init must be a fraction in (0, 1)")
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.angular_samples = angular_samples
        self.radial_samples = radial_samples
        self.num_points = angular_samples * radial_samples
        self.min_valid_points = min_valid_points
        self.fusion_init = float(fusion_init)
        self.chunk_size = chunk_size
        self.use_position = bool(use_position)
        self.use_text_attention = bool(use_text_attention)
        self.use_learnable_fusion = bool(use_learnable_fusion)
        self.use_sparse_selection = bool(use_sparse_selection)

        angles = torch.tensor([-math.pi / 2, math.pi / 2, math.pi, 0,
                               -3 * math.pi / 4, -math.pi / 4,
                               3 * math.pi / 4, math.pi / 4], dtype=torch.float32)
        # Midpoints cover each 45-degree sector without duplicating its edges.
        offsets = ((torch.arange(angular_samples) + 0.5) / angular_samples - 0.5) * (math.pi / 4)
        sub_angles = angles[:, None] + offsets[None]
        self.register_buffer("directions", torch.stack((angles.cos(), angles.sin()), -1), persistent=False)
        self.register_buffer("sector_rays", torch.stack((sub_angles.cos(), sub_angles.sin()), -1), persistent=False)
        self.register_buffer("radial_fractions", (torch.arange(radial_samples) + 0.5) / radial_samples,
                             persistent=False)
        self.visual_projection = nn.Conv2d(hidden_channels, context_dim, 1, bias=False)
        self.visual_norm = nn.LayerNorm(context_dim)
        self.position_mlp = self._mlp(self.position_dim, context_dim) if self.use_position else None
        self.sector_fusion = self._mlp(context_dim * (1 + int(self.use_position)), context_dim)
        needs_text = self.use_text_attention or self.use_learnable_fusion
        self.text_projection = nn.Linear(hidden_channels, context_dim, bias=False) if needs_text else None
        self.sector_key = nn.Linear(context_dim, context_dim, bias=False) if self.use_text_attention else None
        self.sector_value = nn.Linear(context_dim, context_dim, bias=False)
        self.environment_projection = nn.Linear(context_dim, hidden_channels, bias=False)
        fusion_inputs = 2 * hidden_channels + context_dim
        self.fusion_gate = self._mlp(fusion_inputs, 1) if self.use_learnable_fusion else None
        if self.fusion_gate is not None:
            # Initially supply a small background input. This does not make the
            # separate fusion MLP an identity on the original candidate.
            nn.init.normal_(self.fusion_gate[-1].weight, std=1e-3)
            nn.init.constant_(self.fusion_gate[-1].bias, math.log(fusion_init / (1 - fusion_init)))
        # Both learned- and fixed-gate variants learn the same nonlinear fusion.
        self.feature_fusion = self._mlp(2 * hidden_channels, hidden_channels)

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
        exit_distance = ((sizes / 2)[:, :, None, None] /
                         rays.abs().clamp(min=1e-7)[None, None]).min(-1).values
        # Each sub-ray ends at its own intersection with the valid image.
        # Directions parallel to an image side impose no bound on that axis.
        axis_distance = torch.where(rays[None, None] > 0,
                                    image_xy[:, :, None, None] - centers[:, :, None, None],
                                    centers[:, :, None, None])
        to_image_edge = torch.where(
            rays.abs()[None, None] > 1e-7,
            axis_distance / rays.abs().clamp(min=1e-7)[None, None],
            torch.full_like(axis_distance, float("inf")))
        image_exit = to_image_edge.min(-1).values
        span = (image_exit - exit_distance).clamp(min=0)
        distance = (exit_distance[..., None] +
                    span[..., None] * self.radial_fractions.float()[None, None, None, None])
        points = (centers[:, :, None, None, None] +
                  distance[..., None] * rays[None, None, :, :, None])
        b, n = boxes.shape[:2]
        points = points.reshape(b, n, 8, self.num_points, 2)
        # The polar area element is r dr dtheta. Radial span varies across rays,
        # so midpoint quadrature must retain both r and each ray's span.
        area_weights = (distance * span[..., None]).reshape(b, n, 8, self.num_points)
        return points, area_weights, centers, sizes, bounds

    def _sample_sectors(self, visual, boxes, image_sizes, pad_sizes, visual_mask):
        points, area_weights, centers, sizes, bounds = self._sampling_geometry(boxes, image_sizes)
        b, n = boxes.shape[:2]
        image_xy = image_sizes.flip(-1)
        inside_image = ((points >= 0) & (points < image_xy[:, None, None, None])).all(-1)
        inside_box = ((points >= bounds[:, :, None, None, :2]) &
                      (points <= bounds[:, :, None, None, 2:])).all(-1)
        grid = 2 * points / pad_sizes.flip(-1)[:, None, None, None] - 1
        grid = grid.reshape(b, n * 8 * self.num_points, 1, 2).to(visual.dtype)
        sampled = F.grid_sample(visual, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        support = F.grid_sample((~visual_mask[:, None]).to(visual.dtype), grid,
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, n, 8, self.num_points, self.context_dim)
        support = support.reshape(b, n, 8, self.num_points)
        point_valid = inside_image & ~inside_box & (support >= 1 - 1e-6)
        point_valid = point_valid & (sizes > 0).all(-1)[:, :, None, None]
        counts = point_valid.sum(-1)
        valid = counts >= self.min_valid_points
        weights = area_weights * point_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = (sampled * weights[..., None].to(sampled.dtype)).sum(-2)
        features = self.visual_norm(pooled).masked_fill(~valid[..., None], 0)
        centroid = (points * weights[..., None]).sum(-2)
        relative = (centroid - centers[:, :, None]) / image_sizes.min(-1).values[:, None, None, None]
        relative = relative.masked_fill(~valid[..., None], 0)
        direction = self.directions.float()[:, [1, 0]][None, None].expand(b, n, -1, -1)
        position = torch.cat((direction, relative), -1).to(features.dtype)
        return features, position, valid, counts.float() / self.num_points

    @staticmethod
    def _sparse_sector_weights(weights, selectable):
        """Project sentence relevance onto the simplex with exact zero weights.

        A fixed factor of two sharpens relevance before projection; projecting
        probabilities without scaling would just return the dense input.
        Sparsemax can retain any number of sectors, including all;
        it neither fixes top-k nor assigns a minimum weight to excluded sectors.
        Empty rows remain zero. Sorting costs little for just eight sectors.
        """
        scores = 2 * weights
        scores = scores.masked_fill(~selectable, -1e4)
        scores = scores - scores.max(-1, keepdim=True).values
        scores = scores.masked_fill(~selectable, -1e4)
        ordered = scores.sort(-1, descending=True).values
        ranks = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
        cumulative = ordered.cumsum(-1)
        support_size = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True).clamp(min=1)
        threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size
        return (scores - threshold).clamp(min=0).masked_fill(~selectable, 0)

    def _read_sectors(self, features, position, valid, text_queries, text_valid):
        # Only visual and position enter values. Text selects which values to read.
        pieces = [features]
        if self.use_position:
            pieces.append(self.position_mlp(position))
        evidence = F.layer_norm(self.sector_fusion(torch.cat(pieces, -1)), (self.context_dim,))
        active = text_valid[:, None, :, None] & valid[:, :, None, :]
        if self.use_text_attention:
            keys = self.sector_key(evidence)
            logits = (torch.einsum('blc,bndc->bnld', text_queries.float(), keys.float())
                      / math.sqrt(self.context_dim)).float()
            # Normalize over sectors, with safe all-invalid and padding rows.
            logits = logits.masked_fill(~active, -torch.finfo(logits.dtype).max)
            attention = logits.softmax(-1).masked_fill(~active, 0)
        else:
            attention = active.float() / valid.sum(-1)[:, :, None, None].clamp(min=1)
        # Aggregate token relevance first, then select the final set of sectors.
        # Sparsifying only per-token rows could let their union retain all eight.
        weights = attention.sum(2) / text_valid.sum(-1)[:, None, None].clamp(min=1)
        if self.use_text_attention and self.use_sparse_selection:
            selectable = valid & text_valid.any(-1)[:, None, None]
            weights = self._sparse_sector_weights(weights, selectable)
        values = self.sector_value(evidence).masked_fill(~valid[..., None], 0)
        context = (values * weights[..., None].to(values.dtype)).sum(2)
        return context, weights, attention

    def _fuse_context(self, context, queries, sentence, can_use):
        query_norm = F.layer_norm(queries, (self.hidden_channels,))
        background_norm = F.layer_norm(self.environment_projection(context), (self.hidden_channels,))
        # Match the original query RMS before supplying a separate background input.
        # The detached scale is a reference, not an additional trainable weight.
        query_scale = queries.detach().float().norm(dim=-1, keepdim=True) / math.sqrt(self.hidden_channels)
        background = (background_norm.float() * query_scale).to(queries.dtype)
        if self.use_learnable_fusion:
            pieces = [query_norm, background_norm]
            pieces.append(sentence[:, None].expand(-1, queries.shape[1], -1))
            gate = self.fusion_gate(torch.cat(pieces, -1)).sigmoid().to(queries.dtype)
        else:
            gate = queries.new_full((*queries.shape[:2], 1), self.fusion_init)
        gate = gate.masked_fill(~can_use[..., None], 0)
        enhanced = self.feature_fusion(torch.cat((queries, gate * background), -1)).to(queries.dtype)
        # Explicitly preserve identity even when all sectors/text are invalid.
        enhanced = torch.where(can_use[..., None], enhanced, queries)
        return enhanced, gate

    def forward(self, visual, boxes, queries, text, text_mask, img_metas, visual_mask=None):
        """Return enhanced queries and detached eight-sector diagnostics."""
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
        has_text = text_valid.any(1)
        sentence, text_tokens = None, None
        if self.text_projection is not None:
            clean_text = text.masked_fill(~text_valid[..., None], 0)
            text_tokens = self.text_projection(clean_text).masked_fill(~text_valid[..., None], 0)
            sentence = text_tokens.sum(1) / text_valid.sum(1).clamp(min=1)[:, None].to(text_tokens.dtype)

        contexts, direction_weights, valid_sectors, coverage, text_attention = [], [], [], [], []
        for start in range(0, n, self.chunk_size):
            features, position, valid, fraction = self._sample_sectors(
                projected, boxes[:, start:start + self.chunk_size], image_sizes, pad_sizes, visual_mask)
            context, weights, attention = self._read_sectors(features, position, valid, text_tokens, text_valid)
            contexts.append(context)
            direction_weights.append(weights)
            valid_sectors.append(valid)
            coverage.append(fraction)
            text_attention.append(attention)
        context = torch.cat(contexts, 1) if n else queries.new_empty((b, 0, self.context_dim))
        valid_sectors = (torch.cat(valid_sectors, 1) if n
                         else torch.empty((b, 0, 8), dtype=torch.bool, device=queries.device))
        direction_weights = (torch.cat(direction_weights, 1) if n
                             else queries.new_empty((b, 0, 8)))
        text_attention = (torch.cat(text_attention, 1) if n
                          else queries.new_empty((b, 0, text.shape[1], 8)))
        dense_weights = text_attention.sum(2) / text_valid.sum(-1)[:, None, None].clamp(min=1)
        selected = direction_weights > 0
        can_use = valid_sectors.any(-1) & has_text[:, None]
        enhanced, gate = self._fuse_context(context, queries, sentence, can_use)
        correction = enhanced - queries
        ratio = correction.float().norm(dim=-1, keepdim=True) / queries.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return enhanced, {
            "bcem_context": context.detach(),
            "bcem_direction_weights": direction_weights.detach(),
            "bcem_direction_weights_dense": dense_weights.detach(),
            "bcem_selected_sectors": selected.detach(),
            "bcem_selected_sector_count": selected.sum(-1).detach(),
            "bcem_text_sector_attention": text_attention.detach(),
            "bcem_background_weight": gate.detach(),
            "bcem_valid_sectors": valid_sectors.detach(),
            "bcem_sector_coverage": (torch.cat(coverage, 1).detach()
                                     if n else queries.new_empty((b, 0, 8))),
            "bcem_residual_ratio": ratio.detach(),
        }
