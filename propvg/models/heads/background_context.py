"""Candidate-wise background context enhancement before referent scoring.

Boxes are normalized cxcywh against the valid (unpadded) image. Sampling
coordinates are constructed in input pixels, then mapped to the padded feature
map. Only box coordinates are detached; image, text and query gradients remain.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class BackgroundContextEnhancement(nn.Module):
    """BCEM: 32 exterior regions + 24 ordered transitions by default.

    Ablations share the sampler/reader: ``use_transitions=False`` keeps only
    regions, ``use_gru=False`` keeps unordered pair encodings, and
    ``learnable_gate=False`` fixes the background usage weight to one.
    """

    direction_names = ("up", "down", "left", "right", "upper_left",
                       "upper_right", "lower_left", "lower_right")
    position_dim = 11

    def __init__(self, hidden_channels=256, context_dim=128, num_heads=4,
                 scales=(0.10, 0.15, 0.20, 0.25), chunk_size=32,
                 min_valid_points=5, angle_offset_degrees=15.0,
                 context_strength=0.1, use_transitions=True, use_gru=True,
                 learnable_gate=True, attention_dropout=0.0,
                 checkpoint_chunks=True):
        super().__init__()
        scales = tuple(float(value) for value in scales)
        if len(scales) < 2 or any(not math.isfinite(v) or not 0 < v <= 1 for v in scales):
            raise ValueError("BCEM scales must contain at least two fractions in (0, 1]")
        if any(a >= b for a, b in zip(scales, scales[1:])):
            raise ValueError("BCEM scales must be strictly increasing")
        if context_dim <= 0 or num_heads <= 0 or context_dim % num_heads:
            raise ValueError("context_dim must be positive and divisible by num_heads")
        if chunk_size < 1 or not 1 <= min_valid_points <= 9:
            raise ValueError("chunk_size must be positive and min_valid_points must be in [1, 9]")
        if not 0 <= angle_offset_degrees <= 22.5:
            raise ValueError("angle_offset_degrees must be in [0, 22.5]")
        if not math.isfinite(context_strength) or context_strength < 0:
            raise ValueError("context_strength must be finite and nonnegative")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")

        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.min_valid_points = min_valid_points
        self.context_strength = float(context_strength)
        self.use_transitions = bool(use_transitions)
        self.use_gru = bool(use_gru and use_transitions)
        self.learnable_gate = bool(learnable_gate)
        self.attention_dropout = attention_dropout
        self.checkpoint_chunks = checkpoint_chunks
        self.num_directions = 8
        self.num_scales = len(scales)
        self.num_nodes = self.num_directions * self.num_scales
        self.num_transitions = self.num_directions * (self.num_scales - 1) if use_transitions else 0

        angles = torch.tensor([-math.pi / 2, math.pi / 2, math.pi, 0,
                               -3 * math.pi / 4, -math.pi / 4,
                               3 * math.pi / 4, math.pi / 4], dtype=torch.float32)
        offsets = torch.tensor([-1., 0., 1.]) * math.radians(angle_offset_degrees)
        sub_angles = angles[:, None] + offsets[None]
        self.register_buffer("directions", torch.stack((angles.cos(), angles.sin()), -1), persistent=False)
        self.register_buffer("sub_directions", torch.stack((sub_angles.cos(), sub_angles.sin()), -1), persistent=False)
        scale_tensor = torch.tensor(scales, dtype=torch.float32)
        # Nine samples: three sub-rays, each with three radial locations.
        previous_gap = torch.cat((scale_tensor[:1], scale_tensor.diff()))
        next_gap = torch.cat((scale_tensor.diff(), previous_gap[-1:]))
        half_width = torch.minimum(previous_gap, next_gap) * 0.25
        self.register_buffer("scales", scale_tensor, persistent=False)
        self.register_buffer("radial_half_width", half_width, persistent=False)
        self.register_buffer("radial_offsets", torch.tensor([-1., 0., 1.]), persistent=False)

        self.visual_projection = nn.Conv2d(hidden_channels, context_dim, 1, bias=False)
        self.visual_norm = nn.LayerNorm(context_dim)
        self.node_position = self._mlp(self.position_dim, context_dim)
        self.node_type = nn.Parameter(torch.randn(context_dim) * 0.02)
        if self.use_transitions:
            self.pair_position = self._mlp(self.position_dim * 2 + 1, context_dim)
            self.transition_mlp = self._mlp(context_dim * 4, context_dim)
            self.transition_gru = nn.GRUCell(context_dim, context_dim) if self.use_gru else None
            self.transition_type = nn.Parameter(torch.randn(context_dim) * 0.02)
        else:
            self.pair_position = self.transition_mlp = self.transition_gru = None
            self.register_parameter("transition_type", None)
        self.text_query = nn.Linear(hidden_channels, context_dim, bias=False)
        self.evidence_key = nn.Linear(context_dim, context_dim, bias=False)
        self.evidence_value = nn.Linear(context_dim, context_dim, bias=False)
        # This key lives in the projected attention space. Its value is created
        # as an exact zero *after* the value projection, never as a raw token.
        self.null_key = nn.Parameter(torch.randn(context_dim) * 0.02)
        self.reader_output = nn.Linear(context_dim, context_dim, bias=False)
        self.environment_projection = nn.Linear(context_dim, hidden_channels, bias=False)
        self.gate = (nn.Sequential(nn.Linear(hidden_channels * 2 + context_dim, context_dim),
                                   nn.GELU(), nn.Linear(context_dim, 1))
                     if self.learnable_gate else None)

    def _mlp(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, self.context_dim), nn.GELU(),
                             nn.Linear(self.context_dim, output_dim))

    @staticmethod
    def _image_sizes(img_metas, visual):
        if img_metas is None or len(img_metas) != visual.shape[0]:
            raise ValueError("BCEM needs img_shape/pad_shape for every image")
        valid = visual.new_tensor([meta["img_shape"][:2] for meta in img_metas], dtype=torch.float32)
        padded = visual.new_tensor([meta["pad_shape"][:2] for meta in img_metas], dtype=torch.float32)
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
        # Geometry stays float32 even under mixed precision. Feature and
        # attention arithmetic use the caller's floating point dtype.
        boxes = boxes.detach().float()
        image_xy = image_sizes.flip(-1)[:, None]
        centers, sizes = boxes[..., :2] * image_xy, boxes[..., 2:].clamp(min=0) * image_xy
        half_sizes = sizes / 2
        bounds = torch.cat((centers - half_sizes, centers + half_sizes), -1)
        short_side = image_sizes.min(-1).values
        distance = short_side[:, None] * self.scales.float()[None]
        radial_delta = short_side[:, None, None] * self.radial_half_width.float()[None, :, None]
        radial_distance = distance[..., None] + radial_delta * self.radial_offsets.float()[None, None]
        rays = self.sub_directions.float()
        # Intersect each ray with the original axis-aligned candidate box.
        exit_distance = (half_sizes[:, :, None, None] / rays.abs().clamp(min=1e-7)[None, None]).min(-1).values
        length = exit_distance[:, :, :, None, :, None] + radial_distance[:, None, None, :, None, :]
        points = centers[:, :, None, None, None, None] + length[..., None] * rays[None, None, :, None, :, None]
        points = points.reshape(boxes.shape[0], boxes.shape[1], 8, self.num_scales, 9, 2)
        # Middle angle and middle radial location give the nominal region
        # center. Other samples only increase the local observation footprint.
        nominal = points[..., 4, :]
        return points, nominal, centers, sizes, bounds, distance

    def _sample_regions(self, visual, boxes, image_sizes, pad_sizes, visual_mask):
        points, nominal, centers, sizes, bounds, distance = self._sampling_geometry(boxes, image_sizes)
        b, n = boxes.shape[:2]
        image_xy = image_sizes.flip(-1)
        inside_image = ((points >= 0) & (points < image_xy[:, None, None, None, None])).all(-1)
        inside_box = ((points >= bounds[:, :, None, None, None, :2]) &
                      (points <= bounds[:, :, None, None, None, 2:])).all(-1)
        nonempty_box = (sizes > 0).all(-1)
        grid = (2 * points / pad_sizes.flip(-1)[:, None, None, None, None] - 1)
        grid = grid.reshape(b, n * self.num_nodes * 9, 1, 2).to(visual.dtype)
        sampled = F.grid_sample(visual, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        support = F.grid_sample((~visual_mask[:, None]).to(visual.dtype), grid,
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, n, 8, self.num_scales, 9, self.context_dim)
        support = support.reshape(b, n, 8, self.num_scales, 9)
        # Reject interpolation footprints touching masked padding or outside
        # the feature grid, rather than treating padding zeros as evidence.
        point_valid = inside_image & (~inside_box) & (support >= 1 - 1e-6)
        point_valid = point_valid & nonempty_box[:, :, None, None, None]
        counts = point_valid.sum(-1)
        valid = counts >= self.min_valid_points
        pooled = (sampled * point_valid[..., None].to(sampled.dtype)).sum(-2)
        pooled = pooled / counts.clamp(min=1)[..., None].to(pooled.dtype)
        features = self.visual_norm(pooled).masked_fill(~valid[..., None], 0)

        short = image_sizes.min(-1).values[:, None, None, None, None]
        relative = nominal - centers[:, :, None, None]
        closest_delta = (relative.abs() - sizes[:, :, None, None] / 2).clamp(min=0)
        closest_distance = closest_delta.norm(dim=-1, keepdim=True) / short
        # z = [sin(theta), cos(theta), s/S, rho/S, dx/S, dy/S,
        #      x/W, y/H, box_w/W, box_h/H, valid_fraction].
        direction = self.directions.float()[:, [1, 0]][None, None, :, None].expand(b, n, -1, self.num_scales, -1)
        normalized_distance = (distance / image_sizes.min(-1).values[:, None])[:, None, None, :, None]
        normalized_distance = normalized_distance.expand(b, n, 8, -1, -1)
        absolute = nominal / image_xy[:, None, None, None]
        box_scale = (sizes / image_xy[:, None])[:, :, None, None].expand(-1, -1, 8, self.num_scales, -1)
        position = torch.cat((direction, normalized_distance, closest_distance,
                              relative / short, absolute, box_scale,
                              counts[..., None].float() / 9), -1).to(features.dtype)
        return features, position, valid

    def _encode_evidence(self, features, position, node_valid):
        b, n = features.shape[:2]
        nodes = features + self.node_position(position) + self.node_type
        evidence = [nodes.reshape(b, n, self.num_nodes, self.context_dim)]
        masks = [node_valid.reshape(b, n, self.num_nodes)]
        if not self.use_transitions:
            return evidence[0], masks[0], node_valid.new_empty((b, n, 0))

        near, far = features[:, :, :, :-1], features[:, :, :, 1:]
        gap = position[:, :, :, 1:, 2:3] - position[:, :, :, :-1, 2:3]
        pair_position = self.pair_position(torch.cat((position[:, :, :, :-1],
                                                     position[:, :, :, 1:], gap), -1))
        changes = self.transition_mlp(torch.cat((near, far, far - near, pair_position), -1))
        transition_valid = node_valid[:, :, :, :-1] & node_valid[:, :, :, 1:]
        if self.use_gru:
            state = features.new_zeros((b * n * 8, self.context_dim))
            outputs = []
            for step in range(self.num_scales - 1):
                state = self.transition_gru(changes[:, :, :, step].reshape(-1, self.context_dim), state)
                # Reset at missing observations; zero input alone would still
                # propagate a GRU's recurrent state and learned biases.
                state = state * transition_valid[:, :, :, step].reshape(-1, 1).to(state.dtype)
                outputs.append(state.reshape(b, n, 8, self.context_dim))
            changes = torch.stack(outputs, dim=3)
        transitions = changes + pair_position + self.transition_type
        evidence.append(transitions.reshape(b, n, self.num_transitions, self.context_dim))
        masks.append(transition_valid.reshape(b, n, self.num_transitions))
        return torch.cat(evidence, 2), torch.cat(masks, 2), masks[-1]

    def _read_background(self, evidence, evidence_valid, text_queries, text_valid):
        b, n, length, dim = evidence.shape
        key = self.evidence_key(evidence)
        value = self.evidence_value(evidence)
        key = torch.cat((key, self.null_key.to(key.dtype)[None, None, None].expand(b, n, 1, -1)), 2)
        value = torch.cat((value, value.new_zeros((b, n, 1, dim))), 2)
        allowed = torch.cat((evidence_valid, evidence_valid.new_ones((b, n, 1))), -1)
        head_dim = dim // self.num_heads
        q = text_queries[:, None].expand(-1, n, -1, -1).reshape(b * n, -1, self.num_heads, head_dim).transpose(1, 2)
        k = key.reshape(b * n, length + 1, self.num_heads, head_dim).transpose(1, 2)
        v = value.reshape(b * n, length + 1, self.num_heads, head_dim).transpose(1, 2)
        # SDPA boolean masks use True for permitted entries (opposite to a
        # MultiheadAttention key_padding_mask). The null entry is always valid.
        read = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=allowed.reshape(b * n, 1, 1, length + 1),
                                             dropout_p=self.attention_dropout if self.training else 0.0)
        read = read.transpose(1, 2).reshape(b, n, text_queries.shape[1], dim)
        read = self.reader_output(read)
        valid = text_valid[:, None, :, None].to(read.dtype)
        return (read * valid).sum(2) / valid.sum(2).clamp(min=1)

    def _forward_chunk(self, visual, boxes, queries, text_queries, sentence,
                       text_valid, image_sizes, pad_sizes, visual_mask):
        features, position, node_valid = self._sample_regions(visual, boxes, image_sizes, pad_sizes, visual_mask)
        evidence, allowed, transition_valid = self._encode_evidence(features, position, node_valid)
        context = self._read_background(evidence, allowed, text_queries, text_valid)
        if self.learnable_gate:
            sent = sentence[:, None].expand(-1, queries.shape[1], -1)
            gate = torch.sigmoid(self.gate(torch.cat((queries, context, sent), -1)))
        else:
            gate = context.new_ones((*context.shape[:2], 1))
        return context, gate, node_valid.reshape(*queries.shape[:2], self.num_nodes), transition_valid

    def forward(self, visual, boxes, queries, text, text_mask, img_metas, visual_mask=None):
        """Return enhanced queries and detached diagnostics; never rescore boxes.

        Diagnostics expose context, usage gate, valid node/transition masks and
        the actual residual/query norm ratio without retaining autograd graphs.
        """
        b, n, channels = queries.shape
        if channels != self.hidden_channels or boxes.shape != (b, n, 4):
            raise ValueError("BCEM queries and normalized cxcywh boxes must share candidate indices")
        if visual.shape[:2] != (b, self.hidden_channels) or text.shape[0] != b or text.shape[-1] != channels:
            raise ValueError("BCEM visual/text/query channels and batch sizes must agree")
        if text.shape[1] == 0 or (text_mask is not None and text_mask.shape != text.shape[:2]):
            raise ValueError("BCEM needs a nonempty token axis and a matching text mask")
        image_sizes, pad_sizes = self._image_sizes(img_metas, visual)
        if visual_mask is None:
            visual_mask = self._feature_mask(visual, image_sizes, pad_sizes)
        if visual_mask.shape != (b, *visual.shape[-2:]):
            raise ValueError("BCEM visual_mask must match the selected feature map")
        visual_mask = visual_mask.bool()
        projected_visual = self.visual_projection(visual.masked_fill(visual_mask[:, None], 0))
        projected_visual = projected_visual.masked_fill(visual_mask[:, None], 0)
        text_valid = (torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
                      if text_mask is None else ~text_mask.bool())
        # Mask *before* projection/pooling so even extreme padding values
        # cannot affect valid tokens or the sentence-conditioned gate.
        valid_text = text.masked_fill(~text_valid[..., None], 0)
        text_queries = self.text_query(valid_text)
        sentence = valid_text.sum(1) / text_valid.sum(1).clamp(min=1)[:, None].to(text.dtype)

        contexts, gates, nodes, transitions = [], [], [], []
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            args = (projected_visual, boxes[:, start:end].detach(), queries[:, start:end],
                    text_queries, sentence, text_valid, image_sizes, pad_sizes, visual_mask)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                result = checkpoint(self._forward_chunk, *args, use_reentrant=False)
            else:
                result = self._forward_chunk(*args)
            context, gate, node_mask, transition_mask = result
            contexts.append(context)
            gates.append(gate)
            nodes.append(node_mask)
            transitions.append(transition_mask)

        context = torch.cat(contexts, 1) if n else queries.new_empty((b, 0, self.context_dim))
        gate = torch.cat(gates, 1) if n else queries.new_empty((b, 0, 1))
        correction = self.context_strength * gate * self.environment_projection(context)
        enhanced = queries + correction
        ratio = correction.float().norm(dim=-1, keepdim=True) / queries.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        diagnostics = {
            "bcem_context": context.detach(),
            "bcem_gate": gate.detach(),
            "bcem_residual_ratio": ratio.detach(),
            "bcem_valid_nodes": torch.cat(nodes, 1).detach() if n else text_valid.new_empty((b, 0, self.num_nodes)),
            "bcem_valid_transitions": (torch.cat(transitions, 1).detach() if n
                                       else text_valid.new_empty((b, 0, self.num_transitions))),
        }
        return enhanced, diagnostics
