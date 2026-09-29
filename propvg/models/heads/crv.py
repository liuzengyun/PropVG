"""Independent spatial counterfactual verifier; no CRS/decoder parameter sharing."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def gather_candidates(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class CounterfactualReferentVerifier(nn.Module):
    """Detach baseline evidence, compare original/masked scenes, learn fusion."""

    def __init__(self, hidden_channels=256, topk=8, verifier_dim=128,
                 grid_size=24, num_heads=4, num_layers=2,
                 core_expansion=1.5, feather_expansion=2.0,
                 max_residual=2.0, drop_temperature=1.0, initial_alpha=0.1,
                 cf_loss_weight=0.1, rank_margin=0.5,
                 max_negative_gt_coverage=0.05):
        super().__init__()
        if topk < 2 or grid_size < 2 or num_layers < 1:
            raise ValueError("topk/grid_size must be >=2; num_layers must be >=1")
        if verifier_dim % num_heads or verifier_dim % 4:
            raise ValueError("verifier_dim must be divisible by num_heads and 4")
        if not 1 <= core_expansion < feather_expansion:
            raise ValueError("require 1 <= core_expansion < feather_expansion")
        if max_residual <= 0 or drop_temperature <= 0 or rank_margin <= 0:
            raise ValueError("residual bound, temperature and margin must be positive")
        if not 0 < initial_alpha < 1 or cf_loss_weight < 0:
            raise ValueError("initial_alpha must be in (0,1); cf_loss_weight >=0")
        if not 0 <= max_negative_gt_coverage <= 1:
            raise ValueError("max_negative_gt_coverage must be in [0,1]")
        self.topk, self.grid_size = int(topk), int(grid_size)
        self.core_expansion, self.feather_expansion = float(core_expansion), float(feather_expansion)
        self.max_residual, self.drop_temperature = float(max_residual), float(drop_temperature)
        self.cf_loss_weight, self.rank_margin = float(cf_loss_weight), float(rank_margin)
        self.max_negative_gt_coverage = float(max_negative_gt_coverage)

        self.visual_proj = nn.Conv2d(hidden_channels, verifier_dim, 1, bias=False)
        self.text_proj = nn.Linear(hidden_channels, verifier_dim, bias=False)
        self.scene_token = nn.Parameter(torch.zeros(1, 1, verifier_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=verifier_dim, nhead=num_heads, dim_feedforward=verifier_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.scene_matcher = nn.TransformerDecoder(layer, int(num_layers))
        self.scene_head = nn.Sequential(
            nn.LayerNorm(verifier_dim), nn.Linear(verifier_dim, verifier_dim),
            nn.GELU(), nn.Linear(verifier_dim, 1, bias=False),
        )
        # Exact baseline at step 0, without freezing CRV or staged training.
        nn.init.zeros_(self.scene_head[-1].weight)
        # One image-wise weight, shared across candidates; no extra classifier.
        self.fusion_gate = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, math.log(initial_alpha / (1 - initial_alpha)))
        self.register_buffer("independent_version", torch.tensor(2, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Reject old CRV weights even when the project loader uses strict=False.
        if any(key.startswith(prefix) for key in state_dict):
            version = state_dict.get(prefix + "independent_version")
            if version is None or int(version) != 2:
                error_msgs.append("Old/incompatible CRV checkpoint: use enable_crv=False "
                                  "for the baseline, or start a fresh independent-CRV run.")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _soft_box_masks(self, boxes, valid_ratio, height, width, dtype=None):
        """Cell-integrated smooth mask, including boxes smaller than one cell."""
        boxes = boxes.detach().float()
        ratio = valid_ratio.detach().float()[:, None]
        centers, sizes = boxes[..., :2] * ratio, boxes[..., 2:].clamp(min=1e-8) * ratio

        def axis_weights(axis, count):
            center, size = centers[..., axis, None], sizes[..., axis, None]
            outer_left = center - size * self.feather_expansion / 2
            inner_left = center - size * self.core_expansion / 2
            inner_right = center + size * self.core_expansion / 2
            outer_right = center + size * self.feather_expansion / 2
            low = torch.arange(count, device=boxes.device, dtype=torch.float32) / count
            # Integrate only the valid portion of a boundary feature cell.
            high = torch.minimum(low + 1.0 / count, ratio[..., axis, None])

            def primitive(x, start, end):
                span = (end - start).clamp(min=1e-8)
                t = ((x - start) / span).clamp(0, 1)
                return span * (t.pow(3) - 0.5 * t.pow(4)) + (x - end).clamp(min=0)

            def integral(x):
                return primitive(x, outer_left, inner_left) - primitive(x, inner_right, outer_right)

            return ((integral(high) - integral(low)) * count).clamp(0, 1)

        return (axis_weights(1, height)[..., :, None] *
                axis_weights(0, width)[..., None, :]).to(dtype=dtype or boxes.dtype)

    @staticmethod
    def _rectangle_coverage(boxes, ratios, height, width):
        centers = boxes[..., :2] * ratios[:, None]
        sizes = boxes[..., 2:].clamp(min=0) * ratios[:, None]
        low = (centers - sizes / 2).clamp(min=0)
        high = torch.minimum(centers + sizes / 2, ratios[:, None]).clamp(min=0)
        x = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
        y = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
        wx = (torch.minimum(x + 1 / width, high[..., 0, None]) -
              torch.maximum(x, low[..., 0, None])).clamp(min=0) * width
        wy = (torch.minimum(y + 1 / height, high[..., 1, None]) -
              torch.maximum(y, low[..., 1, None])).clamp(min=0) * height
        return wy[..., :, None] * wx[..., None, :]

    def _scene_features(self, visual, visual_mask, boxes, valid_ratios):
        valid = torch.ones_like(visual[:, 0], dtype=torch.bool) if visual_mask is None else ~visual_mask.bool()
        if valid.shape != visual[:, 0].shape or (~valid.flatten(1).any(1)).any():
            raise ValueError("CRV needs a correctly shaped, nonempty valid image region")
        # Keep baseline integer validity masks untouched, but use exact cell
        # areas in this detached branch for means, pooling and interventions.
        unit_box = visual.new_tensor([.5, .5, 1., 1.])[None, None].expand(visual.shape[0], -1, -1)
        weights = self._rectangle_coverage(unit_box.float(), valid_ratios.float(),
                                           *visual.shape[-2:]).to(visual.dtype)
        weights = weights * valid[:, None]
        if (weights.flatten(1).sum(1) <= 0).any():
            raise ValueError("CRV valid ratios and image padding mask disagree")
        neutral = (visual * weights).sum((2, 3), keepdim=True) / weights.sum((2, 3), keepdim=True)
        masks = self._soft_box_masks(boxes, valid_ratios, *visual.shape[-2:], dtype=visual.dtype)
        # Integrated erased area / valid area = replacement strength per cell.
        masks = (masks / weights.clamp(min=1e-8)).clamp(0, 1) * valid[:, None]
        grid = (self.grid_size, self.grid_size)
        pooled_valid = F.adaptive_avg_pool2d(weights, grid)

        def pool(features):
            # Partial padding cells are normalized by valid area.
            return F.adaptive_avg_pool2d(features * weights, grid) / pooled_valid.clamp(min=1e-6)

        original = pool(visual)
        # Mask at native high resolution BEFORE pooling; do not materialize
        # [B,K,C,H,W] or keep its autograd history.
        pooled = [original]
        for index in range(boxes.shape[1]):
            mask = masks[:, index:index + 1]
            pooled.append(pool(visual * (1 - mask) + neutral * mask))
        scenes = torch.stack(pooled, dim=1)
        b, n, c, h, w = scenes.shape
        scenes = self.visual_proj(scenes.reshape(b * n, c, h, w)).reshape(b, n, -1, h, w)
        return scenes, pooled_valid[:, 0] > 0, masks, neutral

    @staticmethod
    def _position(height, width, dim, device, dtype):
        frequency = torch.arange(dim // 4, device=device, dtype=torch.float32)
        frequency = 1 / (10000 ** (frequency / (dim // 4)))
        y = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height
        x = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xp, yp = xx.flatten()[:, None] * frequency * (2 * math.pi), yy.flatten()[:, None] * frequency * (2 * math.pi)
        return torch.cat((xp.sin(), xp.cos(), yp.sin(), yp.cos()), -1).to(dtype)[None]

    def _match_scenes(self, scenes, valid_visual, text, text_mask):
        b, n, c, h, w = scenes.shape
        memory = scenes.flatten(3).permute(0, 1, 3, 2).reshape(b * n, h * w, c)
        memory = memory + self._position(h, w, c, scenes.device, scenes.dtype)
        memory_padding = (~valid_visual).flatten(1)[:, None].expand(-1, n, -1).reshape(b * n, h * w)
        language = self.text_proj(text)
        language = language[:, None].expand(-1, n, -1, -1).reshape(b * n, text.shape[1], c)
        query = torch.cat((self.scene_token.expand(b * n, -1, -1), language), 1)
        query_padding = torch.cat((
            torch.zeros(b * n, 1, dtype=torch.bool, device=scenes.device),
            text_mask[:, None].expand(-1, n, -1).reshape(b * n, text.shape[1]),
        ), 1)
        decoded = self.scene_matcher(query, memory, tgt_key_padding_mask=query_padding,
                                     memory_key_padding_mask=memory_padding)
        return self.scene_head(decoded[:, 0]).reshape(b, n)

    @staticmethod
    def _box_iou(boxes, target):
        low, high = boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2
        target_low, target_high = target[:, None, :2] - target[:, None, 2:] / 2, target[:, None, :2] + target[:, None, 2:] / 2
        # Match inference detector_postprocess: clip to the valid image,
        # not the padded canvas, before assigning refer labels.
        low, high = low.clamp(0, 1), high.clamp(0, 1)
        target_low, target_high = target_low.clamp(0, 1), target_high.clamp(0, 1)
        intersection = (torch.minimum(high, target_high) - torch.maximum(low, target_low)).clamp(min=0).prod(-1)
        union = (high - low).clamp(min=0).prod(-1) + (target_high - target_low).clamp(min=0).prod(-1) - intersection
        return intersection / union.clamp(min=1e-8)

    def _evidence(self, drops):
        # Softsign retains a polynomial gradient at large drops; tanh can
        # become exactly constant in FP32. Keep the original bounded residual.
        return self.max_residual * F.softsign(drops / self.drop_temperature)

    @staticmethod
    def _pairwise_ranking(logits, positive, negative, margin, smooth=False):
        pairs = positive[:, :, None] & negative[:, None, :]
        violations = margin - logits[:, :, None] + logits[:, None, :]
        values = F.softplus(violations) if smooth else F.relu(violations)
        per_image = (values * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp(min=1)
        usable = pairs.flatten(1).any(1)
        return (per_image * usable).sum() / usable.sum().clamp(min=1)

    def _protected_ranking(self, raw_logits, fused_logits, positive, negative):
        """Correct legal CRS mistakes; preserve its already-correct winner.

        Gray candidates remain ignored. No labels are fabricated when Top-K
        misses the target. Stop rewarding improvements once margins are met.
        """
        raw_logits = raw_logits.detach()
        usable = positive.any(1) & negative.any(1)
        correction_images = usable & negative[:, 0]
        preservation_images = usable & positive[:, 0]
        # A masked finite sentinel avoids inf-inf for missing classes; such
        # rows are excluded below but must still have finite graph zeros.
        floor = torch.finfo(fused_logits.dtype).min
        best_positive = fused_logits.masked_fill(~positive, floor).max(1).values
        hardest_negative = fused_logits.masked_fill(~negative, floor).max(1).values
        correction = F.relu(self.rank_margin - best_positive + hardest_negative)
        correction = torch.where(correction_images, correction, torch.zeros_like(correction))
        # Preserve the original CRS winner, not every positive proposal. For
        # close baseline pairs retain their original gap; otherwise require
        # only the normal margin, rather than rewarding ever-larger gaps.
        base_gap = raw_logits[:, :1] - raw_logits
        required_gap = base_gap.clamp(min=0, max=self.rank_margin)
        violations = required_gap - fused_logits[:, :1] + fused_logits
        preservation = F.relu(violations.masked_fill(~negative, floor).max(1).values)
        preservation = torch.where(preservation_images, preservation, torch.zeros_like(preservation))
        active = correction_images | preservation_images
        denominator = active.sum().clamp(min=1)
        correction_loss = correction.sum() / denominator
        preservation_loss = preservation.sum() / denominator
        return (correction_loss + preservation_loss, correction_loss,
                preservation_loss, active)

    def forward(self, visual, visual_mask, boxes, refer_scores, refer_logits,
                text, text_mask, valid_ratios, target_boxes=None, target_valid=None):
        if not self.training and target_boxes is not None:
            raise ValueError("Refer GT interventions are training-only; inference must not receive GT")
        if not torch.isfinite(valid_ratios).all() or (valid_ratios <= 0).any() or (valid_ratios > 1).any():
            raise ValueError("valid_ratios must be exact resize/pad ratios in (0,1]")
        # Fixed, unconditional gradient boundary, not a configurable option.
        visual, boxes, text = visual.detach(), boxes.detach(), text.detach()
        refer_scores, refer_logits = refer_scores.detach(), refer_logits.detach()
        valid_ratios = valid_ratios.detach()
        text_mask = text_mask.bool()
        if text_mask.all(1).any():
            raise ValueError("CRV requires at least one valid full-expression token")
        scores, logits = refer_scores.squeeze(-1), refer_logits.squeeze(-1)
        if boxes.shape[1] == 0:
            raise ValueError("CRV requires at least one predicted candidate")
        # Sigmoid probabilities can quantize distinct large logits into ties.
        indices = logits.argsort(dim=1, descending=True, stable=True)[:, :self.topk]
        selected_boxes = gather_candidates(boxes, indices)
        raw_logits = logits.gather(1, indices)
        k = indices.shape[1]
        gt_valid = None
        if target_boxes is not None:
            target_boxes = target_boxes.detach().to(boxes)
            gt_valid = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device) if target_valid is None else target_valid.detach().bool()
            gt_valid = gt_valid & torch.isfinite(target_boxes).all(1) & (target_boxes[:, 2:] > 0).all(1)
            target_boxes = torch.where(gt_valid[:, None], target_boxes, torch.zeros_like(target_boxes))
            intervention_boxes = torch.cat((selected_boxes, target_boxes[:, None]), 1)
        else:
            intervention_boxes = selected_boxes
        scenes, valid_visual, masks, _ = self._scene_features(visual, visual_mask, intervention_boxes, valid_ratios)
        scene_logits = self._match_scenes(scenes, valid_visual, text, text_mask)
        original_logit, masked_logits = scene_logits[:, :1], scene_logits[:, 1:]
        all_drops = original_logit - masked_logits
        drops = all_drops[:, :k]
        # Relative evidence has no absolute-support sigmoid or calibration loss.
        selected_scores = scores.gather(1, indices)
        statistics = torch.stack((
            selected_scores.mean(1), selected_scores[:, 0],
            selected_scores[:, 0] - selected_scores[:, min(1, k - 1)],
            selected_scores.std(1, unbiased=False), drops.abs().mean(1),
            drops.std(1, unbiased=False),
        ), 1).detach()
        alpha = self.fusion_gate(statistics).sigmoid()
        # Do NOT center drops: that would algebraically cancel the original
        # scene score and remove its gradient from the final objective.
        all_evidence = self._evidence(all_drops)
        evidence = all_evidence[:, :k]
        correction = alpha * evidence
        fused_logits = raw_logits + correction
        zero = (scene_logits.sum() + alpha.sum()) * 0
        loss_final, loss_cf = zero, zero
        loss_correction, loss_preservation = zero, zero
        valid_pair_images = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        positive = negative = selected_ious = gt_drop = None
        if target_boxes is not None:
            selected_ious = self._box_iou(selected_boxes, target_boxes)
            positive = (selected_ious > .5) & gt_valid[:, None]
            negative = (selected_ious == 0) & gt_valid[:, None]
            # Disjoint boxes can still erase the GT after expansion/feathering.
            # Exclude these from the reliable negative set, using refer GT only.
            gt_coverage = self._rectangle_coverage(target_boxes[:, None], valid_ratios, *visual.shape[-2:])
            overlap = (masks[:, :k] * gt_coverage).sum((2, 3)) / gt_coverage.sum((2, 3)).clamp(min=1e-8)
            negative = negative & (overlap <= self.max_negative_gt_coverage)
            loss_final, loss_correction, loss_preservation, valid_pair_images = self._protected_ranking(
                raw_logits, fused_logits, positive, negative)
            gt_drop = all_drops[:, k:k + 1]
            cf_positive = torch.cat((positive, gt_valid[:, None]), 1)
            cf_negative = torch.cat((negative, torch.zeros_like(gt_valid[:, None])), 1)
            # Supervise the same transformed evidence used in fusion: raw
            # drop margins alone do not ensure a usable post-transform gap.
            loss_cf = self._pairwise_ranking(
                all_evidence, cf_positive, cf_negative, self.rank_margin, smooth=False)
        return dict(
            indices=indices, weighted_scores=fused_logits.sigmoid()[..., None],
            raw_crs_logits=raw_logits, fused_logits=fused_logits, fusion_alpha=alpha,
            fusion_residual_logits=correction, support_drop_logits=drops,
            counterfactual_evidence=evidence, valid_pair_images=valid_pair_images,
            original_support_logits=original_logit, masked_support_logits=masked_logits[:, :k],
            gt_support_drop_logits=gt_drop, selected_ious=selected_ious,
            positive_candidates=positive, negative_candidates=negative,
            loss_final=loss_final, loss_counterfactual=loss_cf,
            loss_correction=loss_correction, loss_preservation=loss_preservation,
            loss=loss_final + self.cf_loss_weight * loss_cf + zero,
        )
