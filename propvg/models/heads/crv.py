"""Counterfactual Referent Verifier for proposal-level disambiguation."""

import torch
from torch import nn
from torch.nn import functional as F


def gather_candidates(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class CounterfactualReferentVerifier(nn.Module):
    """Re-rank high-scoring proposals by masking their visual evidence."""

    def __init__(
        self,
        hidden_channels=256,
        topk=8,
        verifier_dim=256,
        grid_size=24,
        num_heads=8,
        num_layers=2,
        verifier_dropout=0.0,
        core_expansion=1.5,
        feather_expansion=2.0,
        neutral_mode="mean",
        margin=0.2,
        positive_drop_margin=0.5,
        ranking_loss_weight=1.0,
        fused_ranking_loss_weight=1.0,
        final_bce_loss_weight=0.5,
        gt_drop_loss_weight=1.0,
        fusion_alpha_init=1.0,
        fusion_alpha_max=2.0,
        fusion_temperature=1.0,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        negative_max_gt_coverage=0.1,
        detach_inputs=True,
    ):
        super().__init__()
        if topk < 2:
            raise ValueError("CRV topk must be at least 2")
        if neutral_mode not in ("zero", "mean", "learnable"):
            raise ValueError("CRV neutral_mode must be 'zero', 'mean' or 'learnable'")
        if core_expansion < 1.0:
            raise ValueError("CRV core_expansion must be at least 1")
        if feather_expansion <= core_expansion:
            raise ValueError("CRV feather_expansion must be greater than core_expansion")
        self.topk = int(topk)
        self.grid_size = int(grid_size)
        self.core_expansion = float(core_expansion)
        self.feather_expansion = float(feather_expansion)
        self.neutral_mode = neutral_mode
        self.margin = float(margin)
        self.positive_drop_margin = float(positive_drop_margin)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.fused_ranking_loss_weight = float(fused_ranking_loss_weight)
        self.final_bce_loss_weight = float(final_bce_loss_weight)
        self.gt_drop_loss_weight = float(gt_drop_loss_weight)
        self.fusion_alpha_max = float(fusion_alpha_max)
        self.fusion_temperature = float(fusion_temperature)
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.negative_max_gt_coverage = float(negative_max_gt_coverage)
        if self.negative_iou_threshold >= self.positive_iou_threshold:
            raise ValueError("CRV negative_iou_threshold must be smaller than positive_iou_threshold")
        self.detach_inputs = bool(detach_inputs)

        self.visual_proj = nn.Conv2d(hidden_channels, verifier_dim, kernel_size=1, bias=False)
        self.text_proj = nn.Linear(hidden_channels, verifier_dim, bias=False)
        self.support_token = nn.Parameter(torch.zeros(1, 1, verifier_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=verifier_dim,
            nhead=num_heads,
            dim_feedforward=verifier_dim * 4,
            # Paired original/masked scenes must not receive independent
            # dropout noise, otherwise their score difference is not purely
            # caused by the visual intervention.
            dropout=float(verifier_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.verifier = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(verifier_dim)
        self.support_head = nn.Sequential(
            nn.Linear(verifier_dim, verifier_dim),
            nn.GELU(),
            nn.Linear(verifier_dim, 1),
        )
        # Start as an exact order-preserving re-ranker: every original/masked
        # support logit and therefore every residual is zero, so the initial
        # final probability is exactly the raw CRS probability.
        nn.init.zeros_(self.support_head[-1].weight)
        nn.init.zeros_(self.support_head[-1].bias)
        if self.fusion_alpha_max <= 0:
            raise ValueError("CRV fusion_alpha_max must be positive")
        if not 0 < fusion_alpha_init < self.fusion_alpha_max:
            raise ValueError("CRV fusion_alpha_init must be between 0 and fusion_alpha_max")
        if self.fusion_temperature <= 0:
            raise ValueError("CRV fusion_temperature must be positive")
        # Bounded positive fusion strength. Although alpha starts non-zero,
        # the zero-initialized support head makes every initial drop exactly
        # zero, hence final_score == raw_CRS_score at initialization.
        alpha_ratio = torch.tensor(float(fusion_alpha_init / self.fusion_alpha_max))
        inverse_sigmoid = torch.logit(alpha_ratio)
        self.fusion_alpha_raw = nn.Parameter(inverse_sigmoid)
        if neutral_mode == "learnable":
            self.neutral_feature = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        else:
            self.register_parameter("neutral_feature", None)

    @property
    def fusion_alpha(self):
        return self.fusion_alpha_max * self.fusion_alpha_raw.sigmoid()

    @staticmethod
    def _masked_mean(values, valid):
        if valid.any():
            return values[valid].mean()
        return values.sum() * 0.0

    @staticmethod
    def _balanced_bce_with_logits(logits, positive, negative):
        positive_loss = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), reduction="none"
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )
        positive_count = positive.sum().to(logits.dtype)
        negative_count = negative.sum().to(logits.dtype)
        positive_mean = (positive_loss * positive.to(logits.dtype)).sum() / positive_count.clamp(min=1.0)
        negative_mean = (negative_loss * negative.to(logits.dtype)).sum() / negative_count.clamp(min=1.0)
        group_count = (positive_count > 0).to(logits.dtype) + (negative_count > 0).to(logits.dtype)
        return (positive_mean + negative_mean) / group_count.clamp(min=1.0)

    def _select(self, scores):
        k = min(self.topk, scores.shape[1])
        return scores.detach().topk(k, dim=1).indices

    def _soft_box_masks(self, boxes, valid_ratio, height, width, dtype, device):
        x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        xx = xx[None, None]
        yy = yy[None, None]

        centers = boxes[..., :2] * valid_ratio[:, None]
        box_sizes = boxes[..., 2:].clamp(min=1e-5) * valid_ratio[:, None]
        inner_sizes = box_sizes * self.core_expansion
        outer_sizes = box_sizes * self.feather_expansion
        inner_x1 = (centers[..., 0] - inner_sizes[..., 0] * 0.5)[..., None, None]
        inner_y1 = (centers[..., 1] - inner_sizes[..., 1] * 0.5)[..., None, None]
        inner_x2 = (centers[..., 0] + inner_sizes[..., 0] * 0.5)[..., None, None]
        inner_y2 = (centers[..., 1] + inner_sizes[..., 1] * 0.5)[..., None, None]
        outer_x1 = (centers[..., 0] - outer_sizes[..., 0] * 0.5)[..., None, None]
        outer_y1 = (centers[..., 1] - outer_sizes[..., 1] * 0.5)[..., None, None]
        outer_x2 = (centers[..., 0] + outer_sizes[..., 0] * 0.5)[..., None, None]
        outer_y2 = (centers[..., 1] + outer_sizes[..., 1] * 0.5)[..., None, None]

        def smoothstep(value):
            value = value.clamp(0.0, 1.0)
            return value * value * (3.0 - 2.0 * value)

        # The whole core (original box through core_expansion) is exactly 1.
        # Only the band from core_expansion through feather_expansion is
        # feathered, so the target and its immediate boundary are completely
        # replaced while the intervention transitions smoothly outside.
        left = smoothstep((xx - outer_x1) / (inner_x1 - outer_x1).clamp(min=1e-6))
        right = smoothstep((outer_x2 - xx) / (outer_x2 - inner_x2).clamp(min=1e-6))
        top = smoothstep((yy - outer_y1) / (inner_y1 - outer_y1).clamp(min=1e-6))
        bottom = smoothstep((outer_y2 - yy) / (outer_y2 - inner_y2).clamp(min=1e-6))
        mask_x = torch.minimum(left, right)
        mask_y = torch.minimum(top, bottom)
        return (mask_x * mask_y).clamp(0.0, 1.0)

    def _scene_features(self, visual, visual_mask, boxes):
        grid = self.grid_size
        batch_size, channels, height, width = visual.shape

        if visual_mask is None:
            valid_highres = torch.ones((batch_size, height, width), dtype=torch.bool, device=visual.device)
            valid_ratio = visual.new_ones((batch_size, 2))
        else:
            valid_highres = ~visual_mask.bool()
            valid_h = (~visual_mask[:, :, 0]).sum(1).to(visual.dtype) / visual_mask.shape[1]
            valid_w = (~visual_mask[:, 0, :]).sum(1).to(visual.dtype) / visual_mask.shape[2]
            valid_ratio = torch.stack((valid_w, valid_h), dim=-1)

        valid_float = valid_highres[:, None].to(visual.dtype)
        if self.neutral_mode == "zero":
            neutral = visual.new_zeros((batch_size, channels, 1, 1))
        elif self.neutral_mode == "mean":
            neutral = (visual * valid_float).sum((2, 3), keepdim=True) / valid_float.sum(
                (2, 3), keepdim=True
            ).clamp(min=1.0)
        else:
            neutral = self.neutral_feature.expand(batch_size, -1, -1, -1)

        # Fill padding before pooling so partially valid boundary cells cannot
        # mix arbitrary padded FPN activations into the verifier input.
        base_visual = visual * valid_float + neutral * (1.0 - valid_float)

        # Apply each intervention on the original high-resolution FPN map.
        # Pool only after the candidate evidence has been removed so that a
        # small aerial target is not mixed into its neighbours beforehand.
        masks = self._soft_box_masks(
            boxes, valid_ratio, height, width, visual.dtype, visual.device
        )
        # Scene 0 is the untouched valid image; scenes 1..K are paired
        # counterfactual interventions scored by the exact same verifier.
        pooled = [F.adaptive_avg_pool2d(base_visual, (grid, grid))]
        for candidate_index in range(boxes.shape[1]):
            mask = masks[:, candidate_index : candidate_index + 1]
            counterfactual = base_visual * (1.0 - mask) + neutral * mask
            pooled.append(F.adaptive_avg_pool2d(counterfactual, (grid, grid)))
        scenes = torch.stack(pooled, dim=1)
        num_scenes = scenes.shape[1]
        scenes = self.visual_proj(
            scenes.reshape(batch_size * num_scenes, channels, grid, grid)
        ).reshape(batch_size, num_scenes, -1, grid, grid)
        valid = ~F.interpolate(
            (~valid_highres)[:, None].float(), size=(grid, grid), mode="nearest"
        ).squeeze(1).bool()
        return scenes, valid

    @staticmethod
    def _box_iou(boxes, target_boxes):
        boxes_xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)
        target_xyxy = torch.cat(
            (target_boxes[..., :2] - target_boxes[..., 2:] / 2,
             target_boxes[..., :2] + target_boxes[..., 2:] / 2),
            -1,
        )[:, None]
        left_top = torch.maximum(boxes_xyxy[..., :2], target_xyxy[..., :2])
        right_bottom = torch.minimum(boxes_xyxy[..., 2:], target_xyxy[..., 2:])
        intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
        box_area = boxes[..., 2:].clamp(min=0).prod(-1)
        target_area = target_boxes[..., 2:].clamp(min=0).prod(-1)[:, None]
        return intersection / (box_area + target_area - intersection).clamp(min=1e-6)

    @staticmethod
    def _target_coverage(boxes, target_boxes):
        """Fraction of each refer-GT area covered by candidate boxes."""
        boxes_xyxy = torch.cat(
            (boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1
        )
        target_xyxy = torch.cat(
            (target_boxes[..., :2] - target_boxes[..., 2:] / 2,
             target_boxes[..., :2] + target_boxes[..., 2:] / 2),
            -1,
        )[:, None]
        left_top = torch.maximum(boxes_xyxy[..., :2], target_xyxy[..., :2])
        right_bottom = torch.minimum(boxes_xyxy[..., 2:], target_xyxy[..., 2:])
        intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
        target_area = target_boxes[..., 2:].clamp(min=0).prod(-1)[:, None]
        return intersection / target_area.clamp(min=1e-6)

    @staticmethod
    def _position_2d(height, width, dim, device, dtype):
        if dim % 4:
            raise ValueError("CRV verifier_dim must be divisible by 4 for 2D position encoding")
        quarter = dim // 4
        frequency = torch.arange(quarter, device=device, dtype=torch.float32)
        frequency = 1.0 / (10000 ** (frequency / max(quarter, 1)))
        y = torch.linspace(0, 1, height, device=device, dtype=torch.float32)
        x = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x_phase = xx.reshape(-1, 1) * frequency[None] * (2.0 * torch.pi)
        y_phase = yy.reshape(-1, 1) * frequency[None] * (2.0 * torch.pi)
        position = torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
        )
        return position.to(dtype=dtype)[None]

    def _scene_support(self, scenes, valid_visual, text, text_mask):
        b, num_scenes, c, h, w = scenes.shape
        visual_tokens = scenes.flatten(3).permute(0, 1, 3, 2).reshape(b * num_scenes, h * w, c)
        visual_tokens = visual_tokens + self._position_2d(h, w, c, visual_tokens.device, visual_tokens.dtype)
        visual_padding = (
            (~valid_visual).flatten(1)[:, None]
            .expand(-1, num_scenes, -1)
            .reshape(b * num_scenes, h * w)
        )

        text = self.text_proj(text)
        text_tokens = (
            text[:, None].expand(-1, num_scenes, -1, -1)
            .reshape(b * num_scenes, text.shape[1], c)
        )
        text_padding = (
            text_mask.bool()[:, None].expand(-1, num_scenes, -1)
            .reshape(b * num_scenes, text.shape[1])
        )
        support_tokens = self.support_token.expand(b * num_scenes, -1, -1)
        queries = torch.cat((support_tokens, text_tokens), dim=1)
        query_padding = torch.cat(
            (torch.zeros((b * num_scenes, 1), dtype=torch.bool, device=text.device), text_padding),
            dim=1,
        )
        decoded = self.verifier(
            tgt=queries,
            memory=visual_tokens,
            tgt_key_padding_mask=query_padding,
            memory_key_padding_mask=visual_padding,
        )
        support_feature = self.output_norm(decoded[:, 0])
        return self.support_head(support_feature).reshape(b, num_scenes)

    def forward(
        self,
        visual,
        visual_mask,
        boxes,
        refer_scores,
        text,
        text_mask,
        target_boxes=None,
        target_valid=None,
    ):
        # By default CRV is a strictly auxiliary re-ranker. Its loss trains
        # only CRV and cannot perturb BEiT-3, SimFPN, DETR, Grab or CRS.
        if self.detach_inputs:
            visual = visual.detach()
            boxes = boxes.detach()
            refer_scores = refer_scores.detach()
            text = text.detach()
        scores = refer_scores.squeeze(-1)
        indices = self._select(scores)
        selected_boxes = gather_candidates(boxes, indices)
        selected_scores = scores.gather(1, indices)

        # During training, append the exact referring GT as one additional
        # intervention.  It guarantees a stable positive counterfactual even
        # when no sufficiently accurate proposal enters raw CRS Top-K.  The GT
        # scene is never constructed during validation or inference.
        candidate_count = selected_boxes.shape[1]
        intervention_boxes = selected_boxes.detach()
        has_gt_intervention = target_boxes is not None
        if has_gt_intervention:
            intervention_boxes = torch.cat(
                (intervention_boxes, target_boxes.detach()[:, None]), dim=1
            )
        scenes, valid_visual = self._scene_features(
            visual, visual_mask, intervention_boxes
        )
        scene_support_logits = self._scene_support(scenes, valid_visual, text, text_mask)
        original_support_logits = scene_support_logits[:, :1]
        masked_support_logits = scene_support_logits[:, 1:1 + candidate_count]
        gt_masked_support_logits = (
            scene_support_logits[:, 1 + candidate_count:2 + candidate_count]
            if has_gt_intervention else None
        )

        # A large positive drop means that removing this candidate destroys
        # scene support for the full expression, so it is counterfactually
        # important. The shared scorer makes the comparison explicitly paired.
        support_drop_logits = original_support_logits - masked_support_logits
        gt_support_drop_logits = (
            original_support_logits - gt_masked_support_logits
            if gt_masked_support_logits is not None else None
        )

        # Treat CRV as a bounded residual correction to the original CRS
        # probability. This keeps final scores in [0, 1], enables direct final
        # BCE supervision, and prevents an extreme drop from overwhelming CRS.
        alpha = self.fusion_alpha.to(support_drop_logits.dtype)
        fusion_residual_logits = alpha * torch.tanh(
            support_drop_logits / self.fusion_temperature
        )
        raw_crs_logits = torch.logit(selected_scores.clamp(min=1e-6, max=1.0 - 1e-6))
        fused_logits = raw_crs_logits + fusion_residual_logits
        weighted_scores = fused_logits.sigmoid().unsqueeze(-1)
        # Kept as a diagnostic odds multiplier: final odds = raw odds * importance.
        importance_scores = fusion_residual_logits.exp()

        loss = visual.sum() * 0.0
        selected_ious = None
        expanded_gt_coverage = None
        if target_boxes is not None:
            selected_ious = self._box_iou(selected_boxes.detach(), target_boxes.detach())
            expanded_boxes = selected_boxes.detach().clone()
            expanded_boxes[..., 2:] *= self.feather_expansion
            expanded_gt_coverage = self._target_coverage(expanded_boxes, target_boxes.detach())
            positive = selected_ious >= self.positive_iou_threshold
            negative = (
                (selected_ious < self.negative_iou_threshold)
                & (expanded_gt_coverage < self.negative_max_gt_coverage)
            )
            if target_valid is not None:
                positive = positive & target_valid[:, None]
                negative = negative & target_valid[:, None]
            if target_valid is None:
                target_valid = torch.ones(
                    support_drop_logits.shape[0], dtype=torch.bool,
                    device=support_drop_logits.device,
                )

            # Purely relative counterfactual supervision.  Removing a true
            # referent must produce a positive drop; removing a reliable
            # distractor should leave support unchanged (drop near zero).
            candidate_positive_loss = self._masked_mean(
                F.relu(self.positive_drop_margin - support_drop_logits), positive
            )
            candidate_negative_loss = self._masked_mean(
                F.smooth_l1_loss(
                    support_drop_logits, torch.zeros_like(support_drop_logits), reduction="none"
                ),
                negative,
            )
            gt_positive_loss = self._masked_mean(
                F.relu(self.positive_drop_margin - gt_support_drop_logits[:, 0]),
                target_valid,
            )
            drop_loss = (
                candidate_positive_loss
                + candidate_negative_loss
                + self.gt_drop_loss_weight * gt_positive_loss
            )

            # Removing a target-covering candidate must cause a larger support
            # drop than removing any reliable high-scoring distractor.
            pair_valid = positive[:, :, None] & negative[:, None, :]
            rank_terms = F.relu(
                self.margin
                - support_drop_logits[:, :, None]
                + support_drop_logits[:, None, :]
            )
            candidate_rank_loss = self._masked_mean(rank_terms, pair_valid)

            # The exact-GT intervention also ranks against every reliable hard
            # distractor, including samples whose Top-K has no positive box.
            gt_rank_terms = F.relu(
                self.margin - gt_support_drop_logits + support_drop_logits
            )
            gt_pair_valid = target_valid[:, None] & negative
            gt_rank_loss = self._masked_mean(gt_rank_terms, gt_pair_valid)
            rank_loss = candidate_rank_loss + gt_rank_loss

            # Optimize the score actually used at inference.  Ranking drops in
            # isolation was insufficient when raw CRS assigned a distractor a
            # much larger score than the correct proposal.
            fused_rank_terms = F.relu(
                self.margin
                - fused_logits[:, :, None]
                + fused_logits[:, None, :]
            )
            fused_rank_loss = self._masked_mean(fused_rank_terms, pair_valid)

            # Directly supervise the exact scores used at inference. Samples
            # with no positive Top-K proposal are skipped: suppressing all of
            # their candidates cannot recover the absent target. Positives and
            # negatives are averaged as two groups to avoid class imbalance.
            bce_sample_valid = positive.any(dim=1)
            final_bce_positive = positive & bce_sample_valid[:, None]
            final_bce_negative = negative & bce_sample_valid[:, None]
            final_bce_loss = self._balanced_bce_with_logits(
                fused_logits, final_bce_positive, final_bce_negative
            )
            loss = (
                drop_loss
                + self.ranking_loss_weight * rank_loss
                + self.fused_ranking_loss_weight * fused_rank_loss
                + self.final_bce_loss_weight * final_bce_loss
            )
        else:
            candidate_positive_loss = loss
            candidate_negative_loss = loss
            gt_positive_loss = loss
            rank_loss = loss
            fused_rank_loss = loss
            final_bce_loss = loss

        return {
            "indices": indices,
            "weighted_scores": weighted_scores,
            "importance_scores": importance_scores,
            "fusion_residual_logits": fusion_residual_logits,
            "fused_logits": fused_logits,
            "fusion_alpha": alpha,
            "original_support_logits": original_support_logits,
            "masked_support_logits": masked_support_logits,
            "relative_support_scores": -support_drop_logits,
            "support_drop_logits": support_drop_logits,
            "gt_masked_support_logits": gt_masked_support_logits,
            "gt_support_drop_logits": gt_support_drop_logits,
            "selected_ious": selected_ious,
            "expanded_gt_coverage": expanded_gt_coverage,
            "loss_candidate_positive": candidate_positive_loss,
            "loss_candidate_negative": candidate_negative_loss,
            "loss_gt_positive": gt_positive_loss,
            "loss_drop_rank": rank_loss,
            "loss_fused_rank": fused_rank_loss,
            "loss_final_bce": final_bce_loss,
            "loss": loss,
        }
