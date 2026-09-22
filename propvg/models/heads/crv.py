"""Counterfactual Referent Verifier."""

import torch
from torch import nn
from torch.nn import functional as F


def gather_candidates(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class CounterfactualReferentVerifier(nn.Module):
    """Re-rank CRS Top-K proposals with paired visual interventions.

    The same complete expression scores the untouched scene and every scene
    in which one candidate is masked. Their score difference is a bounded
    residual correction to the original CRS logit.
    """

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
        positive_drop_margin=0.5,
        rank_margin=0.2,
        positive_iou_threshold=0.5,
        detach_inputs=True,
        **_legacy_options,
    ):
        super().__init__()
        if topk < 2:
            raise ValueError("CRV topk must be at least 2")
        if verifier_dim % num_heads:
            raise ValueError("CRV verifier_dim must be divisible by num_heads")
        if neutral_mode not in ("zero", "mean", "learnable"):
            raise ValueError("CRV neutral_mode must be 'zero', 'mean' or 'learnable'")
        if core_expansion < 1.0:
            raise ValueError("CRV core_expansion must be at least 1")
        if feather_expansion <= core_expansion:
            raise ValueError("CRV feather_expansion must be greater than core_expansion")
        if positive_drop_margin <= 0:
            raise ValueError("CRV positive_drop_margin must be positive")
        if rank_margin <= 0:
            raise ValueError("CRV rank_margin must be positive")

        self.topk = int(topk)
        self.grid_size = int(grid_size)
        self.core_expansion = float(core_expansion)
        self.feather_expansion = float(feather_expansion)
        self.neutral_mode = neutral_mode
        self.positive_drop_margin = float(positive_drop_margin)
        self.rank_margin = float(rank_margin)
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.detach_inputs = bool(detach_inputs)

        self.visual_proj = nn.Conv2d(hidden_channels, verifier_dim, kernel_size=1, bias=False)
        self.text_proj = nn.Linear(hidden_channels, verifier_dim, bias=False)

        # The unchanged complete expression attends to an original/masked scene.
        self.support_token = nn.Parameter(torch.zeros(1, 1, verifier_dim))
        scene_layer = nn.TransformerDecoderLayer(
            d_model=verifier_dim,
            nhead=num_heads,
            dim_feedforward=verifier_dim * 4,
            dropout=float(verifier_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.verifier = nn.TransformerDecoder(scene_layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(verifier_dim)
        self.support_head = nn.Sequential(
            nn.Linear(verifier_dim, verifier_dim),
            nn.GELU(),
            nn.Linear(verifier_dim, 1),
        )
        # At initialization CRV is exactly inactive: residual=0 and final
        # ranking is identical to CRS. There is no fusion alpha.
        nn.init.zeros_(self.support_head[-1].weight)
        nn.init.zeros_(self.support_head[-1].bias)

        if neutral_mode == "learnable":
            self.neutral_feature = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        else:
            self.register_parameter("neutral_feature", None)

    def _select(self, scores):
        k = min(self.topk, scores.shape[1])
        return scores.detach().topk(k, dim=1).indices

    @staticmethod
    def _position_2d(height, width, dim, device, dtype):
        if dim % 4:
            raise ValueError("CRV verifier_dim must be divisible by 4")
        quarter = dim // 4
        frequency = torch.arange(quarter, device=device, dtype=torch.float32)
        frequency = 1.0 / (10000 ** (frequency / max(quarter, 1)))
        y = torch.linspace(0, 1, height, device=device, dtype=torch.float32)
        x = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x_phase = xx.reshape(-1, 1) * frequency[None] * (2.0 * torch.pi)
        y_phase = yy.reshape(-1, 1) * frequency[None] * (2.0 * torch.pi)
        return torch.cat(
            (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
        ).to(dtype=dtype)[None]

    @staticmethod
    def _valid_geometry(visual, visual_mask):
        batch_size, _, height, width = visual.shape
        if visual_mask is None:
            valid = torch.ones(
                (batch_size, height, width), dtype=torch.bool, device=visual.device
            )
            ratio = visual.new_ones((batch_size, 2))
        else:
            valid = ~visual_mask.bool()
            valid_h = valid[:, :, 0].sum(1).to(visual.dtype) / visual_mask.shape[1]
            valid_w = valid[:, 0, :].sum(1).to(visual.dtype) / visual_mask.shape[2]
            ratio = torch.stack((valid_w, valid_h), dim=-1)
        return valid, ratio

    def _neutralize_padding(self, visual, valid):
        valid_float = valid[:, None].to(visual.dtype)
        if self.neutral_mode == "zero":
            neutral = visual.new_zeros((visual.shape[0], visual.shape[1], 1, 1))
        elif self.neutral_mode == "mean":
            neutral = (visual * valid_float).sum((2, 3), keepdim=True) / valid_float.sum(
                (2, 3), keepdim=True
            ).clamp(min=1.0)
        else:
            neutral = self.neutral_feature.expand(visual.shape[0], -1, -1, -1)
        base = visual * valid_float + neutral * (1.0 - valid_float)
        return base, neutral

    def _soft_box_masks(self, boxes, valid_ratio, height, width, dtype, device):
        x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        xx, yy = xx[None, None], yy[None, None]

        centers = boxes[..., :2] * valid_ratio[:, None]
        sizes = boxes[..., 2:].clamp(min=1e-5) * valid_ratio[:, None]
        inner, outer = sizes * self.core_expansion, sizes * self.feather_expansion
        ix1 = (centers[..., 0] - inner[..., 0] / 2)[..., None, None]
        iy1 = (centers[..., 1] - inner[..., 1] / 2)[..., None, None]
        ix2 = (centers[..., 0] + inner[..., 0] / 2)[..., None, None]
        iy2 = (centers[..., 1] + inner[..., 1] / 2)[..., None, None]
        ox1 = (centers[..., 0] - outer[..., 0] / 2)[..., None, None]
        oy1 = (centers[..., 1] - outer[..., 1] / 2)[..., None, None]
        ox2 = (centers[..., 0] + outer[..., 0] / 2)[..., None, None]
        oy2 = (centers[..., 1] + outer[..., 1] / 2)[..., None, None]

        def smoothstep(value):
            value = value.clamp(0.0, 1.0)
            return value * value * (3.0 - 2.0 * value)

        left = smoothstep((xx - ox1) / (ix1 - ox1).clamp(min=1e-6))
        right = smoothstep((ox2 - xx) / (ox2 - ix2).clamp(min=1e-6))
        top = smoothstep((yy - oy1) / (iy1 - oy1).clamp(min=1e-6))
        bottom = smoothstep((oy2 - yy) / (oy2 - iy2).clamp(min=1e-6))
        return (torch.minimum(left, right) * torch.minimum(top, bottom)).clamp(0, 1)

    def _scene_features(self, visual, visual_mask, boxes):
        valid_highres, valid_ratio = self._valid_geometry(visual, visual_mask)
        base_visual, neutral = self._neutralize_padding(visual, valid_highres)
        masks = self._soft_box_masks(
            boxes, valid_ratio, visual.shape[-2], visual.shape[-1], visual.dtype, visual.device
        )

        grid = self.grid_size
        original = F.adaptive_avg_pool2d(base_visual, (grid, grid))
        masked = []
        for candidate_index in range(boxes.shape[1]):
            mask = masks[:, candidate_index : candidate_index + 1]
            counterfactual = base_visual * (1.0 - mask) + neutral * mask
            masked.append(F.adaptive_avg_pool2d(counterfactual, (grid, grid)))
        masked = torch.stack(masked, dim=1)

        original = self.visual_proj(original)
        b, n, c, h, w = masked.shape
        masked = self.visual_proj(masked.reshape(b * n, c, h, w)).reshape(
            b, n, -1, h, w
        )
        valid = ~F.interpolate(
            (~valid_highres)[:, None].float(), size=(grid, grid), mode="nearest"
        ).squeeze(1).bool()
        return original, masked, valid

    def _scene_support(self, scenes, valid_visual, text, text_mask):
        b, n, c, h, w = scenes.shape
        visual_tokens = scenes.flatten(3).permute(0, 1, 3, 2).reshape(b * n, h * w, c)
        visual_tokens = visual_tokens + self._position_2d(
            h, w, c, visual_tokens.device, visual_tokens.dtype
        )
        visual_padding = (~valid_visual).flatten(1)[:, None].expand(-1, n, -1).reshape(
            b * n, h * w
        )
        text_tokens = self.text_proj(text)
        text_tokens = text_tokens[:, None].expand(-1, n, -1, -1).reshape(
            b * n, text.shape[1], c
        )
        text_padding = text_mask.bool()[:, None].expand(-1, n, -1).reshape(
            b * n, text_mask.shape[1]
        )
        support_tokens = self.support_token.expand(b * n, -1, -1)
        queries = torch.cat((support_tokens, text_tokens), dim=1)
        query_padding = torch.cat(
            (
                torch.zeros((b * n, 1), dtype=torch.bool, device=scenes.device),
                text_padding,
            ),
            dim=1,
        )
        decoded = self.verifier(
            tgt=queries,
            memory=visual_tokens,
            tgt_key_padding_mask=query_padding,
            memory_key_padding_mask=visual_padding,
        )
        return self.support_head(self.output_norm(decoded[:, 0])).reshape(b, n)

    def _paired_support(self, original_scene, masked_scenes, valid_visual, text,
                        text_mask):
        """Score the original scene once and all candidate interventions."""
        scenes = torch.cat((original_scene[:, None], masked_scenes), dim=1)
        return self._scene_support(scenes, valid_visual, text, text_mask)

    @staticmethod
    def _box_iou(boxes, target_boxes):
        boxes_xyxy = torch.cat(
            (boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1
        )
        target_xyxy = torch.cat(
            (target_boxes[..., :2] - target_boxes[..., 2:] / 2,
             target_boxes[..., :2] + target_boxes[..., 2:] / 2), -1,
        )[:, None]
        left_top = torch.maximum(boxes_xyxy[..., :2], target_xyxy[..., :2])
        right_bottom = torch.minimum(boxes_xyxy[..., 2:], target_xyxy[..., 2:])
        intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
        box_area = boxes[..., 2:].clamp(min=0).prod(-1)
        target_area = target_boxes[..., 2:].clamp(min=0).prod(-1)[:, None]
        return intersection / (box_area + target_area - intersection).clamp(min=1e-6)

    @staticmethod
    def _masked_mean(values, mask):
        if not mask.any():
            return values.sum() * 0.0
        return values[mask].mean()

    @staticmethod
    def _pairwise_margin(logits, positive, negative, margin):
        """Require every valid positive to outrank every valid negative."""
        valid_pairs = positive[:, :, None] & negative[:, None, :]
        if not valid_pairs.any():
            return logits.sum() * 0.0
        pairwise = F.relu(
            float(margin) - logits[:, :, None] + logits[:, None, :]
        )
        return pairwise[valid_pairs].mean()

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
        # CRV is a strictly detached re-ranker. Neither counterfactual nor
        # final-ranking loss may update CRS, DETR, SimFPN, BEiT-3 or text
        # features; only CRV-local parameters receive these gradients.
        if self.detach_inputs:
            visual = visual.detach()
            boxes = boxes.detach()
            text = text.detach()
        refer_scores = refer_scores.detach()

        scores = refer_scores.squeeze(-1)
        indices = self._select(scores)
        selected_boxes = gather_candidates(boxes, indices)
        selected_scores = scores.gather(1, indices)
        candidate_count = selected_boxes.shape[1]

        # Exact refer-GT intervention exists only during training. It remains
        # a stable positive when no accurate proposal reaches CRS Top-K.
        has_gt_intervention = target_boxes is not None
        intervention_boxes = selected_boxes.detach()
        if has_gt_intervention:
            intervention_boxes = torch.cat(
                (intervention_boxes, target_boxes.detach()[:, None]), dim=1
            )

        original_scene, masked_scenes, valid_visual = self._scene_features(
            visual, visual_mask, intervention_boxes
        )
        pair_logits = self._paired_support(
            original_scene, masked_scenes, valid_visual, text, text_mask
        )
        # The original-scene baseline is deliberately shared by every
        # intervention because the text is not conditioned on a candidate.
        original_scene_logits = pair_logits[:, :1]
        all_masked_logits = pair_logits[:, 1:]
        all_drop_logits = original_scene_logits - all_masked_logits

        original_support_logits = original_scene_logits.expand(-1, candidate_count)
        masked_support_logits = all_masked_logits[:, :candidate_count]
        support_drop_logits = all_drop_logits[:, :candidate_count]
        gt_masked_support_logits = (
            all_masked_logits[:, candidate_count:candidate_count + 1]
            if has_gt_intervention else None
        )
        gt_support_drop_logits = (
            all_drop_logits[:, candidate_count:candidate_count + 1]
            if has_gt_intervention else None
        )

        # Parameter-free fusion: CRV directly predicts a bounded correction
        # in CRS-logit space. There is no learned or validation-searched alpha.
        fusion_residual_logits = torch.tanh(support_drop_logits)
        raw_crs_logits = torch.logit(selected_scores.clamp(min=1e-6, max=1.0 - 1e-6))
        fused_logits = raw_crs_logits + fusion_residual_logits
        weighted_scores = fused_logits.sigmoid().unsqueeze(-1)
        importance_scores = fusion_residual_logits.exp()

        zero = visual.sum() * 0.0
        loss_counterfactual = zero
        loss_final_rank = zero
        selected_ious = None
        if has_gt_intervention:
            if target_valid is None:
                target_valid = torch.ones(
                    support_drop_logits.shape[0], dtype=torch.bool,
                    device=support_drop_logits.device,
                )
            selected_ious = self._box_iou(selected_boxes.detach(), target_boxes.detach())
            # Strict label definition requested for both CRV objectives:
            # IoU > 0.5 is positive, IoU == 0 is negative, and every proposal
            # in between (including exactly 0.5) is ignored.
            positive = (
                (selected_ious > self.positive_iou_threshold)
                & target_valid[:, None]
            )
            negative = (selected_ious <= 0.0) & target_valid[:, None]

            # Exact GT plus all positive proposals are independently pushed
            # above a positive drop margin. False candidates are encouraged
            # to leave scene support unchanged (drop ~= 0), and every
            # positive/negative pair receives a ranking constraint. Positive
            # proposals never compete with one another.
            cf_logits = torch.cat(
                (support_drop_logits, gt_support_drop_logits), dim=1
            )
            cf_positive = torch.cat((positive, target_valid[:, None]), dim=1)
            cf_negative = torch.cat(
                (negative, torch.zeros_like(target_valid[:, None])), dim=1
            )
            loss_positive = self._masked_mean(
                F.relu(self.positive_drop_margin - cf_logits), cf_positive
            )
            loss_negative = self._masked_mean(
                F.smooth_l1_loss(
                    cf_logits, torch.zeros_like(cf_logits), reduction="none"
                ),
                cf_negative,
            )
            loss_cf_rank = self._pairwise_margin(
                cf_logits, cf_positive, cf_negative, self.rank_margin
            )
            loss_counterfactual = loss_positive + loss_negative + loss_cf_rank

            # Supervise the exact fused logits used during inference. Every
            # correct proposal is compared independently with every completely
            # non-overlapping proposal; samples without both classes are skipped.
            loss_final_rank = self._pairwise_margin(
                fused_logits, positive, negative, self.rank_margin
            )

        loss = loss_counterfactual + loss_final_rank
        return {
            "indices": indices,
            "weighted_scores": weighted_scores,
            "importance_scores": importance_scores,
            "fusion_residual_logits": fusion_residual_logits,
            "fused_logits": fused_logits,
            "original_support_logits": original_support_logits,
            "masked_support_logits": masked_support_logits,
            "relative_support_scores": -support_drop_logits,
            "support_drop_logits": support_drop_logits,
            "gt_masked_support_logits": gt_masked_support_logits,
            "gt_support_drop_logits": gt_support_drop_logits,
            "selected_ious": selected_ious,
            "loss_counterfactual": loss_counterfactual,
            "loss_final_rank": loss_final_rank,
            "loss": loss,
        }
