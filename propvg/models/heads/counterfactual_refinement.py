"""Independent local-crop counterfactual refinement guided by grouped CRS entropy."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from propvg.layers.box_ops import box_cxcywh_to_xyxy


def pairwise_iou(boxes, targets):
    """Clipped xyxy boxes in the same (valid-image normalized) coordinates."""
    lo = torch.maximum(boxes[:, None, :2], targets[None, :, :2])
    hi = torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp(min=0).prod(-1)
    return inter / (area[:, None] + target_area[None] - inter).clamp(min=1e-8)


class SceneTextScorer(nn.Module):
    """Shared local scene/text support score, before and after intervention."""
    def __init__(self, channels, dim=128, heads=4, layers=2):
        super().__init__()
        self.visual = nn.Conv2d(channels, dim, 1, bias=False)
        self.text = nn.Linear(channels, dim, bias=False)
        self.position = nn.Linear(2, dim, bias=False)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True)
        self.reader = nn.TransformerDecoder(layer, layers)
        # A shared additive output bias would cancel in every support drop.
        self.score = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1, bias=False))

    def forward(self, scenes, valid, text, text_mask):
        # scenes: S,D,G,G. Invalid/padding cells never become scene evidence.
        count, dim, height, width = scenes.shape
        yy, xx = torch.meshgrid((torch.arange(height, device=scenes.device) + .5) / height,
                                (torch.arange(width, device=scenes.device) + .5) / width, indexing="ij")
        positions = self.position(torch.stack((xx, yy), -1).reshape(-1, 2).to(scenes))
        memory = scenes.flatten(2).transpose(1, 2) + positions[None]
        memory_valid = valid.flatten(1) > 0
        text_valid = ~text_mask.bool()
        usable = memory_valid.any(1) & text_valid.any(1)
        # Make attention finite even for empty inputs, then suppress its result.
        memory_valid = memory_valid.clone()
        text_valid = text_valid.clone()
        memory_valid[~memory_valid.any(1), 0] = True
        text_valid[~text_valid.any(1), 0] = True
        original = self.text(text)
        token_position = torch.arange(text.shape[1], device=text.device, dtype=original.dtype)
        frequency = torch.exp(torch.arange(0, dim, 2, device=text.device, dtype=original.dtype)
                              * (-math.log(10000.) / dim))
        language_position = torch.zeros_like(original[:1])
        language_position[0, :, 0::2] = (token_position[:, None] * frequency).sin()
        language_position[0, :, 1::2] = (token_position[:, None] * frequency).cos()
        read = self.reader(original + language_position, memory,
                           tgt_key_padding_mask=~text_valid, memory_key_padding_mask=~memory_valid)
        weights = (~text_mask.bool()).to(read.dtype)[..., None]
        denominator = weights.sum(1).clamp(min=1.)
        initial = (original * weights).sum(1) / denominator
        interpreted = (read * weights).sum(1) / denominator
        visual_weights = valid.flatten(1).to(scenes.dtype)[..., None]
        visual_mean = (scenes.flatten(2).transpose(1, 2) * visual_weights).sum(1)
        visual_mean = visual_mean / visual_weights.sum(1).clamp(min=1e-8)
        support = self.score(torch.cat((initial, interpreted, interpreted - initial, visual_mean), -1)).squeeze(-1)
        return support * usable


class CounterfactualRefinement(nn.Module):
    def __init__(self, hidden_channels=256, topk=5, uncertainty_topn=5, grid_size=24, crop_size=48,
                 scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=2,
                 inner_scale=1.5, outer_scale=2.0, duplicate_iou=0.8,
                 uncertainty_temperature=1.0, uncertainty_threshold=0.6, gate_initial_probability=0.02,
                 cf_temperature=1.0,
                 start_epoch=1, positive_margin=1.0,
                 negative_gt_coverage=0.01, gt_loss_weight=0.1,
                 cf_loss_weight=0.1, fusion_loss_weight=1.0,
                 ranking_margin=0.5, safety_margin=0.5, safety_weight=2.0, repair_weight=2.0):
        super().__init__()
        if min(topk, uncertainty_topn, grid_size, chunk_size, scorer_layers) < 1:
            raise ValueError("CRV candidate counts, grid, chunks and layers must be positive")
        if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size < grid_size:
            raise ValueError("CRV crop_size must be an integer at least as large as grid_size")
        if not 1 <= inner_scale < outer_scale:
            raise ValueError("CRV requires 1 <= inner_scale < outer_scale")
        if not 0 < duplicate_iou <= 1 or not 0 <= negative_gt_coverage <= 1:
            raise ValueError("Invalid CRV IoU/coverage threshold")
        if min(uncertainty_temperature, cf_temperature) <= 1e-4:
            raise ValueError("CRV temperatures must exceed 1e-4")
        if not 0 <= uncertainty_threshold <= 1:
            raise ValueError("CRV uncertainty_threshold must lie inside [0,1]")
        if not 0 < gate_initial_probability < 1:
            raise ValueError("CRV gate initialization must lie inside (0,1)")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 1:
            raise ValueError("CRV start_epoch must be a positive 1-based integer")
        if scorer_dim % scorer_heads or scorer_dim % 2:
            raise ValueError("CRV scorer_dim must be even and divisible by scorer_heads")
        self.topk, self.uncertainty_topn, self.grid_size = topk, uncertainty_topn, grid_size
        self.crop_size = crop_size
        self.chunk_size = chunk_size
        self.inner_scale, self.outer_scale, self.duplicate_iou = inner_scale, outer_scale, duplicate_iou
        self.uncertainty_temperature = uncertainty_temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.start_epoch, self.positive_margin = start_epoch, positive_margin
        self.negative_gt_coverage, self.gt_loss_weight = negative_gt_coverage, gt_loss_weight
        self.cf_loss_weight, self.fusion_loss_weight = cf_loss_weight, fusion_loss_weight
        self.ranking_margin, self.safety_margin = ranking_margin, safety_margin
        self.safety_weight, self.repair_weight = safety_weight, repair_weight
        self.scorer = SceneTextScorer(hidden_channels, scorer_dim, scorer_heads, scorer_layers)
        self.register_buffer("schema_version", torch.tensor(6, dtype=torch.long))
        # Saved with checkpoints so validation/evaluation before the scheduled
        # start cannot accidentally use a randomly initialized CRV branch.
        self.register_buffer("last_training_epoch", torch.tensor(0, dtype=torch.long))
        # Used only by independent CRV ranking supervision/diagnostics, not fusion.
        self.register_buffer("cf_temperature", torch.tensor(float(cf_temperature)))
        # Only entropy controls fusion. Two scalars replace the 14-D quality MLP.
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_initial_probability / (1 - gate_initial_probability)))

    @staticmethod
    def clipped_boxes(boxes):
        return box_cxcywh_to_xyxy(boxes).clamp(0, 1)

    def stage_ready(self, epoch=None):
        if self.training:
            if epoch is None:
                if self.start_epoch > 1:
                    raise ValueError("Delayed CRV training requires the zero-based training epoch")
                epoch = 0
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("CRV training epoch must be a nonnegative zero-based integer")
            self.last_training_epoch.fill_(epoch + 1)
        return int(self.last_training_epoch) >= self.start_epoch

    @staticmethod
    def support_drop(original_support, masked_support):
        """Positive means deleting this candidate reduces scene/text support."""
        return original_support - masked_support

    def localization_loss(self, importance, ious):
        """GT probability-mass ranking, independent of CRS and fusion alpha."""
        positive = ious >= .5
        if not positive.any():
            return importance.sum() * 0
        scores = importance / self.cf_temperature
        return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive], dim=0)

    @staticmethod
    def spatial_grid(feature, meta):
        height, width = feature.shape[-2:]
        ih, iw = meta["img_shape"][:2]
        ph, pw = meta["pad_shape"][:2]
        yy, xx = torch.meshgrid((torch.arange(height, device=feature.device, dtype=feature.dtype) + .5) * ph / height / ih,
                                (torch.arange(width, device=feature.device, dtype=feature.dtype) + .5) * pw / width / iw,
                                indexing="ij")
        return xx, yy, pw / width / iw, ph / height / ih

    def intervention_masks(self, boxes_xyxy, feature, meta, valid):
        xx, yy, _, _ = self.spatial_grid(feature, meta)
        center = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
        half = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]) / 2
        radius = torch.maximum((xx[None] - center[:, 0, None, None]).abs() / half[:, 0, None, None].clamp(min=1e-8),
                               (yy[None] - center[:, 1, None, None]).abs() / half[:, 1, None, None].clamp(min=1e-8))
        progress = ((radius - self.inner_scale) / (self.outer_scale - self.inner_scale)).clamp(0, 1)
        mask = .5 * (1 + (math.pi * progress).cos())
        return mask * valid[None] * (half > 0).all(-1)[:, None, None]

    def gt_coverage(self, masks, gt, feature, meta):
        # Fractional cell/GT overlap also handles targets smaller than one cell.
        xx, yy, dx, dy = self.spatial_grid(feature, meta)
        overlap_x = (torch.minimum(xx + dx / 2, gt[2]) - torch.maximum(xx - dx / 2, gt[0])).clamp(min=0)
        overlap_y = (torch.minimum(yy + dy / 2, gt[3]) - torch.maximum(yy - dy / 2, gt[1])).clamp(min=0)
        support = overlap_x * overlap_y
        return (masks * support).sum((-1, -2)) / support.sum().clamp(min=1e-8)

    def uncertainty(self, logits, boxes, order):
        # Greedy grouping affects only uncertainty, never detection or ranking.
        kept = []
        for index in order[:self.uncertainty_topn].tolist():
            if not kept or pairwise_iou(boxes[index:index + 1], boxes[kept]).max() < self.duplicate_iou:
                kept.append(index)
        zero = logits.new_tensor(0.)
        if len(kept) < 2:
            return zero, logits.new_tensor(float(len(kept)))
        selected = logits[kept]
        relative = (selected / self.uncertainty_temperature).softmax(0)
        entropy = (-(relative * relative.clamp(min=1e-8).log()).sum() / math.log(len(kept))).clamp(0, 1)
        return entropy, logits.new_tensor(float(len(kept)))

    def crop_origins(self, boxes_xyxy, feature, meta):
        """Integer (x,y) origins on the padded feature map, not image pixels."""
        ih, iw = meta["img_shape"][:2]
        ph, pw = meta["pad_shape"][:2]
        height, width = feature.shape[-2:]
        centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
        centers = centers * centers.new_tensor((iw * width / pw, ih * height / ph))
        return (centers - self.crop_size / 2).floor().long()

    def crop_support(self, origins, feature):
        """Actual intervention footprint, including truncation by a local crop."""
        height, width = feature.shape[-2:]
        yy = torch.arange(height, device=feature.device)[None, :, None]
        xx = torch.arange(width, device=feature.device)[None, None, :]
        x, y = origins[:, 0, None, None], origins[:, 1, None, None]
        return (xx >= x) & (xx < x + self.crop_size) & (yy >= y) & (yy < y + self.crop_size)

    def pooled_scenes(self, projected, valid, masks, origins):
        """Matched original/masked fixed-size crops, THEN valid-aware pooling."""
        # Linear bias-free projection commutes exactly with mean replacement.
        # The fill mean still uses the WHOLE valid image, not a crop's mean.
        v = valid.to(projected.dtype)
        mean = (projected * v).sum((-1, -2), keepdim=True) / v.sum().clamp(min=1.)
        height, width = projected.shape[-2:]
        offset = torch.arange(self.crop_size, device=projected.device)
        yy = origins[:, 1, None, None] + offset[None, :, None]
        xx = origins[:, 0, None, None] + offset[None, None, :]
        inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy, xx = yy.clamp(0, height - 1), xx.clamp(0, width - 1)
        local_valid = (valid[yy, xx] & inside)[:, None].to(projected.dtype)
        original = projected[0, :, yy, xx].permute(1, 0, 2, 3)
        local_masks = masks[torch.arange(len(masks), device=masks.device)[:, None, None], yy, xx][:, None]
        masked = original + (mean - original) * local_masks
        denominator = F.adaptive_avg_pool2d(local_valid, self.grid_size)
        normalize = lambda scenes: (F.adaptive_avg_pool2d(scenes * local_valid, self.grid_size)
                                     / denominator.clamp(min=1e-8)) * (denominator > 0)
        return normalize(original), normalize(masked), denominator

    def fusion(self, base, drops, entropy):
        base = base.detach()
        statistics = entropy.detach().reshape(1)
        gate_logit = self.gate(statistics).squeeze(-1)
        alpha = gate_logit.sigmoid()
        if not bool(entropy > self.uncertainty_threshold):
            alpha = alpha * 0
        # Residual correction of ORIGINAL CRS logits; no softmax mixture or
        # replacement of the base branch, even when alpha approaches one.
        fused = base + alpha * drops
        return fused, alpha, {"base_logits": base, "gate_statistics": statistics}

    def fusion_loss(self, base, fused, ious):
        # Metric-compatible multiple positives; ambiguous IoUs are ignored only
        # by the counterfactual auxiliary loss, not final ranking supervision.
        positive = ious >= .5
        negative = ~positive
        zero = fused.sum() * 0
        if not positive.any() or not negative.any():
            return zero, zero, zero, zero
        base = base.detach()
        original_margin = base[positive].max() - base[negative].max()
        final_margin = fused[positive].max() - fused[negative].max()
        originally_correct = positive[0]  # Top-K is in the original CRS order.
        rank = F.softplus(self.ranking_margin - final_margin)
        rank = rank * torch.where(originally_correct, rank.new_tensor(1.), rank.new_tensor(self.repair_weight))
        safety_target = original_margin.clamp(max=self.safety_margin)
        safety = F.relu(safety_target - final_margin) * originally_correct
        return rank + self.safety_weight * safety, rank, safety, originally_correct.to(base.dtype)

    def counterfactual_loss(self, drops, ious, coverage):
        positive = ious > .5
        negative = (ious == 0) & (coverage <= self.negative_gt_coverage)
        loss = drops.sum() * 0
        if positive.any():
            loss = loss + F.relu(self.positive_margin - drops[positive]).mean()
        if negative.any():
            loss = loss + F.smooth_l1_loss(drops[negative], torch.zeros_like(drops[negative]))
        # Replace the old pair hinge with direct CRV localization supervision.
        # Its metric-compatible labels are separate from intervention anchors.
        loss = loss + self.localization_loss(drops, ious)
        return loss, negative

    def forward(self, visual, visual_mask, boxes, text, text_mask, probabilities, logits, img_metas,
                refer_gt=None, epoch=None):
        # All upstream paths stop here, including the base scores used by fusion.
        visual, boxes, text = visual.detach(), boxes.detach(), text.detach()
        probabilities, logits = probabilities.detach().squeeze(-1), logits.detach().squeeze(-1)
        if not self.stage_ready(epoch):
            # Detached zeros leave ALL CRV gradients None: even AdamW cannot
            # decay/update this branch before its scheduled start epoch.
            zero = logits.new_zeros(())
            losses = {"loss_crv_cf": zero, "loss_crv_fusion": zero}
            info = {"crv_" + key: zero for key in ("alpha", "active", "uncertainty", "corrected", "harmed",
                                                   "reachable", "rejected_negatives", "rank_changed", "group_count", "ready")}
            info["crv_crs_probabilities"] = probabilities
            info["crv_samples"] = [{"skipped": True, "reason": "before_start_epoch"} for _ in range(len(boxes))]
            info["crv_last_training_epoch"] = self.last_training_epoch.detach().clone()
            return probabilities[..., None], None, losses, info
        order = probabilities.argsort(dim=1, descending=True, stable=True)
        final_order = order.clone()
        final_probabilities = probabilities.clone()
        cf_terms, fusion_terms, alphas, active, entropies = [], [], [], [], []
        group_counts, changed = [], []
        corrected, harmed, reachable, rejected = [], [], [], []
        sample_details = []
        for bi in range(len(boxes)):
            clipped = self.clipped_boxes(boxes[bi])
            indices = order[bi, :self.topk]
            base = logits[bi, indices]
            entropy, group_count = self.uncertainty(logits[bi], clipped, order[bi])
            valid = ~visual_mask[bi].bool()
            usable = valid.any() & (~text_mask[bi].bool()).any()
            entropies.append(entropy)
            group_counts.append(group_count)
            route = usable & (entropy > self.uncertainty_threshold)
            active.append(route.float())
            # Train the independent scorer on valid inputs even below threshold,
            # but only ambiguous inputs receive fusion/ranking supervision.
            if not bool(usable) or (not self.training and not bool(route)):
                alphas.append(base.new_tensor(0.))
                changed.append(base.new_tensor(0.))
                sample_details.append({"indices": indices, "skipped": True})
                continue
            projected = self.scorer.visual(visual[bi:bi + 1])
            candidate_masks = self.intervention_masks(clipped[indices], visual[bi], img_metas[bi], valid)
            gt = refer_gt[bi] if self.training and refer_gt is not None else None
            crop_boxes = clipped[indices]
            masks = candidate_masks
            if gt is not None:
                gt = gt.detach().to(visual).clamp(0, 1)
                gt_mask = self.intervention_masks(gt[None], visual[bi], img_metas[bi], valid)
                masks = torch.cat((masks, gt_mask))
                crop_boxes = torch.cat((crop_boxes, gt[None]))
            origins = self.crop_origins(crop_boxes, visual[bi], img_metas[bi])
            # Negative reliability must reflect only evidence actually removed.
            masks = masks * self.crop_support(origins, visual[bi])
            originals, interventions = [], []
            for start in range(0, len(masks), self.chunk_size):
                original, masked, pooled_valid = self.pooled_scenes(
                    projected, valid, masks[start:start + self.chunk_size], origins[start:start + self.chunk_size])
                count = len(original)
                # Encode each half separately to keep scorer batches <= chunk_size.
                for scenes, values in ((original, originals), (masked, interventions)):
                    values.append(self.scorer(scenes, pooled_valid, text[bi:bi + 1].expand(count, -1, -1),
                                              text_mask[bi:bi + 1].expand(count, -1)))
            original_support, masked_support = torch.cat(originals), torch.cat(interventions)
            importance = self.support_drop(original_support, masked_support)
            drops = importance[:len(indices)]
            fused, alpha, branches = self.fusion(base, drops, entropy)
            alphas.append(alpha)
            local_order = fused.argsort(descending=True, stable=True)
            if bool(route):
                final_order[bi, :len(indices)] = indices[local_order]
                final_probabilities[bi, indices] = fused.sigmoid()
            changed.append(((local_order[0] != 0) & route).float())
            zero = fused.sum() * 0
            cf, fusion = zero, zero
            if gt is not None and bool(usable):
                ious = pairwise_iou(clipped[indices], gt[None]).squeeze(-1)
                coverage = self.gt_coverage(masks[:len(indices)], gt, visual[bi], img_metas[bi])
                cf, negative = self.counterfactual_loss(drops, ious, coverage)
                cf = cf + self.gt_loss_weight * F.relu(self.positive_margin - importance[-1])
                fusion, _, _, _ = self.fusion_loss(branches["base_logits"], fused, ious)
                fusion = fusion * route
                before = (ious[0] >= .5)
                after = (ious[fused.argmax()] >= .5) if bool(route) else before
                corrected.append((~before & after).float())
                harmed.append((before & ~after).float())
                reachable.append((ious >= .5).any().float())
                rejected.append(((ious == 0) & ~negative).float().sum() / (ious == 0).float().sum().clamp(min=1.))
            cf_terms.append(cf)
            fusion_terms.append(fusion)
            sample_details.append({"indices": indices, "skipped": False,
                                   "crop_origins": origins[:len(indices)].detach(),
                                   "original_support": original_support[:len(indices)].detach(),
                                   "masked_support": masked_support[:len(indices)].detach(),
                                   "importance": drops.detach(),
                                   "drops": drops.detach(), "fused_logits": fused.detach(),
                                   "base_logits": base.detach(),
                                   # These normalized distributions are diagnostics ONLY;
                                   # they never enter the residual fusion or final scores.
                                   "base_distribution": base.softmax(0).detach(),
                                   "cf_distribution": (drops / self.cf_temperature).softmax(0).detach(),
                                   "fused_distribution": fused.softmax(0).detach(),
                                   "gate_statistics": branches["gate_statistics"], "active": bool(route)})
        # Explicit ordering keys retain ALL candidates, permit only Top-K
        # internal reordering, and cannot accidentally promote original K+1.
        priority = torch.empty_like(logits)
        priority.scatter_(1, final_order, torch.arange(boxes.shape[1], 0, -1, device=boxes.device,
                                                      dtype=logits.dtype)[None].expand_as(logits))
        parameter_zero = next(self.parameters()).reshape(-1)[0] * 0
        mean = lambda values: torch.stack(values).mean() if values else parameter_zero
        losses = {"loss_crv_cf": mean(cf_terms) * self.cf_loss_weight,
                  "loss_crv_fusion": mean(fusion_terms) * self.fusion_loss_weight}
        metrics = {"alpha": mean(alphas), "active": mean(active), "rank_changed": mean(changed),
                   "uncertainty": mean(entropies), "group_count": mean(group_counts),
                   "ready": logits.new_tensor(1.),
                   "corrected": mean(corrected), "harmed": mean(harmed), "reachable": mean(reachable),
                   "rejected_negatives": mean(rejected)}
        info = {"crv_" + key: value.detach() for key, value in metrics.items()}
        info["crv_samples"] = sample_details
        info["crv_crs_probabilities"] = probabilities.detach()
        info["crv_last_training_epoch"] = self.last_training_epoch.detach().clone()
        return final_probabilities[..., None], priority[..., None], losses, info
