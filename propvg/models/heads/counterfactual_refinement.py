"""Entropy-routed counterfactual refinement over complete visual scenes."""
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
    """Shared full-scene/text support score, before and after intervention."""
    def __init__(self, channels, dim=128, heads=4, layers=2):
        super().__init__()
        self.visual = nn.Conv2d(channels, dim, 1, bias=False)
        self.text = nn.Linear(channels, dim, bias=False)
        self.position = nn.Linear(2, dim, bias=False)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True)
        self.reader = nn.TransformerDecoder(layer, layers)
        # A shared additive output bias would cancel in every support drop.
        self.score = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1, bias=False))

    def prepare_text(self, text, text_mask):
        """Project each image's text once, retaining its scorer gradients."""
        text_valid = ~text_mask.bool()
        original = self.text(text)
        dim = original.shape[-1]
        token_position = torch.arange(text.shape[1], device=text.device, dtype=original.dtype)
        frequency = torch.exp(torch.arange(0, dim, 2, device=text.device, dtype=original.dtype)
                              * (-math.log(10000.) / dim))
        language_position = torch.zeros_like(original[:1])
        language_position[0, :, 0::2] = (token_position[:, None] * frequency).sin()
        language_position[0, :, 1::2] = (token_position[:, None] * frequency).cos()
        weights = text_valid.to(original.dtype)[..., None]
        denominator = weights.sum(1).clamp(min=1.)
        initial = (original * weights).sum(1) / denominator
        return original + language_position, text_valid, weights, initial

    def spatial_positions(self, height, width, reference):
        yy, xx = torch.meshgrid((torch.arange(height, device=reference.device) + .5) / height,
                                (torch.arange(width, device=reference.device) + .5) / width, indexing="ij")
        return self.position(torch.stack((xx, yy), -1).reshape(-1, 2).to(reference))

    def forward(self, scenes, valid, text, text_mask, prepared_text=None, positions=None):
        # Scenes are independent batch entries, never one shared attention sequence.
        if positions is None:
            positions = self.spatial_positions(*scenes.shape[-2:], scenes)
        memory = scenes.flatten(2).transpose(1, 2) + positions[None]
        memory_valid = valid.flatten(1) > 0
        language, text_valid, weights, initial = (self.prepare_text(text, text_mask)
                                                if prepared_text is None else prepared_text)
        usable = memory_valid.any(1) & text_valid.any(1)
        # Make attention finite even for empty inputs, then suppress its result.
        memory_valid, text_valid = memory_valid.clone(), text_valid.clone()
        memory_valid[~memory_valid.any(1), 0] = True
        text_valid[~text_valid.any(1), 0] = True
        read = self.reader(language, memory,
                           tgt_key_padding_mask=~text_valid, memory_key_padding_mask=~memory_valid)
        denominator = weights.sum(1).clamp(min=1.)
        interpreted = (read * weights).sum(1) / denominator
        visual_weights = valid.flatten(1).to(scenes.dtype)[..., None]
        visual_mean = (scenes.flatten(2).transpose(1, 2) * visual_weights).sum(1)
        visual_mean = visual_mean / visual_weights.sum(1).clamp(min=1e-8)
        support = self.score(torch.cat((initial, interpreted, interpreted - initial, visual_mean), -1)).squeeze(-1)
        return support * usable


class CounterfactualRefinement(nn.Module):
    def __init__(self, hidden_channels=256, topk=5, uncertainty_topn=5, grid_size=24, crop_size=None,
                 scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=8,
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
        if crop_size is not None:
            raise ValueError("Full-scene CRV requires crop_size=None; remove the legacy local-crop setting")
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
        self.register_buffer("schema_version", torch.tensor(9, dtype=torch.long))
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
            return epoch + 1 >= self.start_epoch
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
        entropy, counts = self.batch_uncertainty(logits[None], boxes[None], order[None])
        return entropy[0], counts[0]

    def batch_uncertainty(self, logits, boxes, order):
        """Same greedy IoU grouping, without per-candidate CUDA-to-host tests."""
        indices = order[:, :self.uncertainty_topn]
        selected_boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        lo = torch.maximum(selected_boxes[:, :, None, :2], selected_boxes[:, None, :, :2])
        hi = torch.minimum(selected_boxes[:, :, None, 2:], selected_boxes[:, None, :, 2:])
        inter = (hi - lo).clamp(min=0).prod(-1)
        area = (selected_boxes[..., 2:] - selected_boxes[..., :2]).clamp(min=0).prod(-1)
        overlaps = inter / (area[:, :, None] + area[:, None, :] - inter).clamp(min=1e-8)
        kept = []
        for index in range(indices.shape[1]):
            keep = torch.ones_like(indices[:, 0], dtype=torch.bool) if not kept else ~(
                (overlaps[:, index, :index] >= self.duplicate_iou) & torch.stack(kept, 1)).any(1)
            kept.append(keep)
        kept = torch.stack(kept, 1)
        counts = kept.sum(1).to(logits.dtype)
        selected = (logits.gather(1, indices) / self.uncertainty_temperature).masked_fill(~kept, -torch.inf)
        relative = selected.softmax(1)
        entropy = -(relative * relative.clamp(min=1e-8).log()).sum(1) / counts.clamp(min=2).log()
        return entropy.clamp(0, 1), counts

    @staticmethod
    def image_mean(projected, valid):
        """Cache the full valid-image mean, not a candidate crop's mean."""
        v = valid[:, None].to(projected.dtype)
        return (projected * v).sum((-1, -2), keepdim=True) / v.sum((-1, -2), keepdim=True).clamp(min=1.)

    def pooled_original(self, projected, valid):
        """One original full-scene grid, validity and fill mean per image."""
        v = valid[:, None].to(projected.dtype)
        denominator = F.adaptive_avg_pool2d(v, self.grid_size)
        original = F.adaptive_avg_pool2d(projected * v, self.grid_size) / denominator.clamp(min=1e-8)
        return original, denominator, self.image_mean(projected, valid)

    def pooled_interventions(self, projected, valid, masks, owners, original, denominator, mean):
        """Full-resolution mean replacement, then valid-aware full-scene pooling.

        Linearity reuses the original pooled grid. Only this chunk's weighted
        features are materialized; there is no crop or intervention size cap.
        """
        if owners is None:
            owners = masks.new_zeros(len(masks), dtype=torch.long)
        masks = (masks * valid[owners])[:, None].to(projected.dtype)
        weighted = F.adaptive_avg_pool2d(projected[owners] * masks, self.grid_size)
        pooled_masks = F.adaptive_avg_pool2d(masks, self.grid_size)
        return original[owners] + (mean[owners] * pooled_masks - weighted) / denominator[owners].clamp(min=1e-8)

    def fusion(self, base, drops, entropy):
        base = base.detach()
        statistics = entropy.detach().reshape(1)
        gate_logit = self.gate(statistics).squeeze(-1)
        alpha = gate_logit.sigmoid()
        alpha = alpha * (entropy > self.uncertainty_threshold).to(alpha.dtype)
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
        clipped_boxes = self.clipped_boxes(boxes)
        entropy_values, count_values = self.batch_uncertainty(logits, clipped_boxes, order)
        valid_images = ~visual_mask.bool()
        usable = valid_images.flatten(1).any(1) & (~text_mask.bool()).any(1)
        routes = usable & (entropy_values > self.uncertainty_threshold)
        # A single host transfer decides routing for the entire batch. Training
        # and inference BOTH skip low-entropy samples before projection/masking.
        routed = routes.nonzero(as_tuple=False).flatten().tolist()
        contexts, supports = {}, {}
        if routed:
            active_valid = valid_images[routed]
            projected = self.scorer.visual(visual[routed])
            original, pooled_valid, fill_mean = self.pooled_original(projected, active_valid)
            all_masks, all_owners = [], []
            offset = 0
            for owner, bi in enumerate(routed):
                indices = order[bi, :self.topk]
                gt = refer_gt[bi] if self.training and refer_gt is not None else None
                intervention_boxes = clipped_boxes[bi, indices]
                if gt is not None:
                    gt = gt.detach().to(visual).clamp(0, 1)
                    intervention_boxes = torch.cat((intervention_boxes, gt[None]))
                masks = self.intervention_masks(intervention_boxes, visual[bi], img_metas[bi], valid_images[bi])
                count = len(masks)
                contexts[bi] = (owner, offset, count, gt, masks)
                all_masks.append(masks)
                all_owners.append(order.new_full((count,), owner))
                offset += count
            masks, owners = torch.cat(all_masks), torch.cat(all_owners)
            interventions = []
            for start in range(0, len(masks), self.chunk_size):
                end = start + self.chunk_size
                interventions.append(self.pooled_interventions(
                    projected, active_valid, masks[start:end], owners[start:end], original, pooled_valid, fill_mean))
            # Each image contributes exactly ONE original scene; all its masked
            # scenes retain the same complete spatial domain and validity mask.
            scenes = torch.cat((original, torch.cat(interventions)))
            scene_owners = torch.cat((torch.arange(len(routed), device=owners.device), owners))
            active_text, active_text_mask = text[routed], text_mask[routed]
            prepared = self.scorer.prepare_text(active_text, active_text_mask)
            positions = self.scorer.spatial_positions(self.grid_size, self.grid_size, scenes)
            encoded = []
            # Batch across images/candidates and original/intervened scenes.
            # Independent batch entries retain independent attention contexts.
            for start in range(0, len(scenes), self.chunk_size):
                end = start + self.chunk_size
                selected = scene_owners[start:end]
                encoded.append(self.scorer(
                    scenes[start:end], pooled_valid[selected], active_text[selected], active_text_mask[selected],
                    prepared_text=tuple(value[selected] for value in prepared), positions=positions))
            encoded = torch.cat(encoded)
            original_support = encoded[:len(routed)]
            masked_support = encoded[len(routed):]
            # Identical scenes must have exactly zero drops, even if different
            # GEMM batch shapes introduce tiny floating-point differences.
            masked_support = torch.where(masks.flatten(1).any(1), masked_support, original_support[owners])
            for bi, (owner, offset, count, *_rest) in contexts.items():
                supports[bi] = (original_support[owner].expand(count), masked_support[offset:offset + count])
        cf_terms, fusion_terms, alphas, active, entropies = [], [], [], [], []
        changed, group_counts = [], []
        corrected, harmed, reachable, rejected = [], [], [], []
        sample_details = []
        for bi in range(len(boxes)):
            clipped = clipped_boxes[bi]
            indices = order[bi, :self.topk]
            base = logits[bi, indices]
            entropy, group_count = entropy_values[bi], count_values[bi]
            entropies.append(entropy)
            group_counts.append(group_count)
            active.append(routes[bi].float())
            if bi not in contexts:
                alphas.append(base.new_tensor(0.))
                changed.append(base.new_tensor(0.))
                # Normalize task losses over the full local batch, with skipped
                # samples contributing detached zero (no AdamW-only updates).
                cf_terms.append(base.new_zeros(()))
                fusion_terms.append(base.new_zeros(()))
                if self.training and refer_gt is not None and refer_gt[bi] is not None:
                    for values in (corrected, harmed, reachable, rejected):
                        values.append(base.new_zeros(()))
                sample_details.append({"indices": indices, "skipped": True})
                continue
            _, _, _, gt, masks = contexts[bi]
            original_support, masked_support = supports[bi]
            importance = self.support_drop(original_support, masked_support)
            drops = importance[:len(indices)]
            fused, alpha, branches = self.fusion(base, drops, entropy)
            alphas.append(alpha)
            local_order = fused.argsort(descending=True, stable=True)
            final_order[bi, :len(indices)] = indices[local_order]
            final_probabilities[bi, indices] = fused.sigmoid()
            changed.append((local_order[0] != 0).float())
            zero = fused.sum() * 0
            cf, fusion = zero, zero
            if gt is not None:
                ious = pairwise_iou(clipped[indices], gt[None]).squeeze(-1)
                coverage = self.gt_coverage(masks[:len(indices)], gt, visual[bi], img_metas[bi])
                cf, negative = self.counterfactual_loss(drops, ious, coverage)
                cf = cf + self.gt_loss_weight * F.relu(self.positive_margin - importance[-1])
                fusion, _, _, _ = self.fusion_loss(branches["base_logits"], fused, ious)
                before = (ious[0] >= .5)
                after = (ious[fused.argmax()] >= .5)
                corrected.append((~before & after).float())
                harmed.append((before & ~after).float())
                reachable.append((ious >= .5).any().float())
                rejected.append(((ious == 0) & ~negative).float().sum() / (ious == 0).float().sum().clamp(min=1.))
            cf_terms.append(cf)
            fusion_terms.append(fusion)
            sample_details.append({"indices": indices, "skipped": False,
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
                                   "gate_statistics": branches["gate_statistics"], "active": True})
        # Explicit ordering keys retain ALL candidates, permit only Top-K
        # internal reordering, and cannot accidentally promote original K+1.
        priority = torch.empty_like(logits)
        priority.scatter_(1, final_order, torch.arange(boxes.shape[1], 0, -1, device=boxes.device,
                                                      dtype=logits.dtype)[None].expand_as(logits))
        parameter_zero = logits.new_zeros(())
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
