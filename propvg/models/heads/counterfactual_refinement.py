"""Independent counterfactual refinement guided only by grouped CRS entropy."""
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
    """Full text reads a scene using two shared, deterministic cross-modal layers."""
    def __init__(self, channels, dim=128, heads=4, layers=2):
        super().__init__()
        self.visual = nn.Conv2d(channels, dim, 1, bias=False)
        self.text = nn.Linear(channels, dim, bias=False)
        self.position = nn.Linear(2, dim, bias=False)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True)
        self.reader = nn.TransformerDecoder(layer, layers)
        self.score = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))

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
        return self.score(torch.cat((initial, interpreted, interpreted - initial, visual_mean), -1)).squeeze(-1) * usable


class CounterfactualRefinement(nn.Module):
    def __init__(self, hidden_channels=256, topk=5, uncertainty_topn=5, grid_size=24,
                 scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=2,
                 inner_scale=1.5, outer_scale=2.0, duplicate_iou=0.8,
                 uncertainty_temperature=1.0, uncertainty_threshold=0.6, gate_initial_probability=0.02,
                 base_temperature=1.0, cf_temperature=1.0,
                 positive_margin=1.0, pair_margin=0.5,
                 negative_gt_coverage=0.01, gt_loss_weight=0.1,
                 cf_loss_weight=0.1, fusion_loss_weight=1.0,
                 ranking_margin=0.5, safety_margin=0.5, safety_weight=2.0, repair_weight=2.0):
        super().__init__()
        if min(topk, uncertainty_topn, grid_size, chunk_size, scorer_layers) < 1:
            raise ValueError("CRV candidate counts, grid, chunks and layers must be positive")
        if not 1 <= inner_scale < outer_scale:
            raise ValueError("CRV requires 1 <= inner_scale < outer_scale")
        if not 0 < duplicate_iou <= 1 or not 0 <= negative_gt_coverage <= 1:
            raise ValueError("Invalid CRV IoU/coverage threshold")
        if min(uncertainty_temperature, base_temperature, cf_temperature) <= 1e-4:
            raise ValueError("CRV temperatures must exceed 1e-4")
        if not 0 <= uncertainty_threshold <= 1:
            raise ValueError("CRV uncertainty_threshold must lie inside [0,1]")
        if not 0 < gate_initial_probability < 1:
            raise ValueError("CRV gate initialization must lie inside (0,1)")
        if scorer_dim % scorer_heads or scorer_dim % 2:
            raise ValueError("CRV scorer_dim must be even and divisible by scorer_heads")
        self.topk, self.uncertainty_topn, self.grid_size = topk, uncertainty_topn, grid_size
        self.chunk_size = chunk_size
        self.inner_scale, self.outer_scale, self.duplicate_iou = inner_scale, outer_scale, duplicate_iou
        self.uncertainty_temperature = uncertainty_temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.positive_margin, self.pair_margin = positive_margin, pair_margin
        self.negative_gt_coverage, self.gt_loss_weight = negative_gt_coverage, gt_loss_weight
        self.cf_loss_weight, self.fusion_loss_weight = cf_loss_weight, fusion_loss_weight
        self.ranking_margin, self.safety_margin = ranking_margin, safety_margin
        self.safety_weight, self.repair_weight = safety_weight, repair_weight
        self.scorer = SceneTextScorer(hidden_channels, scorer_dim, scorer_heads, scorer_layers)
        self.register_buffer("schema_version", torch.tensor(3, dtype=torch.long))
        self.register_buffer("base_temperature", torch.tensor(float(base_temperature)))
        self.register_buffer("cf_temperature", torch.tensor(float(cf_temperature)))
        # Only entropy controls fusion. Two scalars replace the 14-D quality MLP.
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_initial_probability / (1 - gate_initial_probability)))

    @staticmethod
    def clipped_boxes(boxes):
        return box_cxcywh_to_xyxy(boxes).clamp(0, 1)

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

    def pooled_scenes(self, projected, valid, masks):
        # Linear bias-free projection commutes exactly with mean replacement.
        # Mask at full spatial resolution, THEN pool; do not mask a 24x24 map.
        v = valid.to(projected.dtype)
        denominator = F.adaptive_avg_pool2d(v[None, None], self.grid_size)
        mean = (projected * v).sum((-1, -2), keepdim=True) / v.sum().clamp(min=1.)
        base = F.adaptive_avg_pool2d(projected * v, self.grid_size)
        change = F.adaptive_avg_pool2d((mean - projected) * masks[:, None], self.grid_size)
        scenes = (base + change) / denominator.clamp(min=1e-8)
        scenes = scenes * (denominator > 0)
        return scenes, denominator.expand(len(masks), -1, -1, -1)

    def fusion(self, base, drops, entropy):
        base = base.detach()
        base_logp = F.log_softmax(base / self.base_temperature, dim=0)
        cf_logp = F.log_softmax(drops / self.cf_temperature, dim=0)
        statistics = entropy.detach().reshape(1)
        gate_logit = self.gate(statistics).squeeze(-1)
        alpha = gate_logit.sigmoid()
        # Log-domain mixture stays finite even for saturated gates/extreme scores.
        fused = torch.logaddexp(F.logsigmoid(-gate_logit) + base_logp,
                               F.logsigmoid(gate_logit) + cf_logp)
        if not bool(entropy > self.uncertainty_threshold):
            fused, alpha = base_logp, alpha * 0
        return fused, alpha, {"base_logp": base_logp, "cf_logp": cf_logp, "gate_statistics": statistics}

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
        if positive.any() and negative.any():
            loss = loss + F.relu(self.pair_margin - drops[positive].min() + drops[negative].max())
        return loss, negative

    def forward(self, visual, visual_mask, boxes, text, text_mask, probabilities, logits, img_metas,
                refer_gt=None):
        # All upstream paths stop here, including the base scores used by fusion.
        visual, boxes, text = visual.detach(), boxes.detach(), text.detach()
        probabilities, logits = probabilities.detach().squeeze(-1), logits.detach().squeeze(-1)
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
            masks = torch.cat((candidate_masks.new_zeros((1,) + candidate_masks.shape[1:]), candidate_masks))
            if gt is not None:
                gt = gt.detach().to(visual).clamp(0, 1)
                gt_mask = self.intervention_masks(gt[None], visual[bi], img_metas[bi], valid)
                masks = torch.cat((masks, gt_mask))
            values = []
            for start in range(0, len(masks), self.chunk_size):
                scenes, pooled_valid = self.pooled_scenes(projected, valid, masks[start:start + self.chunk_size])
                count = len(scenes)
                values.append(self.scorer(scenes, pooled_valid, text[bi:bi + 1].expand(count, -1, -1),
                                          text_mask[bi:bi + 1].expand(count, -1)))
            support = torch.cat(values)
            drops = support[0] - support[1:1 + len(indices)]
            fused, alpha, branches = self.fusion(base, drops, entropy)
            alphas.append(alpha)
            local_order = fused.argsort(descending=True, stable=True)
            if bool(route):
                final_order[bi, :len(indices)] = indices[local_order]
                final_probabilities[bi, indices] = fused.exp()
            changed.append(((local_order[0] != 0) & route).float())
            zero = fused.sum() * 0
            cf, fusion = zero, zero
            if gt is not None and bool(usable):
                ious = pairwise_iou(clipped[indices], gt[None]).squeeze(-1)
                coverage = self.gt_coverage(candidate_masks, gt, visual[bi], img_metas[bi])
                cf, negative = self.counterfactual_loss(drops, ious, coverage)
                cf = cf + self.gt_loss_weight * F.relu(self.positive_margin - (support[0] - support[-1]))
                fusion, _, _, _ = self.fusion_loss(branches["base_logp"], fused, ious)
                fusion = fusion * route
                before = (ious[0] >= .5)
                after = (ious[fused.argmax()] >= .5) if bool(route) else before
                corrected.append((~before & after).float())
                harmed.append((before & ~after).float())
                reachable.append((ious >= .5).any().float())
                rejected.append(((ious == 0) & ~negative).float().sum() / (ious == 0).float().sum().clamp(min=1.))
            cf_terms.append(cf)
            fusion_terms.append(fusion)
            sample_details.append({"indices": indices, "skipped": False, "original_support": support[0].detach(),
                                   "masked_support": support[1:1 + len(indices)].detach(),
                                   "drops": drops.detach(), "fused_log_probabilities": fused.detach(),
                                   "base_distribution": branches["base_logp"].exp().detach(),
                                   "cf_distribution": branches["cf_logp"].exp().detach(),
                                   "fused_distribution": fused.exp().detach(),
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
                   "corrected": mean(corrected), "harmed": mean(harmed), "reachable": mean(reachable),
                   "rejected_negatives": mean(rejected)}
        info = {"crv_" + key: value.detach() for key, value in metrics.items()}
        info["crv_samples"] = sample_details
        info["crv_crs_probabilities"] = probabilities.detach()
        return final_probabilities[..., None], priority[..., None], losses, info
