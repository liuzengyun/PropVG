"""Jointly trained adaptive-prefix CRV over complete mean-masked scenes."""
import hashlib
import json
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .proposal_selection import adaptive_prefix

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
    """No entropy/gate/score calibration. Baseline CRS loss is left intact."""
    joint_training = True
    metric_names = ("beta", "active", "ready", "target_count", "target_cap_rate",
                    "bce_samples", "rank_samples", "gt_samples", "supervised_samples",
                    "positive_count", "negative_count", "rejected_negatives",
                    "reachable", "delta_abs", "corrected", "harmed", "rank_changed")

    def __init__(self, hidden_channels=256, num_groups=10, kmeans_iters=6,
                 seed_topk=3, min_targets=3, max_targets=32, grid_size=48,
                 scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=8,
                 inner_scale=1.5, outer_scale=2.0, start_epoch=1,
                 initial_strength=0.1, max_strength=2.0,
                 fusion_loss_weight=1.0, cf_loss_weight=0.1, gt_loss_weight=0.1,
                 positive_margin=1.0, ranking_margin=0.5,
                 negative_gt_coverage=0.01, checkpoint_chunks=True):
        super().__init__()
        counts = (hidden_channels, num_groups, kmeans_iters, seed_topk, min_targets,
                  max_targets, grid_size, scorer_dim, scorer_heads, scorer_layers, chunk_size, start_epoch)
        if any(isinstance(v, bool) or int(v) != v or v < 1 for v in counts):
            raise ValueError("CRV dimensions, candidate counts and start_epoch must be positive integers")
        if not seed_topk <= min_targets <= max_targets:
            raise ValueError("CRV requires seed_topk <= min_targets <= max_targets")
        if scorer_dim % scorer_heads or scorer_dim % 2:
            raise ValueError("CRV scorer_dim must be even and divisible by scorer_heads")
        values = (inner_scale, outer_scale, initial_strength, max_strength, fusion_loss_weight,
                  cf_loss_weight, gt_loss_weight, positive_margin, ranking_margin, negative_gt_coverage)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("CRV settings must be finite")
        if not 1 <= inner_scale < outer_scale or not 0 < initial_strength < max_strength:
            raise ValueError("Invalid CRV mask scales or fusion strength")
        if min(fusion_loss_weight, cf_loss_weight, gt_loss_weight, positive_margin, ranking_margin) < 0:
            raise ValueError("CRV weights and margins must be nonnegative")
        if fusion_loss_weight + cf_loss_weight <= 0 or not 0 <= negative_gt_coverage <= 1:
            raise ValueError("CRV requires an enabled loss and valid GT coverage threshold")
        self.num_groups, self.kmeans_iters = int(num_groups), int(kmeans_iters)
        self.seed_topk, self.min_targets, self.max_targets = int(seed_topk), int(min_targets), int(max_targets)
        self.grid_size, self.chunk_size, self.start_epoch = int(grid_size), int(chunk_size), int(start_epoch)
        self.inner_scale, self.outer_scale = float(inner_scale), float(outer_scale)
        self.max_strength = float(max_strength)
        self.fusion_loss_weight, self.cf_loss_weight = float(fusion_loss_weight), float(cf_loss_weight)
        self.gt_loss_weight = float(gt_loss_weight)
        self.positive_margin, self.ranking_margin = float(positive_margin), float(ranking_margin)
        self.negative_gt_coverage, self.checkpoint_chunks = float(negative_gt_coverage), bool(checkpoint_chunks)
        self.scorer = SceneTextScorer(hidden_channels, scorer_dim, scorer_heads, scorer_layers)
        self.strength_logit = nn.Parameter(torch.tensor(math.log(initial_strength/(max_strength-initial_strength))))
        self.capture_diagnostics = False
        self.register_buffer("schema_version", torch.tensor(11, dtype=torch.long))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_training_epoch", torch.tensor(0, dtype=torch.long))
        policy = dict(hidden_channels=hidden_channels, num_groups=num_groups, kmeans_iters=kmeans_iters,
                      seed_topk=seed_topk, min_targets=min_targets, max_targets=max_targets,
                      grid_size=grid_size, scorer_dim=scorer_dim, scorer_heads=scorer_heads,
                      scorer_layers=scorer_layers, inner_scale=inner_scale, outer_scale=outer_scale,
                      max_strength=max_strength, start_epoch=start_epoch,
                      fusion_loss_weight=fusion_loss_weight, cf_loss_weight=cf_loss_weight,
                      gt_loss_weight=gt_loss_weight, positive_margin=positive_margin,
                      ranking_margin=ranking_margin, negative_gt_coverage=negative_gt_coverage,
                      policy="joint_scene_mask_prefix_multi_positive_balanced_bce_v1")
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()
        self.register_buffer("policy_signature", torch.tensor(list(digest), dtype=torch.uint8))

    @torch.no_grad()
    def mark_optimizer_step(self):
        self.trained_steps.add_(1)

    @contextmanager
    def diagnostics(self):
        previous, self.capture_diagnostics = self.capture_diagnostics, True
        try:
            yield self
        finally:
            self.capture_diagnostics = previous

    def stage_ready(self, epoch=None):
        if self.training:
            if epoch is None:
                if self.start_epoch > 1:
                    raise ValueError("Delayed CRV requires the zero-based training epoch")
                epoch = 0
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("CRV epoch must be a nonnegative zero-based integer")
            self.last_training_epoch.fill_(epoch+1)
            return epoch+1 >= self.start_epoch
        return int(self.trained_steps) > 0 and int(self.last_training_epoch) >= self.start_epoch

    @staticmethod
    def support_drop(original_support, masked_support):
        return original_support-masked_support

    @staticmethod
    def clipped_boxes(boxes):
        return box_cxcywh_to_xyxy(boxes.detach()).clamp(0, 1)

    @staticmethod
    def spatial_grid(feature, meta):
        height, width = feature.shape[-2:]
        ih, iw = meta["img_shape"][:2]
        ph, pw = meta["pad_shape"][:2]
        if min(ih, iw, ph, pw) <= 0:
            raise ValueError("CRV requires positive image/padding sizes")
        yy, xx = torch.meshgrid((torch.arange(height, device=feature.device, dtype=feature.dtype)+.5)*ph/height/ih,
                                (torch.arange(width, device=feature.device, dtype=feature.dtype)+.5)*pw/width/iw,
                                indexing="ij")
        return xx, yy, pw/width/iw, ph/height/ih

    @torch.no_grad()
    def intervention_masks(self, boxes_xyxy, feature, meta, valid):
        xx, yy, _, _ = self.spatial_grid(feature, meta)
        center = (boxes_xyxy[:, :2]+boxes_xyxy[:, 2:])/2
        half = (boxes_xyxy[:, 2:]-boxes_xyxy[:, :2])/2
        radius = torch.maximum((xx[None]-center[:, 0, None, None]).abs()/half[:, 0, None, None].clamp(min=1e-8),
                               (yy[None]-center[:, 1, None, None]).abs()/half[:, 1, None, None].clamp(min=1e-8))
        progress = ((radius-self.inner_scale)/(self.outer_scale-self.inner_scale)).clamp(0, 1)
        return .5*(1+(math.pi*progress).cos())*valid[None]*(half > 0).all(-1)[:, None, None]

    @torch.no_grad()
    def gt_coverage(self, masks, gt, feature, meta):
        xx, yy, dx, dy = self.spatial_grid(feature, meta)
        ox = (torch.minimum(xx+dx/2, gt[2])-torch.maximum(xx-dx/2, gt[0])).clamp(min=0)
        oy = (torch.minimum(yy+dy/2, gt[3])-torch.maximum(yy-dy/2, gt[1])).clamp(min=0)
        area = ox*oy
        return (masks*area).sum((-1, -2))/area.sum().clamp(min=1e-8)

    @staticmethod
    def image_mean(projected, valid):
        weight = valid[:, None].to(projected.dtype)
        return (projected*weight).sum((-1, -2), keepdim=True)/weight.sum((-1, -2), keepdim=True).clamp(min=1)

    def pooled_original(self, projected, valid):
        weight = valid[:, None].to(projected.dtype)
        denominator = F.adaptive_avg_pool2d(weight, self.grid_size)
        original = F.adaptive_avg_pool2d(projected*weight, self.grid_size)/denominator.clamp(min=1e-8)
        return original, denominator, self.image_mean(projected, valid)

    def pooled_interventions(self, projected, valid, masks, owners, original, denominator, mean):
        if owners is None:
            owners = masks.new_zeros(len(masks), dtype=torch.long)
        weight = (masks*valid[owners])[:, None].to(projected.dtype)
        removed = F.adaptive_avg_pool2d(projected[owners]*weight, self.grid_size)
        footprint = F.adaptive_avg_pool2d(weight, self.grid_size)
        return original[owners]+(mean[owners]*footprint-removed)/denominator[owners].clamp(min=1e-8)

    @torch.no_grad()
    def packed_masks(self, boxes, owners, grids, valid, ground_truth):
        """Vectorized full-resolution masks/GT coverage; grids cached per image."""
        xx, yy, dx, dy = (value[owners] for value in grids)
        center, half = (boxes[:, :2]+boxes[:, 2:])/2, (boxes[:, 2:]-boxes[:, :2])/2
        radius = torch.maximum((xx-center[:, 0, None, None]).abs()/half[:, 0, None, None].clamp(min=1e-8),
                               (yy-center[:, 1, None, None]).abs()/half[:, 1, None, None].clamp(min=1e-8))
        progress = ((radius-self.inner_scale)/(self.outer_scale-self.inner_scale)).clamp(0, 1)
        mask = .5*(1+(math.pi*progress).cos())*valid[owners]*(half > 0).all(-1)[:, None, None]
        gt = ground_truth[owners]
        ox = (torch.minimum(xx+dx[:, None, None]/2, gt[:, 2, None, None])
              - torch.maximum(xx-dx[:, None, None]/2, gt[:, 0, None, None])).clamp(min=0)
        oy = (torch.minimum(yy+dy[:, None, None]/2, gt[:, 3, None, None])
              - torch.maximum(yy-dy[:, None, None]/2, gt[:, 1, None, None])).clamp(min=0)
        area = ox*oy
        return mask, (mask*area).sum((-1, -2))/area.sum((-1, -2)).clamp(min=1e-8)

    def _score_original(self, scenes, valid, positions, *prepared):
        return self.scorer(scenes, valid, None, None, prepared_text=prepared, positions=positions)

    def _score_masked(self, projected, valid, masks, owners, original, denominator, mean, positions, *prepared):
        scenes = self.pooled_interventions(projected, valid, masks, owners, original, denominator, mean)
        return self.scorer(scenes, denominator[owners], None, None,
                           prepared_text=tuple(value[owners] for value in prepared), positions=positions)

    @staticmethod
    def balanced_bce(logits, ious):
        """Multi-positive IoU labels; only samples with a positive are supervised."""
        pos, neg = ious > .5, ious == 0
        if not bool(pos.any()):
            return logits.sum()*0, False
        terms = [F.softplus(-logits[pos]).mean()]
        if bool(neg.any()):
            terms.append(F.softplus(logits[neg]).mean())
        return torch.stack(terms).mean(), True

    def forward(self, visual, visual_mask, boxes, text, text_mask, probabilities, logits, img_metas,
                candidate_features, refer_gt=None, epoch=None):
        if candidate_features.shape != boxes.shape[:2]+(text.shape[-1],):
            raise ValueError("CRV needs aligned candidate features B,M,C")
        if not boxes.shape[1] or not text.shape[1]:
            raise ValueError("CRV requires nonempty candidate/text slots")
        ready = self.stage_ready(epoch)
        base, probabilities = logits.squeeze(-1), probabilities.squeeze(-1)
        clipped = self.clipped_boxes(boxes)
        valid_boxes = (torch.isfinite(boxes).all(-1) & torch.isfinite(candidate_features).all(-1)
                       & torch.isfinite(base) & torch.isfinite(probabilities)
                       & (clipped[..., 2:] > clipped[..., :2]).all(-1))
        groups, selected, before, prefixes = adaptive_prefix(
            candidate_features, valid_boxes, probabilities, self.num_groups, self.kmeans_iters,
            self.seed_topk, self.min_targets, self.max_targets)
        # Padding is never visual evidence. Invalid image/text samples retain
        # their CRS shortlist without attempting to learn a fictitious drop.
        valid_images = ~visual_mask.bool()
        usable = valid_images.flatten(1).any(-1) & (~text_mask.bool()).any(-1)
        usable &= torch.isfinite(visual).flatten(1).all(-1) & torch.isfinite(text).flatten(1).all(-1)
        gt_values, gt_present = [], []
        for bi in range(len(boxes)):
            gt = refer_gt[bi] if self.training and refer_gt is not None else None
            gt = gt.detach().to(visual).reshape(4).clamp(0, 1) if gt is not None else visual.new_zeros(4)
            present = bool(torch.isfinite(gt).all() and (gt[2:] > gt[:2]).all())
            gt_values.append(torch.nan_to_num(gt)); gt_present.append(present)
        gt_values = torch.stack(gt_values)
        contexts, original_scores = {}, {}
        if ready and bool(usable.any()):
            routed = usable.nonzero(as_tuple=True)[0]
            active_valid = valid_images[routed]
            projected = self.scorer.visual(visual[routed])
            original, denominator, fill = self.pooled_original(projected, active_valid)
            prepared = self.scorer.prepare_text(text[routed], text_mask[routed])
            positions = self.scorer.spatial_positions(self.grid_size, self.grid_size, projected)
            def run(function, *args):
                if self.training and self.checkpoint_chunks:
                    return checkpoint(function, *args, use_reentrant=False)
                return function(*args)
            values = []
            for start in range(0, len(routed), self.chunk_size):
                stop = min(len(routed), start+self.chunk_size)
                values.append(run(self._score_original, original[start:stop], denominator[start:stop],
                                  positions, *(value[start:stop] for value in prepared)))
            orig_support = torch.cat(values)
            owners, intervention_boxes, image_chunks = [], [], {}
            offset = 0
            for owner, bi_tensor in enumerate(routed):
                bi = int(bi_tensor)
                prediction_boxes = clipped[bi, prefixes[bi]]
                add_gt = self.training and gt_present[bi]
                current = torch.cat((prediction_boxes, gt_values[bi:bi+1])) if add_gt else prediction_boxes
                count = len(current)
                if not count:
                    continue
                owners.extend([owner]*count); intervention_boxes.extend(current.unbind(0))
                image_chunks[bi] = (owner, offset, count, add_gt)
                offset += count
                original_scores[bi] = orig_support[owner]
            if owners:
                owners = torch.tensor(owners, device=boxes.device, dtype=torch.long)
                intervention_boxes = torch.stack(intervention_boxes)
                coordinates = [self.spatial_grid(visual[bi], meta) for bi, meta in enumerate(img_metas)]
                grids = (torch.stack([c[0] for c in coordinates]), torch.stack([c[1] for c in coordinates]),
                         visual.new_tensor([c[2] for c in coordinates]), visual.new_tensor([c[3] for c in coordinates]))
                support_values, coverages, footprints = [], [], []
                for start in range(0, len(owners), self.chunk_size):
                    stop = min(len(owners), start+self.chunk_size)
                    owned = owners[start:stop]
                    # Masks/coverage contain no autograd path to box coordinates.
                    mask, coverage = self.packed_masks(intervention_boxes[start:stop], routed[owned],
                                                       grids, valid_images, gt_values)
                    supported = mask.flatten(1).any(-1)
                    value = run(self._score_masked, projected, active_valid, mask, owned,
                                original, denominator, fill, positions, *prepared)
                    value = torch.where(supported, value, orig_support[owned])
                    support_values.append(value); coverages.append(coverage); footprints.append(supported)
                masked_support = torch.cat(support_values)
                coverage, footprint = torch.cat(coverages), torch.cat(footprints)
                for bi, (owner, offset, count, add_gt) in image_chunks.items():
                    sl = slice(offset, offset+count)
                    contexts[bi] = (self.support_drop(orig_support[owner], masked_support[sl]),
                                    coverage[sl], footprint[sl], masked_support[sl], add_gt)
        # No detach on visual/text/CRS logits: this is JOINT optimization.
        beta = self.max_strength*self.strength_logit.sigmoid()
        deltas = []
        bce_terms, rank_terms, gt_terms = [], [], []
        counts = base.new_zeros(4)  # BCE/rank/GT/union global sample counts.
        details, corrected, harmed, changed, reachable = [], [], [], [], []
        positives, negatives, rejected, active_flags = [], [], [], []
        for bi, ids in enumerate(prefixes):
            local_zero = torch.nan_to_num(base[bi]).sum()*0
            drops = base.new_zeros(len(ids))
            ious = (pairwise_iou(clipped[bi, ids], gt_values[bi:bi+1]).squeeze(-1)
                    if gt_present[bi] else base.new_zeros(len(ids)))
            support = base.new_zeros(len(ids))
            footprint = torch.zeros(len(ids), dtype=torch.bool, device=base.device)
            coverage = base.new_zeros(len(ids))
            if bi in contexts:
                all_drops, all_coverage, all_footprint, support, add_gt = contexts[bi]
                drops, coverage, footprint = all_drops[:len(ids)], all_coverage[:len(ids)], all_footprint[:len(ids)]
            delta = base.new_zeros(base.shape[1]).scatter(0, ids, beta*drops)
            deltas.append(delta)
            fused = base[bi, ids]+beta*drops
            order = (ids[fused.detach().argsort(descending=True, stable=True)]
                     if bool((delta[ids] != 0).any()) else ids)
            eligible = ready and bi in contexts and gt_present[bi]
            bce, rank, gt_loss = local_zero, local_zero, local_zero
            bce_valid = rank_valid = gt_valid = False
            pos = (ious > .5) & footprint
            neg = (ious == 0) & footprint & (coverage <= self.negative_gt_coverage)
            if eligible:
                bce, bce_valid = self.balanced_bce(fused, ious)
                if bool(pos.any()) and bool(neg.any()):
                    candidates = neg.nonzero(as_tuple=True)[0]
                    hardest = candidates[fused[candidates].detach().argmax()]
                    rank = F.softplus(self.ranking_margin+drops[hardest]-drops[pos]).mean()
                    rank_valid = True
                if contexts[bi][-1] and bool(contexts[bi][2][-1]):
                    gt_loss = F.softplus(self.positive_margin-contexts[bi][0][-1])
                    gt_valid = True
            bce_terms.append(bce); rank_terms.append(rank); gt_terms.append(gt_loss)
            counts += counts.new_tensor([bce_valid, rank_valid, gt_valid,
                (bce_valid and self.fusion_loss_weight > 0)
                or (rank_valid and self.cf_loss_weight > 0)
                or (gt_valid and self.cf_loss_weight*self.gt_loss_weight > 0)])
            old_good = bool(len(before[bi])) and gt_present[bi] and bool(
                pairwise_iou(clipped[bi, before[bi][:1]], gt_values[bi:bi+1])[0, 0] >= .5)
            new_good = bool(len(order)) and gt_present[bi] and bool(
                pairwise_iou(clipped[bi, order[:1]], gt_values[bi:bi+1])[0, 0] >= .5)
            corrected.append(not old_good and new_good); harmed.append(old_good and not new_good)
            changed.append(bool(len(order)) and bool(before[bi][0] != order[0]))
            reachable.append(gt_present[bi] and bool((ious >= .5).any()))
            positives.append(int((ious > .5).sum()) if gt_present[bi] else 0)
            negatives.append(int((ious == 0).sum()) if gt_present[bi] else 0)
            rejected.append(int(((ious == 0) & (coverage > self.negative_gt_coverage)).sum()) if gt_present[bi] else 0)
            active_flags.append(bi in contexts)
            sample = dict(indices=ids.detach(), groups=groups[bi].detach(),
                          before_order=before[bi].detach(), final_order=order.detach(),
                          base_logits=base[bi, ids].detach(), fused_logits=fused.detach(),
                          original_support=original_scores.get(bi, local_zero).detach(),
                          masked_support=support[:len(ids)].detach(), drops=drops.detach(),
                          ious=ious.detach(), gt_coverage=coverage.detach(),
                          positive_mask=pos.detach(), reliable_negative_mask=neg.detach())
            if self.capture_diagnostics:
                sample = {key: value.cpu() for key, value in sample.items()}
                if bi in contexts and contexts[bi][-1]:
                    sample["exact_gt_drop"] = contexts[bi][0][-1].detach().cpu()
            details.append(sample)
        delta = torch.stack(deltas)
        final = base+delta
        probability = torch.where(delta == 0, probabilities, final.sigmoid())
        priority = torch.zeros_like(final)
        for bi, sample in enumerate(details):
            # Diagnostics may live on CPU; source order is converted only here.
            order = sample["final_order"].to(base.device)
            priority[bi, order] = torch.arange(len(order), 0, -1, device=base.device, dtype=base.dtype)
        if self.training:
            world = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
                world = torch.distributed.get_world_size()
            if ready:
                zero = sum(p.sum()*0 for p in self.parameters())+torch.nan_to_num(final).sum()*0
            else:
                zero = final.new_zeros(())
            def reduced(terms, count):
                return torch.stack(terms).sum()*world/count.clamp(min=1)+zero
            losses = dict(loss_crv_fusion=reduced(bce_terms, counts[0])*self.fusion_loss_weight,
                          loss_crv_cf=reduced(rank_terms, counts[1])*self.cf_loss_weight,
                          loss_crv_gt=reduced(gt_terms, counts[2])*self.cf_loss_weight*self.gt_loss_weight)
        else:
            zero = final.new_zeros(())
            losses = dict(loss_crv_fusion=zero, loss_crv_cf=zero, loss_crv_gt=zero)
        def average(values):
            return base.new_tensor(values, dtype=base.dtype).mean()
        target_counts = selected.sum(-1).to(base.dtype)
        chosen = target_counts.sum().clamp(min=1)
        metrics = dict(beta=beta.detach(), active=average(active_flags), ready=base.new_tensor(float(ready)),
                       target_count=target_counts.mean(),
                       target_cap_rate=(target_counts == min(self.max_targets, base.shape[1])).float().mean(),
                       bce_samples=counts[0], rank_samples=counts[1], gt_samples=counts[2],
                       supervised_samples=counts[3], positive_count=average(positives),
                       negative_count=average(negatives), rejected_negatives=average(rejected),
                       reachable=average(reachable), delta_abs=delta.detach().abs().sum()/chosen,
                       corrected=average(corrected), harmed=average(harmed), rank_changed=average(changed))
        info = {"crv_"+key: value.detach() for key, value in metrics.items()}
        info.update(crv_samples=details, crv_output_mask=selected.detach(),
                    crv_final_logits=final.detach()[..., None],
                    crv_crs_probabilities=probabilities.detach(),
                    crv_last_training_epoch=self.last_training_epoch.detach().clone())
        return probability[..., None], priority[..., None], losses, info
