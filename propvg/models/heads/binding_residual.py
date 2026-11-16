"""Optional text-conditioned semantic/geometry binding residual after CRS.

Selection and clustering never read GT. GT is confined to ``loss``. No image
encoder reruns, external models, sklearn, or removal of target proposals.
"""
import hashlib
import json
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from propvg.layers.box_ops import box_cxcywh_to_xyxy


def pairwise_iou(a, b):
    intersection = (torch.minimum(a[..., :, None, 2:], b[..., None, :, 2:])
                    - torch.maximum(a[..., :, None, :2], b[..., None, :, :2])).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return intersection / (area_a[..., :, None] + area_b[..., None, :] - intersection).clamp(min=1e-8)


def binding_statistics(semantics, geometry, valid):
    """Exact paired-minus-uniform-rebinding expectation, without an R x R matrix.

    Inputs: [..., R, D], [..., R, D], [..., R]. Both marginal means use
    EXACTLY the same reference set. At most one reference cannot bind anything.
    """
    weight = valid.to(semantics.dtype)[..., None]
    count = weight.sum(-2).clamp(min=1)
    h_mean = (semantics * weight).sum(-2) / count
    g_mean = (geometry * weight).sum(-2) / count
    paired = (semantics * geometry * weight).sum((-2, -1)) / count.squeeze(-1)
    null = (h_mean * g_mean).sum(-1)
    # Anchor before centering: identical semantic vectors must yield EXACT
    # zero, even when a float32 mean over e.g. 3 or 7 copies rounds slightly.
    h_shift, g_shift = semantics - semantics[..., :1, :], geometry - geometry[..., :1, :]
    hc = h_shift - ((h_shift * weight).sum(-2) / count)[..., None, :]
    gc = g_shift - ((g_shift * weight).sum(-2) / count)[..., None, :]
    residual = ((hc * gc * weight).sum((-2, -1)) / count.squeeze(-1))
    scale = math.sqrt(semantics.shape[-1])
    residual = torch.where(valid.sum(-1) >= 2, residual / scale, torch.zeros_like(residual))
    return residual, paired / scale, null / scale


class BindingInteractionResidualScoring(nn.Module):
    def __init__(self, hidden_channels=256, reference_topk=64,
                 duplicate_iou=0.8, model_dim=128, num_heads=4,
                 chunk_size=32, initial_strength=0.1, max_strength=2.0,
                 final_loss_weight=1.0, binding_loss_weight=0.1,
                 negative_topk=8, detach_inputs=False, binding_mode="residual",
                 target_topn=5, version="v2", num_groups=8, kmeans_iters=6):
        super().__init__()
        if version not in ("v1", "v2"):
            raise ValueError("BIRS version must be 'v1' (clustering) or 'v2' (CRS Top-N)")
        self.version = version
        if any(int(x) != x or x < 1 for x in
               (reference_topk, model_dim, num_heads, chunk_size, negative_topk, target_topn,
                num_groups, kmeans_iters)):
            raise ValueError("BIRS counts/dimensions must be positive integers")
        if model_dim % num_heads or not 0 < duplicate_iou <= 1:
            raise ValueError("BIRS requires model_dim divisible by num_heads and 0 < duplicate_iou <= 1")
        if not 0 < initial_strength < max_strength or min(final_loss_weight, binding_loss_weight) < 0:
            raise ValueError("BIRS requires 0 < initial_strength < max_strength and nonnegative loss weights")
        if not final_loss_weight + binding_loss_weight > 0:
            raise ValueError("At least one BIRS loss must be enabled")
        if binding_mode not in ("residual", "paired", "crs_control"):
            raise ValueError("Unknown BIRS binding_mode")
        self.target_topn, self.reference_topk = int(target_topn), int(reference_topk)
        self.num_groups, self.kmeans_iters = int(num_groups), int(kmeans_iters)
        self.duplicate_iou = duplicate_iou
        self.chunk_size, self.negative_topk = int(chunk_size), int(negative_topk)
        self.detach_inputs, self.binding_mode = bool(detach_inputs), binding_mode
        self.final_loss_weight, self.binding_loss_weight = final_loss_weight, binding_loss_weight
        self.max_strength = float(max_strength)
        self.capture_diagnostics = False
        self.reference_projection = nn.Linear(hidden_channels, model_dim)
        self.text_projection = nn.Linear(hidden_channels, model_dim)
        self.text_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=0, batch_first=True)
        self.prototype_projection = nn.Linear(hidden_channels, model_dim)
        self.semantic_binding = nn.Sequential(nn.Linear(model_dim * 3, model_dim), nn.GELU(),
                                              nn.LayerNorm(model_dim), nn.Tanh())
        # Nonlinearity is essential: centering a linear (ref_center - target_center)
        # encoder would CANCEL the target center, making every target equivalent.
        self.geometry_projection = nn.Sequential(nn.Linear(9, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.geometry_binding = nn.Sequential(nn.Linear(model_dim * 3, model_dim), nn.GELU(),
                                              nn.LayerNorm(model_dim), nn.Tanh())
        self.strength_logit = nn.Parameter(torch.tensor(math.log(initial_strength / (max_strength - initial_strength))))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))
        # Keep archived v1 and corrected v2 policy signatures unchanged. Schema
        # numbers describe checkpoint semantics, NOT the user-facing v1/v2 name.
        self.register_buffer("schema_version", torch.tensor(1 if version == "v1" else 3, dtype=torch.long))
        policy = dict(hidden_channels=hidden_channels, reference_topk=reference_topk,
                      duplicate_iou=duplicate_iou, model_dim=model_dim, num_heads=num_heads,
                      max_strength=max_strength, binding_mode=binding_mode,
                      target_topn=int(target_topn), final_loss_scope="all_valid_candidates")
        if version == "v1":
            policy.pop("target_topn")
            policy.pop("final_loss_scope")
            policy.update(num_groups=num_groups, kmeans_iters=kmeans_iters)
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

    @staticmethod
    @torch.no_grad()
    def clipped_boxes(boxes, img_metas):
        # Same valid-image clipping as the detector postprocessor, not pad_shape.
        xyxy = box_cxcywh_to_xyxy(boxes.detach()).float().clone()
        for bi, meta in enumerate(img_metas):
            h, w = meta["img_shape"][:2]
            if h <= 0 or w <= 0:
                raise ValueError("BIRS requires positive image dimensions")
            scale = xyxy.new_tensor([w, h, w, h])
            xyxy[bi] *= scale
            xyxy[bi, :, 0::2].clamp_(0, w)
            xyxy[bi, :, 1::2].clamp_(0, h)
            xyxy[bi] /= scale
        valid = torch.isfinite(xyxy).all(-1) & (xyxy[..., 2:] > xyxy[..., :2]).all(-1)
        return torch.nan_to_num(xyxy), valid

    @torch.no_grad()
    def select_references(self, boxes, classes, valid):
        """Detector Top-K FIRST, then greedy near-duplicate suppression; never refill."""
        fg = classes.detach().float().softmax(-1)[..., 0]
        valid = valid & torch.isfinite(classes).all(-1)
        indices = fg.masked_fill(~valid, -float("inf")).argsort(dim=-1, descending=True, stable=True)
        indices = indices[:, :min(self.reference_topk, boxes.shape[1])]
        picked = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        keep = valid.gather(1, indices).clone()
        overlap = pairwise_iou(picked, picked)
        # Batched GPU suppression: no per-image CPU tensor transfer.
        for j in range(indices.shape[1]):
            keep[:, j + 1:] &= ~(keep[:, j:j + 1] & (overlap[:, j, j + 1:] >= self.duplicate_iou))
        return indices, keep, picked, fg.gather(1, indices)

    @torch.no_grad()
    def cluster(self, queries, valid):
        """Archived v1 deterministic spherical K-means; no GT or RNG draws."""
        b, n, _ = queries.shape
        g = min(self.num_groups, n)
        q = F.normalize(torch.nan_to_num(queries.detach().float()), dim=-1)
        mean = F.normalize((q * valid[..., None]).sum(1) / valid.sum(1).clamp(min=1)[:, None], dim=-1)
        distance = (q - mean[:, None]).square().sum(-1).masked_fill(~valid, -1)
        indices = distance.argmax(-1)
        rows = torch.arange(b, device=q.device)
        centers, minimum = [], torch.full_like(distance, float("inf"))
        for _ in range(g):
            center = q[rows, indices]
            centers.append(center)
            minimum = torch.minimum(minimum, (q - center[:, None]).square().sum(-1))
            indices = minimum.masked_fill(~valid, -1).argmax(-1)
        centers = torch.stack(centers, 1)
        for _ in range(self.kmeans_iters):
            assignment = (q @ centers.transpose(1, 2)).argmax(-1)
            weights = F.one_hot(assignment, g).to(q.dtype) * valid[..., None]
            counts = weights.sum(1)
            pooled = weights.transpose(1, 2) @ q / counts.clamp(min=1)[..., None]
            centers = torch.where(counts[..., None] > 0, F.normalize(pooled, dim=-1), centers)
        assignment = (q @ centers.transpose(1, 2)).argmax(-1)
        return assignment.masked_fill(~valid, -1)

    @staticmethod
    def geometry(target, reference, aspect):
        tc, rc = (target[..., :2] + target[..., 2:]) / 2, (reference[..., :2] + reference[..., 2:]) / 2
        tw, rw = (target[..., 2:] - target[..., :2]).clamp(min=1e-5), (reference[..., 2:] - reference[..., :2]).clamp(min=1e-5)
        offset = (rc[:, None] - tc[:, :, None]) * aspect[:, None, None]
        distance = offset.square().sum(-1).sqrt().clamp(min=1e-5)
        local = (rc[:, None] - tc[:, :, None]) / tw[:, :, None]
        ratios = (rw[:, None] / tw[:, :, None]).log().clamp(-8, 8)
        return torch.cat((offset, offset / distance[..., None], torch.log1p(distance)[..., None],
                          ratios, local.tanh()), -1)

    def forward(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                reference_queries=None, reference_boxes=None, reference_classes=None, source_indices=None):
        if self.version == "v1":
            output = self._score_selected(queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                                          reference_queries, reference_boxes, reference_classes, source_indices)
            if "valid" in output:
                output["candidate_mask"] = output["valid"]
                output["target_indices"] = torch.arange(queries.shape[1], device=queries.device)[None].expand(queries.shape[0], -1)
                output["target_valid"] = output["valid"]
                if self.capture_diagnostics:
                    for key in ("candidate_mask", "target_indices", "target_valid"):
                        output["diagnostics"][key] = output[key].detach().cpu()
            return output
        return self._forward_topn(queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                                  reference_queries, reference_boxes, reference_classes, source_indices)

    def _forward_topn(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                      reference_queries, reference_boxes, reference_classes, source_indices):
        if not self.training and int(self.trained_steps) == 0:
            return dict(scores=scores, final_logits=logits, ranking_priority=None)
        xyxy, valid = self.clipped_boxes(boxes, img_metas)
        valid = valid & torch.isfinite(queries).all(-1)
        b, n = valid.shape
        k = min(n, self.target_topn)
        if not k:
            raise ValueError("BIRS requires at least one candidate slot")
        selected, before = [], []
        for bi in range(b):
            ids = valid[bi].nonzero(as_tuple=True)[0]
            order = ids[scores[bi, ids, 0].detach().float().cpu().argsort(descending=True).to(ids.device)]
            before.append(order)
            # Unique invalid slots pad short samples; never repeat a live query
            # index, which would overwrite its residual during scatter-back.
            invalid = (~valid[bi]).nonzero(as_tuple=True)[0]
            selected.append(torch.cat((order, invalid))[:k])
        selected = torch.stack(selected)
        selected_valid = valid.gather(1, selected)

        def gather(value):
            return value.gather(1, selected[..., None].expand(-1, -1, value.shape[-1]))

        source = (torch.arange(n, device=queries.device)[None].expand(b, -1)
                  if source_indices is None else source_indices)
        core = self._score_selected(
            gather(queries), gather(boxes), gather(classes), text, text_mask, gather(scores), gather(logits), img_metas,
            reference_queries=queries if reference_queries is None else reference_queries,
            reference_boxes=boxes if reference_boxes is None else reference_boxes,
            reference_classes=classes if reference_classes is None else reference_classes,
            source_indices=source.gather(1, selected),
        )
        output = dict(core)
        for key in ("raw_binding", "paired", "null", "delta", "eligible", "legal_references"):
            value = core[key]
            shape = (b, n) + tuple(value.shape[2:])
            index = selected.reshape(b, k, *([1] * (value.ndim-2))).expand_as(value)
            output[key] = value.new_zeros(shape).scatter(1, index, value)
        candidate_mask = torch.zeros_like(valid).scatter(1, selected, selected_valid)
        base_scores = scores.detach() if self.detach_inputs else scores
        base_logits = logits.detach() if self.detach_inputs else logits
        final = base_logits.squeeze(-1) + output["delta"]
        probability = torch.where(output["delta"] == 0, base_scores.squeeze(-1), final.sigmoid())
        priority = torch.zeros_like(final)
        after = []
        for bi, old in enumerate(before):
            head = old[:k]
            if bool((output["delta"][bi, head] != 0).any()):
                head = head[final[bi, head].detach().argsort(descending=True, stable=True)]
            # Hard boundary: outsiders cannot enter Top-N, regardless of their
            # uncalibrated probability relative to a corrected candidate.
            order = torch.cat((head, old[k:]))
            after.append(order)
            priority[bi, order] = torch.arange(len(order), 0, -1, device=final.device, dtype=final.dtype)
        output.update(scores=probability[..., None], final_logits=final[..., None], ranking_priority=priority[..., None],
                      boxes=xyxy, valid=valid, groups=torch.where(candidate_mask, 0, -1),
                      candidate_mask=candidate_mask, target_indices=selected, target_valid=selected_valid,
                      before_orders=before, after_orders=after, base_scores=base_scores.detach().squeeze(-1))
        if self.capture_diagnostics:
            output["diagnostics"] = {key: value.detach().cpu() for key, value in output.items() if torch.is_tensor(value)}
            for key in ("reference_boxes", "reference_confidence"):
                output["diagnostics"][key] = core["diagnostics"][key]
        return output

    def _score_selected(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                        reference_queries=None, reference_boxes=None, reference_classes=None, source_indices=None):
        # A newly enabled/untrained branch must not randomize baseline evaluation.
        if not self.training and int(self.trained_steps) == 0:
            return dict(scores=scores, final_logits=logits, ranking_priority=None)
        reference_queries = queries if reference_queries is None else reference_queries
        reference_boxes = boxes if reference_boxes is None else reference_boxes
        reference_classes = classes if reference_classes is None else reference_classes
        if self.detach_inputs:
            queries, reference_queries, text = queries.detach(), reference_queries.detach(), text.detach()
            scores, logits = scores.detach(), logits.detach()
        b, n, _ = queries.shape
        xyxy, valid = self.clipped_boxes(boxes, img_metas)
        ref_xyxy, ref_valid = self.clipped_boxes(reference_boxes, img_metas)
        valid = valid & torch.isfinite(queries).all(-1)
        ref_valid = ref_valid & torch.isfinite(reference_queries).all(-1)
        if self.version == "v1":
            groups = self.cluster(queries, valid)
            g = min(n, self.num_groups)
        else:
            groups = torch.zeros_like(valid, dtype=torch.long).masked_fill(~valid, -1)
            g = 1
        membership = F.one_hot(groups.clamp(min=0), g).to(queries.dtype) * valid[..., None]
        prototypes = membership.transpose(1, 2) @ torch.nan_to_num(queries)
        prototypes = prototypes / membership.sum(1).clamp(min=1)[..., None]
        prototypes = self.prototype_projection(prototypes)
        indices, keep, reference, confidence = self.select_references(ref_xyxy, reference_classes, ref_valid)
        reference_features = reference_queries.gather(1, indices[..., None].expand(-1, -1, queries.shape[-1]))
        v = self.reference_projection(torch.nan_to_num(reference_features))
        t = self.text_projection(torch.nan_to_num(text))
        padding = text_mask.bool().clone()
        text_valid = (~padding).any(1)
        # Avoid all-padding attention NaNs, then disable those samples' residual.
        padding[:, 0] &= text_valid
        t = t.masked_fill(~text_valid[:, None, None], 0)
        read, _ = self.text_attention(v, t, t, key_padding_mask=padding, need_weights=False)
        v = v + read
        semantic = self.semantic_binding(torch.cat((v[:, None].expand(-1, g, -1, -1),
                    prototypes[:, :, None].expand(-1, -1, v.shape[1], -1),
                    v[:, None] * prototypes[:, :, None]), -1))
        sentence = (t.masked_fill(text_mask.bool()[..., None], 0).sum(1) /
                    (~text_mask.bool()).sum(1).clamp(min=1)[:, None])
        aspect = queries.new_tensor([[meta["img_shape"][1] / min(meta["img_shape"][:2]),
                                     meta["img_shape"][0] / min(meta["img_shape"][:2])] for meta in img_metas])
        source = (torch.arange(n, device=queries.device)[None].expand(b, -1)
                  if source_indices is None else source_indices)
        residuals, paired_all, null_all, legal_all = [], [], [], []
        rows = torch.arange(b, device=queries.device)[:, None]
        for start in range(0, n, self.chunk_size):
            stop = min(n, start + self.chunk_size)
            group = groups[:, start:stop].clamp(min=0)
            h = semantic[rows, group]
            geo = self.geometry_projection(self.geometry(xyxy[:, start:stop], reference, aspect))
            condition = (prototypes[rows, group] + sentence[:, None])[:, :, None].expand_as(geo)
            psi = self.geometry_binding(torch.cat((geo, condition, geo * condition), -1))
            legal = keep[:, None] & valid[:, start:stop, None] & text_valid[:, None, None]
            legal = legal & (source[:, start:stop, None] != indices[:, None])
            legal = legal & (pairwise_iou(xyxy[:, start:stop], reference) < self.duplicate_iou)
            residual, paired, null = binding_statistics(h, psi, legal)
            residuals.append(residual)
            paired_all.append(paired)
            null_all.append(null)
            legal_all.append(legal)
        raw = torch.cat(residuals, 1)
        paired, null, legal = torch.cat(paired_all, 1), torch.cat(null_all, 1), torch.cat(legal_all, 1)
        eligible = valid & (legal.sum(-1) >= 2)
        evidence = raw if self.binding_mode == "residual" else paired
        if self.binding_mode == "crs_control":
            evidence = evidence * 0
        weights = membership * eligible[..., None]
        counts = weights.sum(1)
        group_mean = (weights * evidence[..., None]).sum(1) / counts.clamp(min=1)
        centered = evidence - group_mean.gather(1, groups.clamp(min=0))
        eligible = eligible & (counts.gather(1, groups.clamp(min=0)) > 1)
        strength = self.max_strength * self.strength_logit.sigmoid()
        delta = torch.where(eligible, strength * centered, torch.zeros_like(centered))
        base = logits.squeeze(-1)
        final = base + delta
        probability = torch.where(delta == 0, scores.squeeze(-1), final.sigmoid())
        # Rank with logits (no sigmoid saturation). Encode the complete order
        # for the existing postprocessor. Zero-change samples retain EXACT
        # legacy CPU probability sorting, including its historical tie policy.
        priority, before_orders, after_orders = torch.zeros_like(base), [], []
        for bi in range(b):
            slots = valid[bi].nonzero(as_tuple=True)[0]
            old = slots[scores[bi, slots, 0].detach().float().cpu().argsort(descending=True).to(slots.device)]
            order = (slots[final[bi, slots].detach().argsort(descending=True, stable=True)]
                     if bool((delta[bi] != 0).any()) else old)
            priority[bi, order] = torch.arange(len(order), 0, -1, device=base.device, dtype=base.dtype)
            before_orders.append(old)
            after_orders.append(order)
        output = dict(scores=probability[..., None], final_logits=final[..., None], ranking_priority=priority[..., None],
                      raw_binding=raw, paired=paired, null=null, delta=delta, boxes=xyxy, valid=valid,
                      eligible=eligible, groups=groups, reference_indices=indices, reference_valid=keep,
                      legal_references=legal, strength=strength, before_orders=before_orders, after_orders=after_orders,
                      base_scores=scores.detach().squeeze(-1))
        if self.capture_diagnostics:
            output["diagnostics"] = {key: value.detach().cpu() for key, value in output.items()
                                     if torch.is_tensor(value)}
            output["diagnostics"].update(reference_boxes=reference.detach().cpu(),
                                         reference_confidence=confidence.detach().cpu())
        return output

    def loss(self, output, targets, img_metas, refer_labels):
        """Original Hungarian refer labels for final BCE; GT-disjoint hard negatives for binding."""
        z = output["final_logits"].squeeze(-1)
        # Connected zero makes all enabled parameters DDP-safe even on ranks
        # with no referent/no valid reference pairs.
        zero = sum(parameter.sum() * 0 for parameter in self.parameters()) + z.sum() * 0
        ground_truth, present = [], []
        for boxes, meta in zip(targets, img_metas):
            referred = meta["refer_target_index"]
            if len(referred) > 1:
                raise ValueError("BIRS currently supports one referring GT per sample")
            h, w = meta["img_shape"][:2]
            ground_truth.append(boxes.detach().reshape(-1, 4)[referred[0]].to(z) / z.new_tensor([w, h, w, h])
                                if len(referred) else z.new_zeros(4))
            present.append(bool(len(referred)))
        present = torch.tensor(present, device=z.device, dtype=torch.bool)
        iou = pairwise_iou(output["boxes"], torch.stack(ground_truth)[:, None]).squeeze(-1)
        valid, eligible = output["valid"], output["eligible"]
        labels = refer_labels.detach().to(z)
        # Full-candidate localization supervision is independent of shortlist
        # coverage and available references. Outside Top-N, z is the CRS logit.
        active = present & ((labels > 0) & valid).any(1)
        if self.version == "v1":
            # Preserve the first experiment's loss eligibility, including the
            # no-binding fallback; v2 deliberately removed this restriction.
            active = active & eligible.any(1)
        per_sample = (F.binary_cross_entropy_with_logits(z, labels, reduction="none") * valid).sum(1)
        per_sample = per_sample / valid.sum(1).clamp(min=1)
        final_sum = (per_sample * active).sum() + zero
        bind_sum, bind_count = zero, 0
        binding_pairs = []
        for bi in range(z.shape[0]):
            positive = ((labels[bi] > 0) & (iou[bi] >= .5) & eligible[bi] & present[bi]).nonzero(as_tuple=True)[0]
            sample_losses = []
            for pos in positive:
                neg = (eligible[bi] & (iou[bi] == 0) &
                       (output["groups"][bi] == output["groups"][bi, pos])).nonzero(as_tuple=True)[0]
                order = output["base_scores"][bi, neg].argsort(descending=True, stable=True)[:self.negative_topk]
                if neg.numel():
                    sample_losses.append(F.softplus(output["raw_binding"][bi, neg[order]] -
                                                    output["raw_binding"][bi, pos]).mean())
                    if self.capture_diagnostics:
                        binding_pairs.append(dict(sample=bi, positive=int(pos), negative=neg[order].detach().cpu()))
            if sample_losses:
                bind_sum = bind_sum + torch.stack(sample_losses).mean()
                bind_count += 1
        counts = z.new_tensor([int(active.sum()), bind_count])
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
            world = torch.distributed.get_world_size()
        binding_weight = 0.0 if self.binding_mode == "crs_control" else self.binding_loss_weight
        losses = dict(loss_birs_final=final_sum * world / counts[0].clamp(min=1) * self.final_loss_weight,
                      loss_birs_binding=bind_sum * world / counts[1].clamp(min=1) * binding_weight)
        if self.capture_diagnostics and "diagnostics" in output:
            output["diagnostics"].update(gt_iou=iou.detach().cpu(), binding_pairs=binding_pairs,
                                         refer_labels=labels.detach().cpu())
        old_good, new_good, changed = [], [], []
        for bi, (old, new) in enumerate(zip(output["before_orders"], output["after_orders"])):
            old_good.append(bool(len(old)) and bool(iou[bi, old[0]] >= .5))
            new_good.append(bool(len(new)) and bool(iou[bi, new[0]] >= .5))
            changed.append(bool(len(old)) and bool(len(new)) and bool(old[0] != new[0]))
        old, new = z.new_tensor(old_good).bool(), z.new_tensor(new_good).bool()
        denom = present.float().sum().clamp(min=1)
        supervised = counts[0] if self.final_loss_weight > 0 else counts[1] * (binding_weight > 0)
        metrics = dict(supervised_samples=supervised, binding_samples=counts[1],
                       strength=output["strength"].detach(), eligible=eligible.float().mean(),
                       delta_abs=output["delta"].detach().abs().mean(),
                       reference_count=output["reference_valid"].float().sum(1).mean(),
                       corrected=((~old & new) & present).float().sum() / denom,
                       harmed=((old & ~new) & present).float().sum() / denom,
                       rank_changed=z.new_tensor(changed).mean())
        scope = valid & output["candidate_mask"]
        metrics.update(target_count=scope.float().sum(1).mean(),
                       refer_in_topn=(((labels > 0) & scope).any(1) & present).float().sum() / denom)
        return losses, metrics
