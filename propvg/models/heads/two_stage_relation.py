"""Post-CRS, competition-aware reference evidence. No GT is read by forward."""
import hashlib
import json
import math
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from propvg.layers.box_ops import box_cxcywh_to_xyxy


def pairwise_iou(left, right):
    lo = torch.maximum(left[..., :, None, :2], right[..., None, :, :2])
    hi = torch.minimum(left[..., :, None, 2:], right[..., None, :, 2:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    a = (left[..., 2:] - left[..., :2]).clamp(min=0).prod(-1)
    b = (right[..., 2:] - right[..., :2]).clamp(min=0).prod(-1)
    return inter / (a[..., :, None] + b[..., None, :] - inter).clamp(min=1e-12)


def numpy_iou(box, boxes):
    inter = np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0).prod(-1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    return inter / np.maximum(area + areas - inter, 1e-12)


def masked_softmax(values, valid, dim):
    """Exactly zero, finite outputs for fully masked rows/columns."""
    result = values.float().masked_fill(~valid, -1e9).softmax(dim)
    result = result * valid
    return result / result.sum(dim, keepdim=True).clamp(min=1e-12)


class TwoStageRelationRefinement(nn.Module):
    """Preserve CRS; learn signed relational evidence on a shared reference pool.

    Inputs/outputs keep all query slots. A unique ranking priority explicitly
    orders the refined shortlist before the untouched tail. It is NOT a logit
    or confidence; real differentiable logits are returned separately.
    """
    metric_names = ("lambda", "null_weight", "reference_count", "group_count", "discriminability",
                    "correction_abs", "rank_changed", "reachable", "corrected", "harmed",
                    "supervised_samples", "ready")

    def __init__(self, hidden_channels=256, target_topk=32, reference_topk=64,
                 duplicate_iou=.8, model_dim=128, num_heads=4,
                 prior_temperature=1., relation_temperature=1., prior_uniform_mix=.2,
                 discriminability_floor=.1, initial_strength=.1, max_strength=2.,
                 loss_weight=1., aggregation="competitive", use_geometry=True,
                 use_reference_semantics=True, detach_inputs=False):
        super().__init__()
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be True or False")
        for name, value in dict(hidden_channels=hidden_channels, target_topk=target_topk,
                                reference_topk=reference_topk, model_dim=model_dim, num_heads=num_heads).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if model_dim % num_heads or not 0 < duplicate_iou <= 1:
            raise ValueError("model_dim must be divisible by num_heads; duplicate_iou must be in (0,1]")
        if not 0 <= prior_uniform_mix <= 1 or not 0 < discriminability_floor <= 1:
            raise ValueError("Invalid prior_uniform_mix/discriminability_floor")
        for value in (prior_temperature, relation_temperature, max_strength):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Temperatures and max_strength must be finite and positive")
        if not 0 < initial_strength < max_strength or not math.isfinite(loss_weight) or loss_weight <= 0:
            raise ValueError("Invalid initial_strength/loss_weight")
        if aggregation not in ("competitive", "vote", "weighted", "max", "crs_control"):
            raise ValueError("Unknown relation aggregation")
        self.target_topk, self.reference_topk = target_topk, reference_topk
        self.duplicate_iou, self.model_dim = duplicate_iou, model_dim
        self.prior_temperature, self.relation_temperature = prior_temperature, relation_temperature
        self.prior_uniform_mix, self.discriminability_floor = prior_uniform_mix, discriminability_floor
        self.max_strength, self.loss_weight = max_strength, loss_weight
        self.aggregation = aggregation
        self.use_geometry, self.use_reference_semantics = use_geometry, use_reference_semantics
        # Training-only choice: no parameters/buffers or inference policy
        # signature change, so previously trained checkpoints remain testable.
        self.detach_inputs = detach_inputs
        self.capture_diagnostics = False  # Runtime only; not a checkpoint/config policy.
        self.visual = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, model_dim))
        self.text = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, model_dim))
        self.reader = nn.MultiheadAttention(model_dim, num_heads, dropout=0., batch_first=True)
        self.reference = nn.Sequential(nn.Linear(model_dim * 3, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.geometry = nn.Sequential(nn.Linear(6, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim), nn.Tanh())
        self.pair_score = nn.Linear(model_dim, 1, bias=False)
        self.reference_score = nn.Linear(model_dim, 1)
        self.null_score = nn.Linear(model_dim, 1)
        self.strength_logit = nn.Parameter(torch.tensor(math.log(initial_strength / (max_strength - initial_strength))))
        self.register_buffer("schema_version", torch.tensor(1, dtype=torch.long))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))
        policy = dict(target_topk=target_topk, reference_topk=reference_topk, duplicate_iou=duplicate_iou,
                      model_dim=model_dim, num_heads=num_heads, prior_temperature=prior_temperature,
                      relation_temperature=relation_temperature, prior_uniform_mix=prior_uniform_mix,
                      discriminability_floor=discriminability_floor, max_strength=max_strength,
                      loss_weight=loss_weight, aggregation=aggregation, use_geometry=use_geometry,
                      use_reference_semantics=use_reference_semantics)
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()
        self.register_buffer("policy_signature", torch.tensor(list(digest), dtype=torch.uint8))

    @property
    def strength(self):
        return self.max_strength * self.strength_logit.sigmoid()

    @torch.no_grad()
    def mark_optimizer_step(self):
        self.trained_steps.add_(1)

    @contextmanager
    def diagnostics(self):
        previous = self.capture_diagnostics
        self.capture_diagnostics = True
        try:
            yield self
        finally:
            self.capture_diagnostics = previous

    @staticmethod
    @torch.no_grad()
    def clipped_boxes(boxes, img_metas):
        # Match the existing postprocessor: cxcywh conversion precedes the
        # float32 cast; scale/clip by VALID image sizes, never pad_shape.
        pixel = box_cxcywh_to_xyxy(boxes.detach()).float().clone()
        for bi, meta in enumerate(img_metas):
            h, w = meta["img_shape"][:2]
            if h <= 0 or w <= 0:
                raise ValueError("Relation refinement requires positive image dimensions")
            pixel[bi] *= pixel.new_tensor([w, h, w, h])
            pixel[bi, :, 0::2].clamp_(0, w)
            pixel[bi, :, 1::2].clamp_(0, h)
            pixel[bi] /= pixel.new_tensor([w, h, w, h])
        valid = torch.isfinite(pixel).all(-1) & (pixel[..., 2:] > pixel[..., :2]).all(-1)
        return torch.nan_to_num(pixel), valid

    @torch.no_grad()
    def select(self, boxes, reference_boxes, scores, reference_classes, img_metas):
        xyxy, valid = self.clipped_boxes(boxes, img_metas)
        ref_xyxy, ref_valid = self.clipped_boxes(reference_boxes, img_metas)
        b, n = boxes.shape[:2]
        k, r = min(n, self.target_topk), min(reference_boxes.shape[1], self.reference_topk)
        if not n or not r:
            raise ValueError("At least one target/reference proposal slot is required")
        score_cpu = scores.detach().float().reshape(b, n).cpu()
        foreground = reference_classes.detach().float().softmax(-1)[..., 0]
        foreground_cpu = foreground.cpu()
        target_indices = torch.zeros((b, k), dtype=torch.long)
        target_valid = torch.zeros((b, k), dtype=torch.bool)
        group_slots = torch.zeros((b, k), dtype=torch.long)
        group_valid = torch.zeros((b, k), dtype=torch.bool)
        target_groups = torch.zeros((b, k), dtype=torch.long)
        reference_indices = torch.zeros((b, r), dtype=torch.long)
        reference_valid = torch.zeros((b, r), dtype=torch.bool)
        pre_nms_indices = torch.full((b, r), -1, dtype=torch.long)
        orders, first_indices = [], []
        target_numpy, reference_numpy = xyxy.cpu().numpy(), ref_xyxy.cpu().numpy()
        valid_cpu, ref_valid_cpu = valid.cpu(), ref_valid.cpu()
        for bi in range(b):
            slots = valid_cpu[bi].nonzero(as_tuple=True)[0]
            # Exact legacy CPU evaluator probability/tie policy.
            order = slots[score_cpu[bi, slots].argsort(descending=True)]
            orders.append(order.to(boxes.device))
            first_indices.append(int(order[0]) if len(order) else -1)
            selected = order[:k]
            count = len(selected)
            target_indices[bi, :count], target_valid[bi, :count] = selected, True
            representatives = []
            for local, slot in enumerate(selected.tolist()):
                ious = numpy_iou(target_numpy[bi, slot], target_numpy[bi, representatives]) if representatives else []
                matched = np.flatnonzero(np.asarray(ious) >= self.duplicate_iou)
                if len(matched):
                    group = int(matched[0])
                else:
                    group = len(representatives)
                    representatives.append(slot)
                    group_slots[bi, group], group_valid[bi, group] = local, True
                target_groups[bi, local] = group
            ref_slots = ref_valid_cpu[bi].nonzero(as_tuple=True)[0]
            proposed = ref_slots[foreground_cpu[bi, ref_slots].argsort(descending=True, stable=True)[:r]]
            pre_nms_indices[bi, :len(proposed)] = proposed
            kept = []
            for slot in proposed.tolist():
                if not kept or np.all(numpy_iou(reference_numpy[bi, slot], reference_numpy[bi, kept]) < self.duplicate_iou):
                    kept.append(slot)
            if kept:
                reference_indices[bi, :len(kept)] = torch.tensor(kept)
                reference_valid[bi, :len(kept)] = True
        tensors = dict(target_indices=target_indices, target_valid=target_valid, group_slots=group_slots,
                       group_valid=group_valid, target_groups=target_groups, reference_indices=reference_indices,
                       reference_valid=reference_valid, pre_nms_indices=pre_nms_indices)
        result = {name: value.to(boxes.device) for name, value in tensors.items()}
        result.update(boxes=xyxy, reference_boxes=ref_xyxy, valid=valid, foreground=foreground, crs_orders=orders,
                      crs_top1=torch.tensor(first_indices, device=boxes.device))
        return result

    @staticmethod
    def gather(tensor, indices):
        return tensor.gather(1, indices[..., None].expand(*indices.shape, tensor.shape[-1]))

    @staticmethod
    def geometry_vector(targets, references):
        tc = (targets[..., :2] + targets[..., 2:]) / 2
        rc = (references[..., :2] + references[..., 2:]) / 2
        ts = (targets[..., 2:] - targets[..., :2]).clamp(min=1e-6)
        rs = (references[..., 2:] - references[..., :2]).clamp(min=1e-6)
        displacement = rc[:, None] - tc[:, :, None]
        distance = displacement.square().sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
        ratio = (rs[:, None] / ts[:, :, None]).log().clamp(-10, 10)
        return torch.cat((displacement, distance, ratio, pairwise_iou(targets, references)[..., None]), -1)

    def aggregate(self, pair_logits, pair_valid, reference_logits, null_logit, group_logits):
        """Public, testable aggregation; masks are [B,G,R], no GT involved."""
        usable_ref = pair_valid.any(1)
        reference_mask = torch.cat((torch.ones_like(null_logit, dtype=torch.bool), usable_ref), 1)
        a = masked_softmax(torch.cat((null_logit, reference_logits), 1), reference_mask, 1)
        votes = masked_softmax(pair_logits / self.relation_temperature, pair_valid, 1)
        counts = pair_valid.sum(1)
        prior = masked_softmax(group_logits.detach()[..., None] / self.prior_temperature, pair_valid, 1)
        uniform = pair_valid.float() / counts[:, None].clamp(min=1)
        prior = (1 - self.prior_uniform_mix) * prior + self.prior_uniform_mix * uniform
        center = (prior * votes).sum(1, keepdim=True)
        # v lies in [0,1], so variance <= 1/4. Do not manufacture low-entropy
        # pseudo labels: this measures disagreement, NOT truth/reliability.
        discrimination = (4 * (prior * (votes - center).square()).sum(1)).clamp(0, 1)
        discrimination = discrimination * (counts > 1)
        floor = self.discriminability_floor + (1 - self.discriminability_floor) * discrimination
        neutral_votes = counts[:, None] * (votes - uniform) * pair_valid
        if self.aggregation == "competitive":
            raw = (a[:, None, 1:] * floor[:, None] * neutral_votes).sum(-1)
        elif self.aggregation == "vote":
            raw = (a[:, None, 1:] * neutral_votes).sum(-1)
        elif self.aggregation == "weighted":
            raw = (a[:, None, 1:] * pair_logits.tanh() * pair_valid).sum(-1)
        elif self.aggregation == "max":
            values = (a[:, None, 1:] * pair_logits.tanh()).masked_fill(~pair_valid, -1e9)
            raw = values.max(-1).values.masked_fill(~pair_valid.any(-1), 0)
        else:  # Matched additional-final-loss control; no relational correction.
            raw = pair_logits.sum(-1) * 0
        return raw, dict(reference_weights=a[:, 1:], null_weight=a[:, 0], votes=votes,
                         prior=prior, discriminability=discrimination, effective_discriminability=floor)

    def forward(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                reference_queries=None, reference_boxes=None, reference_classes=None,
                source_indices=None):
        reference_queries = queries if reference_queries is None else reference_queries
        reference_boxes = boxes if reference_boxes is None else reference_boxes
        reference_classes = classes if reference_classes is None else reference_classes
        if self.detach_inputs:
            # Cut BOTH paths of the added final loss: feature/text inputs AND
            # the direct residual CRS logit. Selection/box coordinates were
            # already detached independently of this training-mode switch.
            queries = queries.detach()
            reference_queries = reference_queries.detach()
            text = text.detach()
            scores = scores.detach()
            logits = logits.detach()
        b, n, c = queries.shape
        if (boxes.shape != (b, n, 4) or classes.shape != (b, n, 2) or text.shape[0] != b
                or reference_queries.shape[:2] != reference_boxes.shape[:2]
                or reference_classes.shape != (*reference_boxes.shape[:2], 2)):
            raise ValueError("Misaligned relation-refinement proposals/text")
        layout = self.select(boxes, reference_boxes, scores, reference_classes, img_metas)
        indices, ref_indices = layout["target_indices"], layout["reference_indices"]
        selected_boxes = self.gather(layout["boxes"], indices)
        groups = self.gather(selected_boxes, layout["group_slots"])
        references = self.gather(layout["reference_boxes"], ref_indices)
        feature = self.gather(reference_queries, ref_indices)
        token_valid = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool) if text_mask is None else ~text_mask.bool()
        if not text.shape[1]:
            raise ValueError("At least one text token slot is required")
        safe_mask = ~token_valid.clone()
        safe_mask[~token_valid.any(1), 0] = False
        q, t = self.visual(feature), self.text(text)
        # Keep identical kernels with/without diagnostic capture.
        read, attention = self.reader(q, t, t, key_padding_mask=safe_mask, need_weights=True)
        h = self.reference(torch.cat((q, read, q * read), -1))
        if not self.use_reference_semantics:
            h = torch.ones_like(h)  # Geometry-only ablation; not a new teacher.
        geometry = self.geometry_vector(groups, references)
        if not self.use_geometry:
            geometry = torch.zeros_like(geometry)
        pair_logits = self.pair_score(h[:, None] * self.geometry(geometry.to(h))).squeeze(-1).float()
        text_mean = (t * token_valid[..., None]).sum(1) / token_valid.sum(1).clamp(min=1)[:, None]
        reference_logits = self.reference_score(h).squeeze(-1).float()
        null_logit = self.null_score(text_mean).float()
        # Exclude self/same-instance relations; legitimate overlap remains legal.
        pair_valid = (layout["group_valid"][:, :, None] & layout["reference_valid"][:, None]
                      & (pairwise_iou(groups, references) < self.duplicate_iou)
                      & token_valid.any(1)[:, None, None])
        selected_logits = self.gather(logits.float().reshape(b, n, 1), indices).squeeze(-1)
        group_logits = selected_logits.gather(1, layout["group_slots"])
        raw, evidence = self.aggregate(pair_logits, pair_valid, reference_logits, null_logit, group_logits)
        # One target group has no across-target disambiguation problem.
        usable = (layout["group_valid"].sum(1) > 1) & pair_valid.flatten(1).any(1)
        group_delta = self.strength * raw.tanh() * usable[:, None]
        delta = group_delta.gather(1, layout["target_groups"]) * layout["target_valid"]
        final_selected = selected_logits + delta
        probability = scores.float().reshape(b, n).clone()
        priority = torch.zeros_like(probability)
        orders, relation_orders, logit_orders, changed = [], [], [], []
        # Batch host transfers once, rather than synchronizing CUDA for every
        # candidate/sample decision in the ranking loop.
        delta_cpu = delta.detach().cpu()
        final_cpu = final_selected.detach().cpu()
        selected_cpu = selected_logits.detach().cpu()
        for bi, original in enumerate(layout["crs_orders"]):
            count = min(len(original), indices.shape[1])
            selected = indices[bi, :count]
            # Identity really means identity, including legacy probability ties.
            local_cpu = (final_cpu[bi, :count].argsort(descending=True, stable=True)
                         if count and bool((delta_cpu[bi, :count] != 0).any()) else torch.arange(count))
            local = local_cpu.to(boxes.device)
            changed.append(float(count > 0 and int(local_cpu[0]) != 0))
            order = torch.cat((selected[local], original[count:]))
            priority[bi, order] = torch.arange(len(order), 0, -1, device=boxes.device).float()
            probability[bi, selected] = final_selected[bi, :count].sigmoid()
            orders.append(order)
            relation_orders.append(torch.cat((selected[delta_cpu[bi, :count].argsort(descending=True, stable=True).to(boxes.device)], original[count:])))
            logit_orders.append(torch.cat((selected[selected_cpu[bi, :count].argsort(descending=True, stable=True).to(boxes.device)], original[count:])))
        output = dict(scores=probability[..., None], ranking_priority=priority[..., None], layout=layout,
                      selected_logits=selected_logits, final_logits=final_selected, delta=delta,
                      final_orders=orders, relation_orders=relation_orders, logit_orders=logit_orders,
                      pair_logits=pair_logits, pair_valid=pair_valid, geometry=geometry, **evidence)
        output["metrics"] = dict(
            lambda_=self.strength.detach(), null_weight=evidence["null_weight"].mean().detach(),
            reference_count=layout["reference_valid"].sum(1).float().mean(),
            group_count=layout["group_valid"].sum(1).float().mean(),
            discriminability=evidence["discriminability"].sum().detach() / layout["reference_valid"].sum().clamp(min=1),
            correction_abs=(delta.abs() * layout["target_valid"]).sum().detach() / layout["target_valid"].sum().clamp(min=1),
            rank_changed=delta.new_tensor(changed).mean(),
            ready=delta.new_tensor(float(self.training or int(self.trained_steps) > 0)))
        output["metrics"]["lambda"] = output["metrics"].pop("lambda_")
        if self.capture_diagnostics:
            def cpu(value):
                return value.detach().cpu()
            output["diagnostics"] = [dict(
                boxes=cpu(layout["boxes"][bi]), reference_boxes=cpu(layout["reference_boxes"][bi]),
                crs_scores=cpu(scores.reshape(b, n)[bi]), crs_logits=cpu(logits.reshape(b, n)[bi]),
                target_indices=cpu(indices[bi]), target_valid=cpu(layout["target_valid"][bi]),
                target_source_indices=cpu(indices[bi] if source_indices is None else source_indices[bi].gather(0, indices[bi])),
                reference_indices=cpu(ref_indices[bi]), reference_valid=cpu(layout["reference_valid"][bi]),
                pre_nms_indices=cpu(layout["pre_nms_indices"][bi]),
                group_slots=cpu(layout["group_slots"][bi]), group_valid=cpu(layout["group_valid"][bi]),
                target_groups=cpu(layout["target_groups"][bi]), foreground=cpu(layout["foreground"][bi]),
                geometry=cpu(geometry[bi]), pair_logits=cpu(pair_logits[bi]), pair_valid=cpu(pair_valid[bi]),
                votes=cpu(evidence["votes"][bi]), competition_prior=cpu(evidence["prior"][bi]),
                reference_weights=cpu(evidence["reference_weights"][bi]), null_weight=cpu(evidence["null_weight"][bi]),
                discriminability=cpu(evidence["discriminability"][bi]),
                correction=cpu(delta[bi]), final_logits=cpu(final_selected[bi]), strength=cpu(self.strength),
                crs_order=cpu(layout["crs_orders"][bi]), final_order=cpu(orders[bi]),
                relation_order=cpu(relation_orders[bi]), logit_order=cpu(logit_orders[bi]),
                text_attention=cpu(attention[bi]) if attention is not None else None,
            ) for bi in range(b)]
        return output

    def loss(self, output, targets, img_metas):
        """Only original refer GT enters here, AFTER predicted-pool selection."""
        layout, z = output["layout"], output["final_logits"]
        ground_truth, present = [], []
        for boxes, meta in zip(targets, img_metas):
            referred = meta["refer_target_index"]
            if len(referred) > 1:
                raise ValueError("Two-stage relation refinement requires at most one refer GT")
            h, w = meta["img_shape"][:2]
            ground_truth.append(boxes.detach().reshape(-1, 4)[referred[0]].to(z) / z.new_tensor([w,h,w,h])
                                if len(referred) else z.new_zeros(4))
            present.append(bool(len(referred)))
        # No referent != no reference object in the sentence. Only the former
        # is excluded from the added loss and GT diagnostics.
        present = torch.tensor(present, device=z.device, dtype=torch.bool)
        gt = torch.stack(ground_truth)
        ious = pairwise_iou(layout["boxes"], gt[:, None]).squeeze(-1)
        valid = layout["target_valid"]
        positive = (ious.gather(1, layout["target_indices"]) >= .5) & valid & present[:, None]
        active = positive.any(1)
        per_sample = (torch.logsumexp(z.masked_fill(~valid, -1e9), 1)
                      - torch.logsumexp(z.masked_fill(~positive, -1e9), 1))
        sums = torch.where(active, per_sample, torch.zeros_like(per_sample)).sum() + output["pair_logits"].sum() * 0
        count = active.float().sum()
        before = layout["crs_top1"]
        after = torch.stack([order[0] if len(order) else before.new_tensor(-1) for order in output["final_orders"]])
        old = (ious.gather(1, before.clamp(min=0)[:, None]).squeeze(1) >= .5) & (before >= 0)
        new = (ious.gather(1, after.clamp(min=0)[:, None]).squeeze(1) >= .5) & (after >= 0)
        total = count.detach().clone()
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total)
            world = torch.distributed.get_world_size()
        # DDP averages gradients: local sums / GLOBAL valid sample count.
        loss = sums * world / total.clamp(min=1) * self.loss_weight
        metrics = {"supervised_samples": total}
        for name, values in (("reachable", active), ("corrected", ~old & new), ("harmed", old & ~new)):
            metrics[name] = (values & present).float().sum() / present.float().sum().clamp(min=1)
        return loss, metrics
