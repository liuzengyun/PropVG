"""BIRS v4: competitive CF supervision; v3 remains an independent policy.

All geometry/selection/control-validity decisions are detached and GT-free.
Only loss() reads referring GT. No relation labels or external models are used.
"""
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .binding_residual import pairwise_iou
from .binding_residual_v3 import AdaptiveBindingResidualScoring


class CompetitiveBindingResidualScoring(AdaptiveBindingResidualScoring):
    @property
    def crs_bce_weight(self):
        return 1.0

    def __init__(self, version="v4", reference_coverage=0.5,
                 semantic_difference=0.05, cf_negative_topk=8,
                 cf_positive_iou=0.5, control_reference_topk=4, **kwargs):
        if version != "v4":
            raise ValueError("CompetitiveBindingResidualScoring requires version='v4'")
        if (not math.isfinite(reference_coverage) or not 0 < reference_coverage <= 1
                or not math.isfinite(semantic_difference) or not 0 <= semantic_difference < 2
                or not math.isfinite(cf_positive_iou) or not 0 <= cf_positive_iou < 1
                or isinstance(cf_negative_topk, bool) or int(cf_negative_topk) != cf_negative_topk
                or cf_negative_topk < 1 or isinstance(control_reference_topk, bool)
                or int(control_reference_topk) != control_reference_topk or control_reference_topk < 2):
            raise ValueError("Invalid v4 coverage/semantic/CF sampling settings")
        kwargs.setdefault("attention_soft_mix", 0.1)
        kwargs.setdefault("cf_loss_weight", 0.1)
        super().__init__(version="v3", **kwargs)
        self.version = "v4"
        self.reference_coverage = float(reference_coverage)
        self.semantic_difference = float(semantic_difference)
        self.cf_negative_topk = int(cf_negative_topk)
        self.cf_positive_iou = float(cf_positive_iou)
        self.control_reference_topk = int(control_reference_topk)
        policy = dict(parent_signature=self.policy_signature.tolist(), version="v4",
                      reference_coverage=self.reference_coverage,
                      semantic_difference=self.semantic_difference,
                      control_reference_topk=self.control_reference_topk,
                      attention="stable_sparsemax_reference_only_soft_v1",
                      control="swap_centers_keep_sizes_v1",
                      evidence="valid_controls_mean_v1")
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()
        self.policy_signature.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        self.schema_version.fill_(5)

    @staticmethod
    def sparse_weights(logits, allowed):
        # Stable masked sparsemax. Padding contributes zero to cumulative sums,
        # never a finite sentinel that could exceed a very negative legal logit.
        if not bool(allowed.any(-1).all()):
            raise ValueError("Sparsemax requires an allowed slot (the null slot)")
        z = logits.float().masked_fill(~allowed, float('-inf'))
        z = z-z.max(-1, keepdim=True).values
        ordered = z.sort(-1, descending=True).values
        finite = torch.isfinite(ordered)
        cumulative = ordered.masked_fill(~finite, 0).cumsum(-1)
        ranks = torch.arange(1, z.shape[-1]+1, device=z.device, dtype=z.dtype)
        count = ((1+ranks*ordered > cumulative) & finite).sum(-1, keepdim=True).clamp(min=1)
        tau = (cumulative.gather(-1, count-1)-1)/count
        return (z-tau).clamp(min=0).masked_fill(~allowed, 0).to(logits.dtype)

    def reference_weights(self, logits, allowed):
        sparse = self.sparse_weights(logits, allowed)
        if not self.attention_soft_mix:
            return sparse, sparse
        refs = allowed.clone()
        refs[..., -1] = False
        has_reference = refs.any(-1, keepdim=True)
        # If no legal real reference exists, the soft path also goes to null.
        refs[..., -1:] = ~has_reference
        soft = F.softmax(logits.float().masked_fill(~refs, float('-inf')), -1).to(sparse.dtype)
        return (1-self.attention_soft_mix)*sparse+self.attention_soft_mix*soft, sparse

    def filter_references(self, legal, target_boxes, reference_boxes):
        intersection = (torch.minimum(target_boxes[:, :, None, 2:], reference_boxes[:, None, :, 2:])
                        - torch.maximum(target_boxes[:, :, None, :2], reference_boxes[:, None, :, :2]))
        intersection = intersection.clamp(min=0).prod(-1)
        ta = (target_boxes[..., 2:]-target_boxes[..., :2]).clamp(min=0).prod(-1)
        ra = (reference_boxes[..., 2:]-reference_boxes[..., :2]).clamp(min=0).prod(-1)
        coverage = torch.maximum(intersection/ta[:, :, None].clamp(min=1e-10),
                                 intersection/ra[:, None].clamp(min=1e-10))
        return legal & (coverage < self.reference_coverage)

    @property
    def needs_control_masks(self):
        # Inference and training must use the same control validity, even if
        # this experiment disables the auxiliary CF loss.
        return True

    @torch.no_grad()
    def control_reference_mask(self, legal, attention):
        # Use factual text-conditioned evidence, not GT, to focus interventions.
        ranking = attention[..., :-1].detach().masked_fill(~legal, float('-inf'))
        ids = ranking.argsort(dim=-1, descending=True, stable=True)[..., :self.control_reference_topk]
        return torch.zeros_like(legal).scatter(-1, ids, True) & legal

    def control_positions(self, geo, perm, target_boxes, reference_boxes, aspect):
        moved = geo.gather(2, perm[..., None].expand_as(geo))
        tw = (target_boxes[..., 2:]-target_boxes[..., :2]).clamp(min=1e-5)
        local = moved[..., :2]/aspect[:, None, None]/tw[:, :, None]
        # Exchange center positions but retain each reference's own dimensions.
        return torch.cat((moved[..., :5], geo[..., 5:7], local.tanh()), -1)

    @torch.no_grad()
    def valid_control(self, geo, positions, legal, reference, perm, target_boxes, reference_boxes, aspect):
        changed = (positions-geo).abs().amax(-1) > 1e-6
        q = F.normalize(reference.detach().float(), dim=-1)
        expanded = q[:, None].expand(-1, perm.shape[1], -1, -1)
        moved_q = expanded.gather(2, perm[..., None].expand_as(expanded))
        difference = (1-(expanded*moved_q).sum(-1)).clamp(0, 2)
        # normalize(0)==0: two zero vectors are identical, not cosine-distance 1.
        difference = difference.masked_fill((expanded-moved_q).abs().amax(-1) <= 1e-6, 0)
        tc = (target_boxes[..., :2]+target_boxes[..., 2:])/2
        center = tc[:, :, None]+positions[..., :2]/aspect[:, None, None]
        half = (reference_boxes[..., 2:]-reference_boxes[..., :2])[:, None]/2
        inside = ((center-half >= -1e-6) & (center+half <= 1+1e-6)).all(-1)
        # Reject a whole control if any legal moved object leaves the image.
        return ((changed & legal & (difference > self.semantic_difference)).any(-1)
                & (inside | ~legal).all(-1))

    def binding_evidence(self, actual, counter, eligible, control_masks):
        valid = torch.cat(control_masks, 1) & eligible[..., None]
        count = valid.sum(-1)
        mean = (counter*valid).sum(-1)/count.clamp(min=1)
        return torch.where(count > 0, actual-mean, torch.zeros_like(actual))

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        controls = output['cf_valid_controls']
        count = controls.sum(-1)
        mean = (output['selected_controls']*controls).sum(-1)/count.clamp(min=1)
        ids = output['target_indices']
        output['selected_control_mean'] = mean
        output['null'] = mean.new_zeros(output['valid'].shape).scatter(1, ids, mean)
        output['eligible'] = torch.zeros_like(output['valid']).scatter(1, ids, count > 0)
        if 'diagnostics' in output:
            output['diagnostics'].update(null=output['null'].detach().cpu(),
                                         eligible=output['eligible'].detach().cpu(),
                                         selected_control_mean=mean.detach().cpu())
        return output

    def loss(self, output, targets, img_metas, refer_labels):
        # GT is used exclusively here; forward/evaluation has no GT dependency.
        overlaps = []
        for bi, meta in enumerate(img_metas):
            ids = meta['refer_target_index']
            if len(ids) > 1:
                raise ValueError("BIRS v4 requires a single referring GT")
            if len(ids):
                h, w = meta['img_shape'][:2]
                gt = targets[bi].detach().reshape(-1, 4)[ids[0]].to(output['boxes'])
                gt = gt/gt.new_tensor([w, h, w, h])
                overlaps.append(pairwise_iou(output['boxes'][bi], gt[None]).squeeze(-1))
            else:
                overlaps.append(output['boxes'].new_full(output['valid'][bi].shape, -1))
        enriched = dict(output, cf_gt_iou=torch.stack(overlaps))
        return super().loss(enriched, targets, img_metas, refer_labels)

    def counterfactual_terms(self, output, labels, active):
        ids, valid_controls = output['target_indices'], output['cf_valid_controls']
        count = valid_controls.sum(-1)
        gap = output['selected_support']-(output['selected_controls']*valid_controls).sum(-1)/count.clamp(min=1)
        iou = output['cf_gt_iou'].gather(1, ids)
        valid = output['target_valid'] & (count > 0) & active[:, None]
        positive = (labels.gather(1, ids) > 0) & (iou > self.cf_positive_iou) & valid
        negative = (labels.gather(1, ids) == 0) & (iou == 0) & valid
        base = torch.nan_to_num(output['crs_logits'].detach().squeeze(-1).gather(1, ids),
                                nan=0.0, posinf=0.0, neginf=0.0)
        # Hard negatives are top CRS scores, not random background sampling.
        order = base.masked_fill(~negative, float('-inf')).argsort(dim=-1, descending=True, stable=True)
        hard = torch.zeros_like(negative).scatter(1, order[:, :self.cf_negative_topk], True) & negative
        pairs = positive[:, :, None] & hard[:, None, :]
        improvement = output['strength']*(gap[:, :, None]-gap[:, None, :])
        required = self.cf_margin+base[:, None, :]-base[:, :, None]
        hinge = F.relu(required-improvement)
        total = pairs.sum((1, 2)).clamp(min=1)
        terms = (hinge*pairs).sum((1, 2))/total
        mean_gap = (improvement.detach()*pairs).sum((1, 2))/total
        margin_met = ((improvement.detach() >= required)*pairs).sum((1, 2))/total
        used_positive = positive & pairs.any(-1)
        return terms, pairs.any(-1).any(-1), mean_gap, margin_met, used_positive, valid_controls
