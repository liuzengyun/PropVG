"""BIRS v5: candidate-local, zero-overlap neighbours with textual selection.

Keep v3's candidate prefix, parameters, full-candidate BCE and logit residual.
Only v5 uses a per-target nearest-neighbour pool. GT is read only by loss().
"""
import hashlib
import json

import torch
from torch.utils.checkpoint import checkpoint

from .binding_residual import pairwise_iou
from .binding_residual_v3 import AdaptiveBindingResidualScoring


class LocalBindingResidualScoring(AdaptiveBindingResidualScoring):
    def __init__(self, version="v5", neighbour_topk=8, **kwargs):
        if version != "v5":
            raise ValueError("LocalBindingResidualScoring requires version='v5'")
        if isinstance(neighbour_topk, bool) or int(neighbour_topk) != neighbour_topk or neighbour_topk < 1:
            raise ValueError("BIRS v5 neighbour_topk must be a positive integer")
        # This experiment changes reference retrieval, not the v3 objective.
        if kwargs.get("cf_loss_weight", 0) != 0 or kwargs.get("attention_soft_mix", 0) != 0:
            raise ValueError("BIRS v5 keeps cf_loss_weight=0 and attention_soft_mix=0")
        super().__init__(version="v3", **kwargs)
        self.version, self.neighbour_topk = "v5", int(neighbour_topk)
        policy = dict(parent_signature=self.policy_signature.tolist(), version="v5",
                      neighbour_topk=self.neighbour_topk, distance="image_short_side_center_l2",
                      overlap="target_reference_iou_exact_zero", references="nearest_after_topk_nms",
                      controls="fixed_local_set_full_geometry_cyclic_v1",
                      attention="stable_sparsemax_with_null_v1", null="factual_null_only_zero_delta")
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()
        self.policy_signature.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        self.schema_version.fill_(6)

    @staticmethod
    def sparse_weights(logits, allowed):
        """Stable masked sparsemax; the null slot is always legal."""
        z = logits.float().masked_fill(~allowed, float('-inf'))
        z = z-z.max(-1, keepdim=True).values
        ordered = z.sort(-1, descending=True).values
        finite = torch.isfinite(ordered)
        cumulative = ordered.masked_fill(~finite, 0).cumsum(-1)
        ranks = torch.arange(1, z.shape[-1]+1, device=z.device, dtype=z.dtype)
        count = ((1+ranks*ordered > cumulative) & finite).sum(-1, keepdim=True).clamp(min=1)
        tau = (cumulative.gather(-1, count-1)-1)/count
        return (z-tau).clamp(min=0).masked_fill(~allowed, 0).to(logits.dtype)

    @torch.no_grad()
    def select_local_references(self, target_boxes, reference_boxes, legal, aspect):
        """Nearest legal foregrounds in isotropic image coordinates, no GT.

        Boxes are xyxy normalized by width/height. Multiplication by aspect
        restores pixel distance up to a common image-short-side factor.
        Ties retain the foreground bank's deterministic confidence order.
        """
        tc = (target_boxes[..., :2]+target_boxes[..., 2:])/2
        rc = (reference_boxes[..., :2]+reference_boxes[..., 2:])/2
        offset = (rc[:, None]-tc[:, :, None])*aspect[:, None, None]
        distance = offset.square().sum(-1).sqrt()
        legal = legal & (pairwise_iou(target_boxes, reference_boxes) == 0)
        order = distance.masked_fill(~legal, float('inf')).argsort(dim=-1, stable=True)
        ids = order[..., :min(self.neighbour_topk, reference_boxes.shape[1])]
        keep = legal.gather(-1, ids)
        return ids, keep, distance.gather(-1, ids).masked_fill(~keep, 0)

    def _score_local(self, target, reference, geometry, text, padding, sentence, legal):
        # Unlike v3, reference already has a target axis: [B,K,H,D], H<=8.
        b, k, h, d = geometry.shape
        target_expanded = target[:, :, None].expand(-1, -1, h, -1)
        pair = self.pair_encoder(torch.cat((target_expanded, reference, geometry,
                                           target_expanded*reference), -1))
        tokens = pair.reshape(b*k, h, d)
        t = text[:, None].expand(-1, k, -1, -1).reshape(b*k, text.shape[1], d)
        mask = padding[:, None].expand(-1, k, -1).reshape(b*k, -1)
        read, _ = self.text_attention(tokens, t, t, key_padding_mask=mask, need_weights=False)
        read = self.read_norm(tokens+read).reshape(b, k, h, d)
        sentence = sentence[:, None].expand(-1, k, -1)
        null = self.null_logit(torch.cat((target, sentence), -1))
        attention = self.sparse_weights(
            torch.cat((self.evidence_logit(read).squeeze(-1), null), -1),
            torch.cat((legal, legal.new_ones(b, k, 1)), -1))
        pooled = (attention[..., :-1, None]*read).sum(-2)
        support = self.support(torch.cat((target, sentence, pooled), -1)).squeeze(-1)
        return support, attention, (attention[..., :-1] > 0).sum(-1)

    def forward(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                reference_queries=None, reference_boxes=None, reference_classes=None, source_indices=None):
        original_logits = logits
        if reference_queries is None:
            raise ValueError("BIRS v5 requires explicit pre-BCEM reference queries")
        if self.detach_inputs:
            queries, reference_queries, text = queries.detach(), reference_queries.detach(), text.detach()
            scores, logits = scores.detach(), logits.detach()
        xyxy, valid = self.clipped_boxes(boxes, img_metas)
        valid &= torch.isfinite(queries).all(-1) & torch.isfinite(logits).all(-1) & torch.isfinite(scores).all(-1)
        groups, candidate_mask, selected, selected_valid, before = self.select_targets(queries, valid, scores)
        b, n, _ = queries.shape
        k = selected.shape[1]
        ref_xyxy, ref_valid = self.clipped_boxes(reference_boxes, img_metas)
        ref_valid &= torch.isfinite(reference_queries).all(-1)
        ref_indices, ref_keep, ref_boxes, confidence = self.select_references(ref_xyxy, reference_classes, ref_valid)
        reference = self.reference_projection(torch.nan_to_num(reference_queries.gather(
            1, ref_indices[..., None].expand(-1, -1, queries.shape[-1]))))
        target = self.target_projection(torch.nan_to_num(queries.gather(
            1, selected[..., None].expand(-1, -1, queries.shape[-1]))))
        target_boxes = xyxy.gather(1, selected[..., None].expand(-1, -1, 4))
        source = (torch.arange(n, device=queries.device)[None].expand(b, -1)
                  if source_indices is None else source_indices).gather(1, selected)
        t = self.text_projection(torch.nan_to_num(text))
        padding = text_mask.bool().clone()
        text_valid = (~padding).any(-1)
        padding[:, 0] &= text_valid
        t = t.masked_fill(~text_valid[:, None, None], 0)
        sentence = t.masked_fill(text_mask.bool()[..., None], 0).sum(1)/(~text_mask.bool()).sum(1).clamp(min=1)[:, None]
        aspect = queries.new_tensor([[m['img_shape'][1]/min(m['img_shape'][:2]),
                                     m['img_shape'][0]/min(m['img_shape'][:2])] for m in img_metas])
        r, d = reference.shape[1:]
        factual, controls, attentions, control_attentions, legal_parts = [], [], [], [], []
        sparse_counts, local_slots, local_valid, local_distances = [], [], [], []
        for start in range(0, k, self.chunk_size):
            stop = min(k, start+self.chunk_size)
            chunk_target = target_boxes[:, start:stop]
            legal = ref_keep[:, None] & selected_valid[:, start:stop, None] & text_valid[:, None, None]
            legal &= source[:, start:stop, None] != ref_indices[:, None]
            ids, keep, distances = self.select_local_references(chunk_target, ref_boxes, legal, aspect)
            width = stop-start
            local_reference = reference[:, None].expand(-1, width, -1, -1).gather(
                2, ids[..., None].expand(-1, -1, -1, d))
            geo = self.geometry(chunk_target, ref_boxes, aspect).gather(
                2, ids[..., None].expand(-1, -1, -1, 9))
            branch_scores, branch_attentions = [], []
            for number in range(self.permutations+1):
                positions = geo
                if number:
                    # All controls use precisely the factual neighbour set;
                    # do not search again or move reference identities/null.
                    perm = self.permutation_indices(keep, source[:, start:stop], number-1)
                    positions = geo.gather(2, perm[..., None].expand_as(geo))
                inputs = (target[:, start:stop], local_reference, self.geometry_projection(positions),
                          t, padding, sentence, keep)
                if self.training and self.checkpoint_chunks:
                    score, attention, sparse_count = checkpoint(self._score_local, *inputs, use_reentrant=False)
                else:
                    score, attention, sparse_count = self._score_local(*inputs)
                branch_scores.append(score)
                # Keep existing audit tensor axes: original foreground bank + null.
                expanded = attention.new_zeros(b, width, r).scatter(2, ids, attention[..., :-1])
                branch_attentions.append(torch.cat((expanded, attention[..., -1:]), -1))
                if number == 0:
                    sparse_counts.append(sparse_count)
            factual.append(branch_scores[0])
            controls.append(torch.stack(branch_scores[1:], -1))
            attentions.append(branch_attentions[0])
            legal_parts.append(torch.zeros_like(legal).scatter(2, ids, keep))
            local_slots.append(ids); local_valid.append(keep); local_distances.append(distances)
            if self.capture_diagnostics:
                control_attentions.append(torch.stack(branch_attentions[1:], 2))
        actual, counter = torch.cat(factual, 1), torch.cat(controls, 1)
        attention, legal = torch.cat(attentions, 1), torch.cat(legal_parts, 1)
        # A factual null-only selection means no relation evidence, even if a
        # corrupted arrangement could make a real reference receive attention.
        eligible_selected = selected_valid & (legal.sum(-1) >= 2) & (attention[..., :-1].sum(-1) > 0)
        evidence = torch.where(eligible_selected, actual-counter.mean(-1), torch.zeros_like(actual))
        strength = self.max_strength*self.strength_logit.sigmoid()
        selected_delta = evidence*strength
        if not self.training and int(self.trained_steps) == 0:
            selected_delta = selected_delta*0
        delta = logits.new_zeros(b, n).scatter(1, selected, selected_delta)
        final = logits.squeeze(-1)+delta
        probability = torch.where(delta == 0, scores.squeeze(-1), final.sigmoid())
        priority = torch.zeros_like(final)
        after = []
        for bi in range(b):
            ids = selected[bi, selected_valid[bi]]
            order = (ids[final[bi, ids].detach().argsort(descending=True, stable=True)]
                     if bool((delta[bi, ids] != 0).any()) else ids)
            priority[bi, order] = torch.arange(len(order), 0, -1, device=queries.device, dtype=final.dtype)
            after.append(order)

        def scatter(value):
            shape = (b, n)+tuple(value.shape[2:])
            index = selected.reshape(b, k, *([1]*(value.ndim-2))).expand_as(value)
            return value.new_zeros(shape).scatter(1, index, value)

        slots, keep = torch.cat(local_slots, 1), torch.cat(local_valid, 1)
        output = dict(scores=probability[..., None], final_logits=final[..., None], crs_logits=original_logits,
                      ranking_priority=priority[..., None], output_mask=candidate_mask,
                      boxes=xyxy, valid=valid, groups=groups, candidate_mask=candidate_mask,
                      target_indices=selected, target_valid=selected_valid, before_orders=before, after_orders=after,
                      raw_binding=scatter(evidence), paired=scatter(actual), null=scatter(counter.mean(-1)),
                      delta=delta, eligible=scatter(eligible_selected), strength=strength,
                      legal_references=scatter(legal), reference_indices=ref_indices, reference_valid=ref_keep,
                      base_scores=scores.detach().squeeze(-1), selected_attention=attention,
                      selected_support=actual, selected_controls=counter,
                      selected_sparse_reference_count=torch.cat(sparse_counts, 1),
                      local_reference_slots=slots, local_reference_valid=keep,
                      local_reference_indices=ref_indices[:, None].expand(-1, k, -1).gather(2, slots).masked_fill(~keep, -1),
                      local_reference_distances=torch.cat(local_distances, 1))
        if self.capture_diagnostics:
            output['diagnostics'] = {name: value.detach().cpu() for name, value in output.items() if torch.is_tensor(value)}
            output['diagnostics'].update(reference_boxes=ref_boxes.detach().cpu(),
                                         reference_confidence=confidence.detach().cpu(),
                                         selected_control_attention=torch.cat(control_attentions, 1).detach().cpu())
        return output
