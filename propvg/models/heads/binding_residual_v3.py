"""BIRS v3: adaptive prefix, sparse relational reader and rebinding controls.

Selection/geometry/permutations are GT-free and detached. Full-candidate logits
are retained for training; output_mask restricts predictions to the shortlist.
Legacy v1/v2 construction/state dictionaries are intentionally untouched.
"""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .binding_residual import BindingInteractionResidualScoring, pairwise_iou


class AdaptiveBindingResidualScoring(BindingInteractionResidualScoring):
    def __init__(self, hidden_channels=256, version="v3", num_groups=8, kmeans_iters=6,
                 seed_topk=3, min_targets=5, max_targets=32, reference_topk=64,
                 duplicate_iou=0.8, model_dim=128, num_heads=4, chunk_size=8,
                 permutations=2, initial_strength=0.1, max_strength=2.0,
                 final_mix=0.5, detach_inputs=False, checkpoint_chunks=True):
        nn.Module.__init__(self)
        if version != "v3":
            raise ValueError("AdaptiveBindingResidualScoring requires version='v3'")
        counts = (hidden_channels, num_groups, kmeans_iters, seed_topk, min_targets,
                  max_targets, reference_topk, model_dim, num_heads, chunk_size, permutations)
        if any(isinstance(v, bool) or int(v) != v or v < 1 for v in counts):
            raise ValueError("BIRS v3 counts must be positive integers")
        if not seed_topk <= min_targets <= max_targets:
            raise ValueError("BIRS v3 requires seed_topk <= min_targets <= max_targets")
        if model_dim % num_heads or not 0 < duplicate_iou <= 1:
            raise ValueError("Invalid BIRS v3 dimensions or duplicate IoU")
        if not 0 < initial_strength < max_strength or not 0 < final_mix < 1:
            raise ValueError("BIRS v3 requires 0 < initial_strength < max_strength and 0 < final_mix < 1")
        self.version = version
        self.num_groups, self.kmeans_iters = int(num_groups), int(kmeans_iters)
        self.seed_topk, self.min_targets, self.max_targets = int(seed_topk), int(min_targets), int(max_targets)
        self.reference_topk, self.duplicate_iou = int(reference_topk), float(duplicate_iou)
        self.chunk_size, self.permutations = int(chunk_size), int(permutations)
        self.final_mix, self.detach_inputs = float(final_mix), bool(detach_inputs)
        self.max_strength, self.checkpoint_chunks = float(max_strength), bool(checkpoint_chunks)
        self.capture_diagnostics = False
        self.target_projection = nn.Linear(hidden_channels, model_dim)
        self.reference_projection = nn.Linear(hidden_channels, model_dim)
        self.text_projection = nn.Linear(hidden_channels, model_dim)
        self.geometry_projection = nn.Sequential(nn.Linear(9, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.pair_encoder = nn.Sequential(nn.Linear(model_dim * 4, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.text_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=0, batch_first=True)
        self.read_norm = nn.LayerNorm(model_dim)
        self.evidence_logit = nn.Linear(model_dim, 1)
        self.null_logit = nn.Linear(model_dim * 2, 1)
        self.support = nn.Sequential(nn.Linear(model_dim * 3, model_dim), nn.GELU(), nn.Linear(model_dim, 1))
        self.strength_logit = nn.Parameter(torch.tensor(math.log(initial_strength / (max_strength-initial_strength))))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("schema_version", torch.tensor(4, dtype=torch.long))
        policy = dict(version=version, hidden_channels=hidden_channels, num_groups=num_groups,
                      kmeans_iters=kmeans_iters, seed_topk=seed_topk, min_targets=min_targets,
                      max_targets=max_targets, reference_topk=reference_topk, duplicate_iou=duplicate_iou,
                      model_dim=model_dim, num_heads=num_heads, permutations=permutations,
                      max_strength=max_strength, final_mix=final_mix,
                      reference_source="pre_bcem_decoder", permutation="legal_cyclic_v1",
                      output="prefix_only", loss="full_valid_two_stage_bce")
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()
        self.register_buffer("policy_signature", torch.tensor(list(digest), dtype=torch.uint8))

    @staticmethod
    def sparse_weights(logits, allowed):
        """Masked sparsemax, including one always-allowed zero-value null slot."""
        z = logits.float().masked_fill(~allowed, -1e4)
        z = z - z.max(-1, keepdim=True).values
        ordered = z.sort(-1, descending=True).values
        ranks = torch.arange(1, z.shape[-1]+1, device=z.device, dtype=z.dtype)
        cumulative = ordered.cumsum(-1)
        count = (1+ranks*ordered > cumulative).sum(-1, keepdim=True).clamp(min=1)
        tau = (cumulative.gather(-1, count-1)-1)/count
        return (z-tau).clamp(min=0).masked_fill(~allowed, 0).to(logits.dtype)

    @torch.no_grad()
    def select_targets(self, queries, valid, scores):
        groups = self.cluster(queries, valid)
        b, n = valid.shape
        selected = torch.zeros_like(valid)
        orders = []
        cap = min(n, self.max_targets)
        indices = []
        for bi in range(b):
            ids = valid[bi].nonzero(as_tuple=True)[0]
            order = ids[scores[bi, ids, 0].detach().float().cpu().argsort(descending=True).to(ids.device)]
            window = order[:cap]
            seed_groups = groups[bi, window[:self.seed_topk]]
            matches = (groups[bi, window, None] == seed_groups[None]).any(-1)
            farthest = int(matches.nonzero(as_tuple=True)[0][-1])+1 if bool(matches.any()) else 0
            count = min(len(window), max(self.min_targets, farthest))
            prefix = window[:count]
            selected[bi, prefix] = True
            invalid = (~valid[bi]).nonzero(as_tuple=True)[0]
            # Additional valid outsiders may pad the tensor but remain masked.
            padding = torch.cat((order[count:], invalid))
            indices.append(torch.cat((prefix, padding))[:cap])
            orders.append(order)
        indices = torch.stack(indices)
        # Pad only to this batch's largest selected N, not the search cap.
        width = max(1, int(selected.sum(-1).max()))
        indices = indices[:, :width]
        return groups, selected, indices, selected.gather(1, indices), orders

    @staticmethod
    @torch.no_grad()
    def permutation_indices(legal, source, number):
        """Nonidentity rotations ONLY within each target's legal reference set.

        Deterministic at train/test, independent of GT/RNG. Small sets can have
        repeated controls. These are finite controls, NOT a uniform expectation.
        """
        count = legal.sum(-1, keepdim=True)
        order = legal.long().argsort(dim=-1, descending=True, stable=True)
        rank = legal.long().cumsum(-1)-1
        shift = (source[..., None] * (2*number+3) + number+1) % (count-1).clamp(min=1) + 1
        destination = (rank+shift) % count.clamp(min=1)
        moved = order.gather(-1, destination.clamp(min=0))
        identity = torch.arange(legal.shape[-1], device=legal.device).expand_as(moved)
        return torch.where(legal & (count >= 2), moved, identity)

    def _score(self, target, reference, geometry, text, padding, sentence, legal):
        b, k, r, d = geometry.shape
        target_expanded = target[:, :, None].expand(-1, -1, r, -1)
        reference = reference[:, None].expand(-1, k, -1, -1)
        pair = self.pair_encoder(torch.cat((target_expanded, reference, geometry,
                                           target_expanded*reference), -1))
        tokens = pair.reshape(b*k, r, d)
        t = text[:, None].expand(-1, k, -1, -1).reshape(b*k, text.shape[1], d)
        mask = padding[:, None].expand(-1, k, -1).reshape(b*k, -1)
        read, _ = self.text_attention(tokens, t, t, key_padding_mask=mask, need_weights=False)
        read = self.read_norm(tokens+read).reshape(b, k, r, d)
        sentence = sentence[:, None].expand(-1, k, -1)
        null = self.null_logit(torch.cat((target, sentence), -1))
        attention = self.sparse_weights(torch.cat((self.evidence_logit(read).squeeze(-1), null), -1),
                                        torch.cat((legal, legal.new_ones(b, k, 1)), -1))
        pooled = (attention[..., :-1, None]*read).sum(-2)  # Null VALUE is zero.
        support = self.support(torch.cat((target, sentence, pooled), -1)).squeeze(-1)
        return support, attention

    def forward(self, queries, boxes, classes, text, text_mask, scores, logits, img_metas,
                reference_queries=None, reference_boxes=None, reference_classes=None, source_indices=None):
        original_logits = logits
        if reference_queries is None:
            raise ValueError("BIRS v3 requires explicit pre-BCEM reference queries")
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
        factual, controls, attentions, control_attentions, legal_parts = [], [], [], [], []
        for start in range(0, k, self.chunk_size):
            stop = min(k, start+self.chunk_size)
            legal = ref_keep[:, None] & selected_valid[:, start:stop, None] & text_valid[:, None, None]
            legal &= source[:, start:stop, None] != ref_indices[:, None]
            legal &= pairwise_iou(target_boxes[:, start:stop], ref_boxes) < self.duplicate_iou
            geo = self.geometry(target_boxes[:, start:stop], ref_boxes, aspect)
            branch_scores, branch_attentions = [], []
            for number in range(self.permutations+1):
                positions = geo
                if number:
                    perm = self.permutation_indices(legal, source[:, start:stop], number-1)
                    positions = geo.gather(2, perm[..., None].expand_as(geo))
                # The same geometry encoder, relation reader and sparse selector
                # are rerun for both factual and counterfactual arrangements.
                geometry = self.geometry_projection(positions)
                inputs = (target[:, start:stop], reference, geometry, t, padding, sentence, legal)
                if self.training and self.checkpoint_chunks:
                    score, attention = checkpoint(self._score, *inputs, use_reentrant=False)
                else:
                    score, attention = self._score(*inputs)
                branch_scores.append(score)
                branch_attentions.append(attention)
            factual.append(branch_scores[0]); controls.append(torch.stack(branch_scores[1:], -1))
            attentions.append(branch_attentions[0]); legal_parts.append(legal)
            if self.capture_diagnostics:
                control_attentions.append(torch.stack(branch_attentions[1:], 2))
        actual, counter = torch.cat(factual, 1), torch.cat(controls, 1)
        attention, legal = torch.cat(attentions, 1), torch.cat(legal_parts, 1)
        eligible_selected = selected_valid & (legal.sum(-1) >= 2)
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
        output = dict(scores=probability[..., None], final_logits=final[..., None], crs_logits=original_logits,
                      ranking_priority=priority[..., None], output_mask=candidate_mask,
                      boxes=xyxy, valid=valid, groups=groups, candidate_mask=candidate_mask,
                      target_indices=selected, target_valid=selected_valid, before_orders=before, after_orders=after,
                      raw_binding=scatter(evidence), paired=scatter(actual), null=scatter(counter.mean(-1)),
                      delta=delta, eligible=scatter(eligible_selected), strength=strength,
                      legal_references=scatter(legal), reference_indices=ref_indices, reference_valid=ref_keep,
                      base_scores=scores.detach().squeeze(-1), selected_attention=attention,
                      selected_support=actual, selected_controls=counter)
        if self.capture_diagnostics:
            output['diagnostics'] = {name: value.detach().cpu() for name, value in output.items() if torch.is_tensor(value)}
            output['diagnostics'].update(reference_boxes=ref_boxes.detach().cpu(),
                                         reference_confidence=confidence.detach().cpu(),
                                         selected_control_attention=torch.cat(control_attentions, 1).detach().cpu())
        return output

    def loss(self, output, targets, img_metas, refer_labels):
        valid = output['valid']
        z, base = output['final_logits'].squeeze(-1), output['crs_logits'].squeeze(-1)
        labels = refer_labels.detach().to(z)
        present = []
        for meta in img_metas:
            if len(meta['refer_target_index']) > 1:
                raise ValueError('BIRS v3 requires a single referring GT')
            present.append(bool(len(meta['refer_target_index'])))
        present = torch.tensor(present, device=z.device, dtype=torch.bool)
        active = present & ((labels > 0) & valid).any(-1)
        count = active.float().sum()
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            world = torch.distributed.get_world_size()
        zero = sum(p.sum()*0 for p in self.parameters())+torch.nan_to_num(z).sum()*0
        def bce(value):
            value = torch.where(valid, value, torch.zeros_like(value))
            per_sample = (F.binary_cross_entropy_with_logits(value, labels, reduction='none')*valid).sum(-1)/valid.sum(-1).clamp(min=1)
            return (per_sample*active).sum()*world/count.clamp(min=1)+zero
        losses = dict(loss_birs_crs=(1-self.final_mix)*bce(base),
                      loss_birs_final=self.final_mix*bce(z))
        corrected, harmed, changed, reachable = [], [], [], []
        for bi, (old, new) in enumerate(zip(output['before_orders'], output['after_orders'])):
            h, w = img_metas[bi]['img_shape'][:2]
            gt = (targets[bi].detach().reshape(-1, 4)[img_metas[bi]['refer_target_index'][0]].to(z)/z.new_tensor([w,h,w,h])
                  if present[bi] else z.new_zeros(4))
            iou = pairwise_iou(output['boxes'][bi], gt[None]).squeeze(-1)
            a = bool(len(old)) and bool(iou[old[0]] >= .5) and bool(present[bi])
            b = bool(len(new)) and bool(iou[new[0]] >= .5) and bool(present[bi])
            corrected.append(not a and b); harmed.append(a and not b)
            changed.append(bool(len(old)) and bool(len(new)) and bool(old[0] != new[0]))
            reachable.append(bool(len(new)) and bool((iou[new] >= .5).any()) and bool(present[bi]))
        mask = output['target_valid']
        denominator = mask.sum().clamp(min=1)
        attn = output['selected_attention'].detach()
        metrics = dict(supervised_samples=count, strength=output['strength'].detach(),
                       target_count=mask.float().sum(-1).mean(),
                       target_cap_rate=(mask.sum(-1) == min(self.max_targets, valid.shape[1])).float().mean(),
                       reference_count=output['reference_valid'].float().sum(-1).mean(),
                       eligible=output['eligible'].float().sum()/output['candidate_mask'].sum().clamp(min=1),
                       delta_abs=output['delta'].detach().abs().sum()/denominator,
                       null_attention=(attn[..., -1]*mask).sum()/denominator,
                       sparse_reference_count=((attn[..., :-1] > 0).sum(-1)*mask).sum()/denominator,
                       refer_in_topn=(((labels > 0) & output['candidate_mask']).any(-1) & present).float().sum()/present.sum().clamp(min=1),
                       reachable=z.new_tensor(reachable).sum()/present.sum().clamp(min=1),
                       corrected=z.new_tensor(corrected).mean(), harmed=z.new_tensor(harmed).mean(),
                       rank_changed=z.new_tensor(changed).mean())
        return losses, metrics
