"""Detached, observational prediction of the final Top-1 grounding correctness."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from propvg.layers.box_ops import box_cxcywh_to_xyxy


def pair_iou(boxes, target):
    lo = torch.maximum(boxes[:, :2], target[:2])
    hi = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (hi - lo).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(-1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    return intersection / (area + target_area - intersection).clamp(min=1e-6)


class Top1CorrectnessHead(nn.Module):
    """Estimate P(final Top-1 IoU >= .5); never change grounding scores.

    All features are detached here, not just in the caller. Sorting on CPU uses
    exactly the evaluator's tie policy, after clipping/removing empty boxes.
    Only the small detached Top-K list is used for geometric grouping.
    """
    def __init__(self, hidden_channels=256, hidden_dim=128, topk=5,
                 duplicate_iou=.8, loss_weight=1., decision_threshold=.5):
        super().__init__()
        if min(hidden_channels, hidden_dim, topk) < 1 or not 0 < duplicate_iou <= 1:
            raise ValueError("Correctness dimensions/topk must be positive and duplicate_iou in (0,1]")
        if loss_weight <= 0 or not 0 < decision_threshold < 1:
            raise ValueError("Correctness loss_weight must be positive and decision_threshold in (0,1)")
        self.topk = topk
        self.duplicate_iou = duplicate_iou
        self.loss_weight = loss_weight
        self.decision_threshold = decision_threshold
        self.visual = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, hidden_dim), nn.GELU())
        # q, t, q*t, |q-t|, plus 12 explicitly defined scalar observations.
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim * 4 + 12),
                                        nn.Linear(hidden_dim * 4 + 12, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.register_buffer("schema_version", torch.tensor(1, dtype=torch.long))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))

    @staticmethod
    @torch.no_grad()
    def clipped_boxes(boxes, img_metas):
        # Match MIXRefUniModel_OMG.inference + detector_postprocess: normalized
        # cxcywh -> valid-image pixel xyxy -> clip. Padding never rescales GT.
        # Boxes() in the real postprocessor converts xyxy to float32 before
        # pixel scaling, even when the decoder itself runs in fp16.
        result = box_cxcywh_to_xyxy(boxes.detach()).float().clone()
        for index, meta in enumerate(img_metas):
            height, width = meta["img_shape"][:2]
            result[index] *= result.new_tensor([width, height, width, height])
            result[index, :, 0::2].clamp_(0, width)
            result[index, :, 1::2].clamp_(0, height)
        return result

    @torch.no_grad()
    def observations(self, boxes, scores, logits, classes, img_metas, ranking_priority=None):
        clipped = self.clipped_boxes(boxes, img_metas)
        boxes_cpu = clipped.float().cpu()
        scores_cpu = scores.detach().reshape(boxes.shape[:2]).float().cpu()
        logits_cpu = logits.detach().reshape(boxes.shape[:2]).float().cpu()
        foreground = classes.detach().float().softmax(-1)[..., 0].cpu()
        ranking = (ranking_priority.detach().reshape(boxes.shape[:2]).float().cpu()
                   if ranking_priority is not None else scores_cpu)
        selected, scalars, entropies = [], [], []
        for batch, (candidates, meta) in enumerate(zip(boxes_cpu, img_metas)):
            nonempty = ((candidates[:, 2:] - candidates[:, :2]) > 0).all(-1)
            valid = nonempty.nonzero(as_tuple=True)[0]
            order = ranking[batch, valid].argsort(descending=True, stable=ranking_priority is not None)
            indices = valid[order[:self.topk]].tolist()
            if not indices:
                selected.append(-1)
                scalars.append([0.] * 12)
                entropies.append(0.)
                continue
            first = indices[0]
            selected.append(first)
            groups = []
            for index in indices:
                if not groups or pair_iou(candidates[groups], candidates[index]).max() < self.duplicate_iou:
                    groups.append(index)
            z = logits_cpu[batch, groups].double()
            probabilities = scores_cpu[batch, groups]
            relative = z.softmax(0)
            entropy = float(-(relative * relative.clamp(min=1e-30).log()).sum() / math.log(len(groups))) if len(groups) > 1 else 0.
            gap = float(probabilities[0] - probabilities[1]) if len(groups) > 1 else 0.
            logit_gap = float(z[0] - z[1]) if len(groups) > 1 else 0.
            height, width = meta["img_shape"][:2]
            xyxy = candidates[first] / candidates.new_tensor([width, height, width, height])
            center, size = (xyxy[:2] + xyxy[2:]) / 2, xyxy[2:] - xyxy[:2]
            scalars.append([float(probabilities[0]), float(z[0]) / 10., gap, logit_gap / 10.,
                            entropy, float(relative[0]), len(groups) / self.topk,
                            float(foreground[batch, first]), *center.tolist(), *size.tolist()])
            entropies.append(entropy)
        device = boxes.device
        return (torch.tensor(selected, device=device), boxes.new_tensor(scalars),
                boxes.new_tensor(entropies), clipped)

    def forward(self, queries, boxes, text, text_mask, scores, logits, classes,
                img_metas, ranking_priority=None):
        index, scalars, entropy, clipped = self.observations(
            boxes, scores, logits, classes, img_metas, ranking_priority)
        usable = index >= 0
        if queries.shape[1]:
            visual = queries.detach().gather(1, index.clamp(min=0)[:, None, None].expand(-1, 1, queries.shape[-1])).squeeze(1)
            visual = visual * usable[:, None]
        else:
            visual = queries.new_zeros((queries.shape[0], queries.shape[-1]))
        valid = (torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
                 if text_mask is None else ~text_mask.bool())
        language = (text.detach() * valid[..., None]).sum(1) / valid.sum(1).clamp(min=1)[:, None]
        q, t = self.visual(visual), self.text(language)
        evidence = torch.cat((q, t, q * t, (q - t).abs(), scalars.to(q)), dim=-1)
        raw = self.classifier(evidence).squeeze(-1)
        raw = torch.where(usable, raw, raw.new_full(raw.shape, -20.))
        probability = torch.where(usable, raw.sigmoid(), torch.zeros_like(raw))
        return dict(probability=probability, logit=raw, index=index,
                    entropy=entropy, clipped_boxes=clipped)

    def loss(self, observation, targets, img_metas):
        raw = observation["logit"]
        labels = raw.new_full(raw.shape, -1.)
        # GT is used only here, never by forward/observations during inference.
        with torch.no_grad():
            for batch, (boxes, meta) in enumerate(zip(targets, img_metas)):
                indices = meta["refer_target_index"]
                if len(indices) > 1:
                    raise ValueError("Top-1 correctness requires one refer GT, not a multi-target expression")
                if not len(indices):
                    continue
                target = boxes.detach().reshape(-1, 4)[indices[0]].to(observation["clipped_boxes"])
                index = int(observation["index"][batch])
                labels[batch] = (pair_iou(observation["clipped_boxes"][batch, index:index + 1], target)[0] >= .5
                                 if index >= 0 else 0.)
        valid = labels >= 0
        count = valid.float().sum()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            world_size = torch.distributed.get_world_size()
        scale = world_size / count.clamp(min=1.)
        loss = (F.binary_cross_entropy_with_logits(raw, labels.clamp(min=0), reduction="none") * valid).sum() * scale
        predicted = observation["probability"].detach() >= self.decision_threshold
        metrics = dict(valid_samples=count.detach(),
                       accuracy=(((predicted == labels.bool()) & valid).float().sum() * scale).detach(),
                       probability=((observation["probability"].detach() * valid).sum() * scale).detach())
        return loss * self.loss_weight, metrics

    @torch.no_grad()
    def mark_optimizer_step(self):
        self.trained_steps.add_(1)

    def require_trained(self):
        if not self.trained_steps.item():
            raise RuntimeError("Correctness head is untrained. Train it first or set enable_correctness=False for old models.")
