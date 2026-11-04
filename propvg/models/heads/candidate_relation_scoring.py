"""Candidate-centred pairwise text matching with adaptive neighbour pooling.

No GT is accepted. Discrete context selection and geometry use detached boxes;
candidate features and the shared full-text reader remain jointly trainable.
"""
import math

import numpy as np
import torch
from torch import nn


class PairTextLayer(nn.Module):
    """Independent pair queries share text K/V, keeping the image batch axis."""
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim), nn.Dropout(dropout))

    def prepare_memory(self, text, position):
        shape = (text.shape[0], text.shape[1], self.heads, self.head_dim)
        return (self.key(text + position).reshape(shape).transpose(1, 2),
                self.value(text).reshape(shape).transpose(1, 2))

    def forward(self, pair, memory, text_valid):
        # pair: B,P,D; memory: B,H,L,D/H. Do not expand K/V to B*P
        # copies: each image supplies one memory to all of its pair queries.
        b, count, _ = pair.shape
        q = self.query(pair).reshape(b, count, self.heads, self.head_dim).transpose(1, 2)
        key, value = memory
        logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        valid = text_valid[:, None, None, :]
        logits = logits.float().masked_fill(~valid, -torch.inf)
        # Empty text is finite and contributes no memory, not an artificial word.
        logits = torch.where(text_valid.any(-1)[:, None, None, None], logits, torch.zeros_like(logits))
        weights = logits.softmax(-1).to(pair.dtype) * valid.to(pair.dtype)
        context = torch.matmul(self.attention_dropout(weights), value).transpose(1, 2).reshape_as(pair)
        pair = self.norm1(pair + self.output_dropout(self.output(context)))
        return self.norm2(pair + self.ffn(pair)), weights.mean(1)


class CandidateRelationScorer(nn.Module):
    def __init__(self, hidden_channels=256, num_neighbors=6, model_dim=256,
                 geometry_dim=64, num_heads=8, num_layers=2, ffn_dim=512,
                 dropout=0.1, duplicate_iou=0.8, chunk_size=256,
                 geometry_eps=1e-6, log_ratio_clip=5.0,
                 foreground_topk=None, strict_disjoint_neighbors=False,
                 aggregation="adaptive"):
        super().__init__()
        for name, value in dict(hidden_channels=hidden_channels, num_neighbors=num_neighbors,
                                model_dim=model_dim, geometry_dim=geometry_dim, num_heads=num_heads,
                                num_layers=num_layers, ffn_dim=ffn_dim, chunk_size=chunk_size).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Relation scoring {} must be a positive integer".format(name))
        if model_dim % num_heads or model_dim % 2:
            raise ValueError("Relation model_dim must be even and divisible by num_heads")
        if not 0 < duplicate_iou <= 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid relation duplicate_iou/dropout")
        if not math.isfinite(geometry_eps) or geometry_eps <= 0 or not math.isfinite(log_ratio_clip) or log_ratio_clip <= 0:
            raise ValueError("Relation geometry numerical safeguards must be finite and positive")
        if foreground_topk is not None and (isinstance(foreground_topk, bool)
                                            or not isinstance(foreground_topk, int) or foreground_topk < 1):
            raise ValueError("Relation foreground_topk must be None or a positive integer")
        if not isinstance(strict_disjoint_neighbors, bool):
            raise ValueError("Relation strict_disjoint_neighbors must be bool")
        if aggregation not in ("adaptive", "max"):
            raise ValueError("Relation aggregation must be 'adaptive' or 'max'")
        # Defaults intentionally retain the original policy for archived configs.
        self.aggregation = aggregation
        self.foreground_topk = foreground_topk
        self.strict_disjoint_neighbors = strict_disjoint_neighbors
        self.num_neighbors, self.model_dim, self.chunk_size = num_neighbors, model_dim, chunk_size
        self.duplicate_iou, self.geometry_eps, self.log_ratio_clip = duplicate_iou, geometry_eps, log_ratio_clip
        self.visual = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, model_dim))
        self.text = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, model_dim))
        self.geometry = nn.Sequential(nn.Linear(5, geometry_dim), nn.GELU(),
                                      nn.Linear(geometry_dim, geometry_dim), nn.LayerNorm(geometry_dim))
        self.pair = nn.Sequential(nn.Linear(3 * model_dim + geometry_dim, model_dim),
                                  nn.GELU(), nn.LayerNorm(model_dim))
        self.layers = nn.ModuleList([PairTextLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.score = nn.Sequential(nn.Linear(model_dim, max(1, model_dim // 2)), nn.GELU(),
                                   nn.Linear(max(1, model_dim // 2), 1))
        self.weight = nn.Sequential(nn.Linear(model_dim, max(1, model_dim // 4)), nn.GELU(),
                                    nn.Linear(max(1, model_dim // 4), 1))
        nn.init.zeros_(self.weight[-1].weight)
        nn.init.zeros_(self.weight[-1].bias)
        self.register_buffer("schema_version", torch.tensor(1, dtype=torch.long))
        self.register_buffer("trained_steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def select_foreground_indices(self, boxes, classes):
        """Select slots only; REFHead gathers tensors OUTSIDE no_grad.

        No GT or referring scores enter this decision. Invalid boxes rank last
        and serve only as fixed-batch padding when fewer than K are valid;
        normal box postprocessing removes them, as in the original pipeline.
        """
        if self.foreground_topk is None:
            return None
        if classes.shape != boxes.shape[:2] + (2,):
            raise ValueError("Foreground selection needs aligned B,N,2 detector class logits")
        raw = boxes.detach().float()
        xyxy = torch.cat((raw[..., :2] - raw[..., 2:] / 2,
                          raw[..., :2] + raw[..., 2:] / 2), -1).clamp(0, 1)
        valid = torch.isfinite(raw).all(-1) & (xyxy[..., 2:] > xyxy[..., :2]).all(-1)
        foreground = classes.detach().float().softmax(-1)[..., 0]
        valid &= torch.isfinite(foreground)
        order = foreground.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
        return order[:, :min(self.foreground_topk, boxes.shape[1])]

    @torch.no_grad()
    def select_neighbors(self, boxes, classes, img_metas):
        """One CPU transfer, then strict disjoint KNN or archived context NMS.

        This method only selects neighbors; target pruning is performed by
        select_foreground_indices. CPU selection avoids per-candidate CUDA
        scalar synchronization during greedy overlap checks.
        """
        b, n, _ = boxes.shape
        if classes.shape[:2] != (b, n) or classes.shape[-1] != 2 or len(img_metas) != b:
            raise ValueError("Relation scoring requires aligned B,N,2 DETR class logits and image metadata")
        packed = torch.cat((boxes.detach().float(), classes.detach().float().softmax(-1)[..., :1]), -1).cpu().numpy()
        k = self.num_neighbors
        indices = np.full((b, n, k), -1, dtype=np.int64)
        neighbor_valid = np.zeros((b, n, k), dtype=bool)
        geometry = np.zeros((b, n, k, 5), dtype=np.float32)
        context_kept = np.zeros((b, n), dtype=bool)
        candidate_valid = np.zeros((b, n), dtype=bool)
        for bi, (raw, meta) in enumerate(zip(packed, img_metas)):
            height, width = meta["img_shape"][:2]
            if min(height, width) <= 0:
                raise ValueError("Relation scoring requires positive valid-image dimensions")
            if not n:
                continue
            finite = np.isfinite(raw).all(1)
            raw = np.where(np.isfinite(raw), raw, 0).astype(np.float64)
            # New strict checks use the same float32 clipping/scaling as box
            # postprocessing, then float64 areas (no overlap epsilon). Keep
            # archived policy arithmetic unchanged for old-model evaluation.
            dtype = np.float32 if self.strict_disjoint_neighbors else np.float64
            coordinates = raw[:, :4].astype(dtype)
            xyxy = np.concatenate((coordinates[:, :2] - coordinates[:, 2:4] / 2,
                                   coordinates[:, :2] + coordinates[:, 2:4] / 2), 1).clip(0, 1)
            xyxy *= np.array([width, height, width, height], dtype=dtype)
            xyxy = xyxy.astype(np.float64)
            size, center = xyxy[:, 2:] - xyxy[:, :2], (xyxy[:, 2:] + xyxy[:, :2]) / 2
            valid = finite & (size > 0).all(1)
            candidate_valid[bi] = valid
            lo, hi = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2]), np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
            intersection = np.maximum(hi - lo, 0).prod(-1)
            area = np.maximum(size, 0).prod(-1)
            iou = intersection / np.maximum(area[:, None] + area[None] - intersection, 1e-12)
            if self.strict_disjoint_neighbors:
                # No extra global NMS: each target greedily picks mutually
                # disjoint neighbors from the already foreground-ranked pool.
                context_kept[bi] = valid
            else:
                suppressed = np.zeros(n, dtype=bool)
                for index in np.argsort(-raw[:, 4], kind="stable"):
                    if valid[index] and not suppressed[index]:
                        context_kept[bi, index] = True
                        suppressed |= iou[index] >= self.duplicate_iou
            diagonal = math.hypot(width, height)
            displacement = (center[None] - center[:, None]) / diagonal  # neighbour minus centre
            distance = np.sqrt((displacement ** 2).sum(-1))
            overlap_allowed = intersection == 0 if self.strict_disjoint_neighbors else iou < self.duplicate_iou
            eligible = (context_kept[bi][None] & valid[:, None] & overlap_allowed & ~np.eye(n, dtype=bool))
            distances = np.where(eligible, distance, np.inf)
            ranked = np.argsort(distances, axis=1, kind="stable")
            if self.strict_disjoint_neighbors:
                order = np.zeros((n, min(k, n)), dtype=np.int64)
                kept = np.zeros(order.shape, dtype=bool)
                for target in range(n):
                    selected = []
                    for candidate in ranked[target]:
                        if not np.isfinite(distances[target, candidate]):
                            break
                        if selected and np.any(intersection[candidate, selected] > 0):
                            continue
                        selected.append(int(candidate))
                        if len(selected) == k:
                            break
                    order[target, :len(selected)] = selected
                    kept[target, :len(selected)] = True
            else:
                order = ranked[:, :min(k, n)]
                kept = np.isfinite(distances[np.arange(n)[:, None], order])
            length = order.shape[1]
            rows = np.arange(n)[:, None]
            indices[bi, :, :length] = np.where(kept, order, -1)
            neighbor_valid[bi, :, :length] = kept
            normalized_size = np.maximum(size, 0) / diagonal
            ratio = np.log((normalized_size[order] + self.geometry_eps)
                           / (normalized_size[:, None] + self.geometry_eps)).clip(-self.log_ratio_clip, self.log_ratio_clip)
            relation = np.concatenate((displacement[rows, order], distance[rows, order, None], ratio), -1)
            geometry[bi, :, :length] = np.where(kept[..., None], relation, 0)
        convert = lambda value: torch.from_numpy(value).to(boxes.device)
        return {"neighbor_indices": convert(indices), "neighbor_valid": convert(neighbor_valid),
                "geometry": convert(geometry), "context_kept": convert(context_kept),
                "candidate_valid": convert(candidate_valid)}

    @staticmethod
    def text_positions(length, dim, reference):
        position = torch.arange(length, device=reference.device, dtype=reference.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype)
                              * (-math.log(10000.) / dim))
        result = reference.new_zeros(length, dim)
        result[:, 0::2], result[:, 1::2] = (position * frequency).sin(), (position * frequency).cos()
        return result[None]

    def forward(self, queries, boxes, text, text_mask, classes, img_metas):
        if queries.ndim != 3 or boxes.shape != queries.shape[:2] + (4,) or text.shape[0] != queries.shape[0]:
            raise ValueError("Relation scoring requires aligned B,N,C features, B,N,4 boxes and B,L,C text")
        b, n, _ = queries.shape
        selection = self.select_neighbors(boxes, classes, img_metas)
        valid = selection["neighbor_valid"]
        fallback = ~valid.any(-1)
        effective = valid.clone()
        effective[..., 0] |= fallback  # one zero-context pair, not six copies
        if text.shape[1] == 0:
            text = queries.new_zeros(b, 1, queries.shape[-1])
            text_mask = torch.ones(b, 1, device=text.device, dtype=torch.bool)
        text_valid = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool) if text_mask is None else ~text_mask.bool()
        projected = self.visual(queries)
        language = self.text(torch.where(text_valid[..., None], text, torch.zeros_like(text)))
        language = language * text_valid[..., None].to(language.dtype)
        positions = self.text_positions(text.shape[1], self.model_dim, language)
        memories = [layer.prepare_memory(language, positions) for layer in self.layers]
        # Chunk along each image's N*K pairs, not across a flattened B*N*K.
        # A chunk contains B independent groups sharing their own text memory.
        pairs_per_image = n * self.num_neighbors
        pair_scores, pair_weights, pair_attention = [], [], []
        flat_indices = selection["neighbor_indices"].reshape(b, pairs_per_image)
        flat_valid = valid.reshape(b, pairs_per_image)
        flat_geometry = selection["geometry"].reshape(b, pairs_per_image, 5).to(projected)
        batch_indices = torch.arange(b, device=queries.device)[:, None]
        for start in range(0, pairs_per_image, self.chunk_size):
            end = min(pairs_per_image, start + self.chunk_size)
            flat = torch.arange(start, end, device=queries.device)
            centers = flat // self.num_neighbors
            target = projected[:, centers]
            neighbor = projected[batch_indices, flat_indices[:, start:end].clamp(min=0)]
            neighbor = torch.where(flat_valid[:, start:end, None], neighbor, torch.zeros_like(neighbor))
            geometry = self.geometry(flat_geometry[:, start:end])
            pair = self.pair(torch.cat((target, neighbor, neighbor - target, geometry), -1))
            for layer, memory in zip(self.layers, memories):
                pair, attention = layer(pair, memory, text_valid)
            pair_scores.append(self.score(pair).squeeze(-1))
            pair_weights.append(self.weight(pair).squeeze(-1))
            pair_attention.append(attention.detach())
        shape = (b, n, self.num_neighbors)
        if pairs_per_image:
            scores = torch.cat(pair_scores, dim=1).reshape(shape)
            weight_logits = torch.cat(pair_weights, dim=1).reshape(shape)
            attention = torch.cat(pair_attention, dim=1).reshape(*shape, text.shape[1])
        else:
            scores = projected.new_zeros(shape)
            weight_logits = projected.new_zeros(shape)
            attention = projected.new_zeros(*shape, text.shape[1])
        if self.aggregation == "adaptive":
            # Preserve the archived forward/gradient path exactly.
            weights = weight_logits.float().masked_fill(~effective, -torch.inf).softmax(-1).to(scores.dtype)
            logits = (weights * scores).sum(-1, keepdim=True)
        else:
            # Padding cannot win; an empty neighborhood has one effective
            # zero-context pair. Only the winning pair receives score gradients.
            logits, winner = scores.masked_fill(~effective, -torch.inf).max(-1, keepdim=True)
            weights = torch.zeros_like(scores).scatter_(-1, winner, 1)
            # Keep the adaptive weight head registered for old checkpoints and
            # optimizer layouts. It is unused here (training DDP uses
            # find_unused_parameters=True), rather than adding artificial loss.
        # Actual text attention aggregated with the same neighbour weights.
        text_attention = (weights.detach()[..., None] * attention).sum(2)
        details = {"relation_" + key: value.detach() for key, value in selection.items()}
        details.update(relation_fallback=fallback.detach(), relation_effective_mask=effective.detach(),
                       relation_pair_logits=scores.detach(), relation_weights=weights.detach())
        return logits.sigmoid(), logits, text_attention, details

    @torch.no_grad()
    def mark_optimizer_step(self):
        self.trained_steps.add_(1)

    def require_trained(self):
        if not self.trained_steps.item():
            raise RuntimeError("Relation scorer is untrained; train the new head or disable enable_relation_scoring")
