"""GT-free, parameter-free proposal grouping and adaptive prefix selection."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def spherical_groups(queries, valid, num_groups=10, iterations=6):
    b, n, _ = queries.shape
    g = min(num_groups, n)
    q = F.normalize(torch.nan_to_num(queries.detach().float()), dim=-1)
    mean = F.normalize((q*valid[..., None]).sum(1)/valid.sum(1).clamp(min=1)[:, None], dim=-1)
    distance = (q-mean[:, None]).square().sum(-1).masked_fill(~valid, -1)
    indices = distance.argmax(-1)
    rows = torch.arange(b, device=q.device)
    centers, minimum = [], torch.full_like(distance, float('inf'))
    for _ in range(g):
        center = q[rows, indices]
        centers.append(center)
        minimum = torch.minimum(minimum, (q-center[:, None]).square().sum(-1))
        indices = minimum.masked_fill(~valid, -1).argmax(-1)
    centers = torch.stack(centers, 1)
    for _ in range(iterations):
        assignment = (q@centers.transpose(1, 2)).argmax(-1)
        weights = F.one_hot(assignment, g).to(q.dtype)*valid[..., None]
        counts = weights.sum(1)
        pooled = weights.transpose(1, 2)@q/counts.clamp(min=1)[..., None]
        centers = torch.where(counts[..., None] > 0, F.normalize(pooled, dim=-1), centers)
    return (q@centers.transpose(1, 2)).argmax(-1).masked_fill(~valid, -1)


@torch.no_grad()
def adaptive_prefix(queries, valid, probabilities, num_groups=10, iterations=6,
                    seed_topk=3, min_targets=3, max_targets=32):
    groups = spherical_groups(queries, valid, num_groups, iterations)
    selected = torch.zeros_like(valid)
    before, prefixes = [], []
    for bi in range(len(queries)):
        ids = valid[bi].nonzero(as_tuple=True)[0]
        # Match the legacy CPU probability sort, including its tie policy.
        order = ids[probabilities[bi, ids].detach().float().cpu().argsort(descending=True).to(ids.device)]
        window = order[:max_targets]
        seeds = groups[bi, window[:seed_topk]]
        matches = (groups[bi, window, None] == seeds[None]).any(-1)
        end = int(matches.nonzero(as_tuple=True)[0][-1])+1 if bool(matches.any()) else 0
        count = min(len(window), max(min_targets, end))
        prefix = window[:count]
        selected[bi, prefix] = True
        before.append(order); prefixes.append(prefix)
    return groups, selected, before, prefixes
