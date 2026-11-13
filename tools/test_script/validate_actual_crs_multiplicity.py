"""Extra check: actual trained CRS inputs, bias vs explicit repeated K/V.

Reuses the read-only diagnostic entrypoint. Run its CLI with --samples 1 or 2;
the additional check is runtime-local and does not touch model source/weights.
"""
import json
from pathlib import Path
import sys

import torch

import analyze_candidate_multiplicity as diagnostic


def main():
    native_forward = torch.nn.MultiheadAttention.forward
    checks = []

    def checked_forward(module, query, key, value, **kwargs):
        output, weights = native_forward(module, query, key, value, **kwargs)
        mask = kwargs.get("attn_mask")
        if (mask is not None and mask.ndim == 3 and mask.dtype.is_floating_point
                and query.shape[1] == 256 and key.shape[1] == 256):
            for batch in range(query.shape[0]):
                factors = mask[batch * module.num_heads, 0].exp()
                counts = factors.round().to(torch.long)
                if not torch.allclose(factors, counts.to(factors.dtype), atol=1e-5, rtol=1e-5):
                    continue
                explicit, _ = native_forward(module, query[batch:batch + 1],
                    key[batch:batch + 1].repeat_interleave(counts, 1),
                    value[batch:batch + 1].repeat_interleave(counts, 1))
                error = float((output[batch:batch + 1] - explicit).abs().max().cpu())
                torch.testing.assert_close(output[batch:batch + 1], explicit, atol=1e-5, rtol=1e-4)
                checks.append(dict(max_abs_error=error, query_count=query.shape[1],
                    explicit_context_count=int(counts.sum()), largest_multiplicity=int(counts.max())))
        return output, weights

    torch.nn.MultiheadAttention.forward = checked_forward
    try:
        diagnostic.main()
    finally:
        torch.nn.MultiheadAttention.forward = native_forward
    if not checks:
        raise RuntimeError("No actual CRS context-replication checks executed")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = dict(checks=len(checks), max_abs_error=max(row["max_abs_error"] for row in checks),
                  passed=True, cases=checks)
    (output / "explicit_repeat_equivalence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
