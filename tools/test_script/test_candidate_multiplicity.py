"""CPU checks for diagnostic correctness, not model/training changes."""
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from analyze_candidate_multiplicity import ContextIntervention, group_candidates, make_interventions, score_order


class MultiplicityTests(unittest.TestCase):
    def test_log_bias_exactly_matches_repeated_keys_and_values(self):
        torch.manual_seed(7)
        attention = torch.nn.MultiheadAttention(16, 4, dropout=0, batch_first=True).double().eval()
        query, key, value = [torch.randn(1, 5, 16, dtype=torch.double) for _ in range(3)]
        repetitions = torch.tensor([1, 2, 8, 1, 4])
        with torch.no_grad():
            repeated = attention(query, key.repeat_interleave(repetitions, 1),
                                 value.repeat_interleave(repetitions, 1))[0]
            biased = attention(query, key, value,
                attn_mask=repetitions.double().log()[None].expand(5, -1))[0]
        torch.testing.assert_close(repeated, biased, atol=1e-12, rtol=1e-12)

    def test_uniform_repeat_does_not_change_attention(self):
        torch.manual_seed(8)
        attention = torch.nn.MultiheadAttention(16, 4, dropout=0, batch_first=True).double().eval()
        query = torch.randn(1, 5, 16, dtype=torch.double)
        with torch.no_grad():
            baseline = attention(query, query, query)[0]
            # Distinct key VIEW bypasses torch 2.0.0's broken float-mask fast path.
            repeated = attention(query, query.view_as(query), query,
                attn_mask=torch.full((5, 5), math.log(8), dtype=torch.double))[0]
        torch.testing.assert_close(baseline, repeated, atol=1e-12, rtol=1e-12)

    def test_representative_grouping_not_transitive(self):
        boxes = np.asarray([[0, 0, 10, 10], [1, 0, 11, 10], [2, 0, 12, 10]], dtype=float)
        groups, sizes, reps = group_candidates(boxes, np.asarray([.9, .8, .7]), np.ones(3, bool), .8)
        np.testing.assert_array_equal(groups, [0, 0, 1])
        np.testing.assert_array_equal(sizes, [2, 2, 1])
        np.testing.assert_array_equal(reps, [0, 2])

    def test_invalid_boxes_are_singleton_and_never_ranked(self):
        boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]], dtype=float)
        valid = np.asarray([True, True, False])
        groups, sizes, _ = group_candidates(boxes, np.asarray([.8, .9, 1.]), valid, .8)
        self.assertNotEqual(groups[1], groups[2])
        np.testing.assert_array_equal(sizes, [2, 2, 1])
        np.testing.assert_array_equal(score_order(np.asarray([.8, .9, 1.]), valid), [1, 0])

    def test_interventions_have_no_gt_input_and_preserve_output_slots(self):
        boxes = np.asarray([[0, 0, 10, 10], [.1, 0, 10.1, 10], [20, 20, 30, 30]], dtype=float)
        biases, groups, sizes, top1, rival = make_interventions(boxes, np.asarray([.9, .8, .7]),
            np.ones(3, bool), np.asarray([.9, .8, .7]))
        self.assertEqual(top1, 0)
        self.assertEqual(rival, 2)
        self.assertTrue(all(bias.shape == (3,) for bias in biases.values()))
        np.testing.assert_allclose(biases["balance_iou0.8"], -np.log([2, 2, 1]), rtol=1e-6)
        self.assertTrue(np.isneginf(biases["dedup_context_iou0.8"][1]))

    def test_runtime_patch_matches_repetition_and_restores_original(self):
        torch.manual_seed(9)
        attention = torch.nn.MultiheadAttention(16, 4, dropout=0, batch_first=True).double().eval()
        layer = SimpleNamespace(attentions=[SimpleNamespace(attn=attention)])
        crs = SimpleNamespace(word_attn=SimpleNamespace(layers=[layer]))
        query = torch.randn(1, 5, 16, dtype=torch.double)
        repetitions = torch.tensor([1, 2, 8, 1, 4])
        original = attention.forward
        with torch.no_grad():
            expected = original(query, query.repeat_interleave(repetitions, 1),
                                query.repeat_interleave(repetitions, 1))[0]
            with ContextIntervention(crs).installed() as intervention:
                intervention.bias = repetitions.double().log()[None]
                actual = attention(query, query, query)[0]
                self.assertEqual(intervention.attention[0].shape, (1, 5))
        self.assertEqual(attention.forward, original)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_batched_interventions_do_not_mix_heads_or_samples(self):
        torch.manual_seed(10)
        attention = torch.nn.MultiheadAttention(16, 4, dropout=0, batch_first=True).double().eval()
        layer = SimpleNamespace(attentions=[SimpleNamespace(attn=attention)])
        crs = SimpleNamespace(word_attn=SimpleNamespace(layers=[layer]))
        query = torch.randn(3, 5, 16, dtype=torch.double)
        counts = torch.tensor([[1, 1, 1, 1, 1], [1, 8, 1, 1, 1], [2, 1, 4, 1, 8]])
        with torch.no_grad():
            expected = torch.cat([attention(query[b:b + 1],
                query[b:b + 1].repeat_interleave(counts[b], 1),
                query[b:b + 1].repeat_interleave(counts[b], 1))[0] for b in range(3)])
            with ContextIntervention(crs).installed() as intervention:
                intervention.bias = counts.double().log()
                actual = attention(query, query, query)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
