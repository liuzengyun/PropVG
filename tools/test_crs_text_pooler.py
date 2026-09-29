"""Regression checks for CRS sentence pooling with integer padding masks.

Run from the repository root: PYTHONPATH=. python tools/test_crs_text_pooler.py
"""
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")

import torch

from propvg.models.heads.REFHead import CRS


class CRSTextPoolerTests(unittest.TestCase):
    def test_integer_and_bool_masks_select_real_tokens(self):
        features = torch.tensor([[[0.], [1.], [2.], [100.], [200.]]])
        for dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
            with self.subTest(dtype=dtype):
                mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=dtype)
                pooled = CRS.text_pooler(None, features, mask)
                torch.testing.assert_close(pooled, torch.tensor([[2.]]))

    def test_batch_masks_and_channelwise_max(self):
        features = torch.tensor([
            [[-5., -2.], [-3., -4.], [100., 100.], [200., 200.]],
            [[1., 4.], [3., 2.], [2., 5.], [100., 100.]],
        ])
        mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])
        expected = torch.tensor([[-3., -2.], [3., 5.]])
        pooled = CRS.text_pooler(None, features, mask)
        torch.testing.assert_close(pooled, expected)
        changed = features.masked_fill(mask.bool()[..., None], -1e6)
        torch.testing.assert_close(CRS.text_pooler(None, changed, mask), expected)

    def test_gradient_reaches_valid_maxima_only(self):
        features = torch.tensor([[[1., 4.], [3., 2.], [100., 200.]]], requires_grad=True)
        pooled = CRS.text_pooler(None, features, torch.tensor([[0, 0, 1]]))
        pooled.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[[0., 1.], [1., 0.], [0., 0.]]]))

    def test_no_padding_uses_all_tokens(self):
        features = torch.tensor([[[9., 1.], [2., 8.], [3., 4.]]])
        mask = torch.zeros(1, 3, dtype=torch.int64)
        torch.testing.assert_close(CRS.text_pooler(None, features, mask), torch.tensor([[9., 8.]]))

    def test_crs_forward_and_backward_match_bool_mask(self):
        torch.manual_seed(701)
        crs = CRS(32).eval()
        query = torch.randn(2, 4, 32, requires_grad=True)
        text = torch.randn(2, 5, 32, requires_grad=True)
        mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1]])
        integer_scores, _, integer_logits = crs(query, text, mask, return_logits=True)
        integer_grads = torch.autograd.grad(integer_logits.square().mean(), (query, text))
        bool_scores, _, bool_logits = crs(query, text, mask.bool(), return_logits=True)
        bool_grads = torch.autograd.grad(bool_logits.square().mean(), (query, text))
        torch.testing.assert_close(integer_scores, bool_scores, rtol=0, atol=0)
        torch.testing.assert_close(integer_logits, bool_logits, rtol=0, atol=0)
        for actual, expected in zip(integer_grads, bool_grads):
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(actual.abs().sum().item(), 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
