"""CPU checks for diagnostic coordinates, query identity and report rendering."""
if __package__:
    from .. import _bootstrap  # noqa: F401
else:
    import tools._bootstrap as _bootstrap  # noqa: F401

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from visualize_relation_scoring import make_record, pixel_boxes, render_record, write_gallery


class RelationVisualizationTests(unittest.TestCase):
    def example(self):
        # Slot 0 is degenerate after clipping; the evaluator removes it.
        boxes = torch.tensor([[[1.2, .5, .1, .1], [.5, .5, .2, .2], [.2, .2, .1, .1]]])
        meta = dict(img_shape=(100, 200), pad_shape=(200, 200), ori_shape=(200, 400, 3),
                    scale_factor=[.5, .5, .5, .5], filename="example.jpg", expression="A car beside a truck.")
        logits = torch.tensor([[[10.], [.4], [.4]]])  # Tie is stable after clipping.
        weights = torch.zeros(1, 3, 6); weights[..., 0] = .25; weights[..., 1] = .75
        pair_logits = torch.zeros(1, 3, 6); pair_logits[..., 0] = 1; pair_logits[..., 1] = .2
        valid = torch.zeros(1, 3, 6, dtype=torch.bool); valid[..., :2] = True
        indices = torch.full((1, 3, 6), -1, dtype=torch.long); indices[..., 0] = 2; indices[..., 1] = 1
        details = dict(relation_pair_logits=pair_logits, relation_weights=weights,
                       relation_neighbor_indices=indices, relation_neighbor_valid=valid,
                       relation_effective_mask=valid.clone(), relation_fallback=torch.zeros(1, 3, dtype=torch.bool),
                       relation_geometry=torch.zeros(1, 3, 6, 5))
        captured = dict(boxes=boxes, logits=logits, probability=logits.sigmoid(), details=details)
        original, kept = pixel_boxes(boxes[0], meta)
        prediction = dict(boxes=original[kept], ranking_logits=logits[0, kept, 0], scores=logits[0, kept, 0].sigmoid())
        sample = dict(boxes=[original[1].tolist()], refer_target_index=0, line_number=4)
        return sample, meta, prediction, captured

    def test_valid_image_clipping_then_original_pixel_rescale(self):
        _, meta, _, captured = self.example()
        result, valid = pixel_boxes(captured['boxes'][0], meta)
        torch.testing.assert_close(result[1], torch.tensor([160., 80., 240., 120.]))
        self.assertEqual(valid.tolist(), [False, True, True])
        meta['pad_shape'] = (1000, 1000)
        a, b = pixel_boxes(captured['boxes'][0], meta)
        torch.testing.assert_close(a, result)
        torch.testing.assert_close(b, valid)

    def test_real_top_query_is_not_renumbered_after_invalid_box_filter(self):
        sample, meta, prediction, captured = self.example()
        row = make_record(3, sample, meta, prediction, captured)
        self.assertEqual(row['top1_query_slot'], 1)
        self.assertTrue(row['top1_correct'])
        self.assertEqual(row['neighbors'][0]['query_slot'], 2)
        self.assertIsNone(row['neighbors'][2]['query_slot'])
        self.assertEqual(row['neighbors'][2]['weight'], 0)
        self.assertAlmostEqual(sum(n['weighted_logit'] for n in row['neighbors']), row['top1_logit'], places=6)

    def test_refuses_misaligned_prediction_boxes(self):
        sample, meta, prediction, captured = self.example()
        prediction['boxes'] = prediction['boxes'].clone() + 10
        with self.assertRaises(AssertionError):
            make_record(3, sample, meta, prediction, captured)

    def test_zero_context_fallback_is_not_fabricated_as_real_neighbor(self):
        sample, meta, prediction, captured = self.example()
        details = captured['details']
        details['relation_neighbor_indices'].fill_(-1)
        details['relation_neighbor_valid'].fill_(False)
        details['relation_effective_mask'].fill_(False); details['relation_effective_mask'][..., 0] = True
        details['relation_weights'].zero_(); details['relation_weights'][..., 0] = 1
        details['relation_pair_logits'][..., 0] = .4
        details['relation_fallback'].fill_(True)
        row = make_record(3, sample, meta, prediction, captured)
        self.assertTrue(row['fallback'])
        self.assertFalse(row['neighbors'][0]['valid'])
        self.assertTrue(row['neighbors'][0]['used'])
        self.assertIsNone(row['neighbors'][0]['box_xyxy'])

    def test_gallery_and_six_panel_figure_render_without_html_format_errors(self):
        with tempfile.TemporaryDirectory(prefix='propvg_relation_vis_') as temporary:
            root = Path(temporary)
            Image.new('RGB', (400, 200), '#cccccc').save(root / 'source.jpg')
            sample, meta, prediction, captured = self.example()
            meta['filename'] = str(root / 'source.jpg')
            row = make_record(3, sample, meta, prediction, captured)
            row['visualization'] = 'result.jpg'
            render_record(row, root / 'result.jpg')
            image = Image.open(root / 'result.jpg')
            self.assertEqual(image.width, 1500)
            self.assertGreater(image.height, 1400)
            write_gallery(root, [row], dict(samples=1))
            page = (root / 'index.html').read_text()
            self.assertIn('result.jpg', page)
            self.assertIn('A car beside a truck.', page)
            self.assertNotIn('__SUMMARY__', page)


if __name__ == '__main__':
    unittest.main()
