"""Strict diagnostic policy only; these tests do not change production KNN."""
if __package__:
    from .. import _bootstrap  # noqa: F401
else:
    import tools._bootstrap as _bootstrap  # noqa: F401

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from visualize_disjoint_neighbors import (
    intersection_area, select_disjoint_neighbors, render_comparison,
)


class DisjointNeighborVisualizationTests(unittest.TestCase):
    def boxes(self):
        return np.asarray([[40, 40, 60, 60], [59.999, 40, 70, 60],
                           [61, 42, 69, 58], [63, 42, 71, 58],
                           [40, 61, 60, 69], [10, 40, 20, 60],
                           [80, 40, 90, 60], [40, 10, 60, 20],
                           [40, 80, 60, 90], [35, 35, 65, 65]], dtype=np.float64)

    def test_rejects_even_tiny_overlap_and_mutual_overlap_nearest_first(self):
        boxes = self.boxes()
        chosen, _, pool = select_disjoint_neighbors(boxes, np.ones(len(boxes), dtype=bool), 0)
        self.assertEqual(chosen, [2, 4, 5, 6, 7, 8])
        self.assertEqual(pool, 7)
        self.assertNotIn(1, chosen)  # Tiny positive overlap is NOT rounded to zero.
        self.assertNotIn(3, chosen)  # Does not overlap target, but overlaps earlier neighbor 2.
        self.assertTrue((intersection_area(boxes[chosen], boxes[0]) == 0).all())
        for i, query in enumerate(chosen):
            self.assertTrue((intersection_area(boxes[chosen[i + 1:]], boxes[query]) == 0).all())

    def test_touching_edges_are_zero_iou_and_shortfalls_are_not_padded(self):
        boxes = np.asarray([[0, 0, 10, 10], [10, 0, 20, 10], [15, 0, 25, 10], [30, 30, 30, 30.]])
        chosen, _, _ = select_disjoint_neighbors(boxes, np.ones(4, dtype=bool), 0)
        self.assertEqual(chosen, [1])

    def test_uses_model_resized_coordinate_distance_not_padding_or_gt(self):
        boxes = np.asarray([[40, 40, 60, 60], [60, 45, 70, 55], [45, 60, 55, 70.]])
        valid = np.ones(3, dtype=bool)
        chosen, _, _ = select_disjoint_neighbors(boxes, valid, 0, count=1)
        self.assertEqual(chosen, [1])
        chosen, _, _ = select_disjoint_neighbors(boxes, valid, 0, count=1, metric_scale=[2, 1, 2, 1])
        self.assertEqual(chosen, [2])

    def test_foreground_filter_excludes_closest_background_and_preserves_low_confidence_top1(self):
        boxes = self.boxes()
        scores = np.full(len(boxes), .9)
        scores[0] = .01  # The original Top1 is NOT subject to neighbor-pool filtering.
        scores[2] = .2   # Nearest valid candidate is background-like.
        selected, _, _ = select_disjoint_neighbors(boxes, np.ones(len(boxes), dtype=bool), 0,
                                                    foreground_scores=scores, foreground_threshold=.5)
        self.assertNotIn(2, selected)
        self.assertIn(3, selected)  # No longer blocked by its low-confidence duplicate.
        self.assertTrue(all(scores[q] >= .5 for q in selected))
        self.assertNotIn(0, selected)

    def test_foreground_shortfall_never_fills_with_low_confidence_boxes(self):
        boxes = self.boxes()
        scores = np.zeros(len(boxes)); scores[2] = .5
        selected, _, _ = select_disjoint_neighbors(boxes, np.ones(len(boxes), dtype=bool), 0,
                                                    foreground_scores=scores, foreground_threshold=.5)
        self.assertEqual(selected, [2])  # Threshold is inclusive, no forced padding to six.
        scores[2] = float('nan')
        selected, _, _ = select_disjoint_neighbors(boxes, np.ones(len(boxes), dtype=bool), 0,
                                                    foreground_scores=scores, foreground_threshold=.5)
        self.assertEqual(selected, [])

    def test_foreground_filter_arguments_are_validated(self):
        boxes = self.boxes(); valid = np.ones(len(boxes), dtype=bool)
        for options in [dict(foreground_scores=np.ones(len(boxes))), dict(foreground_threshold=.5),
                        dict(foreground_scores=np.ones(2), foreground_threshold=.5),
                        dict(foreground_scores=np.ones(len(boxes)), foreground_threshold=1.1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_disjoint_neighbors(boxes, valid, 0, **options)

    def test_writes_two_individual_figures_and_horizontal_combination(self):
        with tempfile.TemporaryDirectory(prefix='propvg_disjoint_vis_') as temporary:
            root = Path(temporary)
            for folder in ('all_candidates', 'strict_neighbors', 'combined'):
                (root / folder).mkdir()
            Image.new('RGB', (100, 100), '#aaaaaa').save(root / 'original.jpg')
            boxes = self.boxes()
            record = dict(filename=str(root / 'original.jpg'), dataset_index=5, expression='A car beside another car.',
                          top1_query_slot=0, top1_correct=True, top1_iou=1., gt_box_xyxy=boxes[0].tolist(),
                          top1_box_xyxy=boxes[0].tolist(), all_boxes_xyxy=boxes.tolist(),
                          candidate_valid=[True] * len(boxes), top1_disjoint_pool=7,
                          new_neighbors=[dict(slot=1, query_slot=2, box_xyxy=boxes[2].tolist(),
                                              distance_resized_px=15., iou_with_top1=0., max_iou_other_neighbors=0.)])
            paths = render_comparison(record, root, 'example')
            left, right, combined = [Image.open(root / paths[k]) for k in ('all_candidates', 'strict_neighbors', 'combined')]
            self.assertEqual(left.size, right.size)
            self.assertEqual(combined.width, left.width + right.width + 12)
            self.assertEqual(combined.height, left.height)
            (root / 'foreground_candidates').mkdir()
            record.update(foreground_threshold=.5, all_foreground_scores=[.9] * len(boxes), top1_foreground_score=.9)
            record['new_neighbors'][0]['foreground_score'] = .9
            paths = render_comparison(record, root, 'foreground_example')
            self.assertIn('foreground_candidates/', paths['all_candidates'])
            self.assertTrue((root / paths['combined']).exists())


if __name__ == '__main__':
    unittest.main()
