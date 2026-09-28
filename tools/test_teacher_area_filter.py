"""CPU-only checks for the optional 10x refer-GT teacher area filter."""

import json
import tempfile
import unittest
from pathlib import Path

from propvg.datasets.aerialvg import AerialVG


class TeacherAreaFilterTests(unittest.TestCase):
    def make_dataset(self, boxes, enabled=None, cap=None, split="train", with_teacher=True, regions=None):
        temporary = tempfile.TemporaryDirectory(prefix="propvg_teacher_area_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        gt = [0, 0, 10, 10]
        record = {"filename": "image.jpg", "width": 100, "height": 100,
                  "grounding": {"caption": "the target", "regions": regions or [{"bbox": gt, "phrase": "the target"}]}}
        predictions = [{"rank": i + 1, "score": .9 - .1 * i, "bbox_xyxy": box}
                       for i, box in enumerate(boxes)]
        teacher = [{"sample_id": 0, "filename": "image.jpg", "predictions": predictions}]
        annotation = root / "annotation.jsonl"
        teacher_path = root / "teacher.json"
        annotation.write_text(json.dumps(record) + "\n", encoding="utf-8")
        teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
        kwargs = {} if enabled is None else {"teacher_max10": enabled}
        dataset = AerialVG(str(root), str(annotation), [], which_set=split,
                          teacher_annsfile=str(teacher_path) if with_teacher else None,
                          teacher_max_boxes=cap, **kwargs)
        self.assertEqual(dataset.samples[0]["boxes"][0], record["grounding"]["regions"][0]["bbox"])
        self.assertEqual(dataset.samples[0]["refer_target_index"], 0)
        self.assertEqual(json.loads(teacher_path.read_text()), teacher)
        return dataset

    def test_boundary_strictly_above_ten_and_fill_count_slot(self):
        large, exact, small = [20, 0, 75, 20], [20, 0, 70, 20], [80, 0, 90, 10]
        dataset = self.make_dataset([large, exact, small], True, cap=1)
        self.assertEqual(dataset.samples[0]["teacher_boxes"], [exact])
        self.assertAlmostEqual(dataset.samples[0]["teacher_scores"][0], .8)
        self.assertEqual(dataset.teacher_area_filtered_boxes, 1)
        self.assertEqual(dataset.teacher_area_filtered_samples, 1)

    def test_false_and_unspecified_keep_previous_path(self):
        large, small = [20, 0, 75, 20], [80, 0, 90, 10]
        disabled = self.make_dataset([large, small], False, cap=1)
        default = self.make_dataset([large, small], cap=1)
        self.assertEqual(disabled.samples, default.samples)
        self.assertEqual(default.samples[0]["teacher_boxes"], [large])

    def test_all_removed_keeps_sample_and_original_gt(self):
        dataset = self.make_dataset([[20, 0, 75, 20]], True)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0]["teacher_boxes"], [])
        self.assertEqual(dataset.samples[0]["teacher_scores"], [])

    def test_teacher_area_is_measured_after_image_clipping(self):
        # Raw area is 1800 but the supervised clipped area is exactly 1000.
        dataset = self.make_dataset([[50, 0, 140, 20]], True)
        self.assertEqual(dataset.samples[0]["teacher_boxes"], [[50, 0, 100, 20]])
        self.assertEqual(dataset.teacher_area_filtered_boxes, 0)

    def test_reference_is_original_target_not_auxiliary_gt(self):
        regions = [{"bbox": [0, 0, 2, 2], "phrase": "the target"},
                   {"bbox": [0, 0, 80, 80], "phrase": "auxiliary"}]
        dataset = self.make_dataset([[20, 0, 27, 6]], True, regions=regions)
        self.assertEqual(dataset.samples[0]["teacher_boxes"], [])

    def test_without_teacher_and_eval_splits_are_unchanged(self):
        for split in ("train", "val", "test"):
            dataset = self.make_dataset([[20, 0, 75, 20]], True, split=split, with_teacher=False)
            self.assertNotIn("teacher_boxes", dataset.samples[0])
            self.assertEqual(dataset.teacher_area_filtered_boxes, 0)

    def test_zero_cap_still_uses_only_original_gt(self):
        dataset = self.make_dataset([[20, 0, 30, 10]], True, cap=0)
        self.assertEqual(dataset.samples[0]["teacher_boxes"], [])

    def test_switch_rejects_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "teacher_max10"):
            self.make_dataset([[20, 0, 30, 10]], enabled=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
