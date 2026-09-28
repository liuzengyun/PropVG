"""CPU-only regressions for scene splitting and teacher reindexing."""

import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_aerialsense import (SOURCES, balanced_assignment, build_units,
                              make_payload, scene_key, serialize_payload)


def entry(filename, source, scene, digest):
    expression = "the target in " + filename
    box = [1.0, 2.0, 6.0, 8.0]
    return {
        "record": {"filename": filename, "width": 10, "height": 10,
                   "grounding": {"caption": expression, "regions": [{"bbox": box, "phrase": expression}]}},
        "source": source, "scene_id": scene, "image_sha256": digest,
        "source_question": filename + ".json", "original_image_name": filename,
        "previous_split": "val", "previous_sample_id": 99,
        "teacher": {"sample_id": 99, "filename": filename, "caption": expression,
                    "width": 10, "height": 10, "predictions": [{"score": .8, "bbox_xyxy": box}]},
    }


class SplitTests(unittest.TestCase):
    def test_inferred_scene_rules(self):
        self.assertEqual(scene_key("CODrone", "day _park_30m_90c_frame_0.jpg"),
                         scene_key("CODrone", "day_park_30m_90c_frame_1800.jpg"))
        self.assertEqual(scene_key("VisDrone", "9999998_00350_d_0000302.jpg"),
                         scene_key("AerialVG", "2602_9999998_00360_d_0000303.jpg"))
        self.assertEqual(scene_key("University-1652", "0152_image-01.jpeg"),
                         scene_key("University-1652", "0152_image-36.jpeg"))
        self.assertNotEqual(scene_key("refGeo", "a-DJI_001.JPG"), scene_key("refGeo", "b-DJI_001.JPG"))
        with self.assertRaises(ValueError):
            scene_key("VisDrone", "unknown.jpg")

    def test_groups_include_scene_image_and_duplicate_aliases(self):
        entries = [entry("a.jpg", "VisDrone", "shared", "hash1"),
                   entry("b.jpg", "AerialVG", "shared", "hash2"),
                   entry("c.jpg", "CODrone", "other", "hash2"),
                   entry("d.jpg", "refGeo", "independent", "hash3")]
        duplicate_expression = copy.deepcopy(entries[0])
        duplicate_expression["record"]["grounding"]["caption"] += " alternative"
        entries.append(duplicate_expression)
        units = build_units(entries)
        self.assertEqual(sorted(len(u["indices"]) for u in units), [1, 4])

    def test_balancing_is_seeded_and_keeps_every_source(self):
        units = []
        for j, source in enumerate(SOURCES):
            for index in range(16):
                counts = [0] * len(SOURCES)
                counts[j] = 4
                units.append({"id": f"{source}/{index}", "counts": tuple(counts)})
        global_state = random.getstate()
        assigned = balanced_assignment(units, .75, 6666, trials=8)
        self.assertEqual(global_state, random.getstate())
        self.assertEqual(assigned, balanced_assignment(units, .75, 6666, trials=8))
        self.assertNotEqual(assigned, balanced_assignment(units, .75, 7777, trials=8))
        for j in range(len(SOURCES)):
            self.assertEqual(sum(u["counts"][j] for u, flag in zip(units, assigned) if flag), 48)

    def test_missing_independent_source_is_rejected(self):
        units = [{"id": "only", "counts": (4, 4, 4, 4, 4)}]
        with self.assertRaises(ValueError):
            balanced_assignment(units, .75, 6666)

    def test_teacher_reindexing_preserves_boxes_scores_and_refer_gt(self):
        entries = [entry(f"{source}-{index}.jpg", source, f"{source}/{index}", f"{source}:{index}")
                   for source in SOURCES for index in range(2)]
        original = copy.deepcopy(entries)
        units = build_units(entries)
        assigned = [unit["images"][0].endswith("-0.jpg") for unit in units]
        records, teachers, manifest = make_payload(entries, units, assigned, .75, 6666)
        self.assertEqual(entries, original)
        for split in ("train", "val"):
            self.assertEqual([t["sample_id"] for t in teachers[split]], list(range(5)))
            for record, teacher in zip(records[split], teachers[split]):
                source_entry = next(e for e in original if e["record"]["filename"] == record["filename"])
                self.assertEqual(record, source_entry["record"])
                self.assertEqual(teacher["predictions"], source_entry["teacher"]["predictions"])
        payload = serialize_payload(records, teachers)
        self.assertEqual(payload["annotation/vg_val_odvg.jsonl"], payload["annotation/vg_test_odvg.jsonl"])
        self.assertEqual(manifest["counts"]["train"]["annotations"], 5)
        self.assertEqual(manifest["previous_holdout_in_new_train"], 5)

    def test_illegally_split_scene_is_rejected(self):
        entries = [entry(f"{source}-{index}.jpg", source, f"{source}/{index}", f"{source}:{index}")
                   for source in SOURCES for index in range(2)]
        entries[1]["scene_id"] = entries[0]["scene_id"]
        units = [{"id": e["record"]["filename"], "indices": [i]} for i, e in enumerate(entries)]
        with self.assertRaises(AssertionError):
            make_payload(entries, units, [i % 2 == 0 for i in range(len(entries))], .75, 6666)


if __name__ == "__main__":
    unittest.main(verbosity=2)
