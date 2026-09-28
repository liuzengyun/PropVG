"""AerialVG (ODVG jsonl) dataset adapter."""
import json
import os.path as osp
from numbers import Integral

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines.compose import Compose
from propvg.utils import get_root_logger, is_main


@DATASETS.register_module()
class AerialVG(Dataset):
    def __init__(self, imgsfile, annsfile, pipeline, which_set="train", max_samples=None,
                 teacher_annsfile=None, teacher_dedup_iou=0.5,
                 teacher_nms_iou=0.5, teacher_max_boxes=None,
                 teacher_max10=False, **kwargs):
        if which_set not in ("train", "val", "test"):
            raise ValueError("AerialVG supports train, val, and test splits")
        if teacher_annsfile is not None and which_set != "train":
            raise ValueError("Teacher boxes may only be loaded for the AerialVG training split")
        if teacher_max_boxes is not None and (
            isinstance(teacher_max_boxes, bool)
            or not isinstance(teacher_max_boxes, Integral)
            or teacher_max_boxes < 0
        ):
            raise ValueError("teacher_max_boxes must be a non-negative integer or None")
        if not isinstance(teacher_max10, bool):
            raise ValueError("teacher_max10 must be True or False")
        self.teacher_max10 = teacher_max10
        self.teacher_area_filtered_boxes = 0
        self.teacher_area_filtered_samples = 0
        self.which_set = which_set
        self.imgsfile = imgsfile
        self.pipeline = Compose(pipeline)
        self.samples = []

        teacher_records = None
        if teacher_annsfile is not None:
            with open(teacher_annsfile, "r", encoding="utf-8") as handle:
                teacher_data = json.load(handle)
            teacher_records = {}
            for teacher_record in teacher_data:
                sample_id = int(teacher_record["sample_id"])
                if sample_id in teacher_records:
                    raise ValueError(f"Duplicate teacher sample_id: {sample_id}")
                teacher_records[sample_id] = teacher_record

        with open(annsfile, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                regions = record.get("grounding", {}).get("regions", [])
                # regions[0] is always the referring target. Keep valid
                # context boxes in the record so the pipeline can optionally
                # use them for DETR proposal supervision.
                if regions:
                    box = regions[0].get("bbox")
                else:
                    box = None
                boxes = []
                phrases = []
                if box and len(box) == 4:
                    x1, y1, x2, y2 = map(float, box)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        phrases.append(regions[0].get("phrase", ""))
                        for region in regions[1:]:
                            aux = region.get("bbox")
                            if not aux or len(aux) != 4:
                                continue
                            ax1, ay1, ax2, ay2 = map(float, aux)
                            if ax2 > ax1 and ay2 > ay1:
                                boxes.append([ax1, ay1, ax2, ay2])
                                phrases.append(region.get("phrase", ""))
                if boxes:
                    caption = record.get("grounding", {}).get("caption", "")
                    target_phrase = phrases[0] if phrases else ""
                    if not caption.strip() and not target_phrase.strip():
                        continue
                    sample = {
                        "filename": record["filename"],
                        "width": record["width"],
                        "height": record["height"],
                        "boxes": boxes,
                        "expression": caption,
                        "target_phrase": target_phrase,
                        "refer_target_index": 0,
                        "line_number": line_number,
                    }
                    if teacher_records is not None:
                        sample_id = line_number - 1
                        if sample_id not in teacher_records:
                            raise ValueError(f"Missing teacher prediction for sample_id={sample_id}")
                        teacher_record = teacher_records[sample_id]
                        if teacher_record.get("filename") != record["filename"]:
                            raise ValueError(
                                f"Teacher/sample mismatch at sample_id={sample_id}: "
                                f"{teacher_record.get('filename')!r} != {record['filename']!r}"
                            )

                        teacher_boxes = []
                        teacher_scores = []
                        predictions = sorted(teacher_record.get("predictions", []), key=lambda pred: pred["rank"])
                        for prediction in predictions:
                            teacher_box = prediction.get("bbox_xyxy")
                            if teacher_box is None or len(teacher_box) != 4:
                                continue
                            tx1, ty1, tx2, ty2 = map(float, teacher_box)
                            tx1 = min(max(tx1, 0.0), float(record["width"]))
                            tx2 = min(max(tx2, 0.0), float(record["width"]))
                            ty1 = min(max(ty1, 0.0), float(record["height"]))
                            ty2 = min(max(ty2, 0.0), float(record["height"]))
                            if tx2 > tx1 and ty2 > ty1:
                                teacher_boxes.append([tx1, ty1, tx2, ty2])
                                teacher_scores.append(float(prediction.get("score", 0.0)))
                        if not teacher_boxes:
                            raise ValueError(f"No valid teacher boxes for sample_id={sample_id}")

                        # The exact annotated referent is always the only
                        # target supervision for that object. Remove teacher
                        # predictions which overlap it, avoiding duplicate or
                        # conflicting DETR targets for the same instance.
                        teacher_array = np.asarray(teacher_boxes, dtype=np.float64)
                        refer_box = np.asarray(boxes[0], dtype=np.float64)
                        lt = np.maximum(teacher_array[:, :2], refer_box[:2])
                        rb = np.minimum(teacher_array[:, 2:], refer_box[2:])
                        intersection = np.clip(rb - lt, 0.0, None).prod(axis=1)
                        teacher_area = np.clip(teacher_array[:, 2:] - teacher_array[:, :2], 0.0, None).prod(axis=1)
                        refer_area = np.clip(refer_box[2:] - refer_box[:2], 0.0, None).prod()
                        ious = intersection / np.maximum(teacher_area + refer_area - intersection, 1e-6)
                        keep = ious < float(teacher_dedup_iou)
                        if self.teacher_max10:
                            # Compare the actual original-pixel supervision
                            # boxes after clipping teacher boxes to the image.
                            # Use regions[0], never an auxiliary/teacher GT.
                            # Exactly 10x is retained; only strictly larger
                            # boxes are removed, before NMS and count limits.
                            area_keep = teacher_area <= 10.0 * refer_area
                            removed = int((~area_keep).sum())
                            self.teacher_area_filtered_boxes += removed
                            self.teacher_area_filtered_samples += int(removed > 0)
                            keep &= area_keep
                        teacher_boxes = [box for box, keep_box in zip(teacher_boxes, keep) if keep_box]
                        teacher_scores = [score for score, keep_box in zip(teacher_scores, keep) if keep_box]

                        # Predictions are already rank ordered. Greedily keep
                        # the highest-ranked box and suppress later teacher
                        # boxes describing the same object. Without this,
                        # Hungarian matching treats duplicate pseudo boxes as
                        # distinct objects and reduces proposal diversity.
                        nms_boxes = []
                        nms_scores = []
                        for teacher_box, teacher_score in zip(teacher_boxes, teacher_scores):
                            if nms_boxes:
                                kept = np.asarray(nms_boxes, dtype=np.float64)
                                candidate = np.asarray(teacher_box, dtype=np.float64)
                                lt = np.maximum(kept[:, :2], candidate[:2])
                                rb = np.minimum(kept[:, 2:], candidate[2:])
                                intersection = np.clip(rb - lt, 0.0, None).prod(axis=1)
                                kept_area = np.clip(kept[:, 2:] - kept[:, :2], 0.0, None).prod(axis=1)
                                candidate_area = np.clip(candidate[2:] - candidate[:2], 0.0, None).prod()
                                ious = intersection / np.maximum(
                                    kept_area + candidate_area - intersection, 1e-6
                                )
                                if np.any(ious >= float(teacher_nms_iou)):
                                    continue
                            nms_boxes.append(teacher_box)
                            nms_scores.append(teacher_score)
                        # Limit proposal-only teacher supervision after
                        # deduplication, so lower-ranked boxes can fill slots
                        # vacated by predictions overlapping the refer GT.
                        teacher_boxes = nms_boxes[:teacher_max_boxes]
                        teacher_scores = nms_scores[:teacher_max_boxes]

                        sample.update(
                            teacher_boxes=teacher_boxes,
                            teacher_scores=teacher_scores,
                        )
                    self.samples.append(sample)

        if teacher_records is not None and len(teacher_records) != line_number:
            raise ValueError(
                f"Teacher annotation count ({len(teacher_records)}) does not match "
                f"training annotation count ({line_number})"
            )
        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]
        if not self.samples:
            raise ValueError(f"No valid AerialVG regions found in {annsfile}")
        if which_set == "train":
            self._set_group_flag()
        # BEiT3 uses its own fixed SentencePiece vocabulary.
        self.word_emb = None
        self.num_token = 64010
        if is_main():
            get_root_logger().info("AerialVG-%s size: %d", which_set, len(self.samples))
            if teacher_records is not None and self.teacher_max10:
                get_root_logger().info(
                    "Teacher area filter (teacher_max10=True): removed %d boxes "
                    "larger than 10x refer GT across %d samples before NMS/count limits",
                    self.teacher_area_filtered_boxes, self.teacher_area_filtered_samples,
                )

    def _set_group_flag(self):
        self.flag = np.asarray([sample["width"] / sample["height"] > 1 for sample in self.samples], dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.pipeline({"ann": self.samples[index], "imgsfile": self.imgsfile})
        return item
