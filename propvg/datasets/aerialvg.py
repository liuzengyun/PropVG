"""AerialVG (ODVG jsonl) dataset adapter.

Each JSONL record becomes one referring-expression sample.  Only the first
region is the referred target. Other regions are contextual annotation and
are deliberately not passed to the detection loss.
"""
import json
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines.compose import Compose
from propvg.utils import get_root_logger, is_main


@DATASETS.register_module()
class AerialVG(Dataset):
    def __init__(self, imgsfile, annsfile, pipeline, which_set="train", max_samples=None, **kwargs):
        if which_set not in ("train", "val", "test"):
            raise ValueError("AerialVG supports train, val, and test splits")
        self.which_set = which_set
        self.imgsfile = imgsfile
        self.pipeline = Compose(pipeline)
        self.samples = []
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
                    self.samples.append({
                        "filename": record["filename"],
                        "width": record["width"],
                        "height": record["height"],
                        "boxes": boxes,
                        "expression": caption,
                        "target_phrase": target_phrase,
                        "refer_target_index": 0,
                        "line_number": line_number,
                    })
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

    def _set_group_flag(self):
        self.flag = np.asarray([sample["width"] / sample["height"] > 1 for sample in self.samples], dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.pipeline({"ann": self.samples[index], "imgsfile": self.imgsfile})
        return item
