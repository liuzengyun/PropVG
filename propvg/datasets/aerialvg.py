"""AerialVG (ODVG jsonl) dataset adapter.

Each JSONL record becomes one referring-expression sample.  The first region
is the referred target; remaining regions are auxiliary image-level boxes.
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
                boxes, phrases = [], []
                for region in regions:
                    box = region.get("bbox")
                    if not box or len(box) != 4:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
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
