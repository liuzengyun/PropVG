"""Dataset-local pipeline components for AerialVG.

This module deliberately lives outside the original PropVG pipeline package.
AerialVG has boxes only; the rectangle mask below is an adapter artifact that
lets the unchanged segmentation-oriented PropVG interface consume the data.
"""
import os.path as osp

import mmcv
import numpy as np
from transformers import XLMRobertaTokenizer

from .builder import PIPELINES


def _clean_text(text):
    import re
    return re.sub(r"([.,'!?\"()*#:;])", "", text.lower()).replace("-", " ").replace("/", " ")


@PIPELINES.register_module()
class LoadAerialVGAnnotations:
    def __init__(self, max_token=40, color_type="color", backend=None,
                 file_client_cfg=dict(backend="disk"), use_auxiliary_boxes=False,
                 use_teacher_boxes=False):
        if use_auxiliary_boxes and use_teacher_boxes:
            raise ValueError("AerialVG auxiliary GT boxes and teacher boxes are mutually exclusive")
        self.max_token = max_token
        self.color_type = color_type
        self.backend = backend
        self.file_client_cfg = file_client_cfg.copy()
        self.use_auxiliary_boxes = use_auxiliary_boxes
        self.use_teacher_boxes = use_teacher_boxes
        self.file_client = None
        self.tokenizer = XLMRobertaTokenizer("pretrain_weights/beit3.spm")

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_cfg)
        ann = results["ann"]
        filename = osp.join(results["imgsfile"], ann["filename"])
        image = mmcv.imfrombytes(self.file_client.get(filename), flag=self.color_type, backend=self.backend)
        if image is None:
            raise FileNotFoundError(filename)
        # A small number of records have an empty caption; use the referred
        # region phrase as a valid fallback query for those records.
        expression = _clean_text(ann["expression"].strip() or ann.get("target_phrase", ""))
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(expression))[: self.max_token - 2]
        if not tokens:
            raise ValueError(f"Empty AerialVG phrase: {ann['expression']!r}")
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        pad = self.max_token - len(tokens)

        if self.use_teacher_boxes:
            if "teacher_boxes" not in ann:
                raise ValueError("Teacher mode is enabled but this sample has no teacher boxes")
            # The original regions[0] GT remains the sole referring target.
            # Teacher predictions are proposal-only auxiliary supervision.
            selected_boxes = ann["boxes"][:1] + ann["teacher_boxes"]
            refer_target_index = 0
        else:
            selected_boxes = ann["boxes"] if self.use_auxiliary_boxes else ann["boxes"][:1]
            refer_target_index = ann["refer_target_index"]
        results.update(
            filename=filename, img=image, img_shape=image.shape, ori_shape=image.shape,
            expression=expression,
            ref_expr_inds=np.asarray(tokens + [self.tokenizer.pad_token_id] * pad, dtype=np.int64),
            text_attention_mask=np.asarray([0] * len(tokens) + [1] * pad, dtype=np.int64),
            gt_bbox=[np.asarray(box, dtype=np.float64) for box in selected_boxes],
            with_bbox=True, with_mask=False, empty=True,
            refer_target_index=[refer_target_index],
        )
        return results
