_base_ = ["../_base_/misc.py"]
# Register the dataset-local components when this config is loaded.  This
# keeps the original train.py usable for both single- and multi-GPU runs.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
del AerialVG, LoadAerialVGAnnotations

work_dir = "/media/vision/lzy/PropVG/work_dir/aerialvg/0928new-crv"

# AerialVG provides one referred target (regions[0]); optional additional
# region boxes serve only as proposal/teacher-style supervision.
# Select the AerialVG-specific validation branch in the training runner.
dataset = "AerialVG"
dataset_type = "AerialVG"
# Maximum observed AerialVG caption length is 55 SentencePiece tokens;
# 64 leaves room for BEiT3 BOS/EOS tokens without truncation.
max_token = 64

# Fix every controllable Python/NumPy/PyTorch/DataLoader random source.
strict_reproducibility = False

# False: supervise only regions[0] (the referring target).
# True: also use regions[1:] as DETR proposal detection targets.
use_auxiliary_gt = False
# True: keep regions[0] as the referring GT and append retained predictions
# from the configured teacher JSON as proposal-only detection supervision.
# Validation and test never load teacher predictions.
teacher = True
# The teacher file must match the configured training annotation records by
# sample_id and filename. Validation and test do not read it.
teacher_annsfile = "data/AerialVG/teacher_trainset.json"
# Maximum number of extra teacher boxes per training sample after refer-GT
# deduplication and teacher NMS. 0 uses only the original refer GT; None keeps
# every available teacher box. The current JSON contains exactly 5 raw boxes
# per sample, so values above 5 cannot add supervision with this file.
teacher_max_boxes = 2
# Teacher predictions overlapping the exact refer GT at or above this IoU are
# discarded; the original refer GT remains the only target for that instance.
teacher_dedup_iou = 0.001
# Suppress lower-ranked teacher predictions which overlap another retained
# teacher prediction. This prevents duplicate pseudo targets for one object.
teacher_nms_iou = 0.5
if teacher and use_auxiliary_gt:
    raise ValueError("teacher and use_auxiliary_gt are mutually exclusive")
# False: fine-tune all BEiT-3 parameters with lr_vis_enc.
# True: freeze the complete BEiT-3 vision-language backbone.
freeze_beit3 = False

enable_grab=False

# Independent CRV: all new losses stop at its own scorer/fusion gate.
# False uses the unchanged CRS ranking and original full-candidate losses.
enable_crv = True
crv_topk = 8
crv_grid_size = 24  # Mask on the largest 144x144 map, then pool for scoring.
crv_cf_loss_weight = 0.1  # Small paired drop-ranking auxiliary objective.

img_size = 586
patch_size = 16
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token,
         use_auxiliary_boxes=use_auxiliary_gt, use_teacher_boxes=teacher),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "refer_target_index", "empty"]),
]
val_pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token,
         use_auxiliary_boxes=use_auxiliary_gt, use_teacher_boxes=False),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "refer_target_index", "empty"]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(type=dataset_type, which_set="train", annsfile="data/AerialVG/annotation/vg_train_odvg.jsonl",
               imgsfile="data/AerialVG/images", pipeline=train_pipeline,
               teacher_annsfile=teacher_annsfile if teacher else None,
               teacher_dedup_iou=teacher_dedup_iou,
               teacher_nms_iou=teacher_nms_iou,
               teacher_max_boxes=teacher_max_boxes),
    val=dict(type=dataset_type, which_set="val", annsfile="data/AerialVG/annotation/vg_val_odvg.jsonl",
             imgsfile="data/AerialVG/images", pipeline=val_pipeline),
    test=dict(type=dataset_type, which_set="test", annsfile="data/AerialVG/annotation/vg_test_odvg.jsonl",
              imgsfile="data/AerialVG/images", pipeline=val_pipeline),
)

model = dict(
    type="MIXRefUniModel_OMG",
    vis_enc=dict(type="BEIT3", img_size=img_size, patch_size=patch_size, vit_type="base", drop_path_rate=0.1,
                 vocab_size=64010, freeze_layer=-1, freeze_beit3=freeze_beit3,
                 vision_embed_proj_interpolate=False,
                 # Load the base BEiT-3 weights only.  This is not a PropVG
                 # task checkpoint; the AerialVG detection/referring head is
                 # still initialized and trained from scratch.
                 pretrain="pretrain_weights/beit3_base_patch16_224.zip"),
    lan_enc=None,
    fusion=None,
    head=dict(type="REFHead", input_channels=768, hidden_channels=256, num_queries=256,
              detr_loss={"criterion": {"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
                         "matcher": {"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0}},
              loss_weight={"mask": {"dice": 1.0, "bce": 1.0, "nt": 0.2, "neg": 0},
                           "bbox": 0.1, "allbbox": 0.1, "refer": 1.0,
                           "crv": 0.5}, MTD={"K": 250},
              enable_grab=enable_grab, enable_crv=enable_crv,
              crv=dict(topk=crv_topk, verifier_dim=128, grid_size=crv_grid_size,
                       num_heads=4, num_layers=2,
                       core_expansion=1.5, feather_expansion=2.0,
                       initial_alpha=0.1, max_residual=2.0, drop_temperature=1.0,
                       cf_loss_weight=crv_cf_loss_weight, rank_margin=0.5,
                       max_negative_gt_coverage=0.05)),
    post_params={"score_weighted": False, "mask_threshold": 0.5, "score_threshold": 0.7,
                 "with_nms": False, "with_mask": False},
    process_visual=False,
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
lr = 0.0004
optimizer_config = dict(type="Adam", lr=lr, lr_vis_enc=0.00004, lr_lan_enc=lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=2, decay_steps=[24, 32], decay_ratio=0.1, max_epoch=36)
log_interval = 50
start_save_checkpoint = 0
# finetune_from = "PropVG-grefcoco.pth"
