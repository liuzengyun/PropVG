_base_ = ["../_base_/misc.py"]
# Register the dataset-local components when this config is loaded.  This
# keeps the original train.py usable for both single- and multi-GPU runs.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
del AerialVG, LoadAerialVGAnnotations

work_dir = "/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-crv-val"

# AerialVG provides one referred target (regions[0]); optional additional
# region boxes serve only as proposal/teacher-style supervision.
# Select the AerialVG-specific validation branch in the training runner.
dataset = "AerialVG"
dataset_type = "AerialVG"
# Maximum observed AerialVG caption length is 55 SentencePiece tokens;
# 64 leaves room for BEiT3 BOS/EOS tokens without truncation.
max_token = 64

# Fix every controllable Python/NumPy/PyTorch/DataLoader random source.
strict_reproducibility = True

# False: supervise only regions[0] (the referring target).
# True: also use regions[1:] as DETR proposal detection targets.
use_auxiliary_gt = False
# True: keep regions[0] as the referring GT and append retained predictions
# from the configured teacher JSON as proposal-only detection supervision.
# Validation and test never load teacher predictions.
teacher = True
# The teacher file must match the configured training annotation records by
# sample_id and filename. Validation and test do not read it.
teacher_annsfile = "data/AerialVG/teacher_valset.json"
# Maximum number of extra teacher boxes per training sample after refer-GT
# deduplication and teacher NMS. 0 uses only the original refer GT; None keeps
# every available teacher box. The current JSON contains exactly 5 raw boxes
# per sample, so values above 5 cannot add supervision with this file.
teacher_max_boxes = 5
# Teacher predictions overlapping the exact refer GT at or above this IoU are
# discarded; the original refer GT remains the only target for that instance.
teacher_dedup_iou = 0.5
# Suppress lower-ranked teacher predictions which overlap another retained
# teacher prediction. This prevents duplicate pseudo targets for one object.
teacher_nms_iou = 0.5
if teacher and use_auxiliary_gt:
    raise ValueError("teacher and use_auxiliary_gt are mutually exclusive")
# False: fine-tune all BEiT-3 parameters with lr_vis_enc.
# True: freeze the complete BEiT-3 vision-language backbone.
freeze_beit3 = False

enable_grab=False
# Do not allocate segmentation-only modules; shared 512-channel FPN is unchanged.
enable_mask = False
# Independent post-CRS refinement. False preserves the previous experiment.
enable_crv = True
crv_topk = 5  # Reorder only these candidates; Top-5 membership is preserved.
crv_grid_size = 48  # Pool complete original/intervened feature maps; no local crops.
crv_uncertainty_threshold = 0.5  # Only grouped entropy above this enables refinement.
crv_start_epoch = 1  # 1-based: no CRV training or reranking before this epoch; 1 starts immediately.
# Candidate background enhancement before CRS; disable for the PropVG baseline.
enable_bcem = False
# New experiments sample four points per stage directly on each ray.
bcem_sampling_mode = "ray"
bcem_num_rays = 8
# Used only by sector mode; 8 reproduces the original direction order.
bcem_num_sectors = 4

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
    train=dict(type=dataset_type, which_set="train", annsfile="data/AerialVG/annotation/vg_val_odvg.jsonl",
               imgsfile="data/AerialVG/images", pipeline=train_pipeline,
               teacher_annsfile=teacher_annsfile if teacher else None,
               teacher_dedup_iou=teacher_dedup_iou,
               teacher_nms_iou=teacher_nms_iou,
               teacher_max_boxes=teacher_max_boxes),
    val=dict(type=dataset_type, which_set="val", annsfile="data/AerialVG/annotation/vg_test_odvg.jsonl",
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
                           "bbox": 0.1, "allbbox": 0.1, "refer": 1.0}, MTD={"K": 250},
              enable_grab=enable_grab, enable_mask=enable_mask,
              enable_crv=enable_crv,
              crv=dict(topk=crv_topk, uncertainty_topn=5, grid_size=crv_grid_size,
                       start_epoch=crv_start_epoch,
                       scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=8,
                       inner_scale=1.5, outer_scale=2.0, duplicate_iou=0.8,
                       uncertainty_temperature=1.0, gate_initial_probability=0.02,
                       uncertainty_threshold=crv_uncertainty_threshold,
                       cf_temperature=1.0,  # Independent CRV ranking only; fusion uses raw CRS logits.
                       negative_gt_coverage=0.01, gt_loss_weight=0.1,
                       cf_loss_weight=0.1, fusion_loss_weight=1.0,
                       positive_margin=1.0, ranking_margin=0.5,
                       safety_margin=0.5, safety_weight=2.0, repair_weight=2.0),
              enable_bcem=enable_bcem,
              bcem=dict(context_dim=128, num_heads=4, num_sectors=bcem_num_sectors,
                        sampling_mode=bcem_sampling_mode, num_rays=bcem_num_rays,
                        radial_edges=(0.0, 0.10, 0.25, 0.50, 1.0),
                        angular_samples=5, radial_samples=4 if bcem_sampling_mode == "ray" else 2,
                        min_valid_points=2 if bcem_sampling_mode == "ray" else 5,
                        context_strength=0.1, chunk_size=32,
                        use_position=True, use_transitions=True, use_gru=True,
                        use_text_attention=True, use_sparse_selection=True,
                        learnable_gate=True, checkpoint_chunks=True)),
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
swanlab = dict(mode="online", project="propvg", init_timeout=30, finish_wait=1, stats_interval=30)
start_save_checkpoint = 0
# Keep the last 10 epoch snapshots, plus top1_best.pth and latest.pth.
save_interval = 1
max_keep_ckpts = 10
# finetune_from = "PropVG-grefcoco.pth"
# resume_from = '/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-bcem-rays/20261018_112107/epoch_35.pth'
start_test = 27  # 1-based: test each epoch from 27, using the same GPUs as val; None/<=0 disables.
