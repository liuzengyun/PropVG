# Register the AerialVG-compatible dataset adapter and pipeline components.
# AerialSense uses the same ODVG JSONL structure and processing code.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401

del AerialVG, LoadAerialVGAnnotations


work_dir = "/media/vision/lzy/PropVG/work_dir/aerialsense/PropVG-aerialsense"

# Keep the AerialVG evaluation branch because AerialSense is converted to the
# same one-expression/one-referred-box visual-grounding format.
dataset = "AerialVG"
dataset_type = "AerialVG"
start_test = 22  # 1-based: test each epoch from 22, using the same GPUs as val; None/<=0 disables.
max_token = 64

# New source-stratified, scene-grouped 75/25 split (seed 6666).
# Original ordered annotation/teacher files remain unchanged for old runs.
# The held-out set is shared by val/test, as in the previous configuration.
aerialsense_split_dir = "data/AerialSense/splits/scene_seed6666"

# Fix Python, NumPy, PyTorch, CUDA, and DataLoader random sources as far as the
# current training runner supports.
strict_reproducibility = False
seed = 6666
deterministic = False

# AerialSense contains exactly one referred target per annotation.
use_auxiliary_gt = False

# Teacher boxes are optional detection targets; refer GT is always original.
teacher = False
teacher_annsfile = "/media/vision/lzy/PropVG/data/AerialSense/splits/scene_seed6666/teacher_train.json"
teacher_max_boxes = 4
# Filter teacher boxes whose clipped pixel area is >10x the original refer GT.
# False restores previous teacher handling. Exactly 10x is not removed here.
teacher_max10 = True
teacher_dedup_iou = 0.001
teacher_nms_iou = 0.5
if teacher and use_auxiliary_gt:
    raise ValueError("teacher and use_auxiliary_gt are mutually exclusive")

# False fine-tunes the complete BEiT-3 backbone; True freezes it.
freeze_beit3 = False

# Optional GrabVG components already implemented in the current model.
enable_grab = False
# Do not allocate segmentation-only modules; shared 512-channel FPN is unchanged.
enable_mask = False
# Independent post-CRS refinement; disabled to retain the previous pipeline.
enable_crv = False
crv_topk = 5
crv_crop_size = 48  # Candidate-centered local crop in feature cells, not image pixels.
crv_grid_size = 24  # Pool matched original/masked local crops to this scoring grid.
crv_uncertainty_threshold = 0.6  # Only grouped entropy above this enables refinement.
crv_start_epoch = 22  # 1-based: no CRV training or reranking before this epoch; 1 starts immediately.

img_size = 586
patch_size = 16
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

train_pipeline = [
    dict(
        type="LoadAerialVGAnnotations",
        max_token=max_token,
        use_auxiliary_boxes=use_auxiliary_gt,
        use_teacher_boxes=teacher,
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "refer_target_index",
            "empty",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadAerialVGAnnotations",
        max_token=max_token,
        use_auxiliary_boxes=False,
        use_teacher_boxes=False,
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "refer_target_index",
            "empty",
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        which_set="train",
        annsfile=aerialsense_split_dir + "/annotation/vg_train_odvg.jsonl",
        imgsfile="data/AerialSense/images",
        pipeline=train_pipeline,
        teacher_annsfile=teacher_annsfile if teacher else None,
        teacher_dedup_iou=teacher_dedup_iou,
        teacher_nms_iou=teacher_nms_iou,
        teacher_max_boxes=teacher_max_boxes,
        teacher_max10=teacher_max10,
    ),
    val=dict(
        type=dataset_type,
        which_set="val",
        annsfile=aerialsense_split_dir + "/annotation/vg_val_odvg.jsonl",
        imgsfile="data/AerialSense/images",
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        which_set="test",
        annsfile=aerialsense_split_dir + "/annotation/vg_test_odvg.jsonl",
        imgsfile="data/AerialSense/images",
        pipeline=val_pipeline,
    ),
)

model = dict(
    type="MIXRefUniModel_OMG",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=patch_size,
        vit_type="base",
        drop_path_rate=0.1,
        vocab_size=64010,
        freeze_layer=-1,
        freeze_beit3=freeze_beit3,
        vision_embed_proj_interpolate=False,
        # Load only generic BEiT-3 base weights, not a PropVG task checkpoint.
        pretrain="pretrain_weights/beit3_base_patch16_224.zip",
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type="REFHead",
        input_channels=768,
        hidden_channels=256,
        num_queries=128,
        detr_loss=dict(
            criterion=dict(loss_class=1.0, loss_bbox=5.0, loss_giou=2.0),
            matcher=dict(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
        ),
        loss_weight=dict(
            mask=dict(dice=1.0, bce=1.0, nt=0.2, neg=0),
            bbox=0.1,
            allbbox=0.1,
            refer=1.0,
        ),
        MTD=dict(K=250),
        enable_grab=enable_grab,
        enable_mask=enable_mask,
        enable_crv=enable_crv,
        crv=dict(topk=crv_topk, uncertainty_topn=5, grid_size=crv_grid_size, crop_size=crv_crop_size,
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
    ),
    post_params=dict(
        score_weighted=False,
        mask_threshold=0.5,
        score_threshold=0.7,
        with_nms=False,
        with_mask=False,
    ),
    process_visual=False,
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
ema_factor = 0.999

lr = 0.0004
optimizer_config = dict(
    type="Adam",
    lr=lr,
    lr_vis_enc=0.00004,
    lr_lan_enc=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=3,
    decay_steps=[30, 38],
    decay_ratio=0.1,
    max_epoch=45,
)

log_interval = 50
swanlab = dict(mode="online", project="propvg", init_timeout=30, finish_wait=1, stats_interval=30)
log_level = "INFO"
save_interval = -1
resume_from = None
load_from = None
finetune_from = None
evaluate_interval = 1
start_evaluate_epoch = 0
start_save_checkpoint = 0

# Set this only when resuming/fine-tuning from a PropVG task checkpoint.
# finetune_from = "checkpoint.pth"
