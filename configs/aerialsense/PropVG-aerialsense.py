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
