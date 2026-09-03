_base_ = ["../_base_/misc.py"]
# Register the dataset-local components when this config is loaded.  This
# keeps the original train.py usable for both single- and multi-GPU runs.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
del AerialVG, LoadAerialVGAnnotations

work_dir = "/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-aerialvg"

# AerialVG provides one caption query per image and auxiliary region boxes.
# The dataset-local adapter supplies box labels only.
# Select the AerialVG-specific validation branch in the training runner.
dataset = "AerialVG"
dataset_type = "AerialVG"
# Maximum observed AerialVG caption length is 55 SentencePiece tokens;
# 64 leaves room for BEiT3 BOS/EOS tokens without truncation.
max_token = 64
img_size = 386
patch_size = 16
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "refer_target_index", "empty"]),
]
val_pipeline = train_pipeline.copy()

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(type=dataset_type, which_set="train", annsfile="data/AerialVG/annotation/vg_train_odvg.jsonl",
               imgsfile="data/AerialVG/images", pipeline=train_pipeline),
    val=dict(type=dataset_type, which_set="val", annsfile="data/AerialVG/annotation/vg_val_odvg.jsonl",
             imgsfile="data/AerialVG/images", pipeline=val_pipeline),
    test=dict(type=dataset_type, which_set="test", annsfile="data/AerialVG/annotation/vg_test_odvg.jsonl",
              imgsfile="data/AerialVG/images", pipeline=val_pipeline),
)

model = dict(
    type="MIXRefUniModel_OMG",
    vis_enc=dict(type="BEIT3", img_size=img_size, patch_size=patch_size, vit_type="base", drop_path_rate=0.1,
                 vocab_size=64010, freeze_layer=-1, vision_embed_proj_interpolate=False,
                 pretrain=None),
    lan_enc=None,
    fusion=None,
    head=dict(type="REFHead", input_channels=768, hidden_channels=256, num_queries=20,
              detr_loss={"criterion": {"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
                         "matcher": {"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0}},
              loss_weight={"mask": {"dice": 1.0, "bce": 1.0, "nt": 0.2, "neg": 0},
                           "bbox": 0.1, "allbbox": 0.1, "refer": 1.0}, MTD={"K": 250}),
    post_params={"score_weighted": False, "mask_threshold": 0.5, "score_threshold": 0.7,
                 "with_nms": False, "with_mask": False},
    process_visual=False,
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
lr = 0.0005
optimizer_config = dict(type="Adam", lr=lr, lr_vis_enc=lr / 10.0, lr_lan_enc=lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=1, decay_steps=[24, 32], decay_ratio=0.1, max_epoch=36)
log_interval = 50
start_save_checkpoint = 0
# finetune_from = "PropVG-grefcoco.pth"
