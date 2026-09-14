python tools/visualize_aerialvg_dataset.py \
    configs/aerialvg/PropVG-aerialvg.py \
    --checkpoint /media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-aerialvg384——prop/20260910_115145/top1_best.pth \
    --split test \
    --output-dir visualization/aerialvg_test \
    --topk 5