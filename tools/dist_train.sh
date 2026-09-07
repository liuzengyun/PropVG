#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH}"
# Silence all Python warnings for this training launcher.
export PYTHONWARNINGS=ignore

/media/vision/lzy/.conda/envs/propvg/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29500 \
  tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py \
  --launcher pytorch
