#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH}"
# Silence all Python warnings for this training launcher.
export PYTHONWARNINGS=ignore
export PYTHONHASHSEED=6666
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

/media/vision/lzy/.conda/envs/propvg/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29500 \
  tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py \
  --launcher pytorch

/media/vision/lzy/.conda/envs/propvg/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29500 \
  tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py \
  --launcher pytorch

/media/vision/lzy/.conda/envs/propvg/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29500 \
  tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py \
  --launcher pytorch
