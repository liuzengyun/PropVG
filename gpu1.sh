#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONHASHSEED=6666
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore

/media/vision/lzy/.conda/envs/propvg/bin/python tools/train.py configs/aerialvg/PropVG-aerialvg.py
