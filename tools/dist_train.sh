#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2,3
CONFIG="${1:-configs/aerialvg/PropVG-aerialvg.py}"
GPUS="${2:-3}"

PORT=${PORT:-29500}
PYTHON_BIN="${PYTHON_BIN:-/media/vision/lzy/.conda/envs/propvg/bin/python}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_aerialvg.py $CONFIG --launcher pytorch ${@:3}
