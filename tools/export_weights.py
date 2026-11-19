#!/usr/bin/env python3
"""Export an inference-only PropVG checkpoint without changing the original.

Usage (in the propvg environment):
    python tools/export_weights.py /path/to/top1_best.pth
    python tools/export_weights.py /path/to/top1_best.pth -o /path/to/weights.pth

The result is {"state_dict": original_state_dict}, which the existing test
loaders accept. All parameters, buffers, module prefixes, dtypes, and state
dict metadata are retained. Optimizer/scheduler/AMP states and top-level
training metadata are omitted. This is for testing, not seamless resuming.

Only use trusted checkpoint files: torch.load uses pickle. Convert a finished
checkpoint, not a latest.pth that another process is currently writing.
Conversion runs on CPU and can need several GB of host RAM; no model or GPU
is initialized. No existing file is overwritten, including the input file.
"""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import torch


def read_weights(source):
    # Explicit False also supports full training checkpoints in newer PyTorch
    # releases whose default restricted loader may reject training metadata.
    checkpoint = torch.load(str(source), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("输入必须是包含 state_dict 的 checkpoint 或原始 state_dict")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif checkpoint and all(isinstance(key, str) and torch.is_tensor(value)
                            for key, value in checkpoint.items()):
        state = checkpoint
    else:
        raise ValueError("输入中没有找到 state_dict")
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError("state_dict 必须是非空的模型状态字典")
    # Do not turn it into dict(state), filter buffers, cast, or strip module.:
    # keeping the original object also preserves OrderedDict._metadata.
    # Returning only state lets optimizer storage be released before saving.
    return state


def export_weights(checkpoint_path, output_path=None):
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("输入不是文件: {}".format(source))
    destination = (Path(output_path).expanduser().resolve() if output_path else
                   source.with_name(source.stem + "_weights.pth"))
    if destination == source:
        raise ValueError("输出不能是原 checkpoint；原文件必须保留")
    if destination.exists():
        raise FileExistsError("输出已经存在，不会覆盖: {}".format(destination))

    source_bytes = source.stat().st_size
    state = read_weights(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write into the same directory, then publish atomically without replacing
    # an existing output (including one created after the initial check).
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=destination.name + ".tmp-",
                                         dir=str(destination.parent), delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save({"state_dict": state}, handle)
        os.link(str(temporary_path), str(destination))
    finally:
        if temporary_path is not None:
            temporary_path.unlink()

    return source, destination, source_bytes, destination.stat().st_size, len(state)


def main():
    parser = argparse.ArgumentParser(
        description="将完整训练 checkpoint 转为仅含 state_dict 的测试权重；不覆盖原文件。",
        epilog="仅处理可信且已保存完成的 checkpoint；CPU 转换可能需要数 GB 内存。输出不能用于无缝断点续训。",
    )
    parser.add_argument("checkpoint", help="完整 checkpoint 路径")
    parser.add_argument("-o", "--output", help="输出路径，默认在原目录生成 原名称_weights.pth")
    args = parser.parse_args()
    try:
        source, destination, original, compact, entries = export_weights(args.checkpoint, args.output)
    except Exception as exc:
        parser.exit(1, "转换失败: {}\n".format(exc))
    print("原文件（已保留）: {}".format(source))
    print("仅权重文件: {}".format(destination))
    print("大小: {:.3f} GiB -> {:.3f} GiB".format(original / 2**30, compact / 2**30))
    print("保留 {} 项模型参数/buffers；已移除优化器、调度器及其他顶层训练信息。".format(entries))
    print("可使用原测试命令，将 checkpoint 路径换为输出文件；模型配置保持一致。")


if __name__ == "__main__":
    main()
