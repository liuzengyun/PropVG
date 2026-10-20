# SwanLab 实验监控

训练监控使用 SwanLab。云端新版项目要求 SDK ≥ 0.9.0，而训练环境的 Python 3.8 无法安装该版本，因此后台上传使用独立的 Python 3.11 环境 `.venv-swanlab`。训练启动命令不变：

```bash
bash gpu1.sh
# 或原来的多卡启动脚本
bash tools/dist_train.sh
```

默认项目为 `propvg`，实验名为“实验组目录/启动时间”。后台初始化成功后，终端会在下一次日志提交时显示 `SwanLab online: https://...`；也可从本次实验的 `swanlab_tracking.json` 找到 `status.json`，读取其中的网页地址。

这项变更从下次启动训练生效。补传 SwanLab 本地记录可手动运行下文的工具。

换机器时，先用 Python ≥ 3.9 创建上传环境：

```bash
python3.11 -m venv .venv-swanlab
.venv-swanlab/bin/python -m pip install -r requirements-swanlab.txt
.venv-swanlab/bin/swanlab login
```

代码自动优先选择项目内的 `.venv-swanlab/bin/python`；也可用 `PROPVG_SWANLAB_PYTHON` 或配置中的 `swanlab.python` 指定解释器。登录信息沿用同一用户的 SwanLab 账号。

## 配置和开关

AerialVG / AerialSense 配置均使用：

```python
swanlab = dict(
    mode="online",
    project="propvg",
    init_timeout=30,  # 后台初始化时限，训练不等待
    finish_wait=1,    # 训练退出最多等后台 1 秒
    stats_interval=30,
    # workspace="团队名称",       # 可选，默认使用已登录账号
    # experiment_name="实验名称", # 自动追加启动时间
)
```

没有 `swanlab` 字段的旧训练配置也使用上述默认值。`SWANLAB_MODE` 优先于配置：

```bash
SWANLAB_MODE=offline bash gpu1.sh  # 仅保存本地记录
SWANLAB_MODE=disabled bash gpu1.sh # 关闭监控
```

也支持 `local` 模式和 `swanlab = False`。账号凭据由 SwanLab SDK 读取，无需在配置中填写。

## 记录内容

| 内容 | 说明 / 图表横轴 |
|---|---|
| `train/*` | 当前 epoch 累计 loss 与 CRV 诊断均值 / global_step |
| `train_current/*`、`train_window/*` | 最近 batch / 最近日志窗口的指标 / global_step |
| `lr/*`、`time/*`、`perf/*`、`memory/*` | 学习率、耗时、名义吞吐、训练进程的显存缓存统计 / global_step |
| `val/*` | 原有验证指标 / epoch |
| `test_epoch/*` | 从 start_test 开始的当前模型测试 / epoch |
| `test_best/*` | 原流程最终最佳验证 checkpoint 的测试 / checkpoint_epoch |
| `best/*`、`epoch/*` | 最佳验证指标和每轮耗时 / epoch |
| `summary/*` | 参数量、数据规模、最佳指标等数值汇总 / 汇总更新次数 |
| `config.tracking_summary` | 完整汇总，包含最终实验目录等文本信息；本地副本为 `summary.json` |
| 终端日志 | 原训练 logger 的文本，在初始化后由后台打印并交给 SwanLab 捕获；本地副本为 `console.log` |

按指标分别设置 `step`。训练 loss 保持原单位；AerialVG Top-1、Top-5、box mIoU 为百分数。原有平面指标字段也继续记录。

多卡仅 rank0 写 journal 并启动上传进程。训练进程不导入 SwanLab SDK；独立后台进程负责初始化、联网和上传。保留原有训练、验证、测试及 checkpoint 选择逻辑。后台会采集系统资源；训练进程显存使用量以 `memory/*` 为准。

## 本地记录和补传

本地记录默认位于：

```text
swanlog/tracking-时间-ID/
  metadata.json  # 配置及实验信息
  events.jsonl   # 结构化日志，先写本地再上传
  status.json    # 当前模式、网页地址和 SDK 目录
  worker.log    # SDK 诊断输出
  console.log   # 训练文本日志
  summary.json  # 汇总信息（有 summary 事件时生成）
  sdk/          # SwanLab 标准本地记录
```

在线初始化失败或超时会在全新的进程中尝试离线，避免失败 SDK 状态影响回退；SDK 无法使用时保留完整 journal，训练继续。`status.json` 和 `worker.log` 会记录已脱敏的具体错误。离线启动的实验不会中途自动切成在线。后台正常收尾最多等待约 30 秒，超时未传完的记录可从 journal 补传。

使用项目提供的恢复工具创建一个名称带 `replay` 的新 SwanLab 实验：

```bash
python tools/replay_swanlab.py 'work_dir/.../本次实验/swanlab_tracking.json' --mode online
# 也可直接指定 swanlog/tracking-时间-ID 目录
```

可用 `--project` 和 `--workspace` 指定补传目标。工具只读取日志快照，不训练、不读取权重、不修改原记录。对运行中的实验重放时，只包含当时已写入的日志。

接口参考：[SwanLab 初始化](https://docs.swanlab.cn/api/py-init.html)、[指标记录](https://docs.swanlab.cn/api/py-log.html)。
