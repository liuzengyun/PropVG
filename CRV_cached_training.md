# 冻结基线、仅训练 CRV：缓存与在线前向两种模式

## 1. 这一版做什么

目的：保持训练好的 grounding 参数不变，单独验证 CRV 重排序是否有效。可选择离线缓存来省去每个 epoch 的重复基线前向，也可选择在线前向来省磁盘空间。

独立配置：`configs/aerialvg/PropVG-crv-cached.py`，完整配置、不继承其他实验配置。

独立入口：`tools/train_crv_cached.py`。不使用原来的 `tools/train.py` 或 `dist_train.sh`；第一版为单卡，不接受 torchrun 多进程，避免多个进程同时覆盖缓存。

没有修改原训练入口、评估入口、checkpoint 加载器和 CRV 方法。原 `REFHead.forward_test` 只增加默认关闭的 `return_crv_inputs` 参数；不指定时执行原流程。原模型不需要重新训练或转换。

原配置配合 `tools/train.py` 仍可同时训练 grounding 和 CRV：基线由原 detection/refer 等损失优化，CRV 由自己的损失优化，原有 detach 边界不变。这与新独立入口的“冻结 grounding、只训练 CRV”是两条不同路径。

新配置中的开关都放在 `offline_crv` 字典里：

```python
offline_crv = dict(
    use_cache=True,
    cleanup_cache_after_training=True,
    # 其余配置保持原值……
)
```

| 设置 | 特征来源 | 基线状态 | 缓存处理 |
|---|---|---|---|
| `use_cache=True` | 提取一次，之后各轮读磁盘 | 提取时冻结，CRV 训练时不放到 GPU | 成功训练/最终测试/完整权重保存后，默认清理本次使用的缓存 |
| `use_cache=False` | 每个 batch 在线运行一次基线 | 全程冻结、eval、no_grad | 不创建、不读取、不删除特征缓存 |

`cleanup_cache_after_training=False` 可显式保留缓存供后续实验复用。中断、训练报错、评估失败或保存失败时都保留缓存；仅执行 `--stage cache` 也不会删除。

## 2. 基线与数据

默认基线是：

```text
work_dir/aerialvg/PropVG-BASE/
20261015_215412+val_top1=66.49+val_top5=88.06+val_miou=57.81
+test_top1=65.72+test_top5=88.31+test_miou=56.96/top1_best.pth
```

实际配置中的路径是一整行，不包含换行。

架构与该实验一致：BEiT-3 base、输入尺寸 586、256 个候选、Grab/BCEM/分割关闭。完整 checkpoint 已包含 BEiT-3，故 `pretrain=None`，不另外加载预训练权重。

修改基线时必须同时对齐原实验架构、图像尺寸、Grab/BCEM 开关和预处理。加载器逐项检查所有正在使用的参数及形状；仅允许省略模块的历史无用参数，不会默默忽略主干/CRS 的缺失参数。

### 2.1 基线开启 BCEM 时的强制一致性检查

可以加载已经训练好的 BCEM 基线。把 `enable_bcem=True`，并将原实验的**整份** `model.head.bcem` 配置复制到新配置中；BCEM 继续参与基线前向，但其参数、buffers 和运行模式均被冻结，只训练 CRV。

不能只检查权重形状：射线数量、采样方式/位置、径向区间、`context_strength` 等可能不改变 state_dict 形状，却会改变前向结果。因此新入口还会比较开启状态及全部 BCEM 构造参数，包括通道/注意力头数、ray/sector 数量、区间与采样数、有效点阈值、残差强度、功能开关、chunk/checkpoint 设置等。省略参数会按构造默认值展开；等价的列表/元组及 `None` 自动默认值允许一致。

基线配置的确认方式：

1. 新独立入口保存的完整 checkpoint 优先使用内嵌的 `frozen_bcem_config`，即使后来移动权重文件也能检查。
2. 旧 checkpoint 没有这份元数据时，自动查找同目录唯一的 `config.py` 或时间戳配置快照。
3. 若没有快照或有多个候选，必须在 `offline_crv.baseline_config` 指定**原训练配置快照**；不能仅凭权重猜测非参数设置。显式配置也不能覆盖与内嵌元数据冲突的设置。

```python
enable_bcem = True
offline_crv = dict(
    baseline_checkpoint="/原BCEM实验/top1_best.pth",
    baseline_config="/原BCEM实验/20261018_133618_PropVG-aerialvg.py",
    # 其余字段保留……
)
# model.head.bcem 使用上述原实验的完整配置，不要直接沿用默认 BASE 示例。
```

例如你之前的 `PropVG-bcem-rays/20261018_133618...` 实验是 `num_rays=16`，本新配置的 BASE 示例为 8。只开启 BCEM 而不复制原配置，会明确报 `arguments.num_rays: baseline=16, current=8`，不会静默继续。

配置比较后，还会检查 BCEM 权重是否完整、形状是否一致；缓存训练阶段即使不构建完整 grounding 模型，也会执行同样校验。检查结果保存为日志目录的 `baseline_bcem_validation.json`，当前规范化配置也保存在新 checkpoint 中。无法确认或不一致时，在 CRV 训练前报错。此检查只作用于新独立入口，不改变旧训练/测试入口。

这一版默认使用正式划分：

| 阶段 | 标注 |
|---|---|
| CRV 训练 | `data/AerialVG/annotation/vg_train_odvg.jsonl` |
| 每轮验证、挑选模型 | `data/AerialVG/annotation/vg_val_odvg.jsonl` |
| 最优 val 模型的最终测试 | `data/AerialVG/annotation/vg_test_odvg.jsonl` |

缓存和训练只保留原始 refer GT，不加载 teacher 或辅助 GT。基线之前使用 teacher 训练过的效果已在权重中保留，关闭 teacher 标注读取不会改变其前向结果。

`enable_correctness` 可以选择开启：如果基线 checkpoint 已经训练过正确性预测头，就把它作为冻结基线的一部分加载和保存。当前默认 BASE checkpoint 没有此头，因此默认仍为 False；开启时要同时选择包含该头的 checkpoint，并使 `correctness` 的维度配置与预训练一致。入口检查该头的完整参数、schema、trained_steps>0 和参数形状，不允许用随机初始化的头冒充已有能力。

该头在实现上是 CRS 后的独立附加观察头，不改变 CRS 排序。独立 CRV 训练只读取其前面的特征，不重新训练该头，也不将其概率引入路由、融合或新损失；正确性预测头的参数和 buffers 在整个 CRV 训练中保持不变。最终完整 checkpoint 保留该头，使用原 `tools/eval_aerialvg.py` 正常测试时会照常执行并输出相关指标。独立训练入口的三路排序验证目前只统计定位性能，不额外重复执行正确性预测头。

CRV 重排后的 Top-1 分布可能不同于该头最初训练时的 CRS Top-1，因此保留预训练头不代表其对新排序的置信度必然仍准确；需在最终测试中重新检查其准确率、AUROC 和校准情况。

## 3. 缓存模式第一步：离线提取特征

目的：把所有依赖基线参数的计算固定下来，每条“图像—完整描述”只前向一次。

基线所有参数 `requires_grad=False`、整个模型 `eval()`、前向 `no_grad()`。只传图像、文本和尺寸元信息，不把 GT 框传入模型。使用现有推理路径获取：

| 缓存字段 | 单条维度 | 用途 |
|---|---|---|
| `visual` | 256×Hf×Wf，常见 256×144×144 | 最大检测空间特征，尚未经过 CRV 投影 |
| `visual_mask` | Hf×Wf | 排除 padding |
| `text`、`text_mask` | L×256、L，通常 L=64 | 完整文本及 padding |
| `boxes` | N×4 | 有效图像归一化 cxcywh 框 |
| `candidate_features` | N×256 | 实际输入 CRS 的候选特征 |
| `probabilities`、`logits` | N×1 | 原 CRS 分数、未 sigmoid 的分数 |
| `img_meta` | 字典 | 图像尺寸、pad 尺寸、表达式、样本身份等 |
| `refer_gt`、`refer_gt_pixel` | 各 4 维 | 原始 refer GT 的归一化/resize 后像素 xyxy，单独用于监督和评价 |

视觉表示依赖文本，不能把同一张图片的不同描述当作同一条缓存。缓存中不保存经过 CRV 可训练投影/评分器处理后的张量，避免更新参数后缓存失效。

每条缓存单独保存为 `00000000.pt` 等文件，避免每个 epoch 全量读取入内存。切片会复制为独立存储，避免一个样本误存整个 batch 的底层内存。

`manifest.json` 记录基线 checkpoint SHA256、标注文件 SHA256、模型配置、预处理、代码版本摘要、数据划分和缓存精度。已完成的匹配缓存直接复用；不匹配则报错并要求新目录，不擅自删除或覆盖其他来源的缓存。中断后重新执行 `--stage cache`，只推理尚未保存的样本。

缓存身份现在区分“真正影响特征的逻辑”和“训练入口/检查/清理的辅助逻辑”：冻结前向与样本打包函数单独计算代码指纹，基线模型/数据/坐标代码、权重、标注、预处理和精度仍严格检查；整个训练脚本和辅助文件的哈希仅用于审计，不再因新增 BCEM 校验或清理代码而让特征失效。`enable_bcem=False` 时，不参与构建和前向的 `bcem` 参数不影响缓存身份；开启时仍完整比较，不能借此绕过 BCEM 原配置校验。

对此前已经生成的两版已审计缓存，提供只读兼容：验证旧 manifest 完整性、已知提取器/入口版本及相同的实际输入后，继续使用原来的样本指纹，不重写 `.pt` 或 manifest，不重新推理。训练和最终清理都支持该身份；部分缓存补全时新样本也沿用同一指纹。未知旧提取器，或真正的模型、数据、图像处理、精度、特征提取逻辑变化，仍会拒绝复用，不提供无条件忽略校验开关。

缓存假设原图目录内容不变，不逐张计算图片内容哈希。如果原图被替换、重压缩或修改，应主动更换缓存目录。

## 4. 第二步：只训练 CRV

目的：只优化 CRV；缓存模式额外省掉各 epoch 重复的 BEiT-3、SimFPN、检测器和 CRS 前向。

```text
缓存模式：读取缓存 V、有效 mask、T、预测框、Q、CRS 分数
在线模式：图像+文本 → 冻结基线一次前向 → 同一组 V、mask、T、框、Q、CRS 分数
两种模式随后进入同一个 CRV：
 → 原分数前若干候选按 IoU 分组
 → 分组归一化熵 U
 → U > 0.6 才运行 CRV；其他样本保留 CRS
 → 对最大空间特征做候选区域干预
 → 原场景与干预场景输入同一个独立图文评分器
 → 支持下降量 d_i = s_original - s_mask_i
 → CRV 分数校准到 CRS logit 尺度
 → 可学习纠错选择器产生 alpha
 → 融合后在原 Top-K 内重排
 → GT 计算 CRV 损失，只更新 CRV
```

缓存模式的训练阶段不会构建完整 grounding 模型，也不会将其搬到 GPU。基线 state_dict 只在 CPU 保留，供完整 checkpoint 保存。

在线模式只构建一次冻结基线，与独立 CRV 分开维护；每个 batch 执行一次基线前向，之后各候选干预仅复用本 batch 的特征，不重复跑主干。视觉/文本/候选特征直接留在设备上交给 CRV，不先写磁盘或搬回 CPU。CRV 调用 `train()` 不会将基线切回训练模式，优化器也不包含任何基线参数。验证与最终 test 也从真实图像在线提取，不读缓存，GT 不传入基线或 CRV 的评估预测路径。

沿用现有 CRV，不在这次实验中混入新算法改动：

- 最大检测视觉特征上做二维干预，不裁局部窗口。
- 1.5 倍框内使用每通道整幅有效图像均值替换，1.5～2 倍区间余弦羽化。
- 原场景每图评分一次，完整场景有效池化到 48×48；每份干预场景独立评分，不混合不同场景的注意力上下文。
- 文本保持完整，不人工解析。
- 原始 refer GT 仅用于训练标签和精确 GT 干预辅助，不加入推理候选。
- 保留现有 `loss_crv_cf`、`loss_crv_fusion`、`loss_crv_gate`，内部定义见 `CRV_method.md`。仅优化 `head.crv.*` 参数。
- 不计算原检测/CRS 定位/Grab/分割损失，不需要为此做 Hungarian 匹配。

第一版优化器为 Adam，学习率固定 `1e-4`，训练 20 epoch，当前配置 batch=1，梯度裁剪 0.15；这些是独立配置中的初始设置，不承诺最优。开始 epoch 为 1，基线已经训练完成，不需要再等 CRS 稳定。在线模式的 batch 必须同时容纳冻结基线与 CRV；缓存模式可根据显存增大 batch。

若一个 batch 全被路由跳过，损失不含可训练图，直接跳过 backward/step，不因空梯度更新参数。

## 5. 评价与完整模型保存

目的：同时观察纠错能力和原本正确样本受到的伤害，避免只看融合后的总分。

每轮使用 val（随 `use_cache` 选择缓存或在线），输出三种 Top-1 Acc@0.5、Top-5 Acc@0.5、Top-1 box mIoU：

1. `crs`：冻结基线原始排序。
2. `crv`：路由激活样本直接按支持下降量排序；跳过样本仍用 CRS，不加融合门控。
3. `fusion`：正常可学习融合后的最终排序。

同时输出 corrected（原错后对）、harmed（原对后错）、net（两者之差）和路由样本数。评估时不向 CRV 传 refer GT，GT 只在输出后计算指标。

坐标转换、像素缩放、裁剪、退化框过滤、CPU 排序和 IoU 定义与现有 AerialVG 评估一致。K=5 时只允许原 Top-5 内交换，通常 Top-5 命中率不变；严格并列分数的边界需注意原 CRS 与 CRV 的排序 tie policy，不能把这种变化解释为找到新候选。

按 val 的 **fusion Top-1** 选择 `top1_best.pth`，另保留 `latest.pth`，不额外保留全部 epoch。训练结束仅对该最优模型评估一次 test，test 不参与参数更新、门控阈值选择或模型挑选。

两个 checkpoint 均包含：

```text
state_dict = 原基线完整权重（含开启的 BCEM、预训练 correctness，不变） + head.crv.*（训练结果）
crv_optimizer = 仅 CRV 的优化器状态
frozen_bcem_config = 基线的规范化 BCEM 配置（包括非参数设置）
epoch、val 指标、基线来源哈希、缓存身份等
```

保存的不是只有 CRV 的小权重。首次训练会在日志目录备份 `propvg/` 和独立训练入口，保存完整 `config.py`。文件夹名字附加最佳 val 指标和该模型的 test 指标。第一版未提供 CRV 训练中途续训入口；缓存阶段可以断点继续，已有完整模型仍可正常测试。

缓存模式默认在全部流程成功之后清理**本次使用的划分**。若关闭最终 test，则只清理 train/val，不碰未使用的 test 缓存。清理前确认 `latest.pth`、`top1_best.pth`、`config.py` 已保存，并逐个检查缓存 manifest 的来源、完成状态和完整文件清单。任何划分出现陌生文件、来源不匹配、软链接、缺失文件或危险目录，都会拒绝清理，不对缓存目录执行递归删除。

清理只逐文件删除已验证的派生 `.pt` 和 manifest，随后移除空目录；图片、标注、基线权重、完整模型和日志均保留。记录写入日志目录的 `cache_cleanup.json`。已删除特征没有备份，但可以从原图和基线重新生成。缓存目录外会留一个很小的 `.crv.lock` 锁文件，用于防止新入口的两个进程一边读同一缓存、一边删除；它不是特征缓存，也不持有磁盘锁之外的模型数据。

## 6. 如何运行

在 `propvg` 环境、仓库根目录执行。示例使用第二张物理 GPU；该进程内它会映射为 `cuda:0`。

命令不变：`use_cache=True` 时依次缓存、训练、最终测试并清理；`False` 时直接在线冻结前向训练和测试：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/train_crv_cached.py \
  configs/aerialvg/PropVG-crv-cached.py
```

也可以分开执行，便于检查首次缓存是否正常：

```bash
# 一次性缓存 train、val、test；已有匹配缓存不重复推理。
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/train_crv_cached.py \
  configs/aerialvg/PropVG-crv-cached.py --stage cache

# 只读已生成的缓存训练，不再运行基线；成功完成后默认删除所用缓存。
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/train_crv_cached.py \
  configs/aerialvg/PropVG-crv-cached.py --stage train
```

不修改文件而临时选择在线模式：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/train_crv_cached.py \
  configs/aerialvg/PropVG-crv-cached.py \
  --cfg-options offline_crv.use_cache=False
```

在线模式的 `--stage train` 和默认 `all` 都直接训练；`--stage cache`/`--splits` 与在线模式冲突，会明确报错。缓存成功清理后，下一次缓存模式训练需要默认 `all` 重新生成，不能直接 `--stage train`。如计划复用同一缓存做多组实验，可设置 `offline_crv.cleanup_cache_after_training=False`。

若先小规模验证，可用 `--cfg-options data.train.max_samples=16 data.val.max_samples=8 data.test.max_samples=8 offline_crv.cache_dir=/tmp/propvg_crv_smoke`。小样本缓存与完整缓存身份不同，务必使用独立目录。其结果不能当作全量性能。

实际部署/重新测试时使用已有评估入口，不需要缓存，不需要再指定基线 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/eval_aerialvg.py \
  /实际训练目录/config.py \
  --checkpoint /实际训练目录/top1_best.pth --split test
```

测试流程仍是 `图像+文本 → 基线一次前向 → CRV → 最终框`。关闭 CRV 对照原基线：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python tools/eval_aerialvg.py \
  /实际训练目录/config.py \
  --checkpoint /实际训练目录/top1_best.pth --split test \
  --cfg-options model.head.enable_crv=False
```

关闭后仅忽略额外 CRV 权重，基线部分照常加载。以前的配置、checkpoint 和测试命令不需要改。

## 7. 空间与速度边界

目的：避免把“冻结”误认为自动省去每个 epoch 的前向。

只有缓存训练阶段不运行基线。首次缓存仍要完整遍历数据集一次，最终在线测试仍要运行基线一次。

默认 FP32 保留特征数值；单条 256×144×144 视觉特征约 20.25 MiB，完整 train/val/test 可能接近 1 TiB。开始每个划分时会估算剩余写入量并检查可用空间。配置中的缓存目录位于大容量数据盘，不是 `/tmp`。

`cache_dtype="float16"` 可将 V/T/Q 存储减半，框、CRS 分数、GT 仍保留 FP32；加载后转 FP32 训练，但此前的量化误差不可恢复，因此不能声称与在线 FP32 逐位一致。缓存精度改变需要换缓存目录。

缓存训练速度也受磁盘读取和 CRV 评分器本身开销限制，不能保证按冻结参数量等比例加速。默认成功后会清理缓存，但生成和训练过程中仍必须有足够峰值磁盘空间。在线模式不占大容量缓存空间，代价是每个 epoch、每次验证仍需基线前向；`cache_dtype` 对在线模式不生效。

## 8. 验证范围

CPU 回归包含缓存与在线 CRV 的预测/损失/梯度一致性、原始 refer GT 选取、评估不读 GT、缓存身份拒绝、完整权重组装及原基线逐位不变、真实 REFHead 提取接口、退化框/裁剪/排序一致性，以及不构建 grounding 模型的两轮缓存 CRV 训练/保存。新增在线训练循环验证冻结基线每 batch 前向一次、不读写/删除缓存、仍有 CRV 参数更新；清理测试覆盖成功删除、失败保留、全划分预检、陌生文件/软链接/危险目录拒绝、同目录并发锁。开启已训练 correctness 时，两种训练模式都验证其参数和 buffers 不变且保存在完整模型中；缺失、未训练、不完整或形状不兼容的该头会明确报错，真实 REFHead 的完整模型测试可以正常执行保留的该头。

BCEM 校验测试覆盖原配置自动发现、显式原配置、内嵌元数据、非形状参数不匹配、缺少来源/多份来源、权重不完整和形状不匹配。两种 CRV 训练模式的集成测试同时开启 BCEM 与预训练正确性头，确认它们冻结且完整保存；真实 REFHead 的缓存特征提取测试也开启 BCEM。

旧缓存兼容测试覆盖已知旧版本只读复用、不重新推理、不改文件时间/内容、关闭 BCEM 时无效参数变化不影响缓存、开启 BCEM 时参数变化仍失效，以及真实权重/标注/代码/精度变化、损坏元数据和未知旧代码继续报错。

当前环境没有可用 CUDA 驱动，未完成真实图片 CUDA 前向、全量缓存或正式训练；不能将 CPU 合成测试视为实际性能验证。
