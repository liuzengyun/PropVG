# Top-1 定位正确性预测头

## 目的与边界

估计**当前最终 Top-1 框与原始 refer GT 的 IoU 是否 ≥0.5**，不是预测“是否为前景”，也不是预测“是否被匈牙利匹配”。

输出 `q` 是正确概率的模型估计，`1-q` 是错误概率的估计；训练成功与概率校准程度需要独立评估，不能默认 `q=0.8` 就已保证 80% 正确率。

本模块只观察：不改变框、不改变排序、不控制 CRV，也不修改 CRS/Grab/BCEM。所有输入截断梯度，另行裁剪梯度，避免新损失通过输入或全局梯度范数影响基线。

## 配置

AerialVG、AerialSense 的配置中均新增：

```python
enable_correctness = False
correctness = dict(hidden_dim=128, topk=5, duplicate_iou=0.8,
                   loss_weight=1.0, decision_threshold=0.5)
```

- `enable_correctness`：设为 `True` 开启。默认 `False`；旧配置没有该字段时也默认关闭，不分配参数、不消耗初始化随机数。
- `hidden_dim`：新头内部维度，不改变原模型的 256 通道。
- `topk`：提取候选竞争信息时查看的候选数，并不筛掉原模型的候选。
- `duplicate_iou`：仅在计算竞争特征时合并高度重叠的候选，避免重复框被当成独立竞争者。不会改变定位输出。
- `loss_weight`：新头 BCE 权重，只影响新头参数。
- `decision_threshold`：`q≥该值` 判断为正确，反之判断为错误。该阈值不参与训练，不修改定位阈值。

## 数据流与每步目的

1. **原模型正常完成预测**：DETR → 可选 Grab → 可选 BCEM → CRS → 可选 CRV。得到候选框、最终评分以及送入 CRS 前的候选 query。新头不另跑编码器。
2. **所有输入 detach**：候选 query `[B,N,256]`、文本 `[B,L,256]`、坐标、检测分类、CRS logits 和最终排序分数都不向原模型传新梯度。
3. **对齐实际 Top-1**：坐标从有效图像归一化 cxcywh 转为像素 xyxy，裁剪并忽略退化框。CPU 排序使用与 AerialVG 评估器相同的并列规则；如 CRV 提供排序优先级，则使用该优先级。因此开启 CRV 时估计的是 CRV 后的最终 Top-1，而不是原 CRS 第一名。
4. **提取视觉／语言证据**：取实际 Top-1 query `[B,256]`，对文本的非 padding token 做均值池化 `[B,256]`。分别经 LayerNorm、Linear、GELU 投影为 `[B,d]`。它们来自原模型现有表示，不解析文本、不使用外部模型。
5. **提取竞争与检测信息**：在最终排序的 Top-K 内按 IoU 分组，计算 12 个标量 `[B,12]`：Top-1 最终分数、原 CRS logit/10、组间分数差、组间 logit 差/10、分组归一化熵、分组 softmax 第一项、组数/K、Top-1 检测前景分数，以及裁剪后框的归一化中心与宽高。
6. **融合并评分**：拼接 `[q,t,q*t,abs(q-t),scalars]`，得到 `[B,4d+12]`。经 LayerNorm → Linear → GELU → Linear 得到 `[B,1]` logit，sigmoid 后输出概率 `[B]`。没有 dropout，不改变原网络的随机数流。
7. **只在训练时生成标签**：读取 `refer_target_index` 指定的原始 refer GT，计算裁剪后 Top-1 框 IoU。`IoU≥0.5` 为 1，否则为 0。无有效预测框时为错误；没有 refer GT 的样本不参与该损失。teacher/其他辅助框不是正确性 GT。
8. **独立 BCE**：`L_correctness = loss_weight * BCEWithLogits(logit, label)`。多卡按全局有效监督数归一化。该损失与原任务损失相加，但新头输入已截断，梯度裁剪也单独执行。
9. **记录训练状态**：实际优化后累积 `trained_steps`。测试不能把随机初始化的新头当成训练好的预测器。

## 验证与测试

推理只使用图像、文本、模型自身输出和尺寸元数据，不向新头传 GT。GT 仅在模型推理结束后由评估器计算标签。

开启并训练后，单个结果的读取方式为：

```python
q = outputs["pred_bboxes"][sample_index]["top1_correct_probability"]
```

关闭新头时不增加这些输出字段。新头默认约 13.5 万个参数，不增加图像／文本编码器前向。

训练日志增加 `loss_correctness`、`correctness_accuracy`、`correctness_probability`、`correctness_valid_samples`。训练准确率是当前批次监督的诊断量，不代表泛化性能。

每次 AerialVG/AerialSense 验证／测试，原有 Top-1、Top-5、box mIoU 不变，另输出：

- **Acc**：新头判断对错的准确率（百分比）。
- **majority baseline**：总猜多数类即可达到的准确率，防止约 71% 的无效高准确率。
- **error precision**：判为错的样本中实际错的比例。
- **error recall**：真实错误被识别的比例。
- **correct FPR**：真实正确样本被误判为错的比例。
- **AUROC**：以错误为正类、以负正确性 logit 为分数，越大越好。
- **entropy AUROC**：同一批最终 Top-1 标签下，分组熵的区分能力，供公平对比。
- **Brier / ECE**：正确概率的误差与校准表现，越小越好。ECE 固定为 10 个等宽概率区间。

没有正／负样本或没有预测为错的样本时，相关未定义指标显示 `N/A`，不伪造为完美结果。没有自动搜索阈值，也不会依据 test 指标选择参数。多卡仍沿用原评估采样策略，包括其可能的 padding 重复，不改变旧定位指标口径。

## 旧模型兼容

- **原配置／旧 checkpoint 测试**：不写新开关或设为 `False`，完全不调用新头。
- **新头 checkpoint 关闭新头测试**：原定位部分仍可加载，新头参数不参与预测。
- **旧 checkpoint 不包含新头，不能开启新头直接测试**：会明确报错，避免输出随机置信度。应关闭开关。
- **想用旧定位模型权重开始训练新头**：使用 `--finetune-from`，同时将 `resume_from=None`；它不是对旧优化器的原样续训。原模型按原训练方案继续训练，新头同步学习，不要求分阶段训练。

新实验从头训练（仍可加载原有 BEiT-3 基础权重）：

```bash
python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_correctness=True resume_from=None
```

测试原模型，即使当前配置存在新参数，也可以显式关闭：

```bash
python tools/eval_aerialvg.py /path/to/original_experiment_config.py \
  --checkpoint /path/to/old_checkpoint.pth --split val \
  --cfg-options model.head.enable_correctness=False
```

旧模型若还涉及历史 Grab 结构差异，仍应使用其归档配置／代码；新头不改变此前已有的模型结构兼容边界。
