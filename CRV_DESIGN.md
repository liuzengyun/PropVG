# CRV：独立空间反事实评分与自适应融合

更新：2026-09-28。本文件是当前 CRV 的唯一说明；旧 CRV/CFCR 实现、旧文档和旧可视化脚本已从活动代码中移除。删除前的源码保存在 `backups/old_crv_before_independent_20260928.tar.gz`，已有实验、checkpoint 和实验内源码快照没有改动。

当前为独立 CRV v2：修复门控被绝对 BCE 压低、tanh 饱和、概率量化排序和 padding 边界问题。v1 修改前的源码备份为 `backups/independent_crv_v1_before_ranking_fix_20260928.tar.gz`。

## 1. 动机与边界

CRS 进行正向候选—文本匹配，航拍中外观相似的错误目标也可能得高分。CRV 增加一个互补问题：移除候选覆盖的空间视觉证据后，完整文本与剩余场景的匹配是否下降？

本版保留原 CRS，不复用其评分器、不重新运行检测解码器、不对候选 query token 做删除。独立、轻量的图文匹配网络同时评分原场景和 mask 场景，用两者之差修正原 CRS 得分。所有输入先 detach，CRV 的损失只更新独立网络和融合门控。

“对描述重要”不必然等于“定位目标”，mask 参考物体、大框覆盖多个对象、BEiT-3 上下文泄漏以及 mask 分布捷径仍是方法局限。本版按照当前约定不做参考物体保护。它是特征级干预，不应表述为真实像素删除或因果识别保证，也不保证精度提升。

## 2. 整体数据流与维度

B 为 batch，P 为 DETR query 数（AerialVG=256、AerialSense=128），T=64，C=256，K=min(8,N)。N 为进入 CRS 的候选数，Grab 开启时是过滤后数量。以 586×586 输入为例：

| 步骤 | 张量 | 处理 |
| --- | --- | --- |
| 标注加载 | 每条完整 caption + 原始 refer GT | 图像与完整表达式一一对应，regions[0] 是定位目标 |
| resize/pad | 图像 [B,3,586,586] | 保持比例缩放后右侧/底部 padding；记录有效尺寸 |
| 分词 | token IDs、文本 padding mask [B,64] | 不拆解描述；超过 max_token 会截断 |
| BEiT-3 | 图像 tokens [B,1296,768]、文本 [B,64,768] | patch=16，空间格为36×36 |
| 头部投影 | 图像 [B,512,36,36]、文本 [B,64,256] | 保留原 PropVG 的512通道数据流 |
| SimFPN | 四层 [B,512,H,W] | H/W=144、72、36、18；检测使用前256通道 |
| 检测 decoder | 各层 queries [B,P,256] | 原3层 deformable decoder；生成所有 proposal |
| DetHead | boxes [B,P,4]、类别 [B,P,2] | boxes 为有效 resize 图像内归一化 cxcywh |
| 可选 Grab | queries [B,N,256]、boxes [B,N,4] | 原过滤、外观和图结构路径不变 |
| refer_adapter + CRS | logits/scores [B,N,1] | 原评分器和原 loss_refer 保留 |
| 取 CRS Top-K | indices [B,K]、boxes [B,K,4] | 按原始 CRS logit 稳定排序选取，不回传；其他候选不进入最终排序 |
| 独立 CRV | 最大检测图 V=[B,256,144,144]、完整文本 [B,64,256] | V、文本、boxes、CRS logits 全部 detach |
| 场景构造 | 原场景 + K 份 mask 场景 | 训练额外增加精确 refer-GT mask，推理不增加 |
| 评分前池化 | [B,S,256,24,24] | 在144×144上先 mask，再做有效区域加权池化；S=1+K（训练1+K+1） |
| CRV 自身投影 | 场景 tokens [B×S,576,128]、文本 [B×S,64,128] | 不是 CRS tokens，无特征对齐或参数共享 |
| 独立图文交互 | 文本 + 场景汇总 token [B×S,65,128] | 2层 TransformerDecoder，4头，FFN256，dropout=0 |
| 场景评分 | g(V,L) [B,1]、g(V\i,L) [B,K] | 同一个新评分器，场景互不混合，不用绝对 support sigmoid |
| 差值 | d_i=g(V,L)-g(V\i,L) [B,K] | 越大表示该候选被删除后匹配下降越大 |
| 自适应门控 | alpha [B,1] | 每张图一个共享权重，无候选级第二分类器 |
| 最终输出 | probabilities/logits [B,K,1]、boxes [B,K,4] | 概率用于显示，融合 logit 用于排序；最高分框是定位结果 |

24×24 只用于控制独立评分器的注意力成本。输入仍是最大144×144特征图，干预也在该尺寸完成。可以修改 `crv_grid_size`，但池化可能稀释小目标；使用最大图不等于评分网络在全部20736个空间点上执行注意力。原 scene 被评分一次，K 个 mask scene 分别评分。没有 Reader、decoder 重放或共享 CRS。

原512通道 neck 和分割模块构建不改；无 mask GT 时仍跳过分割前向与分割损失。没有新矩形伪 mask、没有修改检测框或提案数量。

## 3. mask 与坐标

标注是原图像素 xyxy；Resize 同步缩放标注。检测损失和预测框使用有效 resize 图像尺寸归一化的 cxcywh。CRV 将中心和宽高再乘：

```text
ratio = [img_width / pad_width, img_height / pad_height]
```

从而映射至 padded 特征图，不能直接把有效图像归一化坐标当作整张 padded 图坐标。ratio 从元数据获得，不用取整后的有效格数代替。

每个通道分别计算当前图像有效区域的空间均值；不跨通道取平均，不包含 padding，也不置零。候选框中心不变：1.5倍范围是全替换核心，1.5至2倍用 smoothstep 羽化，2倍之外不修改。掩码按特征格面积积分，小于一个格的框仍有非零干预。越界部分自然裁剪。

CRV 自己按精确 resize/pad ratio 计算每个格的有效面积权重，边界半格不能当作完整有效格。均值和池化使用该权重；mask 积分裁剪至有效图像范围，再除以格内有效面积得到替换强度。原 DETR/Grab 的有效格 mask 不改，因此基线数据流不受这项修正影响。有效面积处理是对已有特征格的加权近似，不能消除编码器特征已经融合的 padding/上下文信息。

```text
V_mask = (1-M) * V + M * mean_valid(V)
```

原场景和所有 mask 场景使用相同的原始通道均值；不是每次遮挡后重新计算。padding 不进入均值或匹配。mask 不作用于语言，不使用辅助 regions 的语义标签。

## 4. 最终加权

CRS 的原始 logit 为 z_i，CRV 差值为 d_i：

```text
u_i             = d_i / temperature
evidence_i      = max_residual * u_i / (1 + abs(u_i))
alpha          = sigmoid(G(stats))
final_logit_i  = stopgrad(z_i) + alpha * evidence_i
final_score_i  = sigmoid(final_logit_i)
```

这是自适应加权残差融合，而不是丢弃 CRS 或两个概率强行平均。CRS 提供基础匹配，CRV 负责候选修正。默认 max_residual=2、temperature=1。每图门控 G 是6→32→1的 MLP；输入为选中 CRS 概率的均值、最高值、Top1/Top2差值、标准差，以及 drop 的平均绝对值和标准差。统计量也 detach，但 G 参数由最终候选排序损失训练。

使用 softsign 而不是 tanh：仍有界，但大 drop 的导数按多项式衰减，不会像 tanh 在常见的大值（例如11、12）上变成完全相同的FP32常数。辅助损失也监督变换后的 evidence 差距，不能只让原始 drop 达到 margin 却忽略实际融合区分度。

不把 drop 减去候选均值：这种中心化会让共享的原场景分数代数抵消。本版直接使用原始差值，原场景与 mask 场景都在梯度图中；非线性 evidence 差值一般使原场景也收到排序梯度，但不保证每个输入都具有非零原场景梯度（例如零初始化时各候选导数相同）。没有绝对场景支持度监督，g 不是校准后的概率。

独立场景评分器最后一层零初始化，使初始所有 d_i=0、最终 logit 精确等于选中 CRS logit。alpha 初始为0.1，只是门控初始化，不是固定融合系数；无需阶段训练、预热或 val 搜索。场景网络开始学习后，alpha 才获得有效更新信号。

修正是有界的，因此不能保证改正任意大的 CRS 分数差；Top-K 没有正确框时也不能创造目标框。

门控学习的目标是“有助于正负候选分离”，不是“把每个错误候选的绝对概率压到0”。对于一组合法正负候选，设 `Delta_e = evidence_positive - evidence_negative`，则最终排序损失对 alpha 的导数是 `-sigmoid(margin-final_gap)*Delta_e`。因此 Delta_e>0 时推动 alpha 增大，Delta_e<0 时推动 alpha 减小；不依赖两者的共同分数偏移。多个 pair 求均值后是其综合贡献，不保证 alpha 始终增大，更不强迫无效 CRV 参与。没有阶段训练、val 搜索或人为下限。

## 5. 监督：仅两项新增目标

保留原 `loss_det`、`loss_refer` 和可选 Grab `loss_filter`。原 loss_refer 仍在进入 CRV 前的全部 CRS 候选上计算，不能删除它：CRS 分数被 detach 后，不会从新损失获得梯度。

CRV 标签只使用原始 refer GT，绝不把 teacher 框或其他前景框当 refer GT：

计算标签 IoU 前，把预测框和 GT 裁剪至有效图像边界，与推理 detector_postprocess 的裁剪一致；不是裁剪至 padded 区域。

- IoU>0.5：正候选。
- IoU=0：负候选。
- 0<IoU<=0.5（含恰好0.5）：忽略。
- 原框不相交，但扩张后的 mask 删除超过5%的 refer GT 区域：从可靠负候选中移除。比例按 mask 强度加权，不改变评价 IoU 规则。

L_final：融合 logit 上的正负 pairwise logistic ranking：

```text
softplus(margin - final_logit_positive + final_logit_negative)
```

对所有合法正负 pair 求图内均值，再按有效图求均值。没有合法正负双方时，该图为可反向的零，不能凭空监督 alpha。所有正确 proposal 都是正样本，不人为只保留一个，也不让多个正确候选相互竞争。该项替代 v1 的最终 BCE，不是额外增加第三项损失。

L_cf：同图正负 CRV evidence 的轻量 margin ranking：

```text
max(0, margin - evidence_positive + evidence_negative)
```

正集合含 Top-K 正预测框及一份精确 refer-GT 干预；负集合只有 Top-K 的可靠负候选。margin=0.5。先按图内合法 pair 求均值，再按有效图求均值；没有 pair 时为可反向的零。Top-K 没正确预测框时，精确 GT 仍能与可靠负候选提供辅助训练。GT 不成为输出候选。

```text
loss_crv = 0.5 * (L_final + 0.1 * L_cf)
total    = 原基线损失 + loss_crv
```

默认新增有效权重：L_final=0.5，L_cf=0.05。日志 `crv_final`/`crv_cf` 是未加权诊断值，不重复加进总 loss。`loss_crv` 是已加权总项。`crv_alpha` 是门控均值，`crv_pairs` 是有合法正负候选、能够训练门控的图像比例（0至1）；没有正框时 GT 干预仍可训练辅助评分，但不能让该图获得虚假的最终排序标签。没有绝对 support BCE、特征对齐或新增第三项损失。

## 6. 梯度边界

| 损失 | 作用 |
| --- | --- |
| 原 loss_det | 原检测相关参数与主干，保持原逻辑 |
| 原 loss_refer | 原 CRS、refer_adapter、候选特征生成路径及主干 |
| 原 Grab loss_filter | 原 Grab/基线相关参数 |
| L_final | 新独立 CRV 网络 + fusion_gate |
| L_cf | 新独立 CRV 网络；不直接训练 fusion_gate |

全部新损失不更新 BEiT-3、SimFPN、检测 decoder、DetHead、Grab、refer_adapter 或 CRS。基线参数和 CRV 参数分别裁剪梯度，避免新损失改变基线裁剪范数。新增模块初始化使用 fork_rng，不改变基线模块初始化；新评分器 dropout=0，不引入配对噪声或额外随机数消耗。

这保护基线的直接训练路径，不保证最终融合排序不变；排序变化正是新模块目的。严格训练复现实验仍需相同数据、随机种子、batch、优化器、硬件和确定性设置。

## 7. 训练与评估

```python
enable_crv = True     # 新独立 CRV
enable_crv = False    # 只运行原 CRS 流程
crv_topk = 8
crv_grid_size = 24
crv_cf_loss_weight = 0.1
```

AerialVG 配置保持开启；AerialSense 保留原来的 False，需要实验时再改 True。Teacher/Grab、学习率、epoch、图像尺寸、数据划分等设置不因本次变更自动改变。

```bash
python tools/train.py configs/aerialvg/PropVG-aerialvg.py

# 两卡单次训练；不修改你现有的 dist_train.sh
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 tools/train.py configs/aerialvg/PropVG-aerialvg.py --launcher pytorch

# 单卡验证，并同时输出 CRS / 融合 / CRV-only 三路 CSV
CUDA_VISIBLE_DEVICES=2 python tools/eval_crv_only.py configs/aerialvg/PropVG-aerialvg.py --checkpoint /path/to/new/top1_best.pth --split val
```

训练只在 train 用 refer GT 及可选 teacher 检测 GT；val/test 不加载 teacher，CRV 推理调用不传任何 GT。GT 只在模型外计算 Top1 Acc@0.5、Top5 Acc@0.5、box mIoU。三项显示百分数；mIoU 是最高分候选框与 refer GT 的 IoU 平均，不是最佳框 IoU，也不是分割 mIoU。

三路评估使用同一候选集合和同一次前向。CRV-only 按原始 drop 排序，CRS-only 按原始 CRS logit 排序，融合按 final_logit 排序；sigmoid 概率只用于显示，不因大 logit 被量化成并列概率而丢失顺序。后处理额外输出 `ranking_logits`，训练验证、三路评估与数据集可视化都会优先使用该字段。真正相同的 logit 用稳定排序处理。关闭 CRV 时保持原概率输出和原排序逻辑。

CRS-only 不是另训的基线模型。K>=5 时与完整候选集合上的同一 logit 排序一致；与历史概率排序对比时，还要考虑浮点量化并列。K<5 时 Top5 对照受候选截断影响。多卡训练时 val 仍采用原分布式验证流程，最后 test 只在主进程单卡执行；现有分布式 val sampler 的末尾补齐不属于 CRV，本次未改变该逻辑。

checkpoint 的 `independent_version=2`。旧 CRV（包括独立v1）与本版训练/评分语义不兼容，版本标记拒绝静默混用。三路评估还会拒绝缺少 CRV 权重的纯基线 checkpoint，避免把随机初始化模块当作训练结果。不要把旧 optimizer/CRV 权重 resume 到这版。使用原 BEiT-3 初始化从新实验训练；如需历史模型，请使用相应实验里的源码快照/删除前备份。

## 8. 代码入口与功能验证

- `propvg/models/heads/crv.py`：唯一 CRV 类、均值 mask、独立图文网络、门控与两项损失。
- `propvg/models/heads/REFHead.py`：原 CRS 不变，训练收集 refer GT，推理无 GT，只接最大检测特征。
- `propvg/apis/train.py`：累加新 loss、分组梯度裁剪、诊断日志。
- `tools/check_crv.py`：功能、初始化、坐标、监督规则、梯度与 DDP 检查。
- `tools/smoke_crv.py`：真实数据和 BEiT-3 前向、头部反向、无 GT 推理，不启动正式训练。
- `tools/eval_crv_only.py`：当前独立 CRV 三路评价，无旧模式兼容。

当前版本验证结果写入 `reports/independent_crv_v2_*.json`；不把旧v1报告当作本版证据。功能测试不证明 grounding 精度提升，诊断采用 no_grad 主干时也不代表完整训练的显存消耗。检查包括门控正确/错误梯度方向、大 drop 非零梯度、变换后证据差距、精确边界面积、logit 量化并列、后处理及评价、基线梯度隔离与两rank同步。

2026-09-28 本版检查结果：

- [CPU 检查](reports/independent_crv_v2_cpu.json)：28项中27项通过，1项两卡检查按预期跳过，无失败。
- [GPU + 两卡 NCCL 检查](reports/independent_crv_v2_gpu_ddp.json)：28项全部通过；两rank各执行3个优化器步，确认同步。覆盖旧 checkpoint 拒绝和新 checkpoint 单次加载。
- [AerialVG 真实数据检查](reports/independent_crv_v2_aerialvg_smoke.json)：BEiT-3 前向、真实头部反向、CRV 梯度隔离、验证无 GT 输入均通过；256候选输入，8个输出框。
- [AerialSense 真实数据检查](reports/independent_crv_v2_aerialsense_smoke.json)：同样通过；128候选输入，8个输出框。只在检查进程临时开启 CRV，不修改配置开关。

两份真实数据检查中的随机初始化候选未命中 refer GT，因此 `crv_pairs=0`、`crv_final=0`，精确 GT 的辅助项仍为非零；这是监督规则的预期结果，不代表门控已学会。未启动正式训练、未保存训练 checkpoint，不能据此声称准确率提升。
