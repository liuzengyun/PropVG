# 面向航拍外观混淆的关系绑定交互残差评分（BIRS）

## 1. 模块定位与动机

**名称：** Binding Interaction Residual Scoring，关系绑定交互残差评分。

本模块置于原 CRS 之后，以预测候选的外观分组组织竞争者，以完整文本引导参考物语义与相对几何的绑定评分，再向 CRS logit 加入组内中心化的残差。不重跑图像编码器或检测解码器，不替换 CRS，不修改预测框，也不向候选特征回写。

航拍密集场景中，多辆车可能具有相似的颜色、类别和尺度。此前候选聚类可视化提示：高分竞争者可能聚集在相似的特征簇内；这只是少量样本上的诊断，不等同于对整个数据集的统计结论。仅判断“存在黑色 SUV”和“左上方有物体”，无法保证模型判断的是“黑色 SUV 位于左上方”。因此，本模块要检验：**某种参考物语义是否绑定在支持当前目标假设的相对位置上，而不只是两类证据分别存在。**

一个直观例子：两个白车候选周围均有黑车、红车。参考物集合相同不意味着场景配置相同；“黑车在左、红车在右”与“红车在左、黑车在右”应能得到不同的绑定证据。另一方面，当所有参考物的语义都一样时，仅靠交换语义—位置对应不应凭空生成可辨识证据。

### 1.1 与 BCEM、Grab 的分工

- BCEM 在 CRS 之前读取框外空间背景并增强候选表示，BIRS 不访问其空间采样路径，也不读取视觉特征图。
- Grab 可先增强/筛选目标候选；BIRS 的参考支路仍读取未被 Grab 筛除的原检测候选池，避免潜在参考物丢失。
- BIRS 只做评分残差。没有 BCEM、Grab 或 teacher 时也能运行；它们不是开启 BIRS 的前提。

### 1.2 创新性与解释边界

关系建模、候选分组、双线性交互、均值中心化及置换对照本身都不是新概念。本实现的研究假设是：**以外观竞争组为条件，利用语义—相对位置的配对增益，而非独立边际匹配，改进航拍指代评分。**需要实验证明绑定结构而非参数量带来了收益，不能凭代码实现宣称原创性已得到保证。

本模块没有进行真实物理世界干预，也没有识别因果效应；“重绑定对照”是统计关联对照。骨干和 CRS 本来就有隐式上下文交互，不能声称原模型只看外观。

相关工作与后续查新重点：[MAttNet](https://arxiv.org/abs/1801.08186)、[AerialVG](https://arxiv.org/abs/2504.07836)、[CORE-3DVG](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html)、[BindCLIP](https://arxiv.org/abs/2609.23717)。这些工作与关系理解、反事实/组合绑定相关，不应将其已有思想归为本模块首次提出。

## 2. 完整数据流与维度

记批大小为 B，目标候选数 N，原始候选池大小 P，文本长度 L，候选维度 C=256，BIRS 维度 D=128，组数 G=8，参考上限 R=64。不开 Grab 时通常 N=P=256；开启 Grab 后 N 可改变，但不会被 BIRS 再裁成 Top-5/Top-8。

```text
BEiT-3 → 原 SimFPN → 原 Deformable DETR
                         ├─ 检测框 [B,P,4]、检测分类 logits [B,P,2]
                         └─ decoder queries [B,P,256]
                                     ↓
                   可选 Grab → refer_adapter → 可选 BCEM
                                     ↓
                        目标候选 Q [B,N,256]
                                     ├──────── 原 CRS + 完整文本 T [B,L,256]
                                     │                     ↓
                                     │             原 logits ℓ [B,N,1]
                                     ↓
                      stop-grad 球面 K-means 分组 [B,N]
                                     ↓
                       可求导的组原型 [B,G,256] → [B,G,128]

原检测池 → 检测置信度 Top-64 → IoU 去重 → 参考有效掩码 [B,R]
                                     ↓
参考特征 [B,R,256] → 128维投影 → 读取完整文本 → [B,R,128]
                                     ↓
                  组原型条件下的参考语义 H [B,G,R,128]

目标框 × 参考框 → 相对几何 [B,N,R,9]
             → 非线性几何编码 + 文本/组条件 → Ψ [B,N,R,128]
                                     ↓
           实际语义—位置配对 − 均匀重绑定期望 → d [B,N]
                                     ↓
                 有效候选组内中心化 × 可学习强度 → δ [B,N]
                                     ↓
                         ℓ_final = ℓ + δ → sigmoid
                                     ↓
                     全部保留候选按最终得分排序，输出框
```

实际按 `chunk_size=32` 个目标分块计算几何和交互，控制单次算子的临时张量规模；不构造 R×R 的重绑定矩阵。训练时 autograd 仍会保留各块反向所需的激活，不能声称总反向显存降为单个分块大小。主要关系计算量约 O(BNRD)，参考物只读一次文本。没有 sklearn、新外部模型和文本人工拆解。

## 3. 每个操作及其目的

### 3.1 目标候选分组：组织外观竞争，不选择“最高均分簇”

对送入 CRS 的候选特征做 L2 归一化，使用确定性最远点初始化的球面 K-means。聚类分配停止梯度，组原型由组内原特征平均得到；联合训练时，原型平均仍可回传梯度。

分组特征是 `refer_adapter` 后、可选 BCEM 后的特征，不是原始 decoder 特征，也不是 t-SNE 二维坐标。它并非纯外观表征，可能含上下文。G 是上限，实际非空组可以少于 G；组编号没有语义，所有组共享参数。

目的：让相似候选共享同一套参考语义解释，并约束残差主要用于组内消歧。**不根据簇均分丢弃整簇，不根据 GT 分组，不把簇编号当类别。**

### 3.2 参考池：检测置信度 Top-K 后去重

前景置信度沿用当前二分类检测头：`softmax(class_logits)[...,0]`。先取有效原检测框中前 R 个，再按置信度从高到低抑制 IoU≥0.8 的近重复框，不从 Top-R 之外补齐。参考池不由 CRS 的文本匹配得分裁剪。

参考池以固定槽位加掩码保存，不足 R 个时不伪造新参考物。对每个目标候选排除同源 query 与 IoU≥0.8 的自身近重复框；真实的部分重叠仍允许。此处不采用“任何 IoU>0 都删除”，因为真实参考物可能重叠。

目的：减少重复检测造成的证据重复，同时保留文本不直接指向的参考物。框和检测分数的选择始终停止梯度。

### 3.3 完整文本读取与组条件语义

参考特征和文本分别线性投影到 D=128。参考 token 作为 query，对完整有效文本 token 做一次多头交叉注意力，得到文本相关的参考表示 v_j。padding 位置不参与注意力；全 padding 样本不产生残差。

对每个外观组原型 z_g，定义：

$$H_{gj}=\tanh\left(\operatorname{LN}\left(\operatorname{GELU}\left(W_h[v_j;z_g;v_j\odot z_g]\right)\right)\right).$$

目的：相同参考物对不同目标外观组可能具有不同的描述作用。该表示不显式解析主语、颜色词、关系词或参考物标签。

### 3.4 相对几何：必须非线性编码

框从归一化 cxcywh 转为 xyxy，按 `img_shape` 有效图像裁剪，绝不使用 padding 尺寸解释位置。设短边 s=min(W,H)，目标中心 c_i、参考中心 c_j，宽高分别为 w、h。使用 9 维向量：

$$r_{ij}=[\Delta_x,\Delta_y,\Delta_x/d,\Delta_y/d,
\log(1+d),\log(w_j/w_i),\log(h_j/h_i),
\tanh((x_j-x_i)/w_i),\tanh((y_j-y_i)/h_i)].$$

其中前两维为像素中心差除以 s，d 为其欧氏距离；对数宽高比截断到 [-8,8]，分母设置数值下限。图像 y 轴向下，正 Δy 表示参考物更靠下。方向是连续值，不人为添加关系词。

经两层 GELU MLP 后，与“组原型+有效文本投影的平均值”构成条件交互，输出有界的几何嵌入 Ψ_ij∈[-1,1]^D。

**为什么不能只线性编码 dx、dy？** 对同一参考集合，把 `x_j-x_i` 在参考维度中心化会消掉 x_i，导致不同目标候选的几何变得一样。方向归一化、局部尺度 tanh 和非线性 MLP 用于避免这一代数退化；是否学到有用关系仍需检验。

### 3.5 实际绑定与重绑定对照

对目标 i，只在相同的合法参考集合 J_i 内计算所有统计量，M_i=|J_i|。设 g_i 为其组编号：

$$E_i(j,k)=\frac{H_{g_i j}^{\top}\Psi_{ik}}{\sqrt D},\qquad
p_i=\frac1{M_i}\sum_{j\in J_i}E_i(j,j).$$

对参考语义—位置对应做均匀随机置换，其平均得分无需采样即可精确计算：

$$n_i=\frac1{M_i^2}\sum_{j,k\in J_i} E_i(j,k)
=\frac{\bar H_i^{\top}\bar\Psi_i}{\sqrt D}.$$

绑定增益：

$$d_i=p_i-n_i
=\frac1{M_i\sqrt D}\sum_{j\in J_i}
(H_{g_i j}-\bar H_i)^{\top}(\Psi_{ij}-\bar\Psi_i).$$

目的：消除“某种参考语义普遍高分”和“某个位置普遍高分”这两类独立边际因素。若配对分数只有 a_j+b_k+c_i，那么 d_i=0。参考语义完全相同时也不能产生绑定增益。

这不表示所有置换都不合理，也不把它们逐一标为负样本。只有实际目标 GT 监督模型最终该支持哪个候选。少于两个合法参考物时 d_i 和最终残差置零。

### 3.6 残差融合：保留 CRS，而非仅使用新评分

在同组且至少有两个合法参考物的候选中计算均值 μ_g；若组内有效候选不足两个，该组不修正。

$$\delta_i=\beta(d_i-\mu_{g_i}),\qquad
\beta=\beta_{\max}\sigma(a),\qquad
\ell_i^{final}=\ell_i^{CRS}+\delta_i.$$

默认 β 初始 0.1，上界 2.0，通过损失学习。最终概率是 sigmoid(ℓ_final)，用于排序的内部 `ranking_priority` 仅编码最终 logit 的排序，**不是置信度**。

中心化目的：防止给整个外观组统一加分，迫使新增分支提供组内差异；它自身不会生成辨识信息。β 可以减小以抑制无用残差，不强迫模块参与，不承诺能避免所有改错。

残差为零的位置直接保留原概率；整张图残差均为零时保留原评估器的概率排序及 tie 行为。未完成任何有效监督更新的 BIRS 在 eval 下直接回退原 CRS，不以随机权重重排。

## 4. 训练监督、权重与梯度

原 `loss_det`、`loss_refer`、可选 Grab filter / mask / correctness 损失保持原算法与权重。BIRS 新增两项：

| 损失 | 默认权重 | 监督内容 | 目的 |
|---|---:|---|---|
| `loss_birs_final` | 1.0 | 最终融合 logit 对原 Hungarian refer 正负标签的 BCE | 直接检验并优化加残差后的定位结果 |
| `loss_birs_binding` | 0.1 | 同组 GT 正候选与高 CRS 得分、完全不重叠负候选之间的 softplus 排序 | 让绑定分支独立学到目标证据，减少 β 变小导致分支没有学习信号的问题 |

两项日志值已经乘权重，不应再额外乘一次。第一项在有效候选上做均值、再按有效样本平均；没有 referent、原匹配正候选缺失或整个样本无可用残差时跳过新增监督。

绑定排序正样本必须同时是原 Hungarian refer 正标签、IoU≥0.5、可产生有效残差。负样本必须与原 refer GT 的 IoU **恰为 0**、同组且有效，再按原 CRS 得分取前 `negative_topk=8` 个。其他候选不进入这项排序监督，包括 0<IoU<0.5 的候选；没有合法正负对时跳过。正样本仍由 Hungarian 匹配约束，不因 IoU 高就把额外候选改成 refer 正标签。

$$\mathcal L_{binding}=\operatorname{mean}_{(p,n)}\operatorname{softplus}(d_n-d_p).$$

只从 `refer_target_index` 读取原 refer GT；teacher/辅助 GT 继续按原逻辑参与原检测损失，但不作为 BIRS 的目标、参考物标签或关系监督。GT 只在 `loss()` 内读取；前向分组、参考选择、几何、评分和推理均不读取 GT。

### 4.1 联合训练与截断开关

- `detach_inputs=False`（默认）：新增损失可影响 CRS、refer_adapter、BCEM/Grab、检测 query 和骨干的特征路径。预测框、检测分类分数的选择、聚类分配不接收 BIRS 直接梯度；但共享特征的变化仍可能间接影响检测结果。
- `detach_inputs=True`：BIRS 的候选、参考特征、文本、原 CRS logit/概率全部截断，只训练 BIRS。原损失继续训练原模型。梯度裁剪对独立 BIRS 与原模型分别计算，避免通过全局梯度范数间接改变原参数更新尺度。

分组及参考选择没有随机采样；新分支采用独立、恢复现场的 CPU 初始化随机流，不改变已有参数的初始化或 CPU/CUDA RNG。DDP 的两项新增损失按全局有效样本数归一化；无监督的 rank 保留连接所有新参数的零梯度图，避免不一致的分支/collective。

## 5. 配置、训练和旧模型兼容

配置位置：`configs/aerialvg/PropVG-aerialvg.py`。默认如下，**没有自动替你打开新实验**：

```python
enable_birs = False
birs = dict(
    num_groups=8, reference_topk=64, duplicate_iou=0.8,
    model_dim=128, num_heads=4, kmeans_iters=6, chunk_size=32,
    initial_strength=0.1, max_strength=2.0,
    final_loss_weight=1.0, binding_loss_weight=0.1,
    negative_topk=8, detach_inputs=False,
)
# 传入 model.head 的是 enable_birs 和 birs。
```

参数目的：G 控制竞争分组粒度，R 控制参考支路规模，duplicate_iou 用于参考去重及排除自身近重复，D/heads 控制分支容量，kmeans_iters 控制聚类迭代，chunk_size 调整单次算子的临时工作集及调度开销，不改变数学做法、也不保证训练总显存按块数下降；其余为融合强度、监督权重、负样本数及梯度策略。

开启 `enable_birs=True`，保持 `enable_crv=False`、`enable_relation_refinement=False`，为实验设置独立 work_dir 后，沿用原训练命令：

```bash
conda activate propvg
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py
```

若从已训练 baseline 开始新训练，使用 `--finetune-from /绝对路径/top1_best.pth`，不要用 `--load-from` 或 `--resume-from` 冒充已有 BIRS 权重。已有 BIRS 的恢复/测试应使用训练时归档配置与对应 checkpoint。BIRS checkpoint 保存全部原模型和新分支参数，沿用已有保存和最佳 Top-1 选择逻辑。

默认关闭时不构建 BIRS，不新增参数、buffer、损失、DDP collective；旧配置没有 `enable_birs` 时构造函数同样默认 False。旧模型仍按其原归档配置测试，不必补配置。开启时校验新权重完整性、结构版本和推理策略摘要（组数、参考数、去重阈值等），防止相同 tensor 形状但策略不一致被静默加载。纯训练开关 `detach_inputs` 和损失权重不进入推理策略摘要。

## 6. 可验证接口与实验分析

核心代码：`propvg/models/heads/binding_residual.py`；接入：`REFHead._binding_residual()`；训练接入：`propvg/apis/train.py`。单元测试：

```bash
PYTHONPATH=. python tools/test_birs.py
```

按需捕获，不在普通训练时将大张量搬到 CPU：

```python
base = model.module if hasattr(model, "module") else model
with base.head.birs.diagnostics():
    # 正常执行模型前向。
    # REFHead 返回的 extra_dict["birs_samples"] 中可获得以下张量。
    ...
```

记录包括原得分、最终 logit、paired/null/raw_binding、delta、组编号、参考原 query 下标/置信度/框、有效参考掩码及每个目标的合法参考集合。最终候选不做额外过滤；已有后处理继续同步过滤退化框与排序分数。

训练日志新增 `birs_strength`、`birs_eligible`、`birs_delta_abs`、`birs_reference_count`、`birs_corrected`、`birs_harmed`、`birs_rank_changed`、全局有效监督/绑定样本数。corrected/harmed 是当前训练样本上的 Top-1 改对/改错比例，不是验证集准确率，也不是纠错成功概率；DDP 日志按各 rank 的比例平均。

至少验证以下对照，而非只看总 Top-1：

1. 同训练预算的原 CRS vs CRS+BIRS，保持数据划分、teacher、检测损失、图像尺寸、随机种子不变。
2. 净纠正数、改对数、改错数；Top-1/Top-5/mIoU；按组内竞争、参考数、原候选召回率分层统计。
3. `binding_mode="paired"`：只用实际配对、不减重绑定期望。`"crs_control"`：残差和绑定损失强制零，用于接入/额外监督控制；它不是原 baseline，因为联合模式下仍有新增 final BCE。
4. `num_groups=1` 对照取消分组；打乱“参考特征—参考框”对应，但保留边际集合，检验模型是否真正使用正确绑定。
5. 替换为重复相同参考语义后，绑定残差应归零；验证少于两参考时回退；验证不同 padding 不改变几何。
6. 统计 GPU 时间、峰值显存和增参量；比较联合/截断梯度，避免把主干训练变化误认作评分机制本身的提升。

## 7. 局限及失败时的判断

- 特征分组可能不对应外观类别，且组边界随训练变化；组内中心化也不能保证组间绝不发生排名变化。
- Top-64 及去重可能遗漏真实参考物；参考池也可能包含背景候选。模块没有额外目标类别监督，不能保证其都是实体。
- 当前平均配对统计可能稀释少数关键参考物；其优势需通过上述绑定打乱对照证明，不能仅凭公式认定成功。
- 没有显式词语解析或关系标签，方向语义只能从 refer GT 间接学习；无关系描述时应由学习到的残差/强度抑制无用证据，但未保证逐样本精确关闭。
- 标注存在多个同样符合文本的实体时，单框 GT 会把合理候选当负例；BIRS 不能消除这类评测歧义。
- 正候选 IoU 不足、同组无完全不重叠负候选时，绑定损失可能缺少监督，应看 binding_samples，不能只看平均 loss。
- sigmoid 的全局 β 不是逐样本可靠性门控，也不是置信度；当表现下降时先检查配对敏感性与改错数，不应反复增加复杂门控来掩盖分支无辨识力。
- 联合训练且残差很小时，新增 final BCE 近似增加一次 CRS 定位监督，改变有效检测/定位损失比例。必须加入 `crs_control` 对照，不能把重加权带来的收益误认作绑定建模的收益。
- 本实现不保证性能提升、因果正确性或原创性。只有充分消融、误差分析和独立查新才能支撑论文结论。

## 8. 本次实现验证记录（2026-11-12）

- BIRS 独立测试 18 项通过。
- 新分支测试覆盖重绑定精确期望、独立边际消除、合法参考集合一致性、单参考回退、同语义零残差、非线性几何避免平移消失、分块/padding 一致性、梯度路径、损失接入与权重、checkpoint 策略校验、诊断接口以及完整检测头前后向。
- 与本次修改前保存的 `REFHead.py` 快照比较：BIRS 关闭、Grab/BCEM 四种开关组合下，参数、RNG、训练预测/损失、参数梯度、测试预测逐位一致。
- 原 optional pipeline、Grab、BCEM ray、correctness 共 45 项回归测试通过；原 two-stage relation 的 36 项测试通过。
- 空闲第二张 4090 上，以随机 `[1,768,36,36]` 输入、256 queries 运行检测头前向/反向/一次 Adam 优化与测试；覆盖基础、BCEM+截断、Grab+BCEM 三种组合，新分支默认增加 281,601 个参数。随机 Grab 丢失 refer 正候选的情况按设计跳过新增监督，eval 回退；此项不等同于真实数据性能测试。
- 两进程 CPU DDP/Gloo 连续两步：一个 rank 无 refer GT，两个 rank 梯度一致，并与全局有效样本的单进程梯度对齐。
- 没有启动完整训练或跑全量验证集；性能、实际训练吞吐与论文层面的创新性尚未由这些功能测试证明。
