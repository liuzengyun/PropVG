# 不确定性引导的独立反事实候选验证（CRV）

**当前版本：** schema_version=6，2026-10-20。仅用CRS分组熵决定是否采用CRV；对每个候选先裁48×48局部视觉区域，再将原始/干预局部场景池化到24×24，由同一个独立图文评分器分别输出原始支持与residual support，取真实分数差作为重要性；以可学习权重修正原始CRS logit。支持配置启训epoch，不复用CRS评分器，不向主干或CRS反传新增梯度。

## 1 动机与本次简化

CRS依靠候选与完整文本的正向匹配进行排序。航拍图像中外观相似的目标可能获得相近分数；CRV补充判断：删除哪个候选的局部证据，会使场景与完整描述的匹配下降更多？

但CRV不应无条件改动已有结果。v3熵引导实验在4723条样本中改对13、改错6，净增7条；激活子集CRV单独排序16.51%，低于CRS的19.19%，候选分布平均熵0.9915。说明分支已生效，但反事实排序仍弱；这些是旧模型结果，不是本版的性能。

本版保留二维均值填充、精确GT干预、熵门控、独立CRV定位排序监督、最终融合监督和梯度隔离。本次恢复原设计的两项含义：评分器对原始与mask场景各输出一个支持分，以删除前后支持下降作为重要性；融合直接在原CRS分数上加入可学习的反事实修正，不再对两路softmax分布取平均。旧的特征配对比较MLP已删除。门控仍为1→1线性层，仅两个标量，不重新加入多信号质量网络。

保留 `crv_start_epoch`，先按原任务训练CRS，再启用CRV。整个过程仍是一次训练运行，优化器和学习率计划不重建，CRS在启用CRV后仍照常训练，并不是自动冻结CRS。v5引入的候选中心48×48局部读取继续保留，原Top-5范围、mask扩张/羽化规则及24×24池化网格均不改。本次没有改CRV学习率计划、启训epoch、数据划分或其他模块的开关。

## 2 总体数据流与维度

记批大小B、候选数N、完整文本长度L、重排数量K。默认K=5、评分通道D=128、局部裁剪边长C=48个特征格、池化网格G=24，文本长度通常补齐到64。

```text
图像 + 完整描述
       ↓
BEiT-3 → 原512通道SimFPN → 检测侧256通道 → Deformable DETR
                                               ↓
                                     可选Grab → refer_adapter → 可选BCEM → CRS
                                                                              ↓
                                             框/视觉/文本/CRS分数全部detach
                                                                              ↓
                                       启训epoch前：直接回退CRS；启训后进入下述流程
                                                                              ↓
                                               原Top-Nu重复框分组 → 归一化熵U
                                                                              ↓
                        U≤u₀：原CRS顺序不变      U>u₀：原Top-K逐框裁48×48并二维干预
                                                                   ↓
                                                  原/干预同窗池化24×24 → 共享独立图文评分器
                                                                   ↓
                                                  原支持s₀ᵢ − residual support s⁻ᵢ → dᵢ
                                                                   ↓
                                                   熵U → 两参数线性门控 → alpha
                                                                   ↓
                                                    原CRS logit + alpha×dᵢ → Top-K内重排
```

| 张量/步骤 | 形状 | 目的 |
|---|---|---|
| 检测侧视觉V | B×256×Hf×Wf，常见144×144 | 保留二维候选区域的局部证据 |
| 文本T | B×L×256 | 使用完整文本，不人工拆解目标词和关系词 |
| 候选框 | B×N×4，有效图像归一化cxcywh | 按预测位置实施干预，不读取GT选择候选 |
| CRS概率/logit | 各B×N×1 | 保留原候选排序并计算不确定性 |
| 单图高分辨率投影 | 1×128×Hf×Wf | 一次投影复用于所有场景 |
| 候选干预mask | K×Hf×Wf | 在高分辨率上删除视觉证据 |
| 同窗原/干预局部场景 | 每路K×128×48×48 | 仅比较候选周围相同空间范围 |
| 池化后的原/干预场景 | 每路K×128×24×24 | 保留局部细节、控制评分计算量 |
| 训练精确GT场景 | 额外一对128×48×48→128×24×24 | 原/干预均以精确GT为中心裁剪 |
| 分块视觉/文本token | S×576×128 / S×L×128，S≤2 | 各场景独立图文交互 |
| 图文汇总表示 | 每场景512维，即4D | 汇总交互前/后文本及视觉，不跨场景拼接 |
| 原/干预支持分 | 每路K个标量，训练有GT时每路再多1个 | 同一评分器输出两个可比较的局部支持实值 |
| 候选重要性/两路分布 | 每图各K维 | 为原Top-K提供反事实候选评分 |
| 熵/alpha | 每图各1个标量 | 唯一的路由与门控依据 |
| 输出排序优先级 | B×N×1 | 保留所有候选，只允许Top-K内部交换 |

原共享512通道FPN不变。CRV只读取检测侧256通道，不改变分割、Grab、BCEM、teacher、数据划分或基础loss权重。

## 3 仅用分组熵决定是否采用CRV

### 3.1 选取高响应候选

**目的：** 只处理基础模型已经认为可能是目标的候选，不对所有N个框构造场景。

以原CRS sigmoid概率稳定降序排列，选原Top-K用于干预，取min(K,N)。另取原Top-Nu用于熵估计，默认Nu=5。两者均不使用GT。

### 3.2 仅为熵估计合并重复框

**目的：** 避免同一物体的多个重叠proposal被误认为多个不同目标在竞争。

按CRS分数贪心保留代表框；与已有任一代表框IoU≥0.8时不新增代表。每组使用首次出现的最高分候选，不累加组概率。该操作不删除检测候选、不改变干预Top-K，也不是新增检测NMS。

### 3.3 归一化熵与明确的路由规则

**目的：** 仅在原候选评分较分散时允许反事实修正，低歧义样本保持基础排序。

设代表数为M、代表CRS原始logit为z，固定温度tau_u=1：

$$q_j=\operatorname{softmax}_j(z/\tau_u),\qquad
U=-\frac{\sum_jq_j\log q_j}{\log M},\qquad
r=\mathbb1[U>u_0].\tag{1}$$

M<2时U=0。默认u0=0.6，严格大于才激活，等于0.6不激活。熵不是校准过的错误概率；低熵也可能错误，高熵也可能正确。0.6是可改超参数，不是理论最优点，不按测试GT选择每条样本的阈值。

启训节点之前，训练和验证都完全回退CRS。启训之后，推理时r=0跳过图文编码/比较，原概率与原顺序保持不变；文本或视觉全部无效也回退。训练时，对所有图文有效、有refer GT的样本计算CRV自身监督，包括低熵样本；但r=0不重排、不修改输出概率、融合loss为0。

## 4 二维视觉干预

### 4.1 将框映射到考虑padding的空间坐标

**目的：** 避免干预区域与图上框错位。

预测cxcywh转为xyxy并裁剪到有效图像[0,1]。对特征格(u,v)，设有效图像为(H,W)、padding画布为(Hp,Wp)：

$$x_{uv}=\frac{(u+.5)W_p/W_f}{W},\qquad
y_{uv}=\frac{(v+.5)H_p/H_f}{H}.\tag{2}$$

padding位置由原视觉mask排除；不使用padding宽高直接归一化GT或候选框。

### 4.2 扩张与羽化

**目的：** 覆盖候选边缘证据，减少硬边界伪影。

框中心为c、半宽高为a，令rho=max(|x-cx|/ax, |y-cy|/ay)，t=clip((rho-1.5)/(2-1.5),0,1)：

$$M_i(u,v)=\tfrac12(1+\cos(\pi t))A(u,v).\tag{3}$$

1.5倍框内完全替换，1.5～2倍羽化，2倍外不变。A为有效位置，零面积框mask为0。

### 4.3 每通道以整幅有效图像的均值替换

**目的：** 删除局部视觉差异，而非置零，也不把256个通道混成一个数。

$$\mu_c=\frac{\sum_{uv}A_{uv}V_{cuv}}{\sum_{uv}A_{uv}},\qquad
\widetilde V_{i,cuv}=(1-M_i)V_{cuv}+M_i\mu_c.\tag{4}$$

无偏置1×1投影256→128与上述线性替换可交换，因此整图只投影一次。填充值仍由**整幅有效图像**计算，不改为局部窗口均值。二维干预在原特征格分辨率进行，不是先降到24×24再画mask。

### 4.4 候选中心48×48局部裁剪，再池化24×24

**目的：** 让同一个评分器比较同一候选附近的证据变化，减少无关远处背景，并避免将整幅144×144图压缩为24×24导致小目标细节被过度稀释。

候选中心由有效图像坐标映射到padding特征图：cx_f=cx·W·Wf/Wp、cy_f=cy·H·Hf/Hp。窗口左上角取floor((cx_f,cy_f)−C/2)，C默认48。48是**特征格数量**，不是原图48像素。窗口不随候选框大小缩放，越界不平移、不拉伸，按无效位置补齐；padding和越界区域都不参与均值、池化或注意力。大框及其羽化范围可能被窗口截断，这是固定局部方案的明确限制，而不是自动扩大窗口。

令R_i为第i个窗口裁剪操作，Pi_A为按有效覆盖率归一化的24×24自适应平均池化：

$$V_i^0=\Pi_A(R_i(V)),\qquad V_i^-=\Pi_A(R_i(\widetilde V_i)).\tag{4b}$$

两路使用完全相同的窗口、padding mask和池化网格。实现只采样窗口内的原分辨率特征并进行等价均值替换，无需复制完整干预图。均值替换与局部裁剪、最后池化的结果与“完整高分辨率干预→裁同窗→池化”一致。可靠负候选的GT覆盖率只统计**窗口内实际被干预的区域**，不把窗口外没有输入评分器的mask计算进去。训练精确GT干预也采用同样规则。

每次编码最多2个场景，原始与干预两路分别分块，不跨场景混合注意力上下文。每个候选现在有自己的一份原局部场景，因此原场景不再每图只编码一次：推理共2K个场景，训练有GT时为2(K+1)个。视觉token仍为每场景576个，不增加到2304个，但场景数量比整图共用原场景更多，**不保证总计算更快**。

## 5 共享评分器输出原支持与residual support

**目的：** 比较同一完整描述在原场景与候选证据被删除后的匹配变化，不复用或更新CRS。

视觉token为576×128，加入二维位置编码；文本256→128后加入一维正余弦位置编码。文本作为Query、视觉作为Key/Value，经两个Transformer Decoder Layer，每层包括文本self-attention、图文cross-attention和512维FFN，4头，dropout=0。文本/视觉padding不参与读取和汇总。

### 5.1 每个场景独立输出一个支持分

**目的：** 恢复“场景在目标证据被删除后，还能在多大程度上支持完整文本”的直接评分接口，而不是让另一个网络任意解释两组特征差异。

文本交互前/后的有效均值为t和r，场景有效均值为v。拼接 `[t,r,r−t,v]` 得到512维表示，经过Linear(512,128)、GELU、Linear(128,1,bias=False)，直接输出支持分。原始与所有干预场景共享全部评分器参数：

$$s_i^0=S_\theta(V_i^0,T),\qquad s_i^-=S_\theta(V_i^-,T).\tag{5a}$$

每个候选使用自己的原局部场景，不能将A候选的原窗口与B候选的干预窗口配对。训练额外评分精确refer GT的一对原/干预场景，但GT场景不进入预测候选。推理完全不读取GT。二维位置编码仍为窗口内相对位置，两路一致；本次未另加原图绝对坐标。

### 5.2 直接计算删除前后的支持下降

**目的：** 候选重要性有明确的“删除后支持下降多少”的含义，不通过额外MLP重新定义重要性。

$$d_i=s_i^0-s_i^-.\tag{5b}$$

删除正确候选应令支持下降较大；删除可靠干扰候选应基本不改变支持。若删除后支持反而上升，d允许为负，不截断为0。相同场景与自身比较时，直接相减自然为0。评分末层不设bias，因为共享输出偏置会在相减中抵消；不另加零干预比较头。

支持分是**未校准的局部图文匹配实值**，不是0～1概率，不额外经过sigmoid。其绝对零点不由本版监督确定，不能再将support=1解释为“文本100%成立”。GT监督的是下降量、下降量的候选排序及最终融合排序，不新增“所有原窗口支持都为1/所有干预窗口支持都为0”的BCE标签。不同候选的局部窗口未必包含完整描述所需的证据，强加这种绝对标签也不合理。

若所有候选共用同一个原始支持标量，它会在softmax(d)的候选排序中抵消。本版每个候选有自己的原局部窗口，s0_i并非公共常数，因此一般不会抵消；但这也意味着它判断的是局部支持，不等于最初完整场景的全局语义成立性。测试同时覆盖公共标量确实抵消及局部原支持具有排序梯度，不做相反承诺。

## 6 可学习反事实权重直接修正原CRS分数

**目的：** 保留原CRS对候选的直接评分，反事实只作为附加修正；不再用两路归一化概率的平均替代原分数。

$$\alpha=r\,\sigma(w\operatorname{sg}(U)+b),\qquad
z_i^f=\operatorname{sg}(z_i)+\alpha d_i,\qquad
p_i^f=\sigma(z_i^f).\tag{6}$$

仅w和b可学习，初始w=0、b=log(.02/.98)。激活样本初始alpha=.02，未激活样本严格为0；不强制alpha下限。门控不读取分差、组数、绝对分数、跨层框、CRV下降量统计或两路首选是否一致。熵输入detach。

z为原CRS的sigmoid之前logit，不是softmax概率；没有对CRS做额外温度缩放。d>0提升候选，d<0降低候选。零下降时严格保留原logit，即使alpha接近1也不丢弃CRS。等价地，原CRS odds被乘以exp(alpha·d)，实现可学习的反事实加权。最终逐候选sigmoid不要求Top-K概率和为1。

删除旧base_temperature；cf_temperature=1只用于独立CRV排序loss及诊断softmax，不进入融合。tau_u同样固定。非激活样本直接返回原CRS概率和原排序，不以任何局部softmax代替原sigmoid。新残差融合不保证一定比v5纠错更多，alpha仍可能很小，支持下降也可能不可靠，需实验验证。

只在原Top-K内按z_f重排，再接其余候选原顺序。默认K=5，固定候选集合时不改变Top-5命中。所有N个框保留，`ranking_priority`经后处理成为`ranking_logits`，正式评估必须据此排序；不能擅自按更新后的sigmoid在全部N个候选中重新选Top-K。

## 7 监督与梯度

### 7.1 重要性锚定与独立CRV排序监督

**目的：** 使删除真实目标产生较大下降，删除可靠干扰候选时下降接近0。

标签只取原始refer GT。正候选IoU>0.5；可靠负候选IoU严格为0，且实际羽化mask对GT覆盖≤.01；其余在反事实辅助loss中忽略。覆盖按GT与特征格的分数相交面积计算，避免小目标被漏算。

辅助锚定仍使用上述严格正负/忽略集合：

$$L_{anchor}=\operatorname{mean}_{p}[1-d_p]_+
+\operatorname{mean}_{n}\operatorname{SmoothL1}(d_n,0).\tag{7a}$$

独立排序采用评估一致的正集合P_loc={i:IoU>=0.5}：

$$L_{rank}^{cf}=\operatorname{logsumexp}_i(d_i/\tau_{cf})
-\operatorname{logsumexp}_{i\in P_{loc}}(d_i/\tau_{cf}),\qquad
L_{cf}=L_{anchor}+L_{rank}^{cf}+.1[1-d_{gt}]_+.\tag{7b}$$

**独立排序的目的：** 直接最大化正确候选的总概率质量，不需要CRS分数，也不乘融合alpha。即使alpha很小，共享支持评分器也获得定位梯度。允许多个正确框；Top-K没有正确框时该排序项为0，精确GT辅助仍可训练。全为正确框时排序项自然为0。此处loss使用softmax不等于推理时混合两路softmax概率，二者职责不同。

该项替换旧pair hinge，不是再叠加同类pair loss，`pair_margin`参数随旧项删除。最终仍只输出两个CRV loss字段。IoU=0.5及0<IoU<0.5不会用于“删除可靠干扰物应无变化”的锚定，但会按定位评价标准参与独立排序；两个监督职责不同。

对应集合缺失时跳过相应项。精确GT干预仅为训练辅助，其0.1不是原定位任务权重。teacher和辅助GT仍只按原配置用于原有监督。

### 7.2 最终定位与保真约束

**目的：** 让可学习门控根据融合后的定位结果学习，鼓励纠正原错样本、抑制改错。

最终定位正集合P采用评估一致的IoU≥0.5，其余为负。允许多个正确候选。基础与融合间隔直接使用原logit z及残差修正后的z_f：

$$M_0=\operatorname{sg}(\max_p z_p-\max_n z_n),\qquad
M_f=\max_p z_p^f-\max_n z_n^f,\tag{8}$$
$$L_{fusion}=r\left(w_{repair}\operatorname{softplus}(.5-M_f)
+2\mathbb1[\text{原CRS正确}][\min(M_0,.5)-M_f]_+\right).\tag{9}$$

原CRS错误时w_repair=2，否则为1。无正确候选或无错误候选时该loss为0，不伪造正候选。低熵样本融合loss也为0；反事实辅助loss仍训练。安全约束不能保证每条样本都不改错，是否有净收益仍需逐条统计。

$$L=L_{base}+.1L_{cf}+1.0L_{fusion}.\tag{10}$$

日志`loss_crv_cf/loss_crv_fusion`已乘外层权重。原loss_det/loss_refer等不变。CF loss包含单位权重的锚定及独立排序、0.1精确GT辅助，外层整体仍为0.1；它训练共享支持评分器。融合loss外层仍为1，训练共享支持评分器及两个门控标量。均不更新CRS、主干、Grab或BCEM。本次不新增loss字段、不改变这些权重。

入口视觉、文本、框、CRS分数全部detach，不读取上一解码层框。保留独立CPU随机种子1717及初始化RNG恢复、评分器dropout=0、基础模型与CRV独立梯度裁剪。FP32受控回归检查隔离；共享AMP溢出跳步等情况未验证，不宣称FP16绝对不干扰。

## 8 配置、兼容与运行

两份配置保留以下字段，其他实验设置不改。下列值仅为示例，运行时以实际配置为准；本次不替用户调整启训epoch或其他开关：

```python
enable_crv = True  # False完全走原流程；保留用户当前各配置的开关状态
crv_topk = 5
crv_crop_size = 48  # 最大检测特征图上的局部窗口边长，特征格单位
crv_grid_size = 24
crv_uncertainty_threshold = 0.5  # 示例；构造默认仍为0.6
crv_start_epoch = 1  # 示例；1从第一轮启用，25则前24轮只用CRS
# model.head.crv内：
# uncertainty_threshold=crv_uncertainty_threshold
# start_epoch=crv_start_epoch
# crop_size=crv_crop_size, grid_size=crv_grid_size
# uncertainty_topn=5, duplicate_iou=.8, uncertainty_temperature=1.
# gate_initial_probability=.02, cf_temperature=1.  # 后者只用于独立排序loss/诊断
```

原有命令不变。CLI覆盖应修改实际展开的head字段，而不只改顶层变量：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_crv=True \
    resume_from=None load_from=None finetune_from=None
```

不开CRV时不构建新分支、不新增loss、不改变原排序。两个配置均保留用户现有enable_crv和其他开关，不自动开启实验。上例清空任务checkpoint的加载选项，只是从头训练的示例；BEiT-3预训练仍由原配置控制，未被清空。若需临时覆盖启训epoch，追加model.head.crv.start_epoch=25；不追加则读取当前配置，不替换成文档示例值。数据来源、Grab、BCEM和分割开关不跟随CRV自动改变。

本版schema_version=6，支持评分头与v5不同，融合语义也改变，不能直接load/resume旧CRV checkpoint后当成新实验。v1～v5均会被兼容性检查拒绝。本次没有改旧实验归档和分析报告；包括20261020_001919的旧模型应使用其归档代码及配置复现。从不含CRV的基线开始可用原`--finetune-from`，同时清空冲突的resume_from/load_from。本版同配置checkpoint可正常resume。关闭CRV时旧基线模型不受该检查影响。

### 8.1 启训epoch、验证与断点行为

**目的：** 在候选和CRS尚不稳定时不训练CRV，也不让随机CRV污染早期验证与模型选择。

训练runner已有的0-based epoch经targets['epoch']传到CRV，内部转为1-based。配置22时，1～21轮不运行CRV编码、mask或比较，不修改概率、不新增排序键，两个CRV loss为常数0，所有CRV参数梯度为None。即使使用AdamW，CRV参数也不会在此期间因weight decay更新。

第22轮开始，执行CRV自身监督和符合熵条件的融合。CRS及主干继续接受原loss训练，没有被冻结；CRV始终detach上游。优化器从一开始就包含CRV参数，无需在中途重建参数组、优化器或调度器。多卡沿用原runner的find_unused_parameters=True，以处理启训前未使用的CRV参数。共享原head学习率计划，不自动给CRV重置或增加学习率；若启动较晚，学习窗口和剩余学习率可能有限，应结合总轮数配置。

`last_training_epoch`为持久buffer，记录最近进入训练前向的实际epoch，并与模型权重一起保存。验证/测试只有当它达到配置start_epoch时才使用CRV；未进入训练的新模块或启训前保存的最佳模型会回退CRS。载入早期最佳checkpoint不会被错误地当作已启用CRV；断点恢复后按实际epoch继续判断。该buffer不是“优化器已更新次数”的证明。

构造函数省略start_epoch时默认1，正常训练从第一轮启用；若设置大于1但直接调用训练forward时未传epoch，会明确报错，避免静默错过启训。独立评估不需要手动传epoch，以checkpoint中的buffer为准。`start_test`只控制额外逐轮test日志，与`crv_start_epoch`相互独立，不写start_test仍走原测试逻辑。

## 9 日志、验证与局限

新增诊断保留alpha、active、uncertainty、group_count、rank_changed、corrected、harmed、reachable、rejected_negatives，并增加crv_ready。ready=0表示尚未达到启训节点，其余CRV指标为0占位，不计算熵；ready=1后active才表示熵路由比例。

- active：图文有效且U>阈值的样本比例；alpha：包含未激活样本0权重的全批均值。
- group_count只用于核查重复框分组，不是额外门控输入；rank_changed统计实际首选变化。
- corrected/harmed/reachable在训练中按有效有GT样本平均；推理时GT相关字段为0占位，需在模型外逐条对比原CRS与融合结果。
- `crv_samples`提供逐候选original_support、masked_support（residual support）、importance及兼容名drops，严格满足importance=original_support-masked_support；同时输出base_logits、fused_logits及crop_origins。删除v5的pair_reference_score/pair_scores及错误语义的fused_log_probabilities。
- base_distribution/cf_distribution/fused_distribution仅为离线比较保留的诊断softmax，**不参与融合和最终输出分数计算**。单维gate_statistics仍只含熵。crop_origins为原特征图上窗口的整数(x,y)左上角，允许负数。低熵推理标记skipped；启训前标记reason=before_start_epoch。crv_last_training_epoch可检查启训状态。

验证入口：`PYTHONPATH=. python tools/test_crv.py`。检查支持下降方向、同一场景下降为0、逐候选原支持配对、公共标量抵消、原CRS logit残差融合、零下降不改分、门控饱和数值稳定性、独立排序不依赖alpha、启训前AdamW不更新、head传递epoch、checkpoint恢复及v1～v5拒绝。原有裁剪、几何、padding、梯度隔离、Top-K、Prop和Grab+BCEM检查保留。

完整训练收益尚待验证。阈值可能漏掉低熵错误；参考物体被删除、空间池化削弱小目标、相近框删除证据相似、特征空间干预不能完全抹去上下文等局限仍存在。固定48×48还会丢失远处参照物，截断大目标及其干预区域；本版只能描述局部图文支持变化，不能等同于完整场景的语义成立性。恢复标量支持差值与残差加权不保证自然获得语义因果能力，也不保证性能提升；现有loss仍可能利用非语义线索，需用干预与文本敏感性等对照验证。

本版32项CRV测试及合计114项相关回归通过（不包括下述既有CRS池化失败）。覆盖256×144×144真实通道/空间尺寸训练前向、支持分差、同窗零干预、原logit残差公式、最终sigmoid输出、Top-K以外分数不变、门控纠错/保真梯度、关闭alpha时独立评分器仍有梯度，以及Prop/Grab+BCEM基础loss、梯度、Adam更新与随机数不变。CPU/Gloo两进程连续4轮检查通过：启训前AdamW不更新CRV、一进程无GT时仍同步、恢复checkpoint后启训状态正常。编译及git diff --check通过。未启动正式训练或GPU/CUDA/NCCL检查。

原CRS整数mask池化的历史测试失败仍保留，不在本次修改范围内。
