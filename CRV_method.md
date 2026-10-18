# 不确定性引导的独立反事实候选验证（CRV）

**当前版本：** schema_version=3，2026-10-18。仅用CRS重复框分组后的熵决定是否采用CRV，并以熵学习融合权重。不再维护多信号质量门控，不复用CRS评分器，不影响主干和CRS训练。

## 1 动机与本次简化

CRS依靠候选与完整文本的正向匹配进行排序。航拍图像中外观相似的目标可能获得相近分数；CRV补充判断：删除哪个候选的局部证据，会使场景与完整描述的匹配下降更多？

但CRV不应无条件改动已有结果。本次分析的旧复杂版在4723条样本中改对28条、改错30条；门控虽学到了不同权重，却没有产生净收益。这不证明“改用熵阈值后必然提升”，因此本版的目的首先是减少控制变量，恢复清晰、可验证的不确定性引导方案。

本次保留二维均值填充、原/干预场景配对评分、最终定位监督和梯度隔离；删除跨检测解码层框读取、绝对分数/分差/组数/稳定性/反事实统计等门控输入，以及两个可学习温度。原14→64→1门控有1025个参数，现改为1→1线性层，仅有两个可学习标量。组数仅保留为诊断，不参与门控。

仍保留概率分布融合，不恢复旧 `z+alpha*tanh(d)` 的有界logit残差，避免同时引入“最大修正不足”的另一个变量。这不是按字节恢复最早版本，而是简化回仅由不确定性引导的流程。

## 2 总体数据流与维度

记批大小B、候选数N、完整文本长度L、重排数量K。默认K=5、评分通道D=128、池化网格G=24，文本长度通常补齐到64。

```text
图像 + 完整描述
       ↓
BEiT-3 → 原512通道SimFPN → 检测侧256通道 → Deformable DETR
                                               ↓
                                     可选Grab → refer_adapter → 可选BCEM → CRS
                                                                              ↓
                                             框/视觉/文本/CRS分数全部detach
                                                                              ↓
                                               原Top-Nu重复框分组 → 归一化熵U
                                                                              ↓
                        U≤0.6：原CRS顺序不变      U>0.6：原Top-K逐框二维干预
                                                                   ↓
                                                  同一个独立评分器读取原场景/干预场景
                                                                   ↓
                                                    d = 原场景分数 − 干预场景分数
                                                                   ↓
                                                   熵U → 两参数线性门控 → alpha
                                                                   ↓
                                                    CRS/CRV候选分布混合 → Top-K内重排
```

| 张量/步骤 | 形状 | 目的 |
|---|---|---|
| 检测侧视觉V | B×256×Hf×Wf，常见144×144 | 保留二维候选区域的局部证据 |
| 文本T | B×L×256 | 使用完整文本，不人工拆解目标词和关系词 |
| 候选框 | B×N×4，有效图像归一化cxcywh | 按预测位置实施干预，不读取GT选择候选 |
| CRS概率/logit | 各B×N×1 | 保留原候选排序并计算不确定性 |
| 单图高分辨率投影 | 1×128×Hf×Wf | 一次投影复用于所有场景 |
| 候选干预mask | K×Hf×Wf | 在高分辨率上删除视觉证据 |
| 原/候选场景 | (K+1)×128×24×24 | 控制评分计算量，不重新编码图像 |
| 训练精确GT场景 | 额外1×128×24×24 | 辅助学习删除真实目标的后果 |
| 分块视觉/文本token | S×576×128 / S×L×128，S≤2 | 各场景独立图文交互 |
| 下降量/两路分布 | 每图各K维 | 为原Top-K提供反事实候选评分 |
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

推理时r=0直接跳过图文评分器、原概率与原顺序保持不变；文本或视觉全部无效也回退。训练时，对所有图文有效样本仍计算反事实辅助监督，使评分器持续学习；但r=0不重排、不修改输出概率、融合loss为0。这不属于阶段训练。

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

无偏置1×1投影256→128与上述线性替换可交换，因此只投影一次。先在原Hf×Wf分辨率执行等价干预，再按有效覆盖率加权自适应池化至24×24；不是先降到24×24再画mask。每次处理最多2个场景，不跨场景混合注意力上下文。

## 5 独立图文评分与下降量

**目的：** 比较同一完整描述在原场景与候选证据被删除后的匹配变化，不复用或更新CRS。

视觉token为576×128，加入二维位置编码；文本256→128后加入一维正余弦位置编码。文本作为Query、视觉作为Key/Value，经两个Transformer Decoder Layer，每层包括文本self-attention、图文cross-attention和512维FFN，4头，dropout=0。文本/视觉padding不参与读取和汇总。

文本交互前后有效均值为t和h，场景有效均值为v，评分MLP为512→128→1：

$$s=f_\theta(V,T)=\operatorname{MLP}([t,h,h-t,v]),\qquad
d_i=f_\theta(V,T)-f_\theta(\widetilde V_i,T).\tag{5}$$

评分是无sigmoid的实值兼容性，不是绝对“文本成立概率”。原场景每图只评分一次；训练额外计算精确refer GT干预，但GT场景不进入预测候选集合。推理完全不读取GT。

## 6 仅由熵学习融合权重

**目的：** 保留可学习融合，但不重新引入复杂质量预测网络。

$$\alpha=r\,\sigma(w\operatorname{sg}(U)+b),\qquad
P_i^b=\operatorname{softmax}_i(\operatorname{sg}(z)/\tau_b),\qquad
P_i^{cf}=\operatorname{softmax}_i(d/\tau_{cf}),\qquad
P_i^f=(1-\alpha)P_i^b+\alpha P_i^{cf}.\tag{6}$$

仅w和b可学习，初始w=0、b=log(.02/.98)。激活样本初始alpha=.02，未激活样本严格为0；不强制alpha下限。门控不读取分差、组数、绝对分数、跨层框、CRV下降量统计或两路首选是否一致。熵输入detach。

tau_b=tau_cf=1，固定为buffer，不是可学习参数。tau_u同样固定。代码以log_softmax和logaddexp计算混合，避免极端分数和门控饱和引起数值问题。非激活样本直接返回原CRS概率，而不是用局部softmax替代原sigmoid。

只在原Top-K内按log(Pf)重排，再接其余候选原顺序。默认K=5，固定候选集合时不改变Top-5命中。所有N个框保留，`ranking_priority`经后处理成为`ranking_logits`，正式评估必须据此排序。Top-K局部概率与外部原sigmoid不能跨集合重新比较。

## 7 监督与梯度

### 7.1 下降量辅助监督

**目的：** 使删除真实目标产生较大下降，删除可靠干扰候选时下降接近0。

标签只取原始refer GT。正候选IoU>0.5；可靠负候选IoU严格为0，且实际羽化mask对GT覆盖≤.01；其余在反事实辅助loss中忽略。覆盖按GT与特征格的分数相交面积计算，避免小目标被漏算。

$$L_{cf}^{pred}=\operatorname{mean}_{p}[1-d_p]_+
+\operatorname{mean}_{n}\operatorname{SmoothL1}(d_n,0)
+[.5-\min_pd_p+\max_nd_n]_+,\qquad
L_{cf}=L_{cf}^{pred}+.1[1-d_{gt}]_+.\tag{7}$$

对应集合缺失时跳过相应项。精确GT干预仅为训练辅助，其0.1不是原定位任务权重。teacher和辅助GT仍只按原配置用于原有监督。

### 7.2 最终定位与保真约束

**目的：** 让可学习门控根据融合后的定位结果学习，鼓励纠正原错样本、抑制改错。

最终定位正集合P采用评估一致的IoU≥0.5，其余为负。允许多个正确候选。令lb=log(Pb)、lf=log(Pf)：

$$M_0=\operatorname{sg}(\max_p\ell_p^b-\max_n\ell_n^b),\qquad
M_f=\max_p\ell_p^f-\max_n\ell_n^f,\tag{8}$$
$$L_{fusion}=r\left(w_{repair}\operatorname{softplus}(.5-M_f)
+2\mathbb1[\text{原CRS正确}][\min(M_0,.5)-M_f]_+\right).\tag{9}$$

原CRS错误时w_repair=2，否则为1。无正确候选或无错误候选时该loss为0，不伪造正候选。低熵样本融合loss也为0；反事实辅助loss仍训练。安全约束不能保证每条样本都不改错，是否有净收益仍需逐条统计。

$$L=L_{base}+.1L_{cf}+1.0L_{fusion}.\tag{10}$$

日志`loss_crv_cf/loss_crv_fusion`已乘外层权重。原loss_det/loss_refer等不变。CF loss仅训练独立评分器；融合loss训练独立评分器和两个门控标量；均不更新CRS、主干、Grab或BCEM。

入口视觉、文本、框、CRS分数全部detach，不读取上一解码层框。保留独立CPU随机种子1717及初始化RNG恢复、评分器dropout=0、基础模型与CRV独立梯度裁剪。FP32受控回归检查隔离；共享AMP溢出跳步等情况未验证，不宣称FP16绝对不干扰。

## 8 配置、兼容与运行

两份当前配置新增/保留以下字段，其他实验设置不改：

```python
enable_crv = True  # False完全走原流程；保留用户当前各配置的开关状态
crv_topk = 5
crv_grid_size = 24
crv_uncertainty_threshold = 0.6
# model.head.crv内：
# uncertainty_threshold=crv_uncertainty_threshold
# uncertainty_topn=5, duplicate_iou=.8, uncertainty_temperature=1.
# gate_initial_probability=.02, base_temperature=1., cf_temperature=1.
```

原有命令不变。CLI覆盖应修改实际展开的head字段，而不只改顶层变量：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_crv=True model.head.crv.uncertainty_threshold=0.6
```

不开CRV时不构建新分支、不新增loss、不改变原排序。当前AerialVG配置的enable_crv=True保持用户设置；AerialSense也保留其当前开关。数据来源、预训练、Grab、BCEM和分割开关不跟随CRV自动改变。

本版schema_version=3，与旧v1/v2门控不兼容，不能直接resume或把旧CRV权重当成本版测试。旧实验可用其日志目录内归档代码复现，归档未修改。从不含CRV的基线开始可用原`--finetune-from`；这不是阶段训练要求。常规load/resume会检查CRV版本和完整参数，防止静默使用随机分支。

## 9 日志、验证与局限

新增诊断保留alpha、active、uncertainty、group_count、rank_changed、corrected、harmed、reachable、rejected_negatives；删除stability和学习温度日志。

- active：图文有效且U>阈值的样本比例；alpha：包含未激活样本0权重的全批均值。
- group_count只用于核查重复框分组，不是额外门控输入；rank_changed统计实际首选变化。
- corrected/harmed/reachable在训练中按有效有GT样本平均；推理时GT相关字段为0占位，需在模型外逐条对比原CRS与融合结果。
- `crv_samples`保留场景分数、下降量、两路/融合分布、单维熵门控输入等诊断；低熵推理样本标记skipped，不存在场景评分字段。

验证入口：`PYTHONPATH=. python tools/test_crv.py`。覆盖分组熵、0.6严格边界、低熵跳过/训练辅助、仅熵门控、固定温度、配对评分几何、负样本覆盖、监督、分布数值稳定、Top-K保护、GT推理隔离、基础初始化/随机流/loss/梯度/一步Adam更新一致性，以及Prop和Grab+BCEM推理。

完整训练收益尚待验证。阈值方案可能漏掉低熵错误；参考物体被删除、融合两路尺度及特征空间干预不能完全抹除上下文证据等局限仍存在。本次仅做用户要求的简化，不声称已经解决独立评分器的语义判别能力问题。旧实验28改对/30改错不是本版结果。

本次19项CRV单元/集成测试及合计92项相关回归通过。两进程CPU/Gloo测试两步通过：rank0激活CRV、rank1低熵跳过且无GT，所有参数更新后同步。未启动GPU训练或CUDA/NCCL验证。

原CRS整数mask池化的历史测试失败仍保留，不在本次修改范围内。
