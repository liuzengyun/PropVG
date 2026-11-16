# 航拍视觉指代的文本条件竞争集合与关系绑定细化（BIRS）

## 当前主方案：v3 自适应竞争范围与稀疏反事实关系细化

配置选择 `birs_version="v3"`；v1/v2的结构、参数布局和checkpoint策略仍保留，历史说明位于后文。v3是新结构，不能用v1/v2完整权重冒充恢复训练。关闭`enable_birs`仍沿用原PropVG/Grab/BCEM等流程。

### A. 动机与边界

CRS已能将许多正确目标排进前几名，但外观相似的航拍实例仍可能竞争第一名。本模块在CRS之后确定一个自适应竞争范围，显式读取“周围有什么、在当前目标的什么位置”，再比较真实语义—位置绑定与打乱绑定的支持差异。它不重新预测框、不重新运行图像编码器/检测解码器、不解析文本、不引入外部模型或参考物GT。

“先看外观、再看关系”只是直观解释，CRS/检测特征本来已有上下文，BCEM开启时还有显式环境。两阶段和关系注意力本身不是原创；重绑定也只是特征对应对照，并非真实物理场景编辑或已识别的因果效应。该版是待验证方案，不保证性能提升。

### B. 数据流与每一步的目的

设B为batch、M为目标池大小（常用256，Grab开启可能减少）、P为原检测候选数、L为文本长度（常用64）、C=256、D=128。R=min(64,P)为参考槽位数，去重后用mask标记有效槽位；N_b是每图实际选择数，K=max_b(N_b)，全空时张量保留一个无效占位。

```text
BEiT-3 → 512通道共享SimFPN（不改）→ 原检测解码器
                                      ↓
                    检测queries [B,P,256] / 框 / 前景logits
                         ├─ refer_adapter → 原检测参考池（BCEM前）
                         │         ↓ 检测Top64 FIRST → IoU≥0.8去重
                         │      参考特征[B,R,256] / 框[B,R,4]
                         ↓
                  可选Grab → refer_adapter → 可选BCEM
                         ↓
               目标特征Q[B,M,256] → 原CRS(Q,T)
                         ↓ 原logits/probability [B,M,1]
           Q聚类 → 多个高分种子簇 → 有限搜索 → 连续Top-N_b
                         ↓
             选中目标[B,K,256] / 框[B,K,4] / valid[B,K]
                         ↓
        目标投影 / 参考投影[B,R,128] / 9维相对几何[B,K,R,9]
                         ↓
          关系对编码[B,K,R,128] → 读取完整文本[B,L,128]
                         ↓
             sparsemax选择R条关系 + 1个零值空选项
                         ↓
                真实支持r_f[B,K]
                         ↑ 同一评分器、重新计算注意力
         合法参考物内部置换几何对应 → 对照支持r_cf[B,K,P_cf]
                         ↓
              Δ=r_f-mean(r_cf) → βΔ（不做组内中心化）
                         ↓
       回填全M个logit（集合外δ=0，专供完整定位监督）
                         ↓
      仅选中Top-N_b按最终logit排序并输出；不附加集合外候选
```

1. **自适应竞争范围，目的：不预先假设每图只有固定数量竞争者。** 对有效Q做确定性球面K-means（默认8簇、6轮），聚类只产生离散标签，不计算共享原型。按原CRS概率降序，取前`seed_topk=3`涉及的簇；仅在前`max_targets=32`搜索这些簇的成员。最远排名确定N，至少保留`min_targets=5`，不足时保留所有有效候选。保留连续前缀，因此排名1/3/5/6/7同簇时第2/4名也保留。新增中间成员不递归扩展其簇。种子包含多个簇，避免Top1孤立误判排除另一竞争簇。选择不读GT、不反传梯度。BCEM开启时Q是增强后的候选表示，不能声称是纯外观聚类。
2. **独立参考池，目的：参考物不必像目标。** 原检测分类softmax的前景通道0取Top64，再贪心抑制IoU≥0.8，不从64名外补齐。v3无论Grab是否开启，都使用原解码器的refer_adapter特征，位于BCEM之前；目标侧保留原Grab/BCEM流程。同源query及目标IoU≥0.8参考物排除。框、置信度筛选及来源索引截断梯度。
3. **相对几何，目的：明确物体的空间归属。** 使用既有9维编码：短边归一化dx/dy、单位方向2维、log(1+距离)、log宽高比2维、目标尺寸归一化位移的tanh共2维。框是有效图像归一化cxcywh，转裁剪xyxy后使用`img_shape`，不是padding画布；这是二维图像几何，不是米制三维位置。
4. **候选自身条件，目的：每次明确正在评估谁。** 目标q_i、参考v_j分别线性投影为D维，几何经MLP投影为D维。拼接`[q_i,v_j,g_ij,q_i*v_j]`，经Linear/GELU/LayerNorm编码为关系token。不使用Top-N均值原型、簇均值或不同目标之间的自注意力。
5. **完整文本交互，目的：关系证据按描述读取。** 关系token为Query，投影文本为Key/Value，一层多头交叉注意力（4头，dropout=0）加残差LayerNorm；padding不参与。输出[B,K,R,D]。没有显式“黑车”类别标签或词—框监督，不能保证注意力就是正确参考物识别。
6. **稀疏汇总，目的：避免64个参考平均稀释一两个关键证据。** 每条交互后关系token预测选择logit，额外空选项logit由目标+有效文本平均表示预测。对合法关系与空选项一起做masked sparsemax，产生精确零权重；空选项value固定为0。加权关系表示与目标、句子拼接，经MLP输出一个support标量。空选项允许少用关系，但不保证对无关系文本自动完全关闭。权重非零数可变，不强制固定1或2个。
7. **重绑定对照，目的：控制“有什么”和“有哪些位置”，只破坏对应。** 对每个目标独立在其合法参考集合内部循环置换几何；语义、目标、文本及几何集合不变，不把无效/自身槽位引入。相同评分器重新执行关系编码、文本交互、稀疏选择和support评分，不搬用真实注意力。默认2次对照，偏移由原目标query ID与对照编号确定，训练/测试一致、不使用GT/RNG，合法数≥2时非原样，少量参考时对照可能重复。它不是均匀随机置换的精确期望，也不保证是最强破坏，属于可消融的确定性采样策略。
8. **差值与融合，目的：提取真实绑定相对对照的额外支持，保留第一阶段匹配。**

$$\Delta_i=r_i^{f}-\frac{1}{P_{cf}}\sum_p r_i^{cf,p},\qquad
\ell_i^{final}=\ell_i^{CRS}+\beta\Delta_i,\qquad
\beta=\beta_{max}\sigma(a).$$

β初始0.1，上限2；不再做组内中心化，不增加门控。少于2个合法参考、全padding文本或无效目标时δ=0。刚启用尚无有效训练步的eval也令δ=0，但仍执行自适应前缀选择；因此v3未训练评估不是完整256框的原基线。真实分不必高于置换分；没有强迫每次置换为负例。

### C. 损失与梯度：无独立反事实损失

设v为全部M个有效候选掩码、y为原Hungarian refer标签。两阶段都计算全部有效候选上的BCE，每图除以有效候选数，再按全局有效样本数平均。有效样本只要求原refer GT存在且匹配正候选有效，不要求正候选进入Top-N或有参考物。

$$L=L_{det}+w_{refer}\big[(1-\lambda)L_{CRS}+\lambda L_{final}\big]+L_{其他原有},\quad\lambda=0.5.$$

在REFHead中**替换**原`loss_refer`为`(1-λ)*BCE_CRS*w_refer`，日志名仍为`loss_refer`；新增`loss_birs_final=λ*BCE_final*w_refer`。没有`loss_birs_binding`、没有独立CF损失，不把原完整refer BCE再叠加一次。`loss_det`及已有Grab/filter、mask/correctness等权重不改。CF通过最终定位BCE优化真实与对照共享评分器及β；β很小时评分器梯度也可能较弱，这是待观察限制。

集合外logit保持原CRS值，仍被两阶段BCE监督，二者系数相加为1；没有缩小为Top-N-only BCE。选择/框/置换不求导。`detach_inputs=False`联合优化CRS及共享候选、文本路径（含目标侧BCEM）；True时最终分支截断CRS及输入，仅更新v3，但第一阶段BCE仍正常训练原模型。沿用现有独立梯度裁剪逻辑。DDP固定执行一次有效样本数all_reduce，无监督rank保留参数连接零梯度；没有有效样本时不增加trained_steps。

### D. 只输出Top-N，不处理集合外的排序

训练内部保留[B,M,1]以计算全候选BCE；前向返回`output_mask/candidate_output_mask`。`MIXRefUniModel_OMG.get_predictions_parts`在原退化框过滤后，用同一掩码过滤框、检测分、referring分和排序值，然后才执行可选NMS/rescale。输出是每图实际N个有效框，不把原后缀接回去，训练/验证/测试后处理一致。Top-5在N≥5时正常评估，不足5个有效框时按现有评估取可用候选。既有非v3不提供此掩码，沿用原后处理。

### E. 配置与兼容

```python
enable_birs = True
birs_version = "v3"
birs = dict(version="v3", num_groups=8, kmeans_iters=6,
            seed_topk=3, min_targets=5, max_targets=32,
            reference_topk=64, duplicate_iou=0.8,
            model_dim=128, num_heads=4, chunk_size=8,
            permutations=2, initial_strength=0.1, max_strength=2.0,
            final_mix=0.5, detach_inputs=False, checkpoint_chunks=True)
```

`final_mix`是两阶段定位监督混合系数；`permutations`是置换次数；`checkpoint_chunks`只以重算换训练显存。分块只计算至本batch最大实际N，不固定计算至32；仍有batch padding开销。默认配置保留用户已有teacher、Grab、BCEM等设置，只切换BIRS到v3。建议使用独立work_dir。v1/v2分支不接收v3参数，主配置已按版本构造不同字典，不能仅将旧归档字典的version改成v3。

v3独立checkpoint schema=4，策略摘要含聚类/搜索、参考来源、置换、输出与损失规则；错版本/错关键配置明确报错。新实验可从相同BEiT-3初始化训练；完整baseline初始化用`--finetune-from`，不能把旧BIRS完整权重当作v3 resume。v1 schema=1、v2 schema=3及关闭BIRS的旧模型路径保持。v3目前接入当前AerialVG使用的MIXRefUniModel_OMG，不宣称其他未接入head具有同样输出行为。

启动方式不变：`python tools/train.py configs/aerialvg/PropVG-aerialvg.py`，或现有多卡脚本。未自动启动完整训练。

### F. 可验证接口、日志与实验要求

使用原`with model.head.birs.diagnostics():`，在`extra_dict['birs_samples']`读取CPU诊断：全query索引下的groups、candidate_mask、delta、raw_binding（此处为support差值）、paired/null（真实/平均对照support）、原/最终logits、参考池及合法掩码；`target_indices/target_valid`映射变长目标；`selected_attention[B,K,R+1]`、`selected_control_attention[B,K,P_cf,R+1]`、`selected_support[B,K]`与`selected_controls[B,K,P_cf]`用于比较两条路径。支持分是未校准标量，不是概率。未开启diagnostics不保存所有对照注意力到CPU。

主要日志：`birs_target_count`平均N、`birs_target_cap_rate`达到配置/候选数上限比例、`birs_refer_in_topn`原匹配正候选覆盖、`birs_reachable`IoU≥0.5覆盖、`birs_reference_count`去重参考数量、`birs_null_attention`平均空选项权重、`birs_sparse_reference_count`非零参考数量、`birs_delta_abs`按选中目标平均绝对修正、`birs_strength`、`birs_eligible`选中目标可用绑定比例、`birs_corrected/harmed/rank_changed`训练batch的Top1变化、`birs_supervised_samples`全局有效样本数。日志不能代替完整val诊断。

必须比较：固定N/自适应N；真实关系直接评分/真实减置换；普通注意力/稀疏选择；BCEM开关。比较最终分与同checkpoint原CRS时，应同时报告完整CRS与同一自适应前缀口径，不能将筛选召回损失误认为重排收益。跨独立模型仍需同训练配置、同损失总权重及多种子验证。重点统计CRS错但集合可达的样本中残差朝GT推动比例，而非仅观察总体loss。

潜在局限：种子覆盖多簇可能使N长期取满上限；聚类尾部成员可能是噪声；参考漏检/误检；稀疏选择早期全部偏向空项；网络忽略绑定或关注伪线索；确定性少量置换不充分；无关系/多解描述难以产生有用差值；联合梯度可能干扰原模型；β小导致差值网络学习慢。不能为保证下降而把所有置换强行标成负例。

### G. 实现验证记录

- 119项CPU单元/回归测试通过：`PYTHONPATH=.:tools python -m unittest test_birs_v3 test_birs test_optional_pipeline test_grab_paper test_bcem_ray test_correctness test_two_stage_relation`。含多种子/连续前缀/禁止递归扩簇、变长batch、置换合法集合、稀疏空项、全候选BCE/集合外正例、联合与截断梯度、无GT/全padding/无参考/全退化框、checkpoint与分块、真实REFHead训练和推理、输出裁剪/NMS/rescale/correctness对齐及既有模块回归。
- CUDA小规模模块测试：B=2、256候选、64文本token、C=256/D=128，checkpoint开启，前向、两阶段BCE、反向梯度均为有限数；随机输入选中N=30/29。只证明实现可运行，不代表实际数据N分布或完整模型耗时。
- 两进程CPU Gloo DDP两步通过：一个rank没有GT，固定collective正常，全局监督样本数为1，所有新参数梯度存在且有限，优化后两rank权重逐值一致。
- 上述不是完整数据集训练或性能验证，没有自动启动正式训练，也没有给旧实验模型替换权重。

## 历史版本说明：v1 / v2

## 1. 当前方案与动机

当前通过 `birs_version` 区分 **v1：第一版外观聚类** 与 **v2：第二版Top-N关系细化 + 全候选融合BCE**。默认v2，保持修改前的流程与权重策略。BIRS 为 Binding Interaction Residual Scoring（关系绑定交互残差评分），放在 CRS 之后，不替换 CRS，不修改预测框，不重跑图像编码器或检测解码器。

### 两个版本的精确定义

| 项目 | v1：第一版 | v2：第二版（当前默认） |
|---|---|---|
| 绑定计算对象 | 全部有效候选，通常256个 | 原CRS Top-N，默认5个 |
| 分组 | 确定性球面K-means，默认8簇、6次更新 | Top-N为一个竞争集合，不执行聚类 |
| 原型 | 各簇有效候选特征均值 | Top-N特征均值 |
| 中心化 | 各簇内有效绑定候选分别中心化 | Top-N内有效绑定候选中心化 |
| 最终排序 | 全部有效候选按融合logit全局排序 | 仅重排Top-N内部，外部原顺序不变 |
| 融合BCE候选范围与分母 | 全部有效候选 | 全部有效候选，**不是Top-N-only BCE版** |
| 融合BCE样本条件 | 有有效匹配正候选，且至少一个候选可产生绑定 | 有有效匹配正候选，不要求存在绑定 |
| 绑定排序监督 | 原匹配正候选与同簇IoU=0的hard negatives | Top-N内原匹配正候选与IoU=0的hard negatives |
| 对同checkpoint的Top-5影响 | 允许改变Top-5成员 | N=5时成员不变 |

v1从 `20261112_235348` 归档实现恢复，包括原确定性聚类、全局排序及无绑定时跳过新增BCE的规则，不是把所有候选平均成一组，也不是只保留平均分最高的一簇。两版共享参数结构、参考物选择及绑定公式。BCEM、Grab、teacher等开关独立，切换BIRS版本不会改变它们；不开Grab时，两版参考特征均沿用当前接入方式（BCEM开启则参考特征也经过BCEM），本次没有额外改变参考支路。

下面第3–5节的数据流与Top-N细节主要描述v2；v1对应差异以上表为准。

航拍小目标的细粒度外观信息有限，密集场景中经常存在颜色、类别、尺度相似的实例。第一阶段匹配可以找到“像描述的一组目标”，但不一定确定具体实例。广域俯视/斜俯视图像又允许观察较多物体及其配置，因此参考物的语义与相对位置可能提供补充证据。

核心问题不是分别判断“有黑车”和“左侧有物体”，而是判断“黑车是否绑定在左侧”。实际绑定相对于重绑定对照的增益，用于补充原图文匹配。

**v2为何不用外观聚类？** 外观特征相似不等于竞争当前描述。上一轮 epoch34 诊断中，test 有832条“CRS错误但Top-5含正确候选”的样本，正确候选与错误Top-1同簇仅271条。v2直接用 CRS Top-N 定义文本条件下的竞争集合，避免聚类边界限制绑定排序监督；这不是已经证明v2优于v1的结论，两版保留供实验对照。

Top-N 筛选和中心化本身不是创新；关系绑定与条件竞争集合的组合需要消融和查新验证，不能保证原创或性能提升。这是针对航拍困难的设计，不是只适用于无人机的数学机制；重绑定只是统计关联对照，不是已识别的真实因果效应。

## 2. 与现有模块的关系

- 原 CRS 始终保留，原定位损失照常监督。
- 可选 Grab / BCEM 的原流程保持；BCEM 在 CRS 前增强表示，BIRS 不读取其空间路径或图像特征图。
- Grab 若筛掉部分目标候选，参考支路仍读原检测池的 refer_adapter 特征，不从 Grab 筛选结果裁剪参考。
- 不需要 teacher、关系标签、额外参考物 GT、文本人工拆解或外部模型。
- BIRS 与旧 CRV / two-stage relation refinement 互斥，避免多个重排器叠加。
- 关闭 enable_birs 时不构建新模块，不增加损失、参数或collective。

## 3. v2完整数据流与维度

B为批大小，M为进入CRS的候选数，P为原检测候选数，L为文本长度；常用M=P=256、L=64、C=256。K=min(target_topn,M)，默认K=5；参考上限R=64，内部D=128。不足K个有效框时只使用实际有效候选，不复制或注入GT。

```text
BEiT-3 → SimFPN → Deformable DETR
                       ↓
              queries [B,P,256] + 框 + 检测分类
                       ↓
              可选Grab → refer_adapter → 可选BCEM
                       ↓
              原CRS(Q [B,M,256], T [B,L,256])
                       ↓
               原logits / 概率 [B,M,1]
                       ↓
             按原CRS概率取Top-N [B,K]
                       ↓
         目标特征 [B,K,256]、框 [B,K,4]
                       ↓
             平均原型 [B,1,256] → [B,1,128]

原检测池 → 检测置信度Top-64 → 去重 → R个槽位/有效掩码
                       ↓
        参考特征投影 → 读取完整文本 → [B,R,128]
                       ↓
       原型条件的参考语义 H [B,1,R,128]

Top-N目标框 × 参考框 → 相对几何 [B,K,R,9]
                → 文本/原型条件非线性编码 Ψ [B,K,R,128]
                       ↓
             实际绑定 − 重绑定期望 [B,K]
                       ↓
         有效集合内中心化 × 可学习β → 残差 [B,K]
                       ↓
            修正Top-N内部logit并排序
                       ↓
       重排后的Top-N + 原CRS其余候选的原顺序
```

实现只为K个目标计算绑定，不为全部M个目标计算；全部M个输出槽位仍保留用于旧后处理。raw_binding、delta等诊断量回填到原query索引，集合外为零，candidate_mask标明计算范围。参考池始终独立，不随K缩小。目标侧关系计算量约O(BKRD)，参考文本读取约O(BRLD)；没有R×R显式置换矩阵。分块只控制单次临时工作集，训练反向仍保留各块所需激活。

## 4. v2每一步的目的与精确规则

### 4.1 文本条件竞争集合与共享原型

按原评估的有效框过滤、CPU概率降序排序规则取Top-N。选择索引停止梯度；选中特征的梯度按detach_inputs开关决定。没有重新训练一个筛选器，不在训练时加入GT候选。

Top-N特征平均得到一个共享原型z，代替外观聚类原型。目的：让同一条描述的实际竞争者使用一致的参考语义条件。默认不再执行K-means，也不依赖num_groups或kmeans_iters。

### 4.2 独立参考池

检测前景置信度为softmax(class_logits)[...,0]。有效检测框先取Top-64，再按置信度贪心抑制IoU≥0.8的近重复框，不从Top-64外补齐。

对每个目标排除同源query及IoU≥0.8的自身近重复；真实部分重叠允许。框从相对有效图像的cxcywh转换为xyxy并裁剪，几何不用pad_shape。选择、框坐标和检测分数停止梯度。

目的：参考物自身不必像目标，不能也用CRS Top-N作为参考池。

### 4.3 完整文本与语义—几何编码

参考特征与完整文本投影到128维，参考token读取有效文本token，得到v_j。padding不参与；全padding样本不产生有效残差。

参考语义：
$$H_j=\tanh\{\operatorname{LN}[\operatorname{GELU}(W_h[v_j;z;v_j\odot z])]\}.$$

对目标i和参考j，构造9维几何：
$$r_{ij}=[\Delta_x,\Delta_y,\Delta_x/d,\Delta_y/d,\log(1+d),
\log(w_j/w_i),\log(h_j/h_i),\tanh((x_j-x_i)/w_i),\tanh((y_j-y_i)/h_i)].$$

前两维为像素中心差除以有效图像短边，d为该归一化距离；方向坐标向右、向下为正。宽高比对数截断[-8,8]，分母有数值下限。非线性MLP编码后，与“原型+有效文本投影平均值”交互，输出Ψ_ij。

目的：保留“什么物体在什么相对位置”的绑定。不能只线性编码dx、dy：在参考维度中心化时目标中心会被消掉，导致不同目标难以区分。

### 4.4 实际配对减重绑定期望

对目标i的同一个合法参考集合J_i，m_i=|J_i|：
$$p_i=\frac{1}{m_i\sqrt D}\sum_{j\in J_i}H_j^\top\Psi_{ij},\qquad
n_i=\frac{\bar H_i^\top\bar\Psi_i}{\sqrt D},\qquad d_i=p_i-n_i.$$

等价计算：
$$d_i=\frac{1}{m_i\sqrt D}\sum_{j\in J_i}(H_j-\bar H_i)^\top(\Psi_{ij}-\bar\Psi_i).$$

n_i是均匀随机重绑定的精确期望，无需抽样。两侧均值使用相同合法参考集合。若分数只有独立的语义项+位置项，则残差为零；参考语义完全相同时也应为零。少于两个合法参考时不产生绑定。没有把所有置换标成负样本，也没有删除真实物体。

### 4.5 残差与硬边界排序

至少两个目标候选能产生合法绑定时，在这些有效候选内中心化：
$$\delta_i=\beta(d_i-\bar d),\quad\beta=\beta_{max}\sigma(a),\quad
\ell_i^{final}=\ell_i^{CRS}+\delta_i.$$

β初始0.1，上界2，仍通过损失学习；没有新增门控。无有效绑定的候选δ=0。中心化减去共同常数时不改变这些候选内部的排序，主要限制平均logit漂移、影响BCE校准，不应把它当作额外消歧能力。

最终用：
**Top-N内按修正logit重排 + 集合外按原CRS顺序接在后面。**

通过现有ranking_priority接口编码完整顺序。不能简单对所有修正后的概率再次排序，否则集合外候选仍可能插进来。概率仍是sigmoid(logit)，ranking_priority是序数，不是置信度；评估/可视化必须优先使用ranking_logits/ranking_priority。

N=5保证同一checkpoint重排前后Top-5成员相同，因而其Top-5 Acc相同；N>5允许改变Top-5，但Top-N成员不变。联合训练仍可能改变CRS，不能保证相对于独立BASE的Top-5不变。

## 5. 损失、监督与梯度

原loss_det、loss_refer、可选Grab/filter/mask/correctness保持原算法与权重，原CRS损失仍对全部原候选计算。

| 新增损失 | 默认权重 | 范围与目的 |
|---|---:|---|
| loss_birs_final | 1.0 | 全部有效候选上的融合BCE，沿用原Hungarian refer标签，直接监督最终定位 |
| loss_birs_binding | 0.1 | Top-N内合法正负对，softplus(d_neg-d_pos)，独立训练绑定分支 |

日志值已乘权重。设全部候选有效掩码为v_i，原Hungarian refer标签为y_i，融合BCE为：

$$\mathcal L_{final}^{(b)}=\frac{\sum_{i=1}^{M}v_i\operatorname{BCEWithLogits}(\ell_i^{final},y_i)}{\max(1,\sum_{i=1}^{M}v_i)}.$$

再按全局有效样本数平均。**分母是全部有效候选数（通常256），不是Top-N数；有效样本也不要求正候选在Top-N。**只要求存在原refer GT且其原匹配正候选在全部有效框中。不因没有绑定参考而跳过融合BCE。

集合外候选的最终logit就是原CRS logit，因此GT匹配正候选在前五之外时仍得到直接定位梯度，前五中的错误候选也仍受负标签监督。此时绑定分支没有集合外正候选的残差计算，不能把“全候选BCE”解释为“全候选绑定正例监督”。无合法绑定正负对时仅跳过绑定排序损失，不跳过融合BCE。

绑定正候选仍须原Hungarian正标签、IoU≥0.5且可产生合法绑定；负候选须与原refer GT的IoU恰为0且有效，按原CRS得分最多取negative_topk个。其他候选不进入此排序项；没有合法对则跳过。共享集合替代同簇限制，其他正负规则不变。

仅loss()读取原refer GT；teacher或辅助GT不作为BIRS关系标签。DDP按全局有效样本数归一化，缺监督rank也保留新参数的连接零梯度图。

- detach_inputs=False：联合训练；新增损失可回传原CRS及候选/文本特征路径，框和离散选择无直接梯度。
- detach_inputs=True：候选、参考、文本及原CRS概率/logit截断，仅训练BIRS；梯度裁剪独立，原损失继续训练原模型。

本次修正撤销了此前“融合BCE只在Top-5内平均”的变化，恢复全候选监督范围、候选维度分母和全局有效样本覆盖，不额外调损失权重。残差很小时仍相当于额外增加一次定位监督；相对于不含BIRS的BASE，训练目标并不相同，收益不能直接归因为绑定机制。

## 6. 配置、训练与旧模型兼容

当前配置：

```python
enable_birs = True          # False完全沿用原模型
birs_version = "v2"        # "v1" 第一版聚类；"v2" 第二版Top-N
birs_target_topn = 5
birs_num_groups = 8
birs = dict(
    version=birs_version,
    target_topn=birs_target_topn,
    num_groups=birs_num_groups, kmeans_iters=6,
    reference_topk=64, duplicate_iou=0.8,
    model_dim=128, num_heads=4, chunk_size=32,
    initial_strength=0.1, max_strength=2.0,
    final_loss_weight=1.0, binding_loss_weight=0.1,
    negative_topk=8, detach_inputs=False,
)
```

`target_topn`只影响v2；`num_groups/kmeans_iters`只影响v1。要复跑第一版，设置`birs_version="v1"`，其余配置保持对照一致。默认版本为v2，旧的v2配置不写version也仍使用v2；第一版旧归档配置测试时必须显式补上`model.head.birs.version="v1"`。原CRV和relation_refinement保持关闭。启动方式不变，建议设置独立work_dir：

```bash
conda activate propvg
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py
```

兼容范围：

- enable_birs=False（旧配置没有此参数时也默认False）：仍沿用原PropVG/Grab/BCEM/correctness等已有路径，既有非BIRS模型可以继续训练/测试，不要求新增BIRS参数。
- v1恢复第一版聚类计算，使用原schema=1及原策略摘要，可加载匹配的第一版权重；v2保留schema=3及其全候选损失/target_topn摘要，现有修复版v2权重可继续加载。两版显式校验策略，不能交叉加载或悄悄切换排序语义；中间的Top-N-only-BCE/schema=2仍不支持。
- 配置中的用户版本v1/v2与checkpoint内部schema=1/3不是同一套编号。不再提供candidate_mode别名。
- 归档实验源码、归档配置和权重没有删除；如需要复查旧实验，可在对应归档环境运行，而非要求当前代码复现旧BIRS。
- 从相同BEiT-3初始化重新训练本组对照。需要基线权重初始化时，可使用不含BIRS的baseline配合--finetune-from；完整新BIRS训练后仍保存原模型+BIRS的整体参数。
- 未完成有效监督优化的BIRS在eval时回退原CRS。这里trained_steps记录有新增监督的优化步，不保证每一步都有绑定正负对。

## 7. 诊断接口、日志与验证

核心文件：propvg/models/heads/binding_residual.py；接入点仍是REFHead._binding_residual()。保持原诊断接口：

```python
with model.head.birs.diagnostics():
    # 正常前向后查看extra_dict["birs_samples"]
    ...
```

新增candidate_mask、target_indices、target_valid，明确哪些原query参与Top-N。其他字段保留全候选原索引，包括原分数、最终logit、raw_binding、paired/null、delta、组标记（选中为0，未选中为-1）、参考池与合法参考掩码。

日志含义：birs_supervised_samples现在是全局每batch可计算全候选融合BCE的样本数，正常32条有refer GT且匹配正框有效时应接近32，不再等于32×refer_in_topn。birs_binding_samples仍统计有合法绑定对的样本数。birs_target_count与birs_refer_in_topn分别表示计算绑定的目标数、原匹配正候选进集合的比例（不是IoU Top-N召回率）。原birs_eligible与delta_abs仍按全部候选槽位平均，Top-N版数值自然更小，不应误判为分支关闭。

验证重点：

1. N=5时每图前后Top-5集合相同、集合外顺序不变，极端残差也不能让外部候选插入。
2. 只对Top-N执行几何/绑定，不运行聚类；参考物仍能来自Top-N以外。
3. 匹配正候选在Top-N之外时，全候选融合BCE仍非零且对集合外正logit有梯度，只有绑定排序损失跳过；分母仍为全部有效候选。
4. v1/v2各自的checkpoint策略校验；验证False路径的旧参数/RNG/前向/梯度不受影响，既有非BIRS权重可严格加载。
5. 统计与独立BASE的整体性能，以及同checkpoint前后改对/改错；联合训练时二者含义不同。
6. 记录全候选融合监督样本数、匹配正候选Top-N覆盖率、合法绑定对比例、β、残差和CRS分差，不能只看loss。\n7. 无可用参考物或全padding文本时残差为零，但只要存在有效refer正框，融合BCE仍计算；GT不存在或无有效匹配正框时跳过新增监督。
8. 保留paired与crs_control对照；crs_control残差/绑定loss为零，只保留新增全候选BCE，可控制额外定位监督的影响。BASE、本控制和完整模块需同配置独立训练，不能只靠测试时消融区分训练收益。

单元测试：PYTHONPATH=.:tools python -m unittest test_birs。

## 8. 局限与下一步实验

- Top-N里没有正确候选时，重排无法修复；Top-5 Acc≈90%仅表示约90%的样本前五至少有一个正确框。
- 原Hungarian正候选不在集合中时，融合BCE仍监督其CRS logit；绑定排序项缺少集合内正例，仍跳过，不偷偷改标签或向Top-N注入GT。
- 正候选和完全不重叠负候选共存的样本可能有限，要观察有效监督数量。
- 绑定得分方向未必正确，不能仅靠增大β解决；参考物可能是背景，正确配置也未必满足文本。
- 无关系描述、多解标注、参考物漏检、原CRS过度自信仍是限制。
- 硬边界排序与跨集合概率数值未必一致，所有下游排序必须遵循priority接口。
- v2只修正全候选融合BCE口径，没有增加门控、损失类型、外部监督或文本解析。v2推理仍有硬边界，BCE直接监督logit而非排序操作本身。v1/v2切换用于明确比较聚类与Top-N方案，不能保证任一版本性能更高。

## 9. 本次验证范围

106项测试通过，覆盖两个版本的分支选择、checkpoint隔离、v1聚类/空输入/梯度、v2全候选BCE及其梯度、正候选在Top-N外、没有绑定参考时的监督保留、Top-N硬边界、独立参考池及原PropVG/Grab/BCEM/correctness等回归路径。

另直接读取第一版实验归档源码，与恢复的v1比较：residual、paired、crs_control三种绑定模式，分别在联合/截断梯度两种设置下，初始化state_dict（含策略摘要）、前向张量、损失及原有日志指标均逐值一致。此处为小输入CPU验证，不是重新训练或完整数据集性能验证。

GPU两样本测试验证联合/截断梯度下全256候选BCE；集合外正logit的梯度与全候选平均公式一致。两进程CPU DDP连续两步通过：两个rank都参与融合BCE，只有一个rank存在绑定配对，参数梯度仍同步。加入BIRS前BASE归档配置及真实检测头权重严格加载，初始化RNG和检测头推理输出与归档实现逐位一致。

本次未启动完整训练，未停止用户运行中的实验，也未修改既有实验归档。
