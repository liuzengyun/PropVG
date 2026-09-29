# 基于空间视觉证据干预的反事实候选验证与自适应融合

> 本文按博士学位论文的方法章节组织，对应2026年9月29日保护性排序修订后的独立CRV v2。性能改善是方法目标，而非已获得的实验结论。正文给出动机与数学建模，附录保留代码、配置和功能检查说明。本文件为当前CRV的唯一说明文档。

## 1. 研究动机

无人机视觉定位要求模型根据航拍图像和自然语言描述，确定表达式所指向的目标位置。与常规视角相比，航拍目标往往尺度较小、分布密集，同类别实例之间还可能具有较高的外观相似性。例如，同一街道可能出现多辆白色轿车或黑色SUV，仅凭类别和颜色等局部属性难以确定唯一实例。这使模型不仅需要发现与描述匹配的目标，还需要排除同一场景中的合理干扰候选。

现有候选式方法通常首先生成候选框及其视觉表示，再计算候选与语言的相关性，以得分最高的候选作为定位结果。当前基线中的候选相关性评分模块（Candidate Relevance Scoring，CRS）属于这一正向匹配范式，主要回答“当前候选与文本有多匹配”。当多个目标共享类别、颜色或局部外观时，错误候选也可能获得较高的匹配分数，导致候选之间缺乏充分的区分度。

针对这一问题，本文引入一个互补的验证问题：**抑制当前候选对应的空间视觉证据后，原始文本与剩余场景的匹配程度发生了多大变化？** 若被抑制的是定位目标，其相关视觉证据可能被削弱，图文匹配程度随之下降；若被抑制的只是外观相似的干扰目标，则真实目标的主要证据可能仍然存在，匹配下降相对较小。因此，干预前后的图文匹配变化可作为正向候选分数之外的补充证据。

上述分析是一项需要实验验证的建模假设，并非对所有表达式成立的因果规律。描述中的参考物体也可能十分重要，删除参考物体同样可能使匹配下降；候选区域还可能覆盖多个对象。本文将干预后的匹配变化视为可学习的反事实证据，使用定位目标监督约束其用途，不将其直接等同于目标身份，也不声称其具有严格的因果识别保证。

据此，构建反事实指向目标验证模块（Counterfactual Referent Verifier，CRV）。该模块保留原CRS，通过独立轻量图文网络对原始场景与干预场景进行配对评分，再以自适应残差方式修正高响应候选的排序。为避免新增训练目标破坏基线的表示与定位能力，CRV与基线之间采用显式梯度隔离，而非让反事实损失联合更新原有主干和CRS。

## 2. 任务定义与总体框架

### 2.1 任务定义

设输入图像为 \(I\)，指向表达式为 \(E\)，对应的唯一目标框为 \(b^\star\)。视觉定位模型学习映射

$$
\widehat b=\mathcal F(I,E).
\tag{1}
$$

训练阶段使用 \(b^\star\) 进行监督，推理阶段只输入图像与文本。记进入CRS的候选集合为 \(\{(b_i,q_i)\}_{i=1}^{N}\)，其中

$$
b_i\in\mathbb R^4,\qquad q_i\in\mathbb R^{256},
\qquad z_i=f_{\mathrm{CRS}}(\{q_j\}_{j=1}^{N},L)_i,
\qquad p_i=\sigma(z_i).
\tag{2}
$$

\(z_i\) 是CRS原始logit，\(p_i\)是其概率形式。上述表示允许原CRS在多个候选之间交互，不将其简化成必然相互独立的候选分类器。

CRV接收最大尺度检测视觉特征 \(V\) 和原始文本特征 \(L\)。忽略batch维度时，

$$
V\in\mathbb R^{256\times H\times W},\qquad
L\in\mathbb R^{T\times256}.
\tag{3}
$$

以586×586输入为例，当前最大检测图为144×144，文本最大长度为64。本文保留完整表达式的分词顺序，不人工拆分目标、属性或关系；超过配置长度的文本仍按现有分词流程截断。

### 2.2 高响应候选选择

仅对CRS最高响应的候选执行干预：

$$
\mathcal A=\operatorname{TopK}_{i\in\{1,\ldots,N\}}(z_i),
\qquad K=\min(K_0,N),\qquad K_0=8.
\tag{4}
$$

此举将新增计算集中于基础模型认为“可能是目标”的候选之间的竞争，而非全部前景和背景。选择使用原始logit稳定排序，避免大logit经sigmoid后因有限精度变为相同概率。

开启CRV时，最终只在 \(\mathcal A\) 中排序，其余候选不再与干预后的候选混合输出。因此，正确目标进入Top-K是方法能够纠正排序的必要条件；CRV不能创造新的目标框，也不能解决Top-K之外的漏召回。原检测损失和CRS损失仍在各自原有候选集合上计算，不因最终截断而改变。

## 3. 候选级空间视觉证据干预

### 3.1 坐标映射与有效区域

检测框采用有效缩放图像内的归一化中心宽高坐标 \(b_i=(c_{x,i},c_{y,i},w_i,h_i)\)。设有效图像尺寸为 \((W_v,H_v)\)，padding后尺寸为 \((W_p,H_p)\)，定义

$$
r_x=\frac{W_v}{W_p},\qquad r_y=\frac{H_v}{H_p},
\qquad
\widetilde b_i=(r_xc_{x,i},r_yc_{y,i},r_xw_i,r_yh_i).
\tag{5}
$$

\(\widetilde b_i\) 是整张padded特征图上的归一化框，有效区域为
\(\Omega=[0,r_x]\times[0,r_y]\)。该映射避免把有效图像框直接当作padded画布坐标，也避免用取整后的有效特征格数代替精确resize/pad比例。

### 3.2 扩张与平滑羽化

仅删除原预测框内的特征可能残留目标边缘证据；直接置零又可能产生明显的人工空洞。因此，构建保持中心不变的扩张掩码，核心范围为原框的 \(a=1.5\) 倍，羽化至 \(b=2.0\) 倍。

对任一坐标轴，设中心为 \(c\)、原框长度为 \(s\)，令 \(u=2|x-c|/s\)，平滑函数为 \(\psi(t)=3t^2-2t^3\)。定义

$$
m(x;c,s)=
\begin{cases}
1,&u\le a,\\
1-\psi\left(\dfrac{u-a}{b-a}\right),&a<u<b,\\
0,&u\ge b.
\end{cases}
\tag{6}
$$

二维连续掩码为

$$
\mathcal M_i(x,y)=
m(x;\widetilde c_{x,i},\widetilde w_i)
m(y;\widetilde c_{y,i},\widetilde h_i).
\tag{7}
$$

为避免小目标因未覆盖特征格中心而没有干预，采用格内面积积分，而不是只依据格中心判定。设 \(\mathcal C_{uv}\) 为特征格的归一化空间范围，则有效面积比例与替换强度为

$$
A_{uv}=\frac{|\mathcal C_{uv}\cap\Omega|}{|\mathcal C_{uv}|},
\qquad
M_{i,uv}=
\frac{\displaystyle\int_{\mathcal C_{uv}\cap\Omega}
\mathcal M_i(x,y)\,\mathrm dx\mathrm dy}
{|\mathcal C_{uv}\cap\Omega|}.
\tag{8}
$$

无有效面积时令替换强度为零；现有padding mask标记的无效格也不参与计算。实现使用解析积分并按有效面积归一化，使小于一个格的目标仍能产生非零干预，图像边缘部分有效格也能得到一致处理。

### 3.3 逐通道有效均值替换

对每个通道独立计算原始图像有效区域的空间均值：

$$
\mu_c=\frac{\sum_{u,v}A_{uv}V_{cuv}}{\sum_{u,v}A_{uv}},
\qquad c=1,\ldots,256.
\tag{9}
$$

对应的干预场景为

$$
V^{(-i)}_{cuv}
=(1-M_{i,uv})V_{cuv}+M_{i,uv}\mu_c.
\tag{10}
$$

核心区域完全替换为该通道的原图有效均值，羽化区域平滑混合，框外保持原始值。均值不跨通道计算，不包括padding，也不在每次干预后重新计算。

干预在最大144×144特征图上执行，不是删除候选query，也不在像素图像上重新运行编码器。该方式减少重复编码成本，但编码器已传播到框外的目标信息可能仍然保留。因此，本文使用“空间特征证据抑制”的表述，不假设目标信息被彻底清除。

## 4. 独立图文配对评分

### 4.1 场景池化与投影

原始场景和各干预场景在完成原尺寸干预后，执行有效区域加权自适应池化。设 \(\mathcal R_{ab}\) 为池化输出格的输入窗口，则

$$
\overline V_{cab}=
\frac{\sum_{(u,v)\in\mathcal R_{ab}}A_{uv}V_{cuv}}
{\max\left(\sum_{(u,v)\in\mathcal R_{ab}}A_{uv},\varepsilon\right)}.
\tag{11}
$$

无有效面积的输出格通过注意力mask排除。默认池化尺寸为 \(G=24\)，视觉token数为 \(G^2=576\)。在最大空间图上干预、在较小网格上交互，可限制注意力开销，但也可能稀释小目标差异。

通过无偏置1×1卷积和文本线性层，将视觉与语言投影到 \(D=128\) 维：

$$
X^{(s)}=\operatorname{Flatten}(W_V\overline V^{(s)})+P
\in\mathbb R^{G^2\times D},
\qquad
Y=LW_L\in\mathbb R^{T\times D},
\tag{12}
$$

其中 \(s\) 表示场景，\(P\) 为二维正弦位置编码。投影及评分参数均属于CRV，不与CRS共享，也不要求空间token与检测query具有相同分布。

### 4.2 文本引导的场景匹配

将可学习的场景汇总token \(t_s\) 与文本拼接为查询：

$$
Q^{(0)}=[t_s;Y]\in\mathbb R^{(T+1)\times D}.
\tag{13}
$$

以 \(X^{(s)}\) 为视觉memory，采用两层pre-norm Transformer Decoder。第 \(\ell\) 层为

$$
\begin{aligned}
U^{(\ell)}&=Q^{(\ell-1)}
+\operatorname{MSA}(\operatorname{LN}(Q^{(\ell-1)})),\\
R^{(\ell)}&=U^{(\ell)}
+\operatorname{MCA}(\operatorname{LN}(U^{(\ell)}),X^{(s)},X^{(s)}),\\
Q^{(\ell)}&=R^{(\ell)}
+\operatorname{FFN}(\operatorname{LN}(R^{(\ell)})).
\end{aligned}
\tag{14}
$$

查询自注意力在文本和汇总token间交互，跨注意力从空间视觉memory读取信息。每层4个注意力头，前馈隐藏维度256，激活为GELU，dropout为0。语言和视觉padding位置通过mask排除，不同场景作为独立样本处理，不跨场景进行自注意力。

取最后一层汇总token输出 \(h_s\)，得到标量匹配值

$$
g_\theta(V^{(s)},L)=
w_2^\top\operatorname{GELU}
\left(W_1\operatorname{LN}(h_s)+b_1\right).
\tag{15}
$$

同一个CRV内部评分器用于原始和干预场景，但不共享CRS参数。此分数没有绝对support概率监督，不能解释为“表达式成立的概率”，其作用是提供可比较的配对差值。

### 4.3 相对证据与有界变换

候选反事实差值定义为

$$
d_i=g_\theta(V,L)-g_\theta(V^{(-i)},L).
\tag{16}
$$

\(d_i\) 越大，表示干预后图文匹配下降越多。为限制对基线得分的扰动，采用有界softsign变换：

$$
e_i=R\frac{d_i/\tau}{1+|d_i/\tau|},
\qquad R=2,\quad\tau=1.
\tag{17}
$$

理论上 \(e_i\in(-R,R)\)，其导数为

$$
\frac{\partial e_i}{\partial d_i}
=\frac{R}{\tau(1+|d_i|/\tau)^2}.
\tag{18}
$$

较大差值处导数按多项式衰减，减轻tanh在有限精度下过早饱和的问题，但不保证任意极端数值都有足够梯度。辅助监督作用于实际融合使用的 \(e_i\)，避免原始差值达到margin、变换后却仍缺少区分度。

原场景仅评分一次，随后与各干预场景配对。不对 \(d_i\) 减去候选均值，因为中心化会直接消去共同的原场景分数。非线性证据差值通常允许原场景获得排序梯度，但并不保证每个输入都有非零原场景梯度，例如零初始化时不同候选的变换导数相同，梯度可能抵消。

## 5. 自适应残差融合

### 5.1 图像级门控

不同样本的候选歧义及干预可靠性不同，本文为每个图文样本学习一个共享权重 \(\alpha\)，而非构建额外的候选级分类器。对按CRS降序排列的Top-K，构造

$$
t=\left[
\operatorname{Mean}(p),\
p_{(1)},\
p_{(1)}-p_{(2)},\
\operatorname{Std}(p),\
\operatorname{Mean}(|d|),\
\operatorname{Std}(d)
\right]^\top\in\mathbb R^6.
\tag{19}
$$

统计量仅在Top-K上计算，标准差使用总体形式。若仅一个候选，第二候选沿用第一候选。停止梯度后输入6→32→1的MLP：

$$
\alpha=\sigma\left(
W_2^G\operatorname{GELU}(W_1^G\operatorname{sg}(t)+b_1^G)+b_2^G
\right).
\tag{20}
$$

其中 \(\operatorname{sg}\) 为停止梯度算子。最终logit及显示概率为

$$
\widetilde z_i=\operatorname{sg}(z_i)+\alpha e_i,
\qquad
\widetilde p_i=\sigma(\widetilde z_i).
\tag{21}
$$

CRS提供基础匹配，CRV提供有界残差修正。推理按logit排序，概率只用于显示；不要求两个分支的绝对概率具有相同标定。

### 5.2 初始化与修正范围

独立评分器最后一层权重零初始化，因此初始 \(d_i=e_i=0\)，融合logit精确等于选中候选的原CRS logit。门控最后一层权重零初始化、偏置设为 \(\log(0.1/0.9)\)，使初始 \(\alpha=0.1\)。这是单阶段训练的初始化策略，不是先训练CRS再训练CRV，也不是固定权重。

零初始化时，门控没有证据差异可利用，其梯度可以为零；评分器产生差异后，门控获得排序信号。无需权重下限、阶段预热或验证集搜索。

对于一组正负候选，

$$
\widetilde z_p-\widetilde z_n
=(z_p-z_n)+\alpha(e_p-e_n),
\qquad |\alpha(e_p-e_n)|<2\alpha R.
\tag{22}
$$

有界残差只能修正允许幅度内的基线分差，不能保证纠正任意大的CRS排序错误。

## 6. 训练监督与优化目标

### 6.1 可靠正负候选

CRV标签只使用原始表达式对应的目标框 \(b^\star\)，不把teacher框或其他前景框作为定位目标。计算IoU前，将预测框及GT裁剪至有效图像边界，与推理后处理一致。

正集合为

$$
\mathcal P=\{i\in\mathcal A:\operatorname{IoU}(b_i,b^\star)>0.5\}.
\tag{23}
$$

负候选要求与GT完全不重叠，且扩张掩码不显著抑制GT。令 \(A^\star_{uv}\) 为GT在特征格内的面积比例，定义

$$
\kappa_i=
\frac{\sum_{u,v}M_{i,uv}A^\star_{uv}}
{\max(\sum_{u,v}A^\star_{uv},\varepsilon)},
\qquad
\mathcal N=\{i\in\mathcal A:
\operatorname{IoU}(b_i,b^\star)=0,\ \kappa_i\le0.05\}.
\tag{24}
$$

\(0<\operatorname{IoU}\le0.5\) 的候选，以及扩张掩码覆盖GT过大的负候选不参与CRV监督。此规则筛选可靠训练标签，不改变定位评价阈值，也不对参考物体实施特殊保护。

### 6.2 纠错与保留的保护性排序损失

原softplus目标即使排序间隔已满足，仍持续奖励增大正负分差，可能推动门控过度参与。2026年9月29日的修改采用有停止区间的hinge监督，并区分CRS原本正确和原本错误的样本。原始Top-K按CRS排序，令第一名为索引1；只有正负集合均非空时才可能获得最终监督。

若CRS第一名属于可靠负集合，且Top-K含正确候选，使用最好的正候选和最强的可靠负候选：

$$
\ell_{\mathrm{correct}}^{(j)}
=\left[m-\max_{p\in\mathcal P_j}\widetilde z_p
+\max_{n\in\mathcal N_j}\widetilde z_n\right]_+,
\qquad m=0.5,\quad 1\in\mathcal N_j.
\tag{25}
$$

这里只约束实际威胁最高分正确框的hard distractor，而不平均大量易负pair。达到margin后，该项及其门控梯度为零；其他正确框仍由反事实辅助目标学习，不强迫每一个正框都成为最终第一名。

若CRS第一名属于正集合，保护其原有正确排序。对每个可靠负候选定义所需间隔为原始分差与margin的较小者：

$$
\delta_{1n}=\min(m,\max(0,z_1-z_n)),\qquad
\ell_{\mathrm{preserve}}^{(j)}
=\left[\max_{n\in\mathcal N_j}
\{\delta_{1n}-\widetilde z_1+\widetilde z_n\}\right]_+,
\quad 1\in\mathcal P_j.
\tag{26}
$$

初始融合等于CRS时，保护项为零；小于margin的原始正确分差按原值保护，较大分差只需维持margin。因此，该项不奖励无休止扩大原有正确间隔，但在融合侵蚀所需间隔时产生惩罚。它保护的是原始正确第一名，不是固定所有候选的原始分数或禁止任何重排序。

最终损失为两类互斥样本的图像级平均。令有效集合为
\(\mathcal U=\{j:\mathcal P_j,\mathcal N_j\ne\varnothing,\ 1\in\mathcal P_j\cup\mathcal N_j\}\)，有

$$
\mathcal L_{\mathrm{final}}=
\frac{\sum_{j\in\mathcal U}
(\ell_{\mathrm{correct}}^{(j)}+\ell_{\mathrm{preserve}}^{(j)})}
{\max(|\mathcal U|,1)}.
\tag{27}
$$

未适用的分项设为零。Top-K没有正框时不虚构最终标签；CRS第一名处于IoU灰区或属于被GT覆盖筛选排除的负框时，保留原“忽略”规则，不把它额外当作负样本。这类样本仍可获得合法的精确GT辅助监督。

对当前活跃的正负约束，门控梯度与负的证据分差成比例；但间隔满足时梯度为零。纠错中证据方向正确会推动alpha增大，保护中错误证据侵蚀原分差会推动alpha减小，同时训练独立评分器纠正方向。门控输入仍无GT，GT只用于训练目标分支判定。保护项是经验训练约束，不是测试阶段绝不损害基线的保证；参考物体与灰区候选的既有局限仍存在。

### 6.3 精确GT干预辅助损失

训练阶段额外使用精确 \(b^\star\) 构造一份干预场景，得到 \(d_\star\) 和 \(e_\star\)。它只辅助评分器学习，不成为输出候选，也不进入门控统计或最终融合。

令 \(\mathcal P^{\mathrm{cf}}=\mathcal P\cup\{\star\}\)，采用有界证据的margin ranking损失：

$$
\mathcal L_{\mathrm{cf}}^{(j)}
=\frac{1}{|\mathcal P_j^{\mathrm{cf}}||\mathcal N_j|}
\sum_{p\in\mathcal P_j^{\mathrm{cf}}}\sum_{n\in\mathcal N_j}
\max(0,m-e_p+e_n).
\tag{28}
$$

归约同样先求图内pair均值，再求有效图均值，无pair时为可反向的零。Top-K暂未命中GT时，精确GT仍可与可靠负候选提供辅助信号，但不赋予最终输出虚假的正标签。

### 6.4 总目标与梯度边界

基线原有目标记为 \(\mathcal L_{\mathrm{base}}\)，包括原检测、CRS定位及配置开启的其他损失。新增目标与总目标为

$$
\mathcal L_{\mathrm{CRV}}
=0.5\mathcal L_{\mathrm{final}}+0.05\mathcal L_{\mathrm{cf}},
\qquad
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{base}}+\mathcal L_{\mathrm{CRV}}.
\tag{29}
$$

代码形式为 \(0.5(\mathcal L_{\mathrm{final}}+0.1\mathcal L_{\mathrm{cf}})\)。最终目标内部区分纠错与保护，但只有两项顶层新增目标，权重不变；没有绝对support BCE、特征对齐损失或额外门控正则。

记基线、CRV评分器与门控参数为 \(\phi,\theta,\eta\)。CRV输入的视觉、文本、框和CRS分数全部停止梯度，因此

$$
\begin{aligned}
\frac{\partial\mathcal L_{\mathrm{CRV}}}{\partial\phi}&=0,\\
\nabla_\theta\mathcal L_{\mathrm{total}}
&=0.5\nabla_\theta\mathcal L_{\mathrm{final}}
+0.05\nabla_\theta\mathcal L_{\mathrm{cf}},\\
\nabla_\eta\mathcal L_{\mathrm{total}}
&=0.5\nabla_\eta\mathcal L_{\mathrm{final}}.
\end{aligned}
\tag{30}
$$

原CRS继续由其原有定位损失训练；CRV损失不更新BEiT-3、SimFPN、检测解码器、Grab、refer_adapter或CRS。基线与CRV分别裁剪梯度，避免新增梯度改变基线裁剪范数。所有分支在同一次训练中优化，梯度隔离不等于冻结基线。

## 7. 推理流程、维度与复杂度

### 7.1 无GT推理

推理过程为：基线图文编码与检测 → 原CRS赋分 → Top-K选择 → 最大空间图上的K次干预 → 原场景和干预场景独立配对评分 → 差值与有界证据 → 图像级门控融合 → 候选排序。最终结果为

$$
\widehat i=\arg\max_{i\in\mathcal A}\widetilde z_i,
\qquad \widehat b=b_{\widehat i}.
\tag{31}
$$

推理不输入目标GT、teacher框或GT干预。通过现有后处理将预测框裁剪并恢复至原图坐标。模块改变排序，不新增框回归器。

### 7.2 关键数据流

以586×586输入、batch为 \(B\)、\(T=64\)、\(K=8\) 为例，训练场景数 \(S=K+2\)，推理场景数 \(S=K+1\)。

| 阶段 | 张量维度 | 处理 |
| --- | --- | --- |
| BEiT-3 | 图像[B,1296,768]；文本[B,64,768] | 原多模态编码 |
| 头部投影 | 图像[B,512,36,36]；文本[B,64,256] | 保留原512通道路径 |
| 最大检测特征 | [B,256,144,144] | SimFPN最大尺度的检测通道 |
| 检测query、预测框 | [B,P,256]；[B,P,4] | AerialVG的P=256，AerialSense的P=128 |
| CRS候选和分数 | [B,N,256]；[B,N,1] | 可选Grab过滤后N可能小于P |
| Top-K | 框[B,K,4]；logit[B,K] | 所有CRV输入detach |
| 原尺寸干预 | 每份[B,256,144,144] | 先干预，再池化 |
| 池化场景 | [B,S,256,24,24] | 原场景、预测框干预及训练GT干预 |
| 视觉memory | [B×S,576,128] | 投影并加入空间位置编码 |
| 图文查询 | [B×S,65,128] | 64个文本token加汇总token |
| 场景分数 | [B,S] | 不同场景互不混合 |
| 差值、证据 | [B,K]；[B,K] | GT辅助证据单独保留 |
| 门控 | [B,1] | 每图一个共享权重 |
| 输出 | 框[B,K,4]；分数[B,K,1] | 融合logit用于最终排序 |

原512通道neck及已构建的分割模块不因CRV而改动；无mask GT时沿用当前基线跳过分割前向和损失。CRV不生成矩形伪分割标签。

### 7.3 计算开销

设 \(M=G^2\)、\(Q=T+1\)、交互层数为 \(L_d\)，忽略batch维度，图文交互主要量级为

$$
\mathcal O\left(SL_d[Q^2D+QMD+(Q+M)D^2]\right).
\tag{32}
$$

原尺寸干预与池化约为 \(\mathcal O(KCHW)\)。实现顺序构造并池化干预场景，不长期保存全部高分辨率场景。原场景只评分一次，不重复运行图像编码器或完整检测解码器，额外成本随K近似线性增加。默认新增参数约480385个。轻量化不代表零开销，实际时延和显存需要在完整运行条件下测量。

## 8. 方法边界与验证要求

本方法的设计特点包括空间特征干预、原始与干预场景的独立配对评分、最终排序监督的自适应残差，以及保护基线的梯度隔离。这些是本方案的设计要点，不等于已经完成相对于所有文献的创新性论证。

方法仍受候选召回限制；参考物体干预可能造成歧义；编码器上下文可能泄漏目标信息；均值填充可能形成可利用的人工模式；小目标经过池化后的差异可能被削弱；有界修正和门控饱和可能限制排序改善。因此，训练损失下降不能单独证明定位精度提升。

应在独立验证和测试划分上同时比较基线、CRV融合和CRV-only排序，报告Top1、Top5和最高分框的box mIoU。设样本数为 \(J\)，排序后的框为 \(\widehat b_{j,(r)}\)，则

$$
\operatorname{Acc@}k=
\frac{100}{J}\sum_{j=1}^{J}
\mathbf1\left[
\max_{1\le r\le\min(k,K_j)}
\operatorname{IoU}(\widehat b_{j,(r)},b_j^\star)\ge0.5
\right],
\qquad k\in\{1,5\},
\tag{33}
$$

$$
\operatorname{box\ mIoU}=
\frac{100}{J}\sum_{j=1}^{J}
\operatorname{IoU}(\widehat b_{j,(1)},b_j^\star).
\tag{34}
$$

box mIoU不是Top-K最佳框IoU，也不是分割mIoU。训练正标签使用严格IoU>0.5，而评价使用IoU≥0.5，应区分二者。还应统计目标进入Top-K比例、纠正原错误数量、破坏原正确数量及门控分布，避免只观察Top1或训练损失。

## 9. 本节小结

本节针对航拍视觉定位中高相似候选难以区分的问题，提出基于空间视觉证据干预的反事实验证与自适应融合方案。在保留CRS正向匹配的基础上，利用最大检测特征图上的逐通道均值替换构造干预场景，使用独立轻量图文网络衡量干预前后的匹配变化，再通过最终排序监督学习图像级融合权重。精确GT干预仅辅助训练，推理完全依赖预测候选。通过将新增梯度限制在CRV和门控内部，方案可在不改变基线监督方式的条件下进行单阶段训练。其有效性取决于候选召回、干预质量和证据可辨性，最终需要真实定位实验验证。


## 附录：当前实现与复现实验说明

### A.1 代码与日志对应

| 代码入口 | 内容 |
| --- | --- |
| `propvg/models/heads/crv.py` | 掩码、均值替换、独立评分器、证据、门控和两项损失 |
| `propvg/models/heads/REFHead.py` | 原CRS不变，收集refer GT，接入最大检测特征及CRV |
| `propvg/apis/train.py` | 新损失累加、分组梯度裁剪和日志 |
| `propvg/models/det_seg/mix_ref_unimodel_OMG.py` | 框后处理及ranking_logits输出 |
| `propvg/apis/test.py` | Top1、Top5及最高分框box mIoU |
| `tools/eval_crv_only.py` | 同一次前向的CRS、融合、CRV-only三路评价 |
| `tools/check_crv.py`、`tools/smoke_crv.py` | 功能检查及真实数据检查 |

日志中的 `crv_final`、`crv_cf` 是未加权诊断损失，`loss_crv`是已加权总项。新增 `crv_correct`、`crv_preserve` 分别表示纠错与保护对最终损失的贡献，共用有效图数量归一化，两者之和等于 `crv_final`，不重复加入总目标。`crv_alpha`为门控均值，`crv_pairs`为Top-K具有合法正负双方且CRS第一名属于合法正/负集合的图像比例，取值0至1；它包含间隔已满足的图，不是非零梯度图比例。缺少正框或第一名为灰区时最终排序损失为零，但GT辅助项仍可非零。

新增模块使用fork_rng初始化，不改变基线初始化随机数流；评分器dropout为0。这些措施保护直接训练路径，不保证最终融合排序不变。严格复现仍需要相同数据、种子、硬件及确定性设置。

### A.2 配置与运行

`enable_crv=True`启用本模块，False保留原CRS流程。当前AerialVG开启、AerialSense关闭，后者按需开启。默认参数如下：

| 配置或实现参数 | 数值 |
| --- | --- |
| crv_topk / crv_grid_size | 8 / 24 |
| verifier_dim | 128 |
| num_heads / num_layers | 4 / 2 |
| core_expansion / feather_expansion | 1.5 / 2.0 |
| max_residual / drop_temperature | 2.0 / 1.0 |
| initial_alpha / rank_margin | 0.1 / 0.5 |
| max_negative_gt_coverage | 0.05 |
| crv_cf_loss_weight / 外部CRV权重 | 0.1 / 0.5 |

Teacher、Grab、学习率、epoch、图像尺寸和数据划分由各自配置控制，不因CRV自动改变。teacher与辅助GT仅按配置参与原检测监督，不作为CRV的refer标签；val/test不加载teacher，不向CRV传入GT。

```bash
# 单卡训练
python tools/train.py configs/aerialvg/PropVG-aerialvg.py

# 两卡单次训练，不修改现有dist_train.sh
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 tools/train.py configs/aerialvg/PropVG-aerialvg.py --launcher pytorch

# 新版checkpoint的单卡三路评价与CSV输出
CUDA_VISIBLE_DEVICES=2 python tools/eval_crv_only.py configs/aerialvg/PropVG-aerialvg.py --checkpoint /path/to/new/top1_best.pth --split val
```

三路评价使用相同候选和同一次前向：CRS按原始logit，融合按融合logit，CRV-only按原始drop排序。K≥5时，CRS Top5与该次前向全候选的相同logit排序一致；与历史概率排序比较还需考虑量化并列。K<5会截断Top5对照。真正相同的logit使用稳定排序；关闭CRV时保留原概率排序逻辑。

多卡训练的val仍采用原分布式验证；最后test在主进程单卡执行。分布式val sampler末尾补齐逻辑没有在本次修改，严谨报告指标时应注意真实样本计数。

### A.3 版本与功能检查

2026-09-29的本次修改仅涉及最终排序训练目标、诊断日志及对应测试。修改前备份为 `/tmp/crv_before_protected_ranking_20260929.tar.gz`。Teacher配置、检测/CRS监督、空间mask、网络参数量及推理融合均未更改。

当前检查：[CPU报告](reports/crv_protected_ranking_cpu.json)的32项中31项通过、DDP按预期跳过；[GPU与CPU Gloo DDP报告](reports/crv_protected_ranking_gpu_gloo.json)的32项全部通过。GPU反向在空闲3号卡执行，两进程检查使用CPU Gloo，不将其声称为本次两GPU NCCL验证。新增检查覆盖间隔满足后停止门控奖励、原始薄间隔初始化零保护、融合破坏正确排序时的保护梯度、hard负候选筛选，以及灰区/缺失正候选的零监督。

[本次AerialVG真实数据检查](reports/crv_protected_ranking_aerialvg_smoke.json)通过：真实BEiT-3前向、修改后头部反向、基线梯度隔离和无GT验证均正常，新诊断字段可用。仅作功能诊断，不启动优化器训练、不保存训练权重，也不据此声称定位性能提升。

当前checkpoint标记为 `independent_version=2`。本次只改变训练目标，不改网络、融合公式、GT规则和推理流程，因此9月28日的v2权重仍可评估；不会因更换训练代码而自动提高旧checkpoint精度。验证新监督建议开始新实验，而非将旧36轮训练当作新目标的实验结果。更旧的CRV（包括独立v1）不应直接resume，三路评价拒绝无CRV权重或架构版本不匹配的checkpoint。

旧CRV/CFCR备份为 `backups/old_crv_before_independent_20260928.tar.gz`；独立v1备份为 `backups/independent_crv_v1_before_ranking_fix_20260928.tar.gz`。已有实验、checkpoint及实验内源码快照未改动。

2026-09-28修改前的功能检查记录（仅作为历史证据）：

- [CPU检查](reports/independent_crv_v2_cpu.json)：28项中27项通过，两卡检查按预期跳过，无失败。
- [GPU与两卡NCCL检查](reports/independent_crv_v2_gpu_ddp.json)：28项全部通过，两rank各执行3个优化器步。
- [AerialVG真实数据检查](reports/independent_crv_v2_aerialvg_smoke.json)：真实BEiT-3前向、头部反向、梯度隔离和无GT验证通过，256候选输入、8个框输出。
- [AerialSense真实数据检查](reports/independent_crv_v2_aerialsense_smoke.json)：同样通过，128候选输入、8个框输出。

检查覆盖门控梯度方向、大差值数值稳定性、padding边界、logit排序、后处理、基线梯度隔离及checkpoint版本。这些不是精度结论。真实数据诊断使用no_grad主干和头部反向，不代表完整训练显存，也未启动正式训练。随机候选未命中GT时，crv_pairs与crv_final为零而GT辅助项非零，是预期行为。
