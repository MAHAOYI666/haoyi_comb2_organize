# 多因子聚合研究基础规范

本文只规定后续多因子聚合实验中已经确定、默认直接沿用的基础实现。

2026-10-07 文档维护边界：项目根目录的 `README.md` 和 `config.human` 由框架负责人维护，我们不再修改，也不向其中同步本项目的研究规则或环境说明。我们自己的共同研究规范、准则及其修订仅维护在本文件；负责人文档描述框架接口，本文件描述项目研究约定，不能将项目约定当作官方包默认行为。合并负责人更新时保留其文档内容，既有及冻结实验继续遵循各自协议。

具体的网络结构、label、loss 组合等属于算法设计，由各实验单独定义。优化器、训练轮数、训练数据范围和 early stopping 按第 3 节的共同训练配置执行。

2026-09-16 接口更新：第 13 节补充 Combo2 1.0.3 的来源缓存、跨月训练预热和 Dataset 初始化契约。本次更新属于框架接入规范，不修改第 1～12 节的数据、算法、训练或交易定义。

2026-09-20 环境说明更新：第 13 节的当前运行接口以本次运行实际安装的 Combo2 wheel 为准，不固定为某个历史版本。默认实例环境本次核验为 `combo2==1.0.6`；该记录不是永久版本要求，核验与留痕规则见第 13.5.1 节。

2026-09-21 checkpoint 历史更新曾要求 `model_keep_num="-1"` 并保留全部滚动期次模型；该要求已由下述 2026-09-30 更新替代，不再作为后续新实验的默认保留规则。

2026-09-21 交易配置历史更新曾固定部分组合优化参数；该写法已由下述 2026-09-29 更新替代，不再作为后续默认参数来源。

2026-09-24 共同训练规范历史更新曾规定无验证集、训练 IC loss 早停及最多 25 轮；其中早停监控量和轮数上限已由下述 2026-10-03 更新替代，不再作为后续新实验的默认规则。

2026-10-03 共同训练规范更新：取消验证集，整个 dataset 用于训练，仍须满足完整目标成熟性。每个完整 epoch 汇总训练模式下含 dropout、实际用于反向传播的训练 loss（`train_loss`），严格降低才算改善（`min_delta=0`），`patience=5`，最多训练 15 轮。训练集与早停的统一要求见第 3.3～3.4 节；具体算法的 checkpoint 评分由各实验单独定义。既有代码及运行中、已冻结的实验不因文档更新而自动修改，适用范围见第 3.5 节。

2026-10-07 重训频率更新：后续新实验统一按自然季度重训，首次建模后在每年 3、6、9、12 月的最后一个交易日各重训一次。具体调度、目标成熟性与模型生效要求见第 3.7 节；既有及冻结实验按第 3.5 节保留原协议。

2026-09-29 交易默认值规范更新：第 9.3 节改为运行时读取实际安装并加载的 Combo2 包的完整默认 OPT1、strategy 和 backtest 配置。规范不列具体 OPT 默认数值，不从历史 XML、旧回测配置或本地源码手抄默认值；每次运行保存包来源、默认快照和实际生效配置并核验一致性。

2026-09-30 产物保留规范更新：后续新实验默认不要求保存正式 checkpoints，显式使用 `model_keep_num="0"`；epoch 候选权重和续训状态按需临时保存，不默认累积全部历史副本。选模规则继续执行，预测、回测、指标、日志和其他正常 output 仍须完整生成并保留。需要持久模型的实验明确声明例外；具体要求见第 3.4～3.6 节。本次只修改规范，不修改正在运行的实验、代码、配置或既有产物。

---

2026-09-26 相关性评价规范更新：今后用户要求计算 `pnl_corr` 或 `pos_corr` 时，必须使用当前实际安装的官方 `comb-eval`；仅用户明确另行指定，或官方工具确实无法完成所要求的计算时，才允许采用替代方法。执行、留痕与例外规则见第 10.4 节。

# 1. 输入因子与股票轴

## 1.1 因子输入

默认使用当前有序的 625 个 Qsim 因子。

因子名称和顺序保持不变，对应 basename 顺序的 SHA-256：

```text
eae5c2a720e2bffa9ec2abc7c9102d3f61d4f165008722a3c8ca17370347e3e3
```

模型输入统一表示为：

$$
X\in\mathbb R^{B\times T\times N\times F}
$$

其中：

* \(B\)：日期 batch；
* \(T\)：模型使用的历史窗口；
* \(N\)：完整股票轴；
* \(F=625\)：因子数。

历史窗口 \(T\) 属于算法设计，可以根据实验改变。

---

## 1.2 股票主轴

股票维使用 loader 中完整代码轴：

```python
stock_axis = torch.arange(
    len(loader.mask.code),
    dtype=torch.long,
)
```

所有日期保持相同的股票顺序。

模型训练、预测和最终输出都使用这一代码映射。

---

## 1.3 输入可用性

当前参考实现根据最后一个输入日判断股票是否具有模型输入：

```python
available = x[:, -1].abs().sum(dim=-1) > 0
```

其中 `x` 已经过统一预处理。

如果模型需要进行股票间信息交互，`available` 用于控制哪些股票能够作为有效的股票输入。

监督 loss 使用的股票集合由独立的监督 mask `w` 决定。

因此模型中保留两个不同概念：

```text
available    当前是否具有模型输入
w            当前是否具有有效训练监督
```

---

# 2. 输入因子预处理

对每个日期、每个因子独立进行股票截面预处理。

设原始因子为：

$$
x_{i,f}
$$

其中 \(i\) 是股票，\(f\) 是因子。

## 2.1 截面标准化

只使用该因子的有限值计算截面均值和标准差：

$$
z_{i,f}
=
\frac{x_{i,f}-\mu_f}
{\sigma_f+\varepsilon}
$$

其中，\(\sigma_f\) 为有效截面的总体标准差（`ddof=0`）。\(\varepsilon\) 沿用 `cs_zscore` 的计算 dtype 约定：float32/float64 为 `1e-8`，float16/bfloat16 为 `1e-4`。零方差截面由该分母保护处理，全缺失截面在缺失值步骤中填零。

---

## 2.2 截断

$$
z_{i,f}
=
\operatorname{clip}(z_{i,f},-4,4)
$$

---

## 2.3 缺失值

剩余非有限值统一填为：

$$
0
$$

---

## 2.4 数据与计算精度

特征缓存：

```text
float16
```

进入模型以后转为：

```text
float32
```

模型使用前统一：

$$
x\leftarrow x/4
$$

因此正常输入范围约为：

$$
[-1,1]
$$

数据层完成截面标准化、截断、缺失值填零和 float16 转换；模型入口完成 float32 转换和除以 4。普通模型实验按这两个阶段接入，除以 4 只执行一次。

---

# 3. 训练历史与固定训练配置

第 3.1～3.4 节及第 3.7 节规定后续新实验共同沿用的训练规则；既有实验的适用范围见第 3.5 节。

## 3.1 训练历史范围

滚动训练默认最多读取：

```text
max_train_days = 2000
```

这表示一次训练最多使用 2000 个历史输入交易日。

实际监督样本数由：

* 历史窗口 \(T\)；
* label 所需未来区间；
* label 完整成熟时间；

共同决定，因此不固定具体训练样本数量。

所有训练样本都必须满足：

> 构成该样本训练目标所需要的全部未来数据，在本次模型训练发生以前已经完整产生。

## 3.2 固定训练配置

同一基础 benchmark 使用以下共同训练配置：

| 项目 | 固定设置 |
|---|---|
| optimizer | Adam |
| learning rate | 5e-6 |
| weight decay | 1e-6 |
| epoch 上限 | 15，达到早停条件可以提前结束 |
| 验证集 | 无；不划分或留出验证段 |
| 早停监控 | 训练模式下含 dropout 的实际训练 loss（`train_loss`），每个完整 epoch 汇总一次，越小越好 |
| 早停条件 | 连续 5 轮未严格降低；`patience=5`、`min_delta=0` |
| 日期 batch size | 3 |
| 日期顺序 | shuffle，使用 seed 固定的 DataLoader generator |
| 学习率调度 | StepLR，step_size=12，gamma=0.5 |
| scheduler 调用时机 | 每个完整 epoch 结束后调用一次 |
| 梯度裁剪 | L2 norm 上限 5.0 |
| 梯度累积 | 无，每个 batch 一次反向传播和一次 optimizer step |
| 模型计算精度 | float32 |
| 随机种子 | 42 |
| 重训频率 | 自然季度；首次建模后，在每年 3、6、9、12 月的最后一个交易日重训，见第 3.7 节 |
| 模型初始化 | 每次滚动重训重新初始化模型和优化器 |
| checkpoint 保留 | 新实验默认显式设置 `<combo><runtime model_keep_num="0" /></combo>`，不保存正式模型；确需持久模型时按第 3.6 节声明保留范围 |

Adam 未显式指定的参数沿用相应调用的默认值，并记录运行环境版本。

输入窗口 T、模型宽度、输出头、网络内部 dropout 和 loss 的辅助项属于算法设计；Adam 的 weight decay 及表中的训练配置固定。

## 3.3 训练集与监督边界

每次滚动重训取消验证集，也不留出固定天数的验证尾段。在第 3.1 节规定的输入历史范围内，按框架实际训练发生时点筛选目标已完整成熟的样本，构成整个训练 dataset；整个 dataset 用于梯度训练。

不再为训练/验证划分执行边界 purge；目标成熟性要求仍然适用：训练目标的全部依赖必须在本次训练发生前完整可知。目标的可得性证据与任何经用户批准的未核验时间假设由相应实验明确记录，不能把有限标签值自动视为成熟，也不能因取消验证集而纳入未来尚未成熟的标签。

训练样本日期、时点和有效监督股票集合在本次滚动训练开始前确定，所有 epoch 使用同一套样本与监督口径，仅按既定随机种子打乱样本顺序。数据缺失须预先确定并记录，不得根据模型预测表现删除样本或更换股票池。没有有效成熟训练样本时明确报告数据不足。

每个完整 epoch 汇总实际用于反向传播的训练 loss（`train_loss`），使用训练过程中 `model.train()`、dropout 开启时的前向计算值；不另做关闭 dropout 的评估来替代早停监控量。具体训练目标、各 loss 项及其权重由实验预先声明并保持一致，早停使用该实际训练目标，而非另设的验证指标；若训练目标只有 IC loss，则监控量就是对应的 `1 - IC`。IC 项的有效股票 mask、Pearson 和零方差分支按第 4 节执行。

epoch 汇总保持日期等权；多个时点先在日内等权，再按日期等权汇总，不能因股票数或 batch 大小不同而改变日期权重。存在辅助 loss 时，同步记录各分项和总训练 loss，明确其汇总方式。

训练 loss 出现非有限值时记录数值异常，不能通过删除异常预测对应的股票或日期获得一个看似有效的监控值。算法若使用历史反馈状态，仍须按各训练样本自己的时点因果查询。

## 3.4 Early stopping 与 checkpoint

early stopping 统一监控第 3.3 节训练模式下含 dropout 的实际训练 loss（`train_loss`），越小越好，使用以下规则：

1. 每个完整 epoch 结束后汇总一次含 dropout 的训练 loss，最多训练 15 个 epoch。
2. 首个数值正常的训练 loss 建立历史最佳值；此后只有严格低于此前最佳值才视为改善，即 `min_delta=0`，相等不算改善。
3. 改善时重置未改善计数；连续 5 个完整 epoch 没有改善时停止，即 `patience=5`。这不是固定只训练 5 个 epoch。数值异常的训练 loss 不更新最佳值，并计为一次未改善；它不能成为正常候选。全部候选数值异常时明确失败。
4. 早停只决定训练何时结束。最终用于预测的模型按各实验在训练前声明的选模规则，从实际完成且数值有效的候选中确定；选模同样只使用本次训练集和当时已知的信息，不另设验证集，也不能默认把停止轮次当作最终选中轮次。是否将权重保存为 checkpoint 不改变选模结果。
5. 训练日志和独立的指标、选模记录保存实际完成轮数、最佳训练 loss 及其 epoch、未改善计数、停止原因、最终选中 epoch 与对应选模指标；如保存模型 payload，其中也记录这些信息。未执行的轮次不能补作候选，选模证据不能只存在于可选的模型文件中。

本节不将某个实验的专属评分、阈值或分段方式规定为所有实验共同算法。训练期维护足够执行选模规则的最小候选集合，优先使用内存中的完整 `state_dict` 独立副本，避免后续参数更新覆盖候选。能在线确定最佳候选时仅更新该候选；确需期末比较多个候选或内存不足时允许临时落盘，不得为了省空间改变已声明的选择规则。

每次滚动训练正常结束后，必须恢复选中权重到实际用于后续预测的模型，再按原时序生成信号。默认不要求将这个模型持久化为正式 checkpoint；若后续流程必须从磁盘加载权重，按第 3.6 节声明所需模型，不能直接禁用其依赖的保存步骤。

`model_keep_num` 只控制框架正式 checkpoint 的保存与保留，不会关闭模型代码自行写出的 `epoch_candidates/`、`epoch_*.pt` 或 `latest_training_state.pt`，也不会自动实现训练指标统计、早停或候选选择。`ResearchModel.fit()` 必须分别落实选模和候选文件的生命周期。全部有效成熟样本从一开始就属于训练集，没有验证尾段需要在训练结束后加回；如研究方案需要另行 refit，须显式声明为不同的训练流程。

滚动期次的模型 checkpoint 不等同于 epoch 内中断续训状态：精确续训还需 optimizer、scheduler、进度、早停计数及最佳指标、候选状态和随机数等必要状态。未保存所需状态时不得宣称支持精确续训；仅设置 `model_keep_num` 不能提供这一能力。

## 3.5 适用范围与既有实验

本次共同规范适用于后续新构建或明确迁移到该规范的实验。已有基模代码、运行中任务、已完成或已冻结实验继续保留各自原始配置、规则和产物；文档修订不会自动改写其代码，也不能将旧结果重新标注为采用新规则。

算法专属的网络、loss、状态构造及最终选模评分由各实验的代码与配置定义，不在本基础规范中展开。本次仅更新实例内规范及说明文档，不停止或启动任务；已有实例任务和巡检安排保持不变。新保留规则不授权清理任何既有实验，也不自动取消运行中实验对全部候选或历史 checkpoint 的验收要求。

## 3.6 模型文件与正常 output 的保留

后续新实验在配置与运行记录中明确模型文件保留策略，默认采用以下规则：

| 产物 | 默认规则 |
|---|---|
| 正式 `checkpoints/` 中的 `model`、`oldmodel` 等权重 | `model_keep_num="0"`，不要求落盘或长期保留 |
| epoch 候选权重 | 仅服务当前滚动期次的选模；恢复选中权重并完成本期选模核验、确认无后续消费者后，释放内存副本或清理本期临时权重；保留指标和选模记录 |
| optimizer、scheduler、RNG 等续训状态 | 不默认逐轮归档；需要故障恢复时显式启用，滚动替换最新完整状态及恢复所需的候选，成功完成且确认无依赖后按实验声明清理；失败或中断时保留已启用的恢复状态供排查 |
| 正常 `output/` / `output_D/` | 继续生成并保留该实验正常流程要求的全部预测、回测和评估产物，以及日志、配置和可复核记录 |

正常 output 包括 `alpha.parquet`、`alpha_history.pt`、实验要求的各 head 信号、IC/评价表、回测持仓与成交明细、PnL、图片与报告、训练日志、epoch 指标、选模记录及代码和环境来源。关闭模型保存不能成为跳过这些输出的理由；没有 checkpoint 也不应使正常输出验收失败。文件按用途区分，不能按 `.pt` 后缀批量禁用或删除，`alpha_history.pt` 等预测产物仍需保留。

需要独立推理、上线、teacher/student 权重依赖、重新生成预测、历史模型对比或断点恢复时，在实验说明中明确保留用途、期次或数量、路径和结束后的处理方式，再选择相应策略。正数 `model_keep_num` 仅在保留数量足以覆盖所有消费者时使用；确需全历史模型时才显式设置 `-1`。这些例外不恢复为所有实验的默认要求，也不要求为留档复制完整模型输出目录。

新实验接入时，使用实际安装的框架验证关闭 checkpoint 后选中权重仍在正确日期生效，正常预测、回测及评估可以完成，指标中的训练期次与选中 epoch 可追溯。无法仅靠内存完成的流程，采用已声明的最小必要持久化方案。验收以选模、时序和正常 output 的完整性为依据，不默认要求存在每期或每轮权重文件。

临时模型文件的处理仅限新实验明确管理的文件，遵守项目 AGENTS.md 的路径和清理边界；不得递归清空 `output`、删除原始数据或借此清理其他实验。此处调整的是后续实验的产物保留要求，不会自动修改框架默认配置或现有实验代码。

---

## 3.7 季度重训与模型生效

后续新实验统一按自然季度重训。实验起始时没有可用模型的，在首个满足训练历史和完整目标成熟条件的交易日完成首次建模；此后在每年 3、6、9、12 月的最后一个交易日各重训一次，季度内其余交易日沿用已有模型。首次建模恰逢季度末时，当天只训练一次。训练历史范围、模型与优化器重新初始化、早停和选模规则继续执行第 3.1～3.4 节。

季度末依据实际交易日历判定，不以固定交易日间隔替代，也不把回测区间截断后的最后一天自动视为季度末。研究代码须通过 `combo_base_path` 对应的 `ComboBase.isTrainDay()` 或实际安装包提供的等价调度接口明确实现季度频率，并在运行前核对计划训练日期、运行后记录实际训练日期及未完成原因；仅在目录名称或 XML 注释中写“quarterly”不构成实现。

训练时点和模型生效顺序遵循本次实际安装并加载的 combo2 包。框架先生成当期预测再训练时，新模型只能用于训练完成后的后续预测，不得回填或重算训练前已经生成的信号；各期训练仍只使用当时完整成熟的目标。季度重训不规定 `model_smooth_rate` 的取值，该参数由实验显式声明并按实际包语义核验。

用户明确指定其他频率的实验按其要求执行并记录；运行中、已完成及冻结实验继续执行原协议。本节只规定后续新实验的默认频率，不自动更改既有模型配置或触发训练。

---

# 4. Reference IC Loss

当实验使用标准 IC loss 时，统一采用有效股票集合上的 Pearson correlation。

模型输入：

```text
pred    [B, N]
target  [B, N]
w       [B, N]
```

其中：

$$
w_{b,i}\in\{0,1\}
$$

表示股票 \(i\) 是否参与日期 \(b\) 的监督。

---

## 4.1 每个日期独立计算 IC

有效股票数：

$$
n_b
=
\max\left(\sum_iw_{b,i},1\right)
$$

预测均值：

$$
\bar p_b
=
\frac{\sum_ip_{b,i}w_{b,i}}
{n_b}
$$

标签均值：

$$
\bar y_b
=
\frac{\sum_iy_{b,i}w_{b,i}}
{n_b}
$$

中心化：

$$
p^c_{b,i}
=
(p_{b,i}-\bar p_b)w_{b,i}
$$

$$
y^c_{b,i}
=
(y_{b,i}-\bar y_b)w_{b,i}
$$

Pearson IC：

$$
IC_b
=
\frac{
\sum_i p^c_{b,i}y^c_{b,i}
}{
\sqrt{
\sum_i(p^c_{b,i})^2
\sum_i(y^c_{b,i})^2
}
}
$$

分母小于等于 \(10^{-8}\) 时：

$$
IC_b=0
$$

batch loss：

$$
L_{IC}
=
\frac1B
\sum_b(1-IC_b)
$$

---

## 4.2 直接使用的实现

```python
def ic_loss(pred, target, w):
    w = w.to(dtype=pred.dtype)

    n = w.sum(dim=1, keepdim=True).clamp_min(1.0)

    pred_mean = (pred * w).sum(
        dim=1, keepdim=True
    ) / n

    target_mean = (target * w).sum(
        dim=1, keepdim=True
    ) / n

    pred_centered = (pred - pred_mean) * w
    target_centered = (target - target_mean) * w

    numerator = (
        pred_centered * target_centered
    ).sum(dim=1)

    denominator = torch.sqrt(
        pred_centered.square().sum(dim=1)
        * target_centered.square().sum(dim=1)
    )

    corr = torch.where(
        denominator > 1e-8,
        numerator / denominator.clamp_min(1e-8),
        torch.zeros_like(numerator),
    )

    return (1.0 - corr).mean()
```

这里：

* 股票在一个日期内等权；
* 每个日期先独立计算 IC；
* batch 内不同日期等权平均。

---

# 5. 当前 Reference Label

当前基础 benchmark 的训练目标使用 `label1d` 原始标签。

设连续五个原始单日目标为：

$$
\ell_d,\ell_{d+1},\ldots,\ell_{d+4}
$$

当前 Reference Label 为：

$$
R_d
=
5\ell_d
+
4\ell_{d+1}
+
3\ell_{d+2}
+
2\ell_{d+3}
+
\ell_{d+4}
$$

即：

```text
retDays = 5
weights = 5, 4, 3, 2, 1
```

这只是当前 reference training target。

新的 label 设计属于算法研究，可以替换这一部分。

---

# 6. Reference Label 的配套预处理

使用第 5 节的 Reference Label 时，沿用本节流程。设计其他标签时，实验须同时定义其预处理、监督 mask 和完整成熟时间；本节流程不自动应用于 rank label、多任务 label 等其他监督设计。

对聚合完成后的 \(R_d\)，在有效股票截面上依次执行：

### 1. Winsorize

使用：

```text
1% / 99%
```

分位数截断。

### 2. 中心化

$$
R_i
\leftarrow
R_i-\operatorname{median}(R)
$$

### 3. 标准化

$$
R_i
\leftarrow
\frac{R_i}
{\operatorname{std}(R)+10^{-8}}
$$

这里使用有效股票截面的总体标准差（`ddof=0`）。

### 4. 截断

$$
R_i
\leftarrow
\operatorname{clip}(R_i,-3,3)
$$

### 5. Max-abs normalization

$$
R_i
\leftarrow
\frac{R_i}
{\max_j|R_j|+\varepsilon}
$$

此步骤沿用 `normalize_by_max_abs` 的数值保护：最大绝对值大于 0 时除以 `max_abs + eps`；全零截面保持不变。Reference Label 在该预处理阶段使用 float32，`eps=1e-8`。

### 6. 无效股票

无效监督位置：

$$
R_i=0
$$

对应：

$$
w_i=0
$$

训练模型取得的是处理后的 target 和监督 mask：

```text
y, w
```

---

# 7. 股票间模型的统一输入语义

如果模型使用 cross-sectional interaction，则股票关系建立在完整股票主轴上。

对于股票 \(i\)，允许读取当前：

$$
available_j=1
$$

的其他股票信息。

因此股票关系模型的基本输入仍然是：

$$
[B,T,N,F]
$$

模型具体采用：

* full attention；
* latent tokens；
* graph；
* pooling；
* residual cross-sectional branch；
* 其他结构；

属于算法设计。

训练和预测使用相同的股票关系定义。

---

# 8. 模型输出

模型最终必须产生每只股票一个 alpha：

$$
\alpha\in\mathbb R^N
$$

多头、多任务等模型可以拥有内部辅助输出，但必须定义最终用于交易和评价的：

$$
\alpha_i
$$

研究模型自身能力时，默认直接使用当前模型输出，不混合旧模型：

```text
model_smooth_rate = 1.0
```

---

# 9. 下游交易链路

如果实验研究的是 combo 模型本身，则模型输出以后继续使用同一套项目交易链路：

```text
alpha
↓
portfolio optimizer
↓
target orders
↓
execution
↓
actual holdings
↓
PnL
```

组合优化器、股票池、风险约束、成交价、费用、T+1 和回测规则保持同一配置。

这样不同模型产生的实际收益差异才来自 alpha，而不是下游交易条件变化。

## 9.1 VWAP30 成交定价

回测成交价格统一使用采样时点对应的 30 分钟 VWAP。默认采样时点为 `100000`，价格字段为：

```text
IntraVwap.Vwap30.100000
```

该字段表示交易日当天 **10:00–10:30** 窗口的成交量加权平均价，完整价格在窗口结束后才可获得。生成该执行窗口订单的信号只能使用窗口开始前已知的信息。

在 `ResearchLoader.data_requirements()` 的数据声明序列中加入成交价格来源。以下 `root` 使用 `ashare_cache_path(self.config.cache_path)` 取得：

```python
DataItem(
    "execution",
    path=str(root / "1d_IntraVwap" / "IntraVwap.Vwap30.{ti:06d}"),
    delay=0,
)
```

`{ti:06d}` 按当前采样时点展开。成交价格源保持原始价格量纲，不列入 625 因子的 `model_input_sources()`，也不经过因子标准化、截断或除以 4。

XML 的关键连接配置为：

```xml
<config>
  <combo>
    <runtime sample_times="100000" />
  </combo>
  <backtest execution_price="execution:execution" />
</config>
```

此处只展示成交定价所需字段，合入实验的完整 XML 使用。`execution:execution` 的含义为 `source:column`，执行层据此读取当天、当前时点的价格。

## 9.2 交易前估值与日终估值

不同用途的价格按交易链路分别处理：

| 用途 | 价格口径 |
|---|---|
| 订单成交 | 当前执行窗口的 VWAP30 |
| 当日首个执行点交易前的持仓估值 | 当日 pre_close；无效时按框架使用此前有效估值 |
| 后续执行点交易前的持仓估值 | 上一执行点已知的有效价格 |
| 日终资产结算 | 当日有效收盘价，缺失处理沿用回测链路 |

当前窗口的 VWAP30 在窗口结束前尚不可知，不能用于窗口开始前的模型输入或优化器持仓估值。原始成交价无效时按交易有效性规则处理，不用前向填充的历史价格模拟成交。

训练 label 的定义与成交价格接口分别声明。使用第 5 节的 Reference Label 时继续执行对应聚合和预处理；使用 VWAP 派生其他 label 时，由实验明确收益区间、索引和成熟时间。

定价接口参考：`eg-torch/loader.py`、`eg-torch/config.xml`、`config.human` 的“日期口径与训练”和“成交回测与输出”部分。

## 9.3 组合优化器参数与版本口径

后续新实验、聚合 head 回测及已有信号的新增默认回放，除用户明确指定其他口径外，均使用**本次运行的解释器实际安装并加载的 Combo2 包的普通 OPT1 完整默认配置**。本文不规定具体 OPT 默认数值；默认值的唯一来源是该运行环境的安装包，不是本文历史版本、旧实验 XML、旧训练配置、项目根目录或 `vendor` 中的源码，也不是另一环境导出的参数表。

运行前按第 13.5.1 节核实包版本与模块来源。当前接口从安装包的 `config.DEFAULT_OPTIMIZER_CONFIG` 读取完整优化器默认配置，并从 `config.DEFAULT_CONFIG` 读取配套的 `strategy`、`backtest` 默认配置；使用独立深拷贝构造本次配置，不能修改包内全局默认对象。若后续包调整接口，应核对该版本公开接口与自带说明后读取等效的完整默认配置，不得退回手写数值或静默沿用旧配置。

“使用默认 OPT”包含完整 hard/soft 股票池、风险和行业约束，以及信号处理、资金、费率、现金比例和其他配套回测规则；不能只更新少数优化器字段而保留旧 XML 的其他覆盖值。仅按实验设置日期、输出与数据路径及第 9.1 节规定的信号时点和成交价格接入，逐项记录这些必要设置。其他偏离须有用户明确要求并列出差异，不得仍标作完整包默认回测。`runEval --simple` 等会改变约束的模式属于不同交易条件，不能替代普通默认 OPT1。

每次正式运行必须完成以下核验与留痕，产物保存在实例对应实验或回测目录：

1. 保存完整包默认快照、解析后的实际生效配置、包版本、解释器与模块实际路径和可核验的构建哈希；可使用 `opt_parameters.json` 及关联审计文件。数值由本次运行从包中读取并写入快照，不在规范或脚本中维护一份“默认值”。
2. 在正式优化前，逐项比较实际构造出的 optimizer 与包默认配置，包括所有约束列表；同时核对 strategy 和 backtest 的实际生效值，包括 `long_ratio`、`cash`、`fee_rate`、`reserve_cash` 等。除上述已记录的必要实验设置或用户明确指定的差异外，发现不一致时应先修正配置，不能把带有旧覆盖值的结果标为默认回测。
3. 明确标记配置所属阶段及对应输出路径。训练阶段的 `run_environment.json` 不能代替随后独立回测的配置记录；独立回测须保存自身的生效配置与运行时核验，完成记录应能关联到该快照和执行脚本。
4. 同一组模型比较使用一致的包构建、完整默认快照和执行口径；运行中或续跑时不得静默切换包默认值。包升级后，新增实验重新读取当次安装包默认值；复现既有结果则使用原快照与相应运行构建，并注明历史口径。比较不同版本的结果时披露默认值和执行行为差异，必要时在统一条件下重放信号。

第 9.1 节的 `source:column` 成交价接口及 VWAP30 口径继续适用。参数含义可以参考包自带说明和项目参数说明文档，但其示例或历史数值不能替代本次安装包的默认值。

已完成或已经冻结的旧实验保留原配置与产物；不得把旧结果重新标注为新参数口径。复现旧实验时显式使用其原始参数。比较新旧模型时，应把相应预测放到同一套交易参数下重新回测，并将新产物保存在独立的实验输出目录；不能把交易参数变化造成的 PnL、换手或持仓差异全部归因于模型。此处规定组合交易优化器参数，第 3 节用于拟合模型的 Adam、学习率、epoch、batch 和早停规则保持原定义。

---

# 10. 统一评价

模型至少从三个层面评价。

## 10.1 信号质量

统一评价：

* IC；
* RankIC；
* 分层目标表现；
* 按时间区间拆分的表现。

训练 label 可以改变，但同一组候选模型最终使用相同的 evaluation target 比较。

---

## 10.2 模型差异

统一观察：

* prediction correlation；
* poscorr；
* Jaccard / overlap 等当前评价器提供的组合差异指标。

---

## 10.3 实际交易

统一观察：

* gross return；
* net return；
* transaction cost；
* turnover；
* drawdown；
* long number；
* 当前项目评价器中的其他正式指标。

当前已经反复用于研究的：

```text
2020-01-01 ～ 2024-06-30
```

作为开发比较区间使用。

---

## 10.4 pnl_corr 与 pos_corr 的官方计算标准（2026-09-26 起）

本条适用于 `haoyi_models` 下所有模型、实验、回放与分析，以及后续对既有产物的新计算请求。用户未作特殊说明时，`pnl_corr` / `pnlcorr` 和 `pos_corr` / `poscorr` 均遵守以下要求。

1. **必须实际调用当前运行环境安装的官方 `comb-eval` 命令计算并以其结果为准。** 根据当前版本的 `--help` 选择对应子命令与参数；PnL 相关性使用官方 PnL 评价入口，信号或持仓矩阵相关性使用官方矩阵评价入口。不得仅引用官方定义、复制公式或声称“与官方等价”，却使用自行实现的计算替代官方结果。
2. **不得自行替换计算口径。** 不得用 NumPy/Pandas 的相关函数、自写 Pearson/Spearman、涨跌同号率、累计净值相关性、残差相关性等替代用户要求的正式 `pnl_corr` 或 `pos_corr`。官方对日期窗口、股票交集、零值、缺失值、最低有效数量、聚合方式及符号的处理，以实际版本和明确传入的参数为准；不得暗中修改。训练中的可微 loss 或诊断代理不等于官方评价指标，不得作为官方达标证据。
3. **明确输入并保存可复核记录。** 区分原始 alpha、独立多头/双边组合、opt 目标权重和实际成交后持仓；PnL 必须注明所用列、费用和账户口径。按用户要求对齐日期，报告实际有效日期或样本数量。保存包版本、完整命令、输入路径、参数、退出状态及官方输出，产物遵循实例留存规则。需要官方输出显示精度之外的数值时，可补充同一已安装官方包的公开接口结果，并与命令输出核对；不能以自行计算代替官方实现。
4. **仅有两种替代计算例外：** 用户明确要求其他方法；或经核查，官方 `comb-eval` 确实无法完成用户要求的计算。不能因为运行较慢、调用不方便或显示精度有限就改用其他公式。出现例外时，必须说明用户指定的特殊口径，或官方不支持的具体需求及核查证据，并明确标注替代方法、公式、有效范围和与官方口径的差异；替代结果不得标作官方 `pnl_corr` / `pos_corr`。
5. **无定义与失败必须如实保留。** 样本不足、零方差或无有效股票/日期等导致官方无法定义指标时，报告 N/A 和原因；不能为得到一个数字而填零、放宽门槛或静默改公式。安装、路径、输入格式等可修复的调用问题，应先修复官方调用；无法恢复时说明未完成，不得未经说明切换方法。

---

# 11. 留给算法设计的内容

以上基础规范确定以后，下面这些内容由具体实验自由设计：

### 模型

* MLP；
* TCN；
* GRU；
* Transformer；
* temporal model；
* cross-sectional model；
* temporal + cross-sectional；
* spectral / low-rank；
* 其他结构。

### 时间信息

* `tsDays`；
* temporal summaries；
* raw sequence；
* 时间 attention；
* TCN；
* recurrent structure。

### Label

* horizon；
* 未来收益权重；
* residual label；
* rank label；
* volatility-adjusted label；
* multi-task label；
* 其他监督设计。

### Loss

* Reference IC；
* ranking；
* pairwise；
* MSE；
* Huber；
* multi-task；
* diversity；
* portfolio-aware loss。

### 模型内部正则与监督接口

* 网络内部 dropout 和结构性正则；
* loss 中的辅助项及其权重；
* 主预测、辅助输出和监督 mask 的定义；
* 用于训练 IC 统计的预测接口及其与训练目标的对应关系；
* 实验最终模型选择规则、所需候选集合及其临时存储方式。

优化器、learning rate、epoch 上限、batch、仅使用训练集、early stopping 和随机种子按第 3 节固定。模型文件遵循第 3.6 节默认不长期保留、按需声明例外的规则；具体候选评分与最终模型选择由各实验单独声明，不改变无验证集和含 dropout 的训练 loss 早停规则。

---

# 12. 使用原则

以后设计具体算法时，默认直接继承本文中的：

```text
625 因子及其顺序
股票主轴
输入预处理
训练历史信息边界
固定 optimizer、epoch、无验证集与含 dropout 的训练 loss early stopping 等训练配置
Reference IC 实现
下游交易链路
统一评价体系
```

然后只需要明确：

```text
本实验具体设计什么？
```

固定的基础设施与训练配置按本文执行；在算法设计范围内，没有被实验主动修改的部分继续沿用本文默认定义。

新实验的文件组织和框架接口按第 13 节接入。

---

# 13. 框架接入与文件分工

## 13.1 数据源声明

输入因子和标签的数据源在 `ResearchLoader.data_requirements()` 中通过 `DataItem` 声明，包括来源路径、字段及日期延迟等信息。

`ResearchLoader.model_input_sources()` 指定作为模型输入的数据源及顺序；`ResearchLoader.model_target()` 指定训练标签的数据源和字段。

XML 的 `<loader>` 用于配置 dtype、历史起点、压缩和 `cacheDays` 等读取参数。当前接口使用大小写准确的 `cacheDays`，不接受旧的 `registry_cache_days` 配置键；具体版本支持范围按第 13.5.1 节核验。**当前配置解析器不接受 `<combo><data>...</data></combo>`，数据源清单应放在 `ResearchLoader` 中。** 参考旧模型文件时，需要将旧 XML 中的因子与标签声明转换为这一接口，同时保留因子顺序和日期映射；只在旧 `<data>` 上增加 `cacheDays` 不能完成接口迁移。

## 13.2 实验文件组织

一个实验可以由 `model.py` 和 `config.xml` 两个文件组织。`model.py` 中分别定义以下类：

| 类 | 职责 |
|---|---|
| `ResearchModel` | 网络、loss、训练循环、选模、保存加载和预测 |
| `ResearchLoader` | 继承 `ComboDataLoader`，声明数据源、输入顺序和标签；需要时覆盖标签处理方法 |
| `ResearchDataset` | 继承 `ComboTrainDataset`，规定股票轴等样本组织方式 |

XML 中对应的连接片段为：

```xml
<combo>
  <paths
    model_path="model.py"
    research_loader_path="model.py"
    research_dataset_path="model.py"
  />
  <runtime model_keep_num="0" />
  <loader dtype="float16" compression="none"
          data_start_ds="20160101" cacheDays="3000" />
</combo>
```

该片段只展示接口连接及新实验默认不保存正式 checkpoint 的设置，完整实验配置还需设置训练参数、日期及下游交易配置；模型代码自行保存的候选权重仍须按第 3.6 节单独管理。

这些类也可以分别放在 `loader.py`、`dataset.py` 中，XML 指向对应文件；文件拆分不改变接口职责。

## 13.3 公共数据层与实验扩展

文件读取、缓存、截面特征预处理、Reference Label 聚合及其预处理由公共数据层提供，标准实验继承这些实现。

设计其他标签时，在实验的 `ResearchLoader` 中按需覆盖 `gen_raw_target()`、`gen_target()` 或 `preprocess_target()`，并定义对应监督 mask 和时间语义。模型入口的 float32 转换、除以 4 和股票输入可用性判断由模型侧执行。

完整股票轴通过实验的 `ResearchDataset._build_validinsts()` 实现：

```python
def _build_validinsts(self):
    return torch.arange(
        len(self.loader.mask.code),
        dtype=torch.long,
    )
```

公共 Dataset 的默认股票选择是历史 base universe 并集，使用完整轴时必须接入上述覆盖方法。

## 13.4 样本接口与训练配置执行

当前默认 Dataset 的单样本返回格式为六项：

```text
(idx, ds, ti, x, y, w)
```

模型的 DataLoader、collate 和训练循环按此接口衔接。旧版本模型中可能存在四项样本解包，复用时需要核对，不能直接套用旧 collate。

XML 提供训练参数值，`ResearchModel` 负责读取并执行。第 3 节规定的 Adam、StepLR、整个 dataset 用于训练且不留出验证段、训练模式下含 dropout 的实际训练 loss 严格降低与 patience=5 早停、15 轮上限，以及本实验已声明的最终模型选择，需要在训练循环中落实；仅填写 XML 不会自动替换模型自身的训练逻辑。相关参数必须实际传入模型并在日志及独立运行记录中留痕；如保存模型 payload，也在其中记录。

正常训练、回测与接口解释以实际安装的官方 combo2 release 为准。2026-10-07 核验时，项目 `bin/runCombo` 转接虚拟环境中的安装包入口，包含该版本的 Dataset 与运行时初始化；显式执行项目根目录的 `python runCombo.py` 才使用仓库源码。显式要求冻结 wheel 的既有实验仍使用其锁定构建并核对来源；实际导入路径与有效配置必须一起记录。

## 13.5 运行环境与入口

本地 Windows 不具备完整因子环境；本地语法检查不能替代实例验证。实例先激活已配置环境并核实 Python、Combo2 版本及模块实际路径：

```bash
source /home/mahaoyi/.local/bin/haoyi-env.sh
cd /home/mahaoyi/projects/haoyi_comb2_organize/haoyi_models/<experiment>
unset PYTHONPATH
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MOSEKLM_LICENSE_FILE=/home/mahaoyi/mosek/mosek.lic
command -v python runCombo runEval
python -c 'import sys, importlib.metadata as m, runCombo, comb2; print(sys.executable); print(sys.version); print(m.version("combo2")); print(runCombo.__file__); print(comb2.__file__)'
runCombo --config config.xml
```

`<experiment>` 替换为实际实验目录。激活脚本将项目 `bin/` 放在 PATH 前部；2026-10-07 核验时，`runCombo` 使用安装包，项目 `runEval`、`combo-hello-world` 仍使用仓库源码，`comboOpt1` 使用安装包。需要官方评估入口时，在激活环境后显式执行 `python -P "$VIRTUAL_ENV/bin/runEval" ...`。核实 `command -v`、解释器、安装版本和实际导入路径，不能仅凭 distribution 版本推断运行源码。不要因一次模型启动顺带更新依赖。

同日安装包为 combo2 1.1.6，wheel 位于 `/home/data/shareddata/comb_pkg/`。该版本 `malloc_top_pad_mb` 默认 1024，需将分配器保留量计入内存预算；`fixbs` 自 1.1.5 起默认 true，会改变固定 book、成交及换手口径。同日直接读取包默认配置的 `fee_rate` 为 0.0015（单边 15bp）。这些是核验快照，正式实验仍按第 9.3 节读取并保存当次完整默认值；历史结果保留其原始版本与配置。

经用户授权的长期队列使用 `nohup`，stdout/stderr 写入项目 `manual_logs`，队列记录退出码。开发、修改与验证直接在 Notebook 项目完成；Windows 目录仅为历史副本。比较旧实验时记录版本与接口差异。

### 13.5.1 Wheel 以实际运行环境为准

Combo2 发布包目录为 `/home/mahaoyi/.local/share/haoyi-dependencies/20261001/wheelhouse/`。运行前核实实际安装版本；需要更新环境时，先检查该目录中实际可用的 wheel、Python ABI 和平台，不根据本文历史版本号拼接包名，也不因目录中出现新包而自动升级。

```bash
find /home/mahaoyi/.local/share/haoyi-dependencies/20261001/wheelhouse -maxdepth 1 -type f -name 'combo2-*.whl' -print -exec sha256sum {} \;
python -m pip show combo2
python -c 'import importlib.metadata as m; d = m.distribution("combo2"); print(d.read_text("direct_url.json") or "安装来源未记录，需另行核对"); print(d.locate_file("comb2_templates/config.human"))'
```

以上命令在第 13.5 节激活的同一环境中执行。发布目录中的 wheel、已安装 distribution 和实际加载的模块是三个需要核对的对象：目录中有某个包不代表当前环境正在使用它；仅核对版本号也不能排除同版本不同构建或仓库源码遮蔽。

每次正式实验在实例的对应实验目录中记录：核验时间、Python 版本和解释器路径、Combo2 安装版本、实际加载的模块路径、所用 wheel 的完整文件名与 SHA-256、关键依赖版本和解析后的有效配置。`direct_url.json` 可辅助核对安装来源；来源无法确认时明确记录，不能把目录里任意同版本 wheel 的哈希当作安装包证据。对比实验使用一致的运行构建，或明确披露版本及默认行为差异。

2026-09-21 本地源码的 `VERSION` 已更新为 `1.0.8`，这是历史源码记录；组合优化器默认值按第 9.3 节从本次实际加载的安装包读取。源码同步与 wheel 安装是两个步骤；不得根据仓库 `VERSION` 推断实例已经运行 1.0.8，也不得用本地默认值代替实际 wheel 的有效配置。新增 `docs/PROFILING.md` 是其他环境的历史性能报告，其中的临时 profiling 代码已撤回；报告中的命令并非当前可直接使用的框架入口，其实验训练参数和耗时不替代本规范或当前实验的实测结果。

2026-09-20 核验快照：默认环境 `/home/mahaoyi/.venvs/haoyi_comb2_py313` 使用 Python `3.13.15`、已安装 `combo2==1.0.6`，`runCombo` 与 `comb2` 均来自该环境的 `site-packages`；发布目录包含 `combo2-1.0.6-cp313-cp313-linux_x86_64.whl`，其 SHA-256 为 `ca4aa0979268b39406a111040eedd62e0c3bf94086e3d8600690f0c69907dc7b`。已安装 distribution 的 `direct_url.json` 指向该文件，来源哈希与实际文件一致。这是当次核验记录，不是未来必须安装 1.0.6 的要求。

本文保留的 Combo2 1.0.3 和 2026-09-16 验证记录属于历史证据。升级后重新核对配置解析、Dataset 契约、训练/预测时序及交易接口；涉及缓存或 Dataset 接口变化时执行第 13.9 节验收。“以实际 wheel 为准”只确定框架接口与运行事实，不放宽第 1～12 节的 benchmark 要求，也不表示新 wheel 已自动通过全部实验验证。

## 13.6 训练窗口与读取分块

当前 `LoaderConfig` 使用 `load_chunk_days` 控制读取分块；XML 在 `runtime.load_chunk_days` 中配置。`cacheDays` 已移除，不再传入配置或用于验收断言。

| 项目 | 作用与边界 |
|---|---|
| `runtime.max_train_days` | 训练历史输入窗口；按对应实验协议设置，不随读取分块改变 |
| `runtime.load_chunk_days` | 单次来源读取的交易日数；按实际可用内存设置 |
| Dataset 的 `X/Y/W` 与监督张量 | 训练样本存储；保留实验的 dtype、标签和掩码定义 |

```xml
<combo>
  <runtime max_train_days="2000" load_chunk_days="32" />
  <loader dtype="float16" compression="none" data_start_ds="20160101" />
</combo>
```

来源缓存通过作用域管理并释放工作数据；不能把历史缓存容量当作当前内存上限。评估内存须合计来源数据、Dataset、监督张量、模型及 batch。float16 约定仅针对预处理后的模型输入；原始价格和 float32 监督保持各自精度。

## 13.7 Dataset 必须在 fit 前完成缓存读取记录

2026-09-16 在 Combo2 1.0.3 中核验的训练调用顺序包含以下步骤；后续 wheel 需按第 13.5.1、13.9 节复核，实验仍须在 Dataset 构造返回前完成真实读取记录：

```text
构造 ResearchDataset
→ 读取并复制 dataset.cache_reads
→ ResearchModel.fit(dataset)
→ 使用读取清单为后续训练准备来源缓存
```

2026-09-16 的风险残差 B 组曾在此处报错：

```text
AttributeError: 'ResearchDataset' object has no attribute 'cache_reads'
```

原因是实验 Dataset 只在构造函数中设置样本坐标，等 `fit()` 收到 XML 参数后才调用公共 Dataset 构造函数。新版框架在此之前就需要读取清单，因此延迟初始化破坏了接口契约。

**不得用 `cache_reads={}`、吞掉异常或移除预热来绕过问题。** 框架会在 `fit()` 前复制该清单；即使稍后补上真实读取记录，先前复制的空清单仍会使后续训练预热失效。

对于依赖 XML 模型参数才能构建监督的实验，按以下方式接入：

1. 使用实验自己的 `ComboBase` 子类，在公共初始化完成、Loader 已创建后，将 `node.model_config` 传给 Loader。需要时通过 `configure_experiment()` 保存并校验实验定义。
2. 在 `ResearchDataset.__init__()` 中调用公共 Dataset 初始化，完成实际特征读取、标签/掩码构建和 `cache_reads` 记录，再将 Dataset 返回给框架。
3. 保留公共样本坐标、dtype/codec、`X/Y/W` 和必要元数据；多头监督可在同一次构造中另外保存 float32 张量。每个监督样本只构建一次。
4. `fit()` 如仍调用准备方法，该方法只验证定义一致并直接返回，不能重复分配或重算。模型与优化器仍按第 3 节规定初始化。

此类实验需要额外连接自定义训练入口：

```xml
<paths model_path="model.py"
       research_loader_path="model.py"
       research_dataset_path="model.py"
       combo_base_path="model.py" />
```

只有实现了相应 `ComboBase` 子类的文件才能配置为 `combo_base_path`；普通实验若直接满足公共 Dataset 构造契约，无须新增子类。自定义入口应继续调用公共训练、预测和预热流程，不复制一份旧版训练循环。若同一文件还支持保存 alpha 的回放，应明确区分训练与回放分支，回放不得意外触发重训。

## 13.8 预取作用域与跨月复用

批量 `prefetch_features()`、`prefetch_targets()` 的预取和数据消费应位于同一个 `loader.cache_scope()` 中。公共 Dataset 已处理自己的读取作用域；实验新增的批量读取应遵守同一约定。单次 `source_history()`、`source_field()` 会自动建立请求作用域，不能把不同作用域的临时工作缓存当作永久存储。

`cache_reads` 必须来自真实读取并包括训练所依赖的所有来源、日期及采样时点。除输入因子外，还可能包括标签、公共掩码、风险字段、成交价及标签成熟性检查来源。风险残差实验的完整记录为 636 个来源，不只是 625 个输入因子；其他实验的来源数按实际声明和读取确定。

跨月缓存复用是同一进程内对重叠历史的复用，保留窗口以外的数据、新日期或显式刷新仍可能需要重新读取。缓存只改变读取与复用方式，不改变 `DataItem.delay`、标签成熟边界、监督 mask、因子预处理或股票映射。原始来源被更新后，不得无条件复用旧 Dataset 或旧缓存；按运行接口刷新或重建。

## 13.9 缓存接入的验收要求

缓存或自定义 Dataset 接口修改后，应完成以下检查，而非只验证 XML 可解析或 loader 能构造：

1. 使用实例当前 wheel 解析配置，确认 `cacheDays` 传递到真实 Loader/Registry，并核对完整交易日覆盖。
2. 在 Dataset 构造返回时核对真实 `cache_reads`。运行公共 `ComboBase.Train`，覆盖至少两次滚动训练与后续预热，不能仅手动调用 `prepare_training_targets()` 后声称训练入口通过。
3. 对仍处于保留窗口且没有源刷新的一段真实历史重复读取，检查 `last_scope_stats.raw_points/raw_chunks`；重复请求应命中缓存，向后扩展应只读取未缓存部分。不能只用耗时或操作系统页缓存判断命中。
4. 对修复前后同一数据构造方式核对样本日期/时点、股票顺序、特征、目标和掩码；纯缓存修复不得顺带改变模型、loss、训练超参数或交易条件。
5. 使用完整因子与股票轴做有界的真实数据及 CUDA 前反向检查，并确认正式运行越过原报错位置。小规模通过不等于完整回测通过。
6. 验收尽量使用内存数据和内存检查点；临时回放文件应在本次检查结束时清理。不得以验证缓存为由留下测试产物，或删除当前运行及成功实验的产物。

2026-09-16 已完成的接入验证包括：baseline 与风险残差 Loader 各抽查 3 个真实因子、滚动 54 个月；单因子 2594 日全区间留存和重复读取零新增读盘；B 两项各两次合成滚动 Train、636 来源预热；九组小样本在复现旧延迟初始化与新构造方式之间逐元素一致；真实 625 因子、5642 股票的标签及 CUDA 检查。以上证据验证缓存接口与数据一致性，不构成完整实验收益或样本外有效性的结论。
