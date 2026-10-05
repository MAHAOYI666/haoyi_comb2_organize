# comb2-organize

combo2 将研究员的 Memmap 数据、源级降维、数组模型、训练、预测、opt2 成交回测和评估接到同一流程。每天一个或多个 sample_times 使用相同接口。

发布包名为 combo2，版本由 VERSION 管理；发布记录见 RELEASE.md。

## Notebook 运行环境

项目位于 `/home/mahaoyi/projects/haoyi_comb2_organize`，相关模型位于相邻的 `combo26q4`。

```bash
source /home/mahaoyi/.local/bin/haoyi-env.sh
cd /home/mahaoyi/projects/haoyi_comb2_organize
runCombo --help
runEval --help
```

激活后项目 `bin/` 中的命令使用当前源码。实验显式导入的已安装 combo2 保持独立，冻结实验仍按其包版本与哈希验证。Python 解释器为 `/home/mahaoyi/.venvs/haoyi_comb2_py313/bin/python`。

行情缓存默认 `/mnt/cache`，因子位于 `/mnt/factors/QsimPool` 和 `/mnt/factors/ZsimPool`；这些输入只读。运行输出使用各配置声明的项目内目录。`cache_path` 指向 AshareCache 的父目录，XML 路径支持 `~` 和环境变量展开；相对路径以 XML 所在目录为基准。

许可证由 `MOSEKLM_LICENSE_FILE` 指定，默认 `/home/mahaoyi/mosek/mosek.lic`；`runEval --mosek` 可显式覆盖。GPU 训练须在获得 GPU 的 Notebook 或作业中执行，CUDA 依赖已安装不代表当前有 GPU。

## 安装方式

下载这个仓库即可。当前仓库已经临时内置所需源码：

```text
vendor/
  comb2/
    comb2/
  comb2-pcmaster/
    comb2_pcmaster/
```

`vendor/` 只包含运行所需源码，不包含原仓库 git history、构建产物、缓存和历史输出。

默认持仓策略依赖 `Mosek==11.0.25`。受保护 wheel 已声明该依赖；直接从源码运行时需在当前 Python 环境安装 MOSEK，并通过环境变量提供有效许可证：

```bash
export MOSEKLM_LICENSE_FILE="$HOME/mosek/mosek.lic"
```

源码仓库根目录包含已脱敏的 `mosek.lic`。受保护 wheel 不内嵌许可证；wheel 部署环境仍需通过 `MOSEKLM_LICENSE_FILE` 指向获准使用的副本。

完整配置和研究员可重写接口说明见 `config.human`。新 research 目录建议保留一份同名文件，作为模型、loader、dataset 的接口手册。

### 已安装 combo2 1.1.4：单信号回测

Notebook 的 Python 环境已升级为负责人提供的 `combo2==1.1.4` wheel。安装包与仓库源码独立：项目 `bin/runCombo`、`bin/runEval` 仍运行当前源码；新增 `comboOpt1` 来自虚拟环境的安装包。仓库 `VERSION` 记录源码版本，不随 wheel 安装改写。

```bash
source /home/mahaoyi/.local/bin/haoyi-env.sh
comboOpt1 signal.parquet --config config.xml
```

`--config` 可省略，此时使用包内默认参数。信号研究、汇报和讨论建议在所用 XML 的 `<backtest>` 上显式设置 `fixbs="true"`；1.1.4 的默认值仍为 `false`。最小配置示例：

```xml
<config>
  <constants cache_path="/mnt/cache" />
  <backtest fixbs="true" />
</config>
```

根据 1.1.4 包内说明，`fixbs=true` 固定每日 book 为 `cash`，盈亏不滚入本金，使用零碎股数且不受现金约束；并非仅修改收益曲线的展示方式。以上配置适用于新版安装包，不代表当前仓库源码已同步支持该参数。

1.1.4 同时修复了回测问题，重跑结果可能与旧版不同；历史结果保留原版本和口径，比较时注明版本及 `fixbs` 设置。依赖 1.1.3 或其二进制哈希的冻结实验，回放前需使用独立的匹配环境，不修改历史校验以适配新包。

### 默认优化器与 simple 模式

默认 opt1 参数的逐项说明见 [opt1_parameters.md](opt1_parameters.md)。

当前默认配置使用简化 daily-VA optimizer 参数：`maxtvr=0.08`、`max_weight=0.008`、`min_participation_ratio=0.1`、`parti_penalty=0.05`，并保留完整的 hard/soft universe、risk 和 industry group 列表。

直接 daily VA 评估默认使用上述参数；显式传入 `runEval ... --simple` 时，保留上述四项数值并清空各 hard/soft 列表。默认和 simple 的评估结果会写入不同目录。

`config.eg.old.xml` 保留了旧版完整 optimizer 样例，供迁移和口径对比使用。它不会被自动加载；使用时请复制其中的 optimizer 属性，并按实际实验修改路径、日期和输出目录。

## 使用

```bash
combo-hello-world -y
runCombo config.xml
runEval config.xml
runEval myposition.parquet target.parquet
runEval myposition.parquet target.parquet --simple --ti 093000
```

在 ResearchLoader.data_requirements 中声明数据与 delay，在 process_source 中对分钟数据按时点降维。通过 model_input_sources、model_target 选择 X、Y；配置 cache_path 时默认交集使用 delay=1 的 BaseUnivMask、NoNewStockMask、LimitMask，model_validity_source 可显式替换默认筛选。Python 数据路径相对研究员文件解析。

数据集返回 (idx, ds, ti, x, y, w)。x 是 [tsDays, stock, feature] 普通张量，窗口取过去 tsDays 个交易日的同一时点快照。模型实现 fit、predict(x_window, di=..., ti=...)、save、load。

完整配置、索引示例、扩展接口及优化器参数见 [config.human](config.human)。示例见 [eg-torch](eg-torch) 和 [eg-lgbm](eg-lgbm)。

### 通用训练与早停规则

后续新实验取消验证集，整个 dataset 用于训练，不留出验证尾段。在各实验声明的监督历史范围内，按训练发生时点筛选完整成熟的目标样本，所有满足监督条件的样本均参与梯度训练。不再执行训练/验证边界 purge；目标的全部依赖仍须在本次训练发生前完整可知。训练日期、时点与有效监督股票集合在训练前固定。

每个完整 epoch 汇总训练模式下含 dropout、实际用于反向传播的训练 loss（`train_loss`），不另做关闭 dropout 的评估来替代该监控量。loss 各项及其权重由实验预先声明；多个时点先在日内等权，再按日期等权汇总。只有严格低于历史最佳值才算改善（`min_delta=0`），相等不算改善；连续 5 轮未改善时停止（`patience=5`），最多训练 15 轮。早停只控制训练时长，最终 checkpoint 遵循各实验明确声明的选择规则。完整要求见[基础规范第 3.3 节](haoyi_models/UNIFIED_RESEARCH_BENCHMARK_PROTOCOL_20260912.md#33-训练集与监督边界)和[第 3.4 节](haoyi_models/UNIFIED_RESEARCH_BENCHMARK_PROTOCOL_20260912.md#34-early-stopping-与-checkpoint)。

这项规范适用于后续新实验；现有代码不会因文档更新自动改变，运行中或已经冻结的旧实验保留原规则。本次仅调整本地文档，不部署、不启动新任务，也不改变现有任务及巡检。

haoyi_models 后续新实验默认显式配置 `model_keep_num="0"`，不要求保存正式 checkpoints；epoch 候选权重仅为选模临时保留，续训状态按需启用。预测、回测、指标、日志和其他正常 output 继续完整生成并保留，`alpha_history.pt` 等预测文件不属于可省略的模型权重。需要独立推理、teacher 权重或恢复能力时，在实验中声明必要的保留范围；运行中及既有实验不自动迁移。完整规则见[基础规范第 3.6 节](haoyi_models/UNIFIED_RESEARCH_BENCHMARK_PROTOCOL_20260912.md#36-模型文件与正常-output-的保留)。

## 回测和评估

execution_price 显式指定 source:column。日内使用 opt2，执行层保留 T+1 锁定、真实持仓和累计换手，每天结算一次。原始成交价、涨跌停和停牌状态共同决定可交易池；不可交易旧持仓冻结，策略若返回池外订单会立即失败。`StockMask2.StockListedDays` 从有效变为缺失时，已有持仓在优化前按零值核销并写入 `settlements.csv`；临时停牌仍沿用最后估值。默认策略的风险、行业和相对方差使用可配置 benchmark，默认 000905.SH；ZZ500 股票池约束和回测报告评价基准独立。

alpha.parquet 使用 (date,time) 索引。`runEval` 分为两部分：`runEval alpha1.parquet alpha2.parquet run/read` 的双 parquet 日频分组回测与 PnL/VA 是主流程，实现在 `runEval.py`；其余 config overall、`--sim`、`--pnl`、`--va`、`--corr`、`--exposure` 模式实现在 `evals/comb_eval/run_eval_other.py`，保留原 CLI 兼容。`--skip-exposure` 和 `--skip-deciles` 只影响 overall。overall 中的十档统计是原始 target 的分组均值，不是各档实际成交 PnL。

双 parquet VA 的所有 PnL 路径（0.00 target、各混合权重、1.00 信号）使用同一组日期，不一致时报错。未传 `--start` 时，起点是两个信号都有非零值的第一个共同日；传入 `--start`/`--end` 时严格使用该窗口，便于不同种子或变体在同一路径上比较：信号开头最多允许 5 个全零日（当天只用另一个信号），超过即报错；窗口内缺少任一交易日也报错，不会悄悄缩短路径。

## 内存和监控

<combo><loader compression="fp4" /></combo> 配置训练特征和预测缓冲区的编码。来源数据仅在当前读取块中复用，作用域结束后释放；训练数据集保存滚动窗口各时点的 X/Y/W，取样时切片、解码并执行窗口变换；预测按时点复用滚动缓冲区。none/fp4/fp8 使用现有张量 Codec；成交价格和原始目标不经过特征压缩。需按训练天数、时点数、股票数、特征数、模型和临时读取块评估内存；源数据更新后需重建数据集。

monitor 可记录读取、训练、预测和回测耗时及内存。runCombo 在各主要阶段、回测循环每月末和每次训练开始/结束输出一行 `[STAGE|时刻]`，附累计耗时。长任务须按 cgroup 可用资源评估峰值。
