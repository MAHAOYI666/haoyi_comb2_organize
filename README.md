# comb2-organize

`comb2-organize` 用来把 research 模型接到内置的 `comb2` 和 `comb2-pcmaster` 源码上。日频模式完成训练、每日信号生成和回测；日内模式完成训练、逐时点信号生成和 IC 分析。

发布记录、版本号和当前 wheel 安装目标统一记录在 `RELEASE.md`。
唯一发布包名为 `combo2`，版本号只从仓库根目录 `VERSION` 读取。

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
export MOSEKLM_LICENSE_FILE=/path/to/comb2_organize/mosek.lic
```

源码仓库根目录包含已脱敏的 `mosek.lic`。受保护 wheel 不内嵌许可证；wheel 部署环境仍需通过 `MOSEKLM_LICENSE_FILE` 指向获准使用的副本。

完整配置和研究员可重写接口说明见 `config.human`。新 research 目录建议保留一份同名文件，作为模型、loader、dataset 的接口手册。

## research 流程示例

下面以 `eg-lgbm` 为例说明完整流程。

### 1. 准备 research 目录

示例目录结构如下：

```text
eg-lgbm/
  model.py
  config.xml
  output/
```

其中：
- `model.py`：research 模型实现
- `config.xml`：运行配置
- `output/`：输出目录

### 2. 编写 research 模型

可以参考 `eg-lgbm/model.py`。模型需要提供一个 `ResearchModel` 类，并实现以下接口：

- `fit(dataset)`
- 日频：`predict(x_window)`
- 日内：`predict(x_window, di=..., ti=...)`
- `save(path_or_buffer)`
- `load(path_or_buffer)`

日频训练样本是 `(idx, x, y, w)`；日内训练样本是 `(idx, di, ti, x, y, w)`。两种 `predict` 接口中的 `x_window` 都是按频率分组的 `FeatureGroups`，不是单个 concat tensor。常见 shape：

- `x_window["1d"]`: `[ts_days, stock, feature]`
- `x_window["5m"]`: `[ts_days, stock, 49, feature]`
- `x_window["1m"]`: `[ts_days, stock, 239, feature]`

如果 config 没有声明某个频率的输入，对应 key 不会存在。模型初始化时会额外收到 `freq`、`freqs`、`num_features_by_freq`、`num_features`；日内模式还会收到 `target_freq` 和 `target_times`。

### 3. 编写配置文件

可以参考 `eg-lgbm/config.xml`。

所有 XML 里的相对路径都会按 `config.xml` 所在目录解析，不依赖运行命令时的当前目录。换机器时，通常只需要修改 `<constants>` 中的机器相关根目录。

关键配置包括：
- `constants.cache_path`：`AshareCache` 的父目录；行情、mask、label、Barra 和回测路径均从这里派生
- `constants.output_root`：唯一输出根目录；日志、alpha、checkpoint、回测和评估目录自动从这里派生
- `constants.freq`：执行模式，支持 `1d`、`5m`、`1m`，缺省为 `1d`
- `strategy.optimizer`：默认 MOSEK 持仓优化器参数；可在 `<strategy><optimizer ... /></strategy>` 中逐项覆盖
- `combo.paths.model_path`：指向 research 目录下的 `model.py`

`freq="1d"` 时，数据区使用 `role="factor"` 和最多一个 `role="label"`。`freq="5m"`/`"1m"` 时，必须有且仅有一个同频 `role="target"`，其余 item 都是模型输入；配置与 target 频率不一致会直接报错。

`builtin.factorsim` 的路径规则：

- 绝对路径：直接读取
- 相对路径：相对 `config.xml` 或 data-pack 文件所在目录解析

普通 factor pool 使用 `<item path="...">` 中的绝对路径，或者相对当前 XML/data-pack 的路径。固定的 label、日内 returns target、mask 和 Barra 数据由 `constants.cache_path` 统一派生。

示例里：
- `model_path="model.py"`
- `output_root="output"`

两种模式都会使用 `output/train.log`、`output/alpha_history.pt`、`output/alpha.parquet` 和 `output/checkpoints/`。日频另有 `output/daily_ic` 与 `output/backtest/`；日内另有 `output/intraday_ic.csv` 与 `output/ic_by_time.csv`。

### 4. 运行 organize 入口

可以从任意目录执行：

```bash
python3 /path/to/comb2-organize/runCombo.py /path/to/research/config.xml
```

或者：

```bash
python3 /path/to/comb2-organize/runCombo.py --config /path/to/research/config.xml
```

研究员只需要把命令里的 `runCombo.py` 和 `config.xml` 换成自己的实际路径。

### 5. 评估已有输出

`runEval.py` 当前只评估 `freq="1d"` 的日期索引 alpha；日内输出直接查看 `intraday_ic.csv` 和 `ic_by_time.csv`。日频评估可以和 `runCombo.py` 一样从任意目录执行：

```bash
/path/to/comb2-organize/runEval.py /path/to/research/config.xml
```

或者：

```bash
/path/to/comb2-organize/runEval.py --config /path/to/research/config.xml
```

如果当前就在 `comb2-organize` 仓库目录下，也可以直接：

```bash
./runEval.py /path/to/research/config.xml
```

入口会先检查 config 对应的必要输出文件是否齐全：

```text
<output_root>/alpha.parquet
```

如果缺失或为空，会打印不齐全的文件列表并退出。齐全后会基于 `alpha.parquet` 和 config 指向的 label/cache 重新计算 IC、PNL、分组回测、Barra 暴露和 CAP corr，并在 `<output_root>/eval_report/` 下生成 summary、检查表和 `signal_analysis.png` 长图，不依赖已有 `daily_ic` 或 `backtest/daily_pnl.csv`。

默认会从 `constants.cache_path/AshareCache/1d_DailyLabel` 读取 `DailyLabel.vwap30_label1d` 和 `DailyLabel.vwap30_label5d`。如果要显式指定本地 label 表：

```bash
./runEval.py /path/to/research/config.xml \
  --label /path/to/vwap30_label1d.csv --label-is-table \
  --label-5d /path/to/vwap30_label5d.csv --label-5d-is-table
```

不依赖 `config.xml` 的单项模式统一改为直接读取本地 parquet/csv：

```bash
./runEval.py --sim /path/to/daily_ic.parquet
./runEval.py --pnl /path/to/daily_pnl.parquet
./runEval.py --corr /path/to/left.parquet /path/to/right.parquet
./runEval.py --va /path/to/base_daily_pnl.parquet /path/to/new_daily_pnl.parquet
./runEval.py --exposure /path/to/alpha.parquet --cache-path /path/to/Cache
```

这些单项模式不再读取 `config.xml` 兜底补参数；如果需要 `config` 驱动的整体评估，继续使用 `runEval config.xml`。

## 数据压缩选项

实际使用时，只需要在 XML 的 `<combo><data ...>` 节点里增加 `compression` 属性即可：

```xml
<data dtype="float16" compression="fp4" data_start_ds="20160101" data_offset="1024" />
```

可选值：
- `none`：默认值，不压缩，行为与原始流程一致
- `fp4`：压缩特征存储，主要用于降低 CPU 内存占用
- `fp8`：使用 PyTorch float8 存储；是否可用取决于当前 PyTorch 运行环境对 CPU float8 相关算子的支持

压缩只作用于 comb2 内部的特征缓存存储：
- `ComboTrainDataset.X`
- `ComboBuffer.buffer`

不压缩 `Y`、`W`、mask、label、因子文件、模型输入输出，也不改变 `cs_zscore`、`truncate`、`nan_to_num` 等数据预处理逻辑。模型侧拿到的仍然是解码后的浮点张量，所以体感上主要就是在 `config.xml` 里指定压缩方式。

缓存说明：

- 3D `builtin.factorsim`（例如 `1m_Grid1mBar/*`、`5m_Intv5mBar/*`）必须在 `<item>` 声明 `freq="1m"` 或 `freq="5m"`，loader 会保持完整 cube 维度，不在数据层降维。
- 模型侧拿到的是按频率分组的 `FeatureGroups`：`x["1d"]` 为 `[stock, feature]` 或 `[ts_days, stock, feature]`，`x["5m"]` / `x["1m"]` 为 `[stock, bar, feature]` 或 `[ts_days, stock, bar, feature]`。
- 当前固定时间轴：`1m` 每天 `239` 个时点，`5m` 每天 `49` 个时点，且 `5m` 序列包含最后一个单独的 `15:00:00` 时点；source 时间轴必须完全一致。
- 不支持 `nbar`，也不支持 `last/mean/std/sum/max/min` 这类会改变维度的 data op。日内聚合应放在 `ResearchModel` 或自定义 `ResearchLoader` 中。

## 可重写接口

可在 `<combo><paths>` 中指定：

```xml
<paths
  model_path="model.py"
  research_loader_path="loader.py"
  research_dataset_path="dataset.py"
/>
```

`loader.py` 里定义 `ResearchLoader(ComboDataLoader)`，常用 hook：

- `preprocess_feature_group(freq, feature, ds)`：每个频率 group 的单日预处理；`1d=[stock,F]`，`5m/1m=[stock,bar,F]`。
- `preprocess_daily_features(feature, ds)`：只改日频默认预处理。
- `preprocess_label(label_values, valid_mask, ds, ret_days)`：改 label 标准化和样本权重。
- `preprocess_target(target_values, valid_mask, ds)`：改日内 target 标准化和样本权重。
- `transform_feature_window(feature_window, target_ti=..., stage=...)`：改训练/预测窗口处理；日内模式必须保留目标时点的因果裁剪。

在 loader hook 中读取已声明的数据，使用 `self.registry.get_data(name, start_ds, end_ds)`；返回保留 date 维：`1d=[R,N]`，`5m=[R,49,N]`，`1m=[R,239,N]`。单次日期范围不能超过 `registry_cache_days`。

`dataset.py` 里定义 `ResearchDataset(ComboTrainDataset)`，常用 hook：

- `_build_validinsts()`：改训练股票池。
- `__getitem__(idx)`：改训练样本结构；如果改返回值，必须同步修改 `ResearchModel.fit()`。

更完整的输入、输出和 shape 契约见 `config.human` 的 “ResearchLoader 和 ResearchDataset” 章节。

## 运行结果

执行后通常会产生这些输出：

- `output/train.log`
- `output/alpha_history.pt`
- `output/alpha.parquet`
- `output/checkpoints/` 下的模型文件
- 日频：`output/daily_ic`、`output/backtest/daily_pnl.csv`
- 日内：`output/intraday_ic.csv`、`output/ic_by_time.csv`，不调用日频回测

## 性能监控

性能监控默认关闭，不影响训练、预测和日频回测流程。需要时可在 XML 顶层加入：

```xml
<monitor enabled="true" output_path="output/perf_metrics.csv" collect_gpu="true" sync_cuda="false" />
```

启用后会记录 `setup`、`combine`、`alpha_convert`、`backtest_step`、`backtest_finalize`、`alpha_analysis` 以及可选的 `combo_*` 阶段，包含耗时、CPU 进程内存和可用的 PyTorch CUDA 显存信息。默认不做 CUDA synchronize，避免明显改变运行时序。

## 当前目录说明

- `runCombo.py`：research 运行入口；日频接入 `comb2-pcmaster` 回测，日内生成逐时点 alpha 和 IC
- `runEval.py`：日频整体评估入口，检查 config 输出是否齐全并生成本地评估报告
- `runEval.py --sim/--pnl/--corr/--va/--exposure`：单项模式，直接读取本地 parquet/csv
- `runEval.py --corr left.parquet right.parquet`：默认只统计最近 240 个重叠交易日的日频截面相关
- `config.py`：配置解析与默认参数
- 默认 config 使用内置 MOSEK `AlphaStrategy`：以原始零点划分正负 alpha 并分别归一化，在中证 500、BarraCNE5、换手、单票、有效持股数和相对方差约束下生成 long-only 目标仓位；换手基准使用真实成交持仓并归一化到股票 book
- `vendor/comb2`：临时内置的 `comb2` 源码
- `vendor/comb2-pcmaster`：临时内置的 `comb2-pcmaster` 源码
- `vendor/perf_monitor.py`：可选性能监控模块

## 第 1 轮 2026-07-10 - Eval 分层 IC

### 目标

以分层单调性 `layerIC` 取代总体评估中的 `percic`，并同步提供 `layerIC.ir` 与 Q10--Q1 收益差。

### 已读材料 / 输入澄清

已读取 `config.human`、本 README 和 `evals/README.md`。用户已明确分层定义、使用当前交易掩码和有效 alpha/label 样本，并要求替换 eval 中 `percic` 的位置；无未决实现选择。

### 实现

复用 `runEval` 既有的 alpha、label 与交易掩码对齐流程。在有效样本中使用不拆分并列值的 10 等频桶；不能形成完整 Q1--Q10 的日期记为缺失，从而保持 `layerSpread=Q10-Q1` 的严格口径。IC 汇总、检查规则、报告关键列、图表和文档均改用 `layerIC` / `layerSpread`；历史 `percic` 文件仍可由单项 `--sim` 模式读取。

### 主指标与后续

主指标为 `layerIC.avg`，辅助输出为 `layerIC.ir` 和 `layerSpread.avg`。L1/L2 的数值阈值暂沿用被替代的 `percic.avg` 阈值，后续应基于历史样本的分布和通过率重新校准；本轮不涉及训练、模型或基线变更。

## 第 2 轮 2026-07-10 - 发布 0.1.8

### 目标 / 输入澄清

用户要求将版本调整为 `0.1.8`，编译受保护 wheel 并安装到当前 `python3`。本轮复用了仓库根目录 `VERSION` 和 `packaging/build_protected_wheel.py` 的既有发布流程，不涉及训练、模型、配置或评估定义变更。

### 实现 / 验证

版本源更新为 `0.1.8`，并同步更新 `RELEASE.md`。构建环境使用 `/root/autodl/local-gcc`，其 `bin` 和 `lib` 已写入 `~/.bashrc`；生成并验证 `dist_protected/combo2-0.1.8-cp313-cp313-linux_x86_64.whl`，随后以当前 Python 3.13 的 `pip --no-deps --force-reinstall` 替换已安装的 `combo2 0.1.7`。

## 第 3 轮 2026-07-10 - lIC / lIR

### 目标 / 输入澄清

用户要求将分层 IC 的公开名称改为 `lIC` / `lIR`，并移除 percentile IC。

### 实现 / 验证

日频字段改为 `lic`，原生汇总为 `lIC.avg` 与 `lIC.ir`；`percic` 已从生成、汇总和单项输入处理中移除。构建并安装 `combo2 0.1.9`，源码测试与项目 `eval.py` 入口均通过；候选模型的 ALL 行为 `lIC=0.345341`、`lIR=0.629311`。

## 第 4 轮 2026-07-15 - Eval CAP Corr

### 目标 / 输入澄清

为 `runEval config.xml` 加入 CAP corr。用户确认市值与 alpha 按同一交易日对齐，只使用当日可得市值；不沿用旧实现的下一交易日市值偏移。

### 实现

复用总体评估已经应用 BaseUniv/Limit、label 和 alpha 的对齐结果，读取 `AshareCache/1d_DailyFdm/DailyFdm.mkt_cap`。每日市值做截面 rank；alpha 做中位数中心化并按多头、空头分别归一化，再求 Pearson 相关。评估报告新增 `cap_corr_summary.csv`，输出年度及全样本的均值、日频 IR 和标准差，并在命令行文本和长图摘要展示；`--skip-exposure` 同时跳过 Barra 与 CAP 指标。

### 验证 / Handoff

新增同日对齐和全样本日频加权汇总测试。`python3 -m pytest -q evals/tests/test_exposure.py evals/tests/test_precision.py evals/tests/test_correlation.py` 通过（12 passed），`python3 -m pytest -q tests evals/tests` 也已完成。全仓收集仍受未跟踪目录 `0714.search.bad.performance/` 中同名 `test_model.*.py` 冲突影响。
