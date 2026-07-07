# comb2-organize

`comb2-organize` 用来把 research 模型接到内置的 `comb2` 和 `comb2-pcmaster` 源码上，完成训练、信号生成和回测。

发布记录、版本号和当前 wheel 安装目标统一记录在 `RELEASE.md`。
发布包名统一为 `Combo2`；wheel 文件名会按 Python 打包规范显示为小写 `combo2-...`；仓库目录名和内部模块名保持现状。

## 安装方式

下载这个仓库即可。当前仓库已经临时内置所需源码：

```text
vendor/
  comb2/
    comb2/
    src/
  comb2-pcmaster/
    comb2_pcmaster/
```

`vendor/` 只包含运行所需源码，不包含原仓库 git history、构建产物、缓存和历史输出。

## research 流程示例

下面以 `eg-lgbm` 为例说明完整流程。

### 1. 准备 research 目录

示例目录结构如下：

```text
eg-lgbm/
  model.py
  config.xml
  output/
  checkpoints/
```

其中：
- `model.py`：research 模型实现
- `config.xml`：运行配置
- `output/`：输出目录
- `checkpoints/`：模型 checkpoint 目录

### 2. 编写 research 模型

可以参考 `eg-lgbm/model.py`。模型需要提供一个 `ResearchModel` 类，并实现以下接口：

- `fit(dataset)`
- `predict(x_window)`
- `save(path_or_buffer)`
- `load(path_or_buffer)`

### 3. 编写配置文件

可以参考 `eg-lgbm/config.xml`。

所有 XML 里的相对路径都会按 `config.xml` 所在目录解析，不依赖运行命令时的当前目录。换机器时，通常只需要修改 `<constants>` 中的机器相关根目录。

关键配置包括：
- `constants.cache_path`：行情、mask、label 等 `AshareCache` 数据根目录
- `constants.output_root`：日志、alpha、回测输出的根目录
- `constants.checkpoint_root`：模型 checkpoint 输出目录
- `combo.paths.model_path`：指向 research 目录下的 `model.py`
- `combo.paths.checkpoint_root`：checkpoint 输出目录
- `combo.paths.output_dir`：日志和中间结果输出目录
- `backtest.output_path`：回测结果输出目录

`builtin.factorsim` 的路径规则：

- 绝对路径：直接读取
- 相对路径：相对 `config.xml` 或 data-pack 文件所在目录解析

普通 factor pool 不再通过 `constants` 配置隐式根目录；需要在 `<item path="...">` 中写绝对路径，或者相对当前 XML/data-pack 的路径。`AshareCache` 内的数据也建议写成绝对路径或相对 data-pack 的路径。

示例里：
- `model_path="model.py"`
- `checkpoint_root="checkpoints"`
- `output_dir="output"`
- `output_path="output/backtest"`

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

`runEval.py` 和 `runCombo.py` 一样可以从任意目录执行：

```bash
/path/to/comb2-organize/runEval.py /path/to/research/config.xml
```

或者：

```bash
/path/to/comb2-organize/runEval.py --config /path/to/research/config.xml
```

`runEval.py` 的 shebang 固定使用 `/root/autodl/python310fs/bin/python3`。如果当前就在 `comb2-organize` 仓库目录下，也可以直接：

```bash
./runEval.py /path/to/research/config.xml
```

入口会先检查 config 对应的必要输出文件是否齐全：

```text
<output_root>/alpha.parquet
```

如果缺失或为空，会打印不齐全的文件列表并退出。齐全后会基于 `alpha.parquet` 和 config 指向的 label/cache 重新计算 IC、PNL、分组回测，并在 `<output_root>/eval_report/` 下生成 summary、检查表和 `signal_analysis.png` 长图，不依赖已有 `daily_ic` 或 `backtest/daily_pnl.csv`。

默认会从 config 的 `combo.loader.ashare_data_path` 下读取 `DailyLabel.vwap30_label1d` 和 `DailyLabel.vwap30_label5d`。如果要显式指定本地 label 表：

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
./runEval.py --exposure /path/to/alpha.parquet --ashare-cache-path /path/to/AshareCache
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

- `processed_feature_cache="true"` 缓存的是最终预处理后的单日 feature，不是原始 1m/5m 日内矩阵。
- 3D `builtin.factorsim`（例如 `1m_Grid1mBar/*`、`5m_Intv5mBar/*`）会按请求日期即时读取成 cube，在 `nbar` 选窗后进入 item ops pipeline，不再长期保留原始分钟级 `source cache`。
- 当前本机实测：`1m_Grid1mBar` 每天 `239` 个时点，`5m_Intv5mBar` 每天 `49` 个时点，且 `5m` 序列包含最后一个单独的 `15:00:00` 时点。

## 运行结果

执行后通常会产生这些输出：

- `output/train.log`
- `output/alpha_history.pt`
- `output/alpha.parquet`
- `output/daily_ic`
- `output/backtest/daily_pnl.csv`
- `checkpoints/` 下的模型文件

## 性能监控

性能监控默认关闭，不影响原有训练和回测流程。需要时可在 XML 顶层加入：

```xml
<monitor enabled="true" output_path="output/perf_metrics.csv" collect_gpu="true" sync_cuda="false" />
```

启用后会记录 `setup`、`combine`、`alpha_convert`、`backtest_step`、`backtest_finalize`、`alpha_analysis` 以及可选的 `combo_*` 阶段，包含耗时、CPU 进程内存和可用的 PyTorch CUDA 显存信息。默认不做 CUDA synchronize，避免明显改变运行时序。

## 当前目录说明

- `runCombo.py`：research 运行入口，负责加载配置、调用 `comb2`、再接入 `comb2-pcmaster` 回测
- `runEval.py`：整体评估入口，检查 config 输出是否齐全并生成本地评估报告
- `runEval.py --sim/--pnl/--corr/--va/--exposure`：单项模式，直接读取本地 parquet/csv
- `runEval.py --corr left.parquet right.parquet`：默认只统计最近 240 个重叠交易日的日频截面相关
- `config.py`：配置解析与默认参数
- 默认 config 使用内置 `AlphaStrategy`：先对有效 alpha 减去当日截面中位数，再持有调整后为正的 alpha，并按调整后正值归一化生成 long-only 仓位
- `vendor/comb2`：临时内置的 `comb2` 源码
- `vendor/comb2-pcmaster`：临时内置的 `comb2-pcmaster` 源码
- `vendor/perf_monitor.py`：可选性能监控模块
