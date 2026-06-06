# comb2-organize

`comb2-organize` 用来把 research 模型接到内置的 `comb2` 和 `comb2-pcmaster` 源码上，完成训练、信号生成和回测。

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
- `constants.cache_path`：行情、mask、label 等缓存数据根目录
- `constants.factor_root`：相对因子路径的根目录，程序不会自动追加 `ZsimPool`
- `constants.output_root`：日志、alpha、回测输出的根目录
- `constants.checkpoint_root`：模型 checkpoint 输出目录
- `combo.paths.model_path`：指向 research 目录下的 `model.py`
- `combo.paths.checkpoint_root`：checkpoint 输出目录
- `combo.paths.output_dir`：日志和中间结果输出目录
- `backtest.output_path`：回测结果输出目录

如果因子在 `ZsimPool` 下，需要显式写在 `constants.factor_root` 里，例如：

```xml
<constants factor_root="/path/to/FactorData/ZsimPool" />
```

也可以保持 `factor_root` 为更上层目录，然后在 `<path>` 中显式写相对路径：

```xml
<path>ZsimPool/yz_20250219_02</path>
```

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

## 数据压缩选项

实际使用时，只需要在 XML 的 `<combo><loader ...>` 节点里增加 `compression` 属性即可：

```xml
<loader dtype="float16" compression="fp4" data_start_ds="20160101">
  ...
</loader>
```

可选值：
- `none`：默认值，不压缩，行为与原始流程一致
- `fp4`：压缩特征存储，主要用于降低 CPU 内存占用
- `fp8`：使用 PyTorch float8 存储；是否可用取决于当前 PyTorch 运行环境对 CPU float8 相关算子的支持

压缩只作用于 comb2 内部的特征缓存存储：
- `ComboTrainDataset.X`
- `ComboBuffer.buffer`

不压缩 `Y`、`W`、mask、label、因子文件、模型输入输出，也不改变 `cs_zscore`、`truncate`、`nan_to_num` 等数据预处理逻辑。模型侧拿到的仍然是解码后的浮点张量，所以体感上主要就是在 `config.xml` 里指定压缩方式。

## 运行结果

执行后通常会产生这些输出：

- `output/train.log`
- `output/alpha_history.pt`
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
- `config.py`：配置解析与默认参数
- 默认 config 使用内置 `AlphaStrategy`：只持有正 alpha，按正 alpha 权重归一化生成 long-only 仓位
- `vendor/comb2`：临时内置的 `comb2` 源码
- `vendor/comb2-pcmaster`：临时内置的 `comb2-pcmaster` 源码
- `vendor/perf_monitor.py`：可选性能监控模块
