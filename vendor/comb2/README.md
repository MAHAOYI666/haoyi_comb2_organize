# comb2 模型接入说明

## Data Compression Options

`comb2` supports optional feature storage compression through the data XML attribute
`compression`.

```xml
<data dtype="float16" compression="fp4" />
```

Supported values are `none` (default passthrough), `fp8`, and `fp4`. Compression is
applied only to `ComboTrainDataset.X` and `ComboBuffer.buffer` after existing feature
preprocessing has completed. Labels, weights, masks, models, and factor files are not
compressed.

FP4/FP8 support `float16`, `float32`, and `bfloat16` logical tensors; `float64` is
supported only by `compression="none"`. FP8 requires the local PyTorch build to support
CPU casts to and from `torch.float8_e4m3fn`; unsupported environments raise instead of
falling back.

FP8 uses `torch.float8_e4m3fn` as the encoded format, but stores the physical buffer as
`torch.uint8` raw bytes. This keeps 1 byte per element while avoiding unsupported CPU
float8 advanced indexing kernels.

Source-path validation:

```powershell
$env:PYTHONPATH = "vendor/comb2"
.\.venv\Scripts\python.exe -c "from comb2.codec import build_codec; print('ok')"
.\.venv\Scripts\python.exe -m pytest vendor/comb2/tests/test_codec.py
```

See `docs/ARCHITECTURE.md` for internal components and `../../config.human` for
researcher-facing configuration.

本文档面向研究员，说明如何在 `comb2` 框架中接入自己的模型并完成训练、预测与回测。

目标是让你只关注三件事：
- 写好自己的 `ResearchModel`
- 在实验目录里准备好模型文件和 XML 配置
- 如需改数据预处理或训练样本，再接入 `ResearchLoader` / `ResearchDataset`

不需要了解框架内部的训练调度、数据缓存或回测实现细节。

补充说明：
- 3D `builtin.factorsim` 日内数据必须声明 `freq="5m"` 或 `freq="1m"`，框架会保持 cube 维度，不在数据层降维。
- 更完整的配置和 hook 契约见 `../../config.human`。

## 1. 推荐目录组织

建议不要把自己的实验直接写进 `comb2` 包内部，而是在外部单独建一个实验目录，例如：

```text
/my_experiment/
  ├── my_model.py
  └── experiment.xml
```

其中：
- `my_model.py`：你自己的模型实现
- `experiment.xml`：这个模型对应的实验配置

框架入口脚本会读取 XML，再根据 XML 中的 `model_path` 加载你的模型文件。

---

## 2. 你需要提供什么

你需要在一个 Python 文件里定义一个类：`ResearchModel`。

框架会按统一接口调用它，因此你的模型文件只需要满足下面这几个要求：

### 必须提供的类

```python
class ResearchModel:
    def __init__(self, config: dict):
        ...

    def fit(self, dataset):
        ...

    def predict(self, x_window):
        ...

    def save(self, path_or_buffer):
        ...

    def load(self, path_or_buffer):
        ...
```

其中：
- `__init__(config)`：接收配置字典，完成模型初始化
- `fit(dataset)`：用训练数据完成训练
- `predict(x_window)`：对单日截面输出预测值
- `save(path_or_buffer)`：保存模型
- `load(path_or_buffer)`：加载模型

类名必须是 `ResearchModel`，否则框架无法识别。

---

## 3. 配置如何传给模型

框架会把 XML 里的模型参数整理成一个 `config` 字典传给你的 `ResearchModel(config)`。

通常会包含以下字段：

```python
{
    "dtype": ...,
    "tsDays": ...,
    "freqs": ...,
    "num_features_by_freq": ...,
    "num_features": ...,
    "device": ...,
    "hidden_size": ...,
    "lr": ...,
    "epochs": ...,
    "num_leaves": ...,
    "feature_fraction": ...,
    "bagging_fraction": ...,
    "bagging_freq": ...,
    "min_data_in_leaf": ...,
    "num_threads": ...,
    "seed": ...,
}
```

并不是所有字段都必须使用。

你的模型只需要读取自己关心的参数，例如：

```python
self.lr = float(config.get("lr", 1e-3))
self.epochs = int(config.get("epochs", 10))
self.device = config.get("device", "cpu")
```

建议始终使用 `config.get(..., default)`，这样即使配置里暂时没有某个字段，也可以正常运行。

---

## 4. 输入输出约定

### 4.1 fit(dataset)

`fit(dataset)` 的输入是框架提供的训练集对象。

你可以把它理解为：
- `len(dataset)` 表示可迭代的训练样本数
- `dataset[idx]` 可以取出单个样本

当前示例模型使用的解包方式是：

```python
_, x, y, w = dataset[idx]
```

这意味着单个样本通常至少包含：
- `x`：特征窗口，类型是按频率分组的 `FeatureGroups`
- `y`：标签
- `w`：样本权重

研究员只需要在自己的 `fit` 里按需要读取并整理这些数据即可。

一个典型的写法如下：

```python
def fit(self, dataset):
    xs = []
    ys = []
    ws = []
    for idx in range(len(dataset)):
        _, x, y, w = dataset[idx]
        xs.append(x)
        ys.append(y)
        ws.append(w)

    # 在这里把 xs / ys / ws 整理成你的模型需要的格式
    # 然后完成训练
    return self
```

如果你的模型不需要权重，也可以忽略 `w`。

### 4.2 predict(x_window)

`predict(x_window)` 的输入是某一天对应的特征窗口，类型是 `FeatureGroups`，不是单个 concat tensor。

常见 shape：

- `x_window["1d"]`: `[tsDays, stock, feature]`
- `x_window["5m"]`: `[tsDays, stock, 49, feature]`
- `x_window["1m"]`: `[tsDays, stock, 239, feature]`

如果 config 没有声明某个频率的 factor，对应 key 不会存在。模型应使用 `config["freqs"]` 或 `if "5m" in x_window` 判断。

你需要返回该日所有股票的预测结果，要求：
- 返回结果长度与当日股票数一致
- 返回类型最好是 `torch.Tensor`
- 如果返回的是 `numpy.ndarray` 或其他数组类型，框架通常也能处理，但建议统一返回 `torch.Tensor`

推荐写法：

```python
def predict(self, x_window):
    pred = ...
    return torch.as_tensor(pred, dtype=self.dtype)
```

如果你的模型天然输出的是二维数组，请在返回前压成一维，确保每只股票对应一个分数。

---

## 5. 保存和加载

为了支持断点续跑和历史回测，模型需要实现 `save` 和 `load`。

要求很简单：
- `save(...)` 能把当前模型状态完整保存下来
- `load(...)` 能恢复到可继续预测的状态

推荐约定：
- `load(...)` 最后返回 `self`
- `save(...)` 和 `load(...)` 同时兼容文件路径与内存 buffer

一个常见写法：

```python
def save(self, path_or_buffer):
    payload = {
        "model": ...,
        "config": self.config,
    }
    torch.save(payload, path_or_buffer)


def load(self, path_or_buffer):
    payload = torch.load(path_or_buffer, map_location="cpu")
    self.model = payload["model"]
    return self
```

只要你的保存格式和加载格式自洽即可。

---

## 6. 最小可用模板

下面是一个最小可用模板。这个模板不代表最佳效果，只是说明接口应该怎么写。

```python
from __future__ import annotations

from typing import Any

import torch
from comb2 import FeatureGroups


class ResearchModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dtype = config.get("dtype", torch.float32)
        self.device = config.get("device", "cpu")
        self.lr = float(config.get("lr", 1e-3))
        self.epochs = int(config.get("epochs", 10))
        self.freqs = tuple(config.get("freqs", ("1d",)))
        self.num_features_by_freq = dict(config.get("num_features_by_freq", {"1d": 1}))
        self.model = None

    def fit(self, dataset):
        return self

    def predict(self, x_window: FeatureGroups):
        pred = torch.zeros(x_window.stock_count(), dtype=self.dtype)
        return pred

    def save(self, path_or_buffer):
        payload = {
            "model": self.model,
            "config": self.config,
        }
        torch.save(payload, path_or_buffer)

    def load(self, path_or_buffer):
        payload = torch.load(path_or_buffer, map_location="cpu")
        self.model = payload.get("model")
        return self
```

你可以直接复制这个文件，再把 `fit` / `predict` 改成自己的版本。

---

## 7. XML 配置格式

研究员推荐使用一个 XML 文件描述实验。顶层分为三部分：
- `strategy`
- `combo`
- `backtest`

一个最小示例如下：

```xml
<config>
  <constants cache_path="/path/to/Cache" output_root="/my_experiment/output" />
  <strategy start_ds="20160111" end_ds="20200101" />

  <combo>
    <paths
      model_path="/my_experiment/my_model.py"
      combo_base_path=""
    />

    <output enable_alpha_analysis="true" />

    <runtime
      snaptime="exp_demo"
      snap_ti="150000"
      livetrading="false"
      trainDelay="0"
      retDays="1"
      tsDays="8"
      model_smooth_rate="0.7"
      model_keep_num="2"
      max_train_days="2000"
    />

    <model
      device="cpu"
      lr="0.05"
      epochs="100"
      num_leaves="31"
      feature_fraction="0.8"
      bagging_fraction="0.8"
      bagging_freq="1"
      min_data_in_leaf="100"
      num_threads="-1"
    />

    <data
      dtype="float16"
      data_start_ds="20160101"
      data_offset="1024"
      valid_path="/path/to/valid"
      filtered_path="/path/to/filtered"
    >
      <item name="alpha.factor_1" module="builtin.factorsim" path="/path/to/factor_1" role="factor" />
      <item name="alpha.factor_2" module="builtin.factorsim" path="/path/to/factor_2" role="factor" />
      <item name="label.default" path="vwap30_label1d" role="label" />
    </data>
  </combo>

  <backtest
    daily_metrics_file="daily_pnl.csv"
    cash="10000000"
    fee_rate="0.0015"
    reserve_cash="0.95"
    verbose="false"
    universe="base"
  />
</config>
```

说明：
- XML 中未填写的字段会回退到框架默认值
- 所有输出都从 `constants.output_root` 派生，包括日志、alpha、checkpoint 和 backtest
- `runtime.snaptime` 是实验/checkpoint 名；`runtime.snap_ti` 是默认数据时点
- `dtype` 当前建议使用：`float16`、`float32`、`float64`、`bfloat16`
- 布尔值建议写成：`true` / `false`
- 多个因子通过多个 `<data><item role="factor" ... /></data>` 声明
- 日内因子必须显式写 `freq="5m"` 或 `freq="1m"`；缺省 `freq` 等价于 `1d`
- 不支持 `nbar`，也不支持 `last/mean/std/sum/max/min` 这类会改变维度的 data op
- `<import path="...">` 只支持根节点为 `<data-pack>` 或 `<data>` 的纯 data 声明文件

---

## 8. 如何接入你自己的模型文件

假设你的实验目录是：

```text
/my_experiment/
  ├── my_model.py
  └── experiment.xml
```

那么只需要在 XML 中把 `combo.paths.model_path` 指向你的模型文件：

```xml
<paths model_path="/my_experiment/my_model.py" />
```

只要这个文件中定义了 `ResearchModel`，框架就会自动加载它。

建议一个模型文件只放一个主要模型实现，避免把实验性代码、临时脚本和模型入口混在一起。

---

## 9. 可选：重写 Loader 和 Dataset

如果只写 `ResearchModel` 不够，可以在 XML 里指定：

```xml
<paths
  model_path="my_model.py"
  research_loader_path="loader.py"
  research_dataset_path="dataset.py"
/>
```

`loader.py` 必须定义 `ResearchLoader(ComboDataLoader)`。常用 hook：

- `preprocess_feature_group(freq, feature, ds)`：处理每个频率的单日 feature group，`1d=[stock,F]`，`5m/1m=[stock,bar,F]`。
- `preprocess_daily_features(feature, ds)`：只改日频默认处理。
- `preprocess_label(label_values, valid_mask, ds, ret_days)`：改 label 标准化和样本权重。
- `transform_feature_window(feature_window, stage=...)`：改训练/预测窗口处理；默认会 mask 最后一天未来日内 bar。

在 loader hook 中读取已声明的 factor / label / aux 数据，使用 `self.registry.get_data(name, start_ds, end_ds)`；返回保留 date 维：`1d=[R,N]`，`5m=[R,49,N]`，`1m=[R,239,N]`。

`dataset.py` 必须定义 `ResearchDataset(ComboTrainDataset)`。常用 hook：

- `_build_validinsts()`：改训练股票池。
- `__getitem__(idx)`：改训练样本结构；如果改返回值，必须同步修改 `ResearchModel.fit()`。

优先重写这些小 hook，不要直接改 `preprocess_features()`；它是遍历所有 group 的总控函数。完整输入、输出和 shape 契约见 `../../config.human`。

---

## 10. 如何运行

### 训练

```bash
python /root/autodl-tmp/comb2/train_test.py --config /my_experiment/experiment.xml
```

### 回测

```bash
python /root/autodl-tmp/comb2-organize/run_backtest.py --config /my_experiment/experiment.xml
```

如果不传 `--config`，脚本会使用框架内置默认配置。

---

## 11. 研究员开发建议

### 建议 1：先保证接口跑通，再优化效果

第一次接入时，先确保以下几点：
- `ResearchModel` 能被成功导入
- `fit(dataset)` 能完整跑完
- `predict(x_window)` 能输出正确长度的结果
- `save/load` 后模型还能继续预测

先跑通，再做调参和建模优化，效率会更高。

### 建议 2：predict 只返回分数，不做交易逻辑

模型职责只是输出每只股票的预测分数，不需要在模型里处理仓位、交易费用、调仓限制等逻辑。

### 建议 3：注意数值有效性

训练和预测时建议自行处理：
- `NaN`
- `inf`
- 空样本
- 全部权重为 0 的情况

如果模型训练依赖严格的数据格式，最好在 `fit` 里先做一次清洗。

### 建议 4：保证 save/load 一致

很多运行问题都来自保存和加载格式不一致。

最简单的原则是：
- `save` 存什么
- `load` 就按同样结构读回来

---

## 11. 常见问题

### Q1：类名可以自定义吗？

不可以。入口类名必须是 `ResearchModel`。

### Q2：一定要用 PyTorch 吗？

不一定。你可以在模型内部使用任意框架，例如：
- PyTorch
- LightGBM
- XGBoost
- sklearn
- 纯 numpy

只要最终实现统一接口即可。

### Q3：predict 必须返回 torch.Tensor 吗？

推荐返回 `torch.Tensor`。

如果你内部使用的是 numpy，也建议在返回前转成 `torch.Tensor`，这样最稳妥。

### Q4：模型文件一定要放在 `comb2` 目录里吗？

不需要。更推荐放在你自己的实验目录里，然后通过 XML 里的 `model_path` 指向它。

### Q5：怎么确认自己接入成功？

最直接的方法是：
- 运行训练脚本，确认 `fit` 被调用
- 运行回测脚本，确认 `predict` 正常输出结果
- 检查是否成功生成模型保存结果和回测输出

---

## 12. 推荐流程

建议按下面顺序接入：

1. 在外部新建实验目录
2. 写好 `my_model.py`
3. 写好 `experiment.xml`
4. 在 XML 中把 `model_path` 指向你的模型文件
5. 先做一次短区间运行，确认训练、预测、保存、加载都正常
6. 再扩大时间区间做正式实验

如果你只是第一次接入，优先追求“能稳定跑通”；如果已经跑通，再开始做特征处理、模型结构和超参数优化。
