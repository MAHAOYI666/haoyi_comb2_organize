# combo2 框架资料与 Torch 示例

更新日期：2026-09-11。本文档包依据本地仓库 `D:\Codex\haoyi_comb2_organize` 的当前代码整理，版本为 **combo2 1.0.2**，提交为 `9210547b79265c42a61fcb0f33d15a61818bab27`。版本取自仓库根目录 `VERSION`；本次没有连接远程实例确认其部署状态。

combo2 把多源因子、源级特征处理、模型训练与预测、持仓优化、实际成交回测和评估接到同一流程。每天一个或多个采样时点使用相同接口。

## 本目录如何阅读

| 文件 | 用途 |
|---|---|
| [现在的负责点.txt](现在的负责点.txt) | 研究目标、工作重点和实验交付口径 |
| [combo框架整体架构.txt](combo框架整体架构.txt) | 从数据到成交及评估的完整说明，包含本例实际参数 |
| [config.xml](config.xml) | 当前仓库 `eg-torch/config.xml` 的原样副本 |
| [model.py](model.py) | 当前 Torch 聚合模型的原样副本 |
| [loader.py](loader.py) | 新接口必需的数据、目标和成交价声明，取自 `eg-torch` |
| [dataset.py](dataset.py) | 与配置配套的数据集扩展入口，取自 `eg-torch` |
| [config.human](config.human) | 当前仓库完整的研究接口和优化器参数手册 |

原文件名中的下载序号已去掉：`现在的负责点 (2).txt` → `现在的负责点.txt`，`combo框架整体架构 (1).txt` → `combo框架整体架构.txt`，`config (1).xml` → `config.xml`。原始五份文件仍保留在 Downloads。新增 loader、dataset 和接口手册，是为了补齐新版示例的必要说明与依赖。

本目录是文档与研究示例包，未包含框架源码、因子数据、AshareCache、模型 checkpoint 或许可证。放到任意目录并不等于数据路径已经可用；运行前需配置该环境中的真实路径。

## 相对旧资料的主要变化

| 旧资料 | 当前实现 |
|---|---|
| XML `<combo><data><item ...>` 声明因子和标签 | `ResearchLoader.data_requirements()` 声明 `DataItem`；读取参数移到 `<combo><loader>` |
| `role`、`freq`、`FeatureGroups` | 来源经 `process_source()` 处理为 `[date, stock, feature]`，模型接收普通张量 |
| `(idx, x, y, w)` | `(idx, ds, ti, x, y, w)` |
| `predict(x_window)` | `predict(x_window, *, di, ti)` |
| `snap_ti` 或固定前一日预测流程 | `runtime.sample_times`；以当前逻辑 `(date,time)` 预测，来源分别按 `delay` 读取 |
| `vwap30_label1d` 静态标签项 | 本例 `builtin.snap_label`、`delay=1`；目标来自当前采样时点的 VWAP 标签 |
| 本例 `trainDelay=0` | 本例 `trainDelay=2`；框架未写参数时默认仍是 0 |
| 减中位数后的正 alpha 归一化持仓 | 原始零点分正负、两侧分别归一化，再由 MOSEK 优化；本例默认 `opt1` |
| `execution_price="vwap30"` | XML 使用 `execution_price="execution:execution"` 引用 loader 字段 |
| 日期索引 alpha、固定 1d/5d IC | `(date,time)` 索引 alpha、研究员原始目标 IC |
| 评估从 alpha 重建 PnL，可用 CLI 覆盖标签 | 配置评估读取实际成交日报；目标统一在 loader 定义，不接受单独标签覆盖 |

旧接口不是可以直接混用的另一种配置格式；当前解析器会拒绝 `<combo><data>`。

## 配套示例的实际配置

- 时间：`data_start_ds=20160101`，运行区间 `20160111` 至 `20200101`，只迭代该范围内实际交易日。
- 采样：`sample_times="100000"`，每天 10:00 一个时点。
- 训练：`trainDelay=2`、`retDays=1`、`tsDays=8`、`max_train_days=2000`。
- 数据：8 个日频因子，全部 `delay=1`；模型特征存储 `float16`，`compression=none`。
- 模型：CPU，hidden 512，fc 256，dropout 0.5，Adam 学习率 `2e-6`，15 epochs，batch 3。
- 回测：默认 `opt1`，初始资金 1000 万元，`reserve_cash=0.95`，本例费率 `0.0015`（买卖各 15bp）。**框架默认费率是 `0.00075`（各 7.5bp），本例显式覆盖默认值。**
- 输出：相对配置目录的 `output-torch/`。本次没有改变基线因子、网络结构或调参结果。

多时点实验同时修改 `sample_times` 和优化器，例如：

```xml
<strategy start_ds="20241028" end_ds="20241029">
  <optimizer type="opt2" />
</strategy>
<!-- 在原有 runtime 节点中将 sample_times 改为 "100000,110000" -->
```

这只是多时点配置方式说明，不是本目录 `config.xml` 当前启用的设置。

## 数据和路径约定

XML 相对路径以 XML 所在目录为基准；`DataItem` 的 Python 相对路径以 `ResearchLoader` 定义文件所在目录为基准。

因此本例的 `factors/yz_20250219_02` 等路径在迁移后指向新研究目录下的 `factors/`。移动 XML 到 Optuna trial 目录不会自动改变 loader 文件对应的因子基准目录。

`constants.cache_path="data/Cache"` 指向 `AshareCache` 的父目录。当前示例需要以下类别的数据：

- loader 声明的 8 个因子。
- `AshareCache/1d_IntraVwap/IntraVwap.Vwap30.100000`、标签计算用复权收盘价和复权因子。
- `StockMask2.BaseUnivMask`、`NoNewStockMask`、`LimitMask` 等有效性及行情状态。
- 回测行情、上市状态，以及默认优化器需要的收益、基准权重、行业、Barra 和 universe 数据；具体来源由配置的约束决定。

来源 `delay=1` 表示请求逻辑 D 行时读取物理 D−1 交易日，只应用一次。源级处理应保证所用数据在决策时点可知；分钟数据需先处理为 `[date, stock, feature]`，原始 bar 轴不会直接送入模型。

## 模型扩展接口

```python
idx, ds, ti, x, y, w = dataset[i]
# x: [tsDays, stock, feature]
# y / w: [stock]

class ResearchModel:
    def __init__(self, config): ...
    def fit(self, dataset): ...
    def predict(self, x_window, *, di, ti): ...
    def save(self, path_or_buffer): ...
    def load(self, path_or_buffer): ...
```

窗口取连续 `tsDays` 个交易日的同一采样时点。模型初始化接收 `dtype`、`tsDays`、`num_features`、`feature_names`、`sample_times` 等信息。预测返回股票截面，框架用 `trainii` 回填全股票轴。

loader 主要入口为 `data_requirements`、`process_source`、`model_input_sources`、`model_target`、`model_validity_source`、`gen_raw_target`、`preprocess_features`、`preprocess_target` 和 `transform_feature_window`。详细签名及 shape 见 `config.human`。

## 运行方式

完整训练和回测在具备因子环境的远程实例进行。下面是运行仓库原有 `eg-torch` 示例的命令；若使用本资料包建立新研究目录，应替换配置路径，并先修改其中的数据路径。

```bash
source /root/autodl-tmp/.venvs/haoyi_comb2_py313/bin/activate
cd /root/autodl-tmp/haoyi_comb2_organize
command -v python
python -c "import sys; print(sys.executable); print(sys.version)"
# 仅在此文件是当前环境获准使用的有效许可证时采用该路径。
export MOSEKLM_LICENSE_FILE=/root/autodl-tmp/haoyi_comb2_organize/mosek.lic
python runCombo.py eg-torch/config.xml
python runEval.py eg-torch/config.xml
```

默认策略依赖 `Mosek==11.0.25` 及有效许可证；受保护 wheel 不包含许可证。已安装对应 wheel 时可使用 `runCombo config.xml`、`runEval config.xml` 命令。Windows 本地检查统一使用仓库的 `.\.venv\Scripts\python.exe`；本地不具备完整因子库。

本项目运行代码的后续修改先在本地仓库完成，再按项目约定同步：

```powershell
wsl bash /mnt/d/Codex/deploy_comb2_to_autodl.sh
```

本次只是资料导出，未修改运行代码，也未执行部署、训练或回测。

## 回测与评估输出

```text
output-torch/
  train.log
  alpha_history.pt
  alpha.parquet                 # (date,time) × stock
  daily_ic                      # (date,time) 下的 ic、count
  ic_by_time.csv                # 每个 time 的 mean、std、count
  checkpoints/mlp_minimal/<训练目标日>/model
  checkpoints/mlp_minimal/<训练目标日>/oldmodel  # 已有旧模型时
  backtest/
    daily_pnl.csv
    pnl_summary.csv
    executions.csv
    settlements.csv             # 有退市核销时
    AlphaStrategy_<开始日>_<结束日>_position.csv
    AlphaStrategy_<开始日>_<结束日>_holdings.csv
    AlphaStrategy_<开始日>_<结束日>_yield.csv
    return.jpg / ex_return.jpg / trade_cost.jpg  # 绘图依赖可用时
  eval_report/                  # 运行 runEval 后
    daily_ic.csv / ic_summary.csv / pnl_summary.csv
    decile_summary.csv
    barra_exposure_summary.csv / cap_corr_summary.csv
    signal_analysis.png / report.json
```

以上 alpha 分析文件由 `enable_alpha_analysis=true` 控制；评估分层和暴露文件取决于相应选项。`runEval config.xml` 需要 `alpha.parquet` 和实际成交日报，使用 loader 的 `gen_raw_target` 和有效性定义重新计算 IC；分层输出是原始目标均值，不能自动解释成各层实际投资收益。

```bash
python runEval.py eg-torch/config.xml --skip-exposure --skip-deciles
python runEval.py --sim /path/to/daily_ic.csv
python runEval.py --pnl /path/to/daily_pnl.csv
python runEval.py --corr /path/to/left.parquet /path/to/right.parquet
python runEval.py --va /path/to/base_daily_pnl.csv /path/to/new_daily_pnl.csv
python runEval.py --exposure /path/to/alpha.parquet --cache-path /path/to/Cache
```

独立表格工具的历史 IC 字段兼容能力，不等于当前完整配置评估会自动生成固定的 1d/5d 标签指标。使用同一份代码和数据版本评估已有 alpha，才能保持目标口径一致。

## 内存、监控与验证边界

`<combo><loader compression="fp4" registry_cache_days="64" ... />` 控制特征编码和来源缓存。`none/fp4/fp8` 编码训练 X 和预测缓冲区，模型取得解码张量；原始来源、目标和成交价不做这种特征压缩。

训练数据集一次性保存各时点的训练特征及 Y/W。`registry_cache_days` 限制来源缓存，不限制整个训练数据集内存。无压缩时 X 窗口可能共享存储，Y/W 始终可能共享存储；自定义窗口和样本处理若要原地修改，应先明确是否需要 clone。来源变更后需重建训练数据集。

monitor 默认关闭；可在原配置节点启用并设置 `output_path`。峰值内存应同时考虑训练数据、模型、batch、缓存及新旧模型，不能仅按 LRU 大小估算。

本次本地检查已通过：8 份文件可按 UTF-8 读取，3 份 Python 文件语法正确，5 份复制文件与仓库逐字节一致，README 链接有效，配置可由当前解析器加载。合成数据检查通过了 6 元样本训练、带 di/ti 的张量预测、trainii、模型保存/加载一致性，以及 IC 损失对无效股票的屏蔽。

未核验远程因子库存，未运行实际 MOSEK 成交链路；不以本地检查代替远程验证。仓库发布记录中的历史测试结果属于该版本记录，不是本次重新执行的结果。

## 核对来源

本文档的优先依据是当前实现：`eg-torch/`、`config.py`、`runCombo.py`、`vendor/comb2/comb2/{DataRegistry,DataLoader,ComboBase}.py`、`vendor/comb2-simbase/comb2_simbase/snap_labels.py`、`vendor/comb2-pcmaster/comb2_pcmaster/{default_strategy,backtest}.py`、`evals/comb_eval/report.py`；结合根目录 `README.md`、`config.human`、`VERSION`、`RELEASE.md` 和架构说明核对。这里的路径均相对来源仓库，不代表这些框架文件也已包含在资料包内。
