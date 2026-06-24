# comb_eval

`comb_eval` 是一个不依赖 PySim 外部服务的本地评估工具包，目标是把 `metric.md` 中的 evaluation 拆成可单独运行的模块。

## 已有模块

- `pnl`: 汇总本地 pnl 文件，计算 `ret_pct`、`tvr_pct`、`ir`、`sharpe`、`dd_pct`、`win_pct`、`margin`、`fitness`、`lnum_ratio` 等，并支持外部提供的 `pnlzz500` benchmark。
- `ic`: 汇总本地 daily IC / pnlsuper 文件，计算各字段的 `avg`、`std`、`ir`、`min_each_year` 等检查输入。
- `correlation`: 计算 pnl correlation、position correlation、trade correlation，以及 pool 级别的 `maxcorr` / `avgcorr` / `topcorrN`。
- `value_add`: 计算 guidance-style `IR_candidate - corr * IR_solid`、pool value-add 和 netting residual。
- `constraints`: 本地实现 short-side equalization、universe coverage、trading-limit exposure 检查。
- `exposure`: 计算 signal 对 Barra 风格因子的日频暴露，支持相关暴露和回归 beta 暴露两种口径。
- `eval`: 对 `pnl` / `ic` 汇总结果执行本地 L1/L2 阈值检查。

## CLI

```bash
python -m comb_eval.cli pnl /path/to/pnl.tsv
python -m comb_eval.cli pnl /path/to/pnl.tsv --pnlzz500 /path/to/pnlzz500.tsv
python -m comb_eval.cli ic /path/to/daily_ic --normalize-names
python -m comb_eval.cli eval --pnl /path/to/pnl.tsv --pnlzz500 /path/to/pnlzz500.tsv --ic /path/to/daily_ic
```

可选日期过滤：

```bash
python -m comb_eval.cli eval --pnl /path/to/pnl.tsv --ic /path/to/daily_ic --start 20160101 --end 20240101
```

配置输出评估：

```bash
python -m comb_eval.cli eval --config /path/to/config.xml
```

该入口会读取 comb2-organize XML config，检查 `constants.output_root` 下至少存在一个 `alpha.parquet`：

```text
<output_root>/alpha.parquet
```

PNL、IC、分组回测都会基于 `alpha.parquet` 和 config 指向的 label/cache 重新计算，不依赖已有 `daily_ic` 或 `backtest/daily_pnl.csv`，也不会 dump daily pnl / daily IC 中间文件。

默认输出到 `<output_root>/eval_report/`：

```text
ic_summary.csv
pnl_summary.csv
ic_checks.csv
pnl_checks.csv
decile_summary.csv
barra_exposure_summary.csv
top10_excess.csv
signal_analysis.png
report.json
```

其中 `signal_analysis.png` 是信号综合长图，包含 IC 统计、PNL 统计、信号 10 分位数时间序列、10 组分组回测、Barra 风格暴露范围，以及 top10% 回测相对基准的累计超额收益。PNL 与分组回测使用与 `evals/tools/rundailypnl.py` 相同的本地 daily-pnl 逻辑，IC 从 1d/5d label 重新计算。

常用可选项：

```bash
python -m comb_eval.cli eval --config /path/to/config.xml \
  --report-dir /tmp/eval_report \
  --plot-output /tmp/eval_report/signal_analysis.png \
  --start 20160101 --end 20240101 \
  --skip-exposure
```

如果 label 不走 config 默认 cache，可以显式指定 1d / 5d label 表：

```bash
python -m comb_eval.cli eval --config /path/to/config.xml \
  --label /path/to/vwap30_label1d.csv --label-is-table \
  --label-5d /path/to/vwap30_label5d.csv --label-5d-is-table
```

相关性 / value-add：

```bash
python -m comb_eval.cli corr /path/to/candidate_pnl.tsv /path/to/pool_1.tsv /path/to/pool_2.tsv
python -m comb_eval.cli matrix-corr /path/to/candidate_pos.parquet /path/to/pool_pos.parquet
python -m comb_eval.cli value-add /path/to/candidate_pnl.tsv /path/to/pool_1.tsv /path/to/pool_2.tsv
python -m comb_eval.cli netting /path/to/candidate_pnl.tsv /path/to/pool_portfolio_pnl.tsv
```

约束检查：

```bash
python -m comb_eval.cli equalize-short /path/to/alpha.parquet /tmp/equalized.parquet --percentile 0.5
python -m comb_eval.cli universe /path/to/alpha.parquet /path/to/universe_mask.parquet
python -m comb_eval.cli trade-limit /path/to/trades.parquet /path/to/trading_limit_mask.parquet
```

AshareCache 读取入口只使用 `Memmaper2(path).load(start_ds, end_ds, df_type)[:]`：

```bash
python -m comb_eval.cli cache-ic /home/jovyan/ml-data1-pvc/factorsim_data/Cache/AshareCache/<cache_path> 20160101 20240101 df
```

## 本地输入格式

### pnl

支持 csv/tsv/parquet。需要日期索引或 `date` 列。推荐列名：

```text
pnl long short ret sh_hld sh_trd n_long n_short
```

也兼容部分 PySim 风格列名：

```text
PNL Long Short Return Holdvalue Tradevalue Longcount Shortcount
```

如果没有 `ret`，会用 `pnl / long` 估算；如果没有 `sh_hld`，会用 `abs(long)+abs(short)` 估算；缺失 `sh_trd` 时 turnover、margin、fitness 会是 NaN。

### daily IC / pnlsuper

支持 csv/tsv/parquet。需要日期索引或 `date` 列。直接支持 `simsuper2.py` 风格字段，例如：

```text
1d_IC 5d_IC 10d_IC barra1d_IC barra5d_IC barra10d_IC coverage
```

也兼容 `comb2_organize` 输出，使用 `--normalize-names` 时会把：

```text
ic 5dic rankic percic coverage
```

映射为 `1d_IC 5d_IC rankic percic coverage`。

### matrix

position / trade / alpha matrix 支持 csv/tsv/parquet。需要日期索引或 `date` 列，其他列为 instrument，值为对应持仓、交易或 alpha 值。

## 外部数据边界

评估层只消费以下外部输入，不依赖外部系统计算逻辑：

1. `pnl` / `pnlzz500` / pool pnl 文件路径或 DataFrame。
2. candidate 和 pool 的 position/trade 矩阵。
3. 需要对比的 alpha 名单和对应文件路径。
4. date range / snaptime 列表。

`10d IC`、`Barra IC`、`universe mask` 等 AshareCache 数据读取必须经由 `comb_eval.io.read_cache_array()`，该函数内部只调用 `from comb2_simbase import Memmaper2` 和 `Memmaper2(path).load(start_ds, end_ds, df_type)[:]`。

## Barra 风格暴露接口

提供两个函数：

1. `compute_style_factor_exposure(signal, style_factors, mode=0)`
   - 输入 signal DataFrame 和 `{style_name: style_df}` 风格因子字典。
   - `mode=0` 返回横截面相关暴露。
   - `mode=1` 返回横截面回归 beta 暴露。
   - 返回值是 `date x style` 的 DataFrame。

2. `compute_barra_style_exposure(signal, start_ds=None, end_ds=None, mode=0, ashare_cache_path=...)`
   - 自动从 `AshareCache/1d_BarraCNE5` 读取全部 Barra 风格因子。
   - 输入:
     - `signal`: `date x code` 的 DataFrame
     - `start_ds`: 起始日期，默认自动匹配 signal 与 Barra 的最长交集
     - `end_ds`: 结束日期，默认自动匹配 signal 与 Barra 的最长交集
     - `mode`: `0` 为相关暴露，`1` 为 beta 暴露
   - 返回值是 `date x style` 的 DataFrame，列名为 `BETA/BTOP/...`
