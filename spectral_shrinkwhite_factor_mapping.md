# Spectral Ridge / shrink-white 因子映射表

生成时间：2026-06-21  
用途：将已扫描到的辅助因子库字段，映射到 Spectral Ridge / shrink-white 所需的数据类别。  

> 说明：本文件只根据扫描结果中的名称、路径和当前 combo 框架说明做映射。对于需要进一步读取字段含义、需要二次加工、或当前未扫描到的数据，均显式标注。

## 路径缩写

| 缩写 | 根路径 |
|---|---|
| `A` | `/home/shared/data1/factorsim_data/Cache/AshareCache` |
| `B` | `/home/shared/data1/factorsim_data/Cache/BaseCache` |
| `F` | `/home/shared/data1/factorsim_data/Cache/FdmCache` |
| `XROOT` | `config.xml` 中的 `/home/shared/data1/factorsim_data/Factor/FactorData/ZsimPool`；若实盘机器实际使用 `/root/ml_data/ZsimPool`，以实盘路径为准 |

## 状态定义

| 状态 | 含义 |
|---|---|
| 直接可用 | 扫描结果中存在明确字段，语义和目标需求基本一致 |
| 需再处理 | 字段存在，但需要 rolling、聚合、相除、回归、缺失率统计等加工 |
| 需确认 | 名称可推断，但具体口径需要进一步查字段定义或读取样本 |
| 当前未扫到 | 本次扫描结果中没有看到对应字段 |
| 输出目录获取 | 不属于因子缓存，需要从 combo 模型输出或回测输出目录中读取 |

## 关键诊断确认

### 缓存读取结构

这些缓存字段不是普通单文件，而是统一的 memmap-like 分片结构：

```text
字段目录/
  meta.npy
  columns.npy
  index.npy
  0.ares
  1.ares
  ...
```

`meta.npy` 的含义可按以下方式理解：

```text
[dtype, ndim, n_dates, n_stocks, chunk_size, n_chunks]
```

本次诊断确认的关键参数：

| 项目 | 结论 |
|---|---|
| dtype | `numpy.float64` |
| n_stocks | `5642` |
| chunk_size | `125` |
| `.ares` 含义 | 按日期切块存储，不是“每个 `.ares` 一个因子” |
| 正确读取方式 | 按 `float64` 和 `meta.npy` 的分片规则读取 |
| 错误读取风险 | 用 `float32` 直接读 `.ares` 会得到错误混杂值，不可用 |

### 矩阵形态

| 字段类型 | 结构 | 说明 |
|---|---|---|
| `StockMask2.*` | `date x stock` | 每天每只股票一个 mask 或数值 |
| `SwIndMask.*` | `date x stock` | 每天每只股票一个行业编码 |
| `BarraCNE5.*` | `date x stock` | 每天每只股票一个 Barra 暴露值 |
| `IndexWeight.*` | `date x stock` | 每天每只股票一个指数权重 |
| `DailyLabel.*` | `date x stock` | 每天每只股票一个未来收益 label |

这些字段不是 `date x stock x industry`，行业字段也不是 one-hot。

### 日期范围

| 字段组 | 日期范围 | 日期数 | 结论 |
|---|---:|---:|---|
| `StockMask2.*` | `20110104 -> 20260618` | 3753 | mask 覆盖最早、最长 |
| `SwIndMask.SW2021_L1/L2/L3` | `20110104 -> 20260618` | 3753 | 申万行业覆盖完整 |
| `BarraCNE5.*` | `20150601 -> 20260618` | 2685 | Barra 从 2015-06 起 |
| `IndexWeight.*` | `20150601 -> 20260618` | 2685 | 指数权重从 2015-06 起 |
| `DailyLabel.vwap30_label5d` | `20150601 -> 20260610` | 2679 | 5日 label 因为需要未来收益，最后几天缺失 |

正式计算必须取日期交集：

```text
valid_dates = alpha_dates ∩ label_dates ∩ mask_dates ∩ barra_dates
```

尤其不能把 `20260618` 的 Barra 暴露直接配 `20260618` 的 `vwap30_label5d`。

### mask 方向

`StockMask2.*` 中除 `StockListedDays` 外，均按正向 mask 使用：

```text
有限值 1.0 = 通过 / 可用 / 允许
NaN = 不通过 / 剔除
```

| 字段 | 诊断结论 |
|---|---|
| `BaseUnivMask` | `1.0` 表示在 base universe 内 |
| `LimitMask` | `1.0` 表示通过涨跌停/交易限制；`NaN` 表示不可交易或受限 |
| `NoNewStockMask` | `1.0` 表示不是需要剔除的新股 |
| `STStock` | 虽然名字像 ST 标记，但语义是正向 mask：`1.0` 表示可用/非 ST 风险 |
| `SuspendStock` | 虽然名字像停牌标记，但语义是正向 mask：`1.0` 表示可用/未停牌 |
| `StockListedDays` | 不是 mask，是上市天数数值 |

建议训练/IC 评估权重：

```text
w_t = isfinite(BaseUnivMask)
    & isfinite(LimitMask)
    & isfinite(NoNewStockMask)
    & isfinite(STStock)
    & isfinite(SuspendStock)
    & isfinite(label)
```

第一版快速诊断的最小可用版本：

```text
w_t = isfinite(BaseUnivMask)
    & isfinite(LimitMask)
    & isfinite(label)
```

`BaseUnivMask` 是核心股票池；`NoNewStockMask / STStock / SuspendStock` 基本覆盖全部 base universe；`LimitMask` 会额外剔除一小部分 base 股票。

### 行业编码

`BarraCNE5.INDUSTRY` 和 `SwIndMask.*` 都是 `date x stock -> 行业 code`，不是行业哑变量矩阵。

| 字段 | 编码范围 | 解释 |
|---|---:|---|
| `BarraCNE5.INDUSTRY` | 大致 `1-31` | Barra 行业编码 |
| `SwIndMask.SW2021_L1` | `1-31` | 申万一级行业编码 |
| `SwIndMask.SW2021_L2` | 最高约 `124` | 申万二级行业编码 |
| `SwIndMask.SW2021_L3` | 最高约 `258/260` | 申万三级行业编码 |

做行业中性化或风险解释时必须先：

```text
industry_code -> one-hot industry dummy
```

不能直接把行业 code 当连续数值回归。

### Barra 风格、收益、协方差

`BarraCNE5.SIZE / BETA / LIQUIDTY` 等字段确认是真正的 Barra 横截面暴露，且已经标准化：

```text
mean ≈ 0
std ≈ 1
```

因此可以直接进入风险暴露矩阵 `B_t`。实际回归前仍建议在有效样本内做一次稳健处理。

`B/BarraCNE5_RET/CNE5FactorRet.parquet` 确认为 Barra 风险模型因子收益输出：

| 类型 | 维度 |
|---|---:|
| Barra 风格因子收益 | 10 |
| country 因子 | 1 |
| 行业因子收益 | 31 |
| 合计 | 42 |

字段包括：

```text
MOMENTUM, RESVOL, BETA, LIQUIDTY, LEVERAGE, GROWTH,
BTOP, EARNYILD, SIZE, SIZENL, country, 1...31
```

`B/BarraCNE5_COV/*.parquet` 确认为每天一个 `42 x 42` 协方差矩阵。`B/BarraCNE5_RET/CNE5SpecRet.parquet` 确认为 `date x stock` 的 Barra 特异收益/残差收益矩阵。

### 指数权重

`IndexWeight.*` 均为 `date x stock -> index weight`。原始权重是百分比制，正权重加总约为 `100`，使用时要除以 `100`：

```text
index_weight = raw_weight / 100
```

| 字段 | 成分数量 | 权重和 | 结论 |
|---|---:|---:|---|
| `IndexWeight.000852.SH` | 约 1000 | 约 100 | 中证1000 |
| `IndexWeight.000905.SH` | 500 | 约 100 | 中证500 |
| `IndexWeight.000906.SH` | 约 800 | 约 100 | 800 成分指数 |
| `IndexWeight.399300.SZ` | 约 300 | 约 100 | 沪深300口径 |
| `IndexWeight.399303.SZ` | 约 2000 | 约 100 | 2000 成分宽基，正式名称仍需确认 |

### label 与回测输出

`DailyLabel.vwap30_label5d` 确认为连续未来收益值，不是分类标签：

```text
date x stock -> future return
```

IC/ICIR 中的 `y_t` 应使用这个横截面未来收益向量，且必须与 alpha、mask、Barra 暴露取日期交集。

多个模型输出目录下的 `daily_pnl.csv` 字段已确认一致：

```text
date, total_asset, pnl, trade_cost, reserve_cash, tvr, long_num
```

| 字段 | 用途 |
|---|---|
| `pnl` | 日 PnL |
| `trade_cost` | 当日交易成本 |
| `tvr` | 换手率 |
| `long_num` | 持仓股票数 |
| `total_asset` | 资产曲线 |

这些字段足够支持“模型提升是否只是因为降低换手/成本”的诊断。

### FdmCache

FdmCache 的 `.nc` 文件结构已确认：

| 表 | shape |
|---|---|
| `balance_sheet/20240130.nc` | `5642 x 20 x 139` |
| `cash_flow/20240130.nc` | `5642 x 20 x 90` |
| `income/20240130.nc` | `5642 x 20 x 87` |

含义基本是：

```text
股票 x 财报期数 x 财务字段
```

质量、成长、杠杆、现金流质量等理论上可以从这里构造。但当前尚未解析出财务字段坐标对应的真实字段名，所以第一版不依赖 FdmCache。

## 0. Spectral Ridge / shrink-white 本体数据

| 需要数据 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 原始 alpha 因子 `X_t` | `config.xml` 中 `<factor_paths>` 的 `cyz_/wjx_/xk_/yz_*` 因子 | `XROOT/<factor_name>` | 直接可用 | 用于构造 `Z_t`，估计 `C, U, Lambda`；本次 inventory 未扫描 alpha 库，只按 config 映射 |
| 当前训练 label `y_t` | `DailyLabel.label1d` | `A/1d_DailyLabel/DailyLabel.label1d` | 直接可用 | combo 当前读单日 label，再按 `retDays=5` 线性衰减聚合 |
| 5日 label | `DailyLabel.label5d` | `A/1d_DailyLabel/DailyLabel.label5d` | 直接可用 | 可用于独立 Spectral IC/ICIR 检验 |
| VWAP label | `DailyLabel.vwap15_label1d` | `A/1d_DailyLabel/DailyLabel.vwap15_label1d` | 直接可用 | 贴近 VWAP 交易口径 |
| VWAP label | `DailyLabel.vwap30_label1d` | `A/1d_DailyLabel/DailyLabel.vwap30_label1d` | 直接可用 | 贴近当前 30min VWAP 撮合口径 |
| VWAP label | `DailyLabel.vwap30_label2d` | `A/1d_DailyLabel/DailyLabel.vwap30_label2d` | 直接可用 | 2日 VWAP 口径 |
| VWAP label | `DailyLabel.vwap30_label5d` | `A/1d_DailyLabel/DailyLabel.vwap30_label5d` | 直接可用 | 5日 VWAP 口径；确认是连续未来收益值，日期到 `20260610` |
| VWAP label | `DailyLabel.vwap30_label10d` | `A/1d_DailyLabel/DailyLabel.vwap30_label10d` | 直接可用 | 10日 VWAP 口径 |
| VWAP label | `DailyLabel.vwap60_label1d` | `A/1d_DailyLabel/DailyLabel.vwap60_label1d` | 直接可用 | 60min VWAP 口径 |
| mask/weight `w_t` | `StockMask2.BaseUnivMask` | `A/1d_StockMask2/StockMask2.BaseUnivMask` | 直接可用 | 正向 mask：`1.0=在 base 股票池内`，`NaN=剔除` |
| mask/weight `w_t` | `StockMask2.LimitMask` | `A/1d_StockMask2/StockMask2.LimitMask` | 直接可用 | 正向 mask：`1.0=通过涨跌停/交易限制`，`NaN=受限`；不区分涨停/跌停 |
| mask/weight `w_t` | `StockMask2.NoNewStockMask` | `A/1d_StockMask2/StockMask2.NoNewStockMask` | 直接可用 | 正向 mask：`1.0=非需剔除新股` |
| mask/weight `w_t` | `StockMask2.SuspendStock` | `A/1d_StockMask2/StockMask2.SuspendStock` | 直接可用 | 正向 mask：`1.0=可用/未停牌`，不是停牌标记 |
| mask/weight `w_t` | `StockMask2.STStock` | `A/1d_StockMask2/StockMask2.STStock` | 直接可用 | 正向 mask：`1.0=可用/非 ST 风险`，不是 ST 标记 |
| baseline alpha | `alpha_history.pt` | 通常为 `output-torch/alpha_history.pt` | 输出目录获取 | 用于 residual IC，判断是否对 NN 有增量 |
| 持仓/换手/成本 | `daily_pnl.csv` | 通常为 `output-torch/backtest/daily_pnl.csv` | 输出目录获取 | 已确认字段：`date,total_asset,pnl,trade_cost,reserve_cash,tvr,long_num` |

## 1. 第一优先级：风险暴露类

### 1.1 市值

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 总市值 | `DailyFdm.mkt_cap` | `A/1d_DailyFdm/DailyFdm.mkt_cap` | 直接可用 | 原始市值口径 |
| 流通股本 | `DailyFdm.floatshare` | `A/1d_DailyFdm/DailyFdm.floatshare` | 直接可用 | 不是流通市值本身 |
| 流通市值 | `DailyFdm.floatshare` + 价格 | `A/1d_DailyFdm/DailyFdm.floatshare` + `A/1d_DailyKline/DailyKline.close` | 需再处理 | 需要用流通股本乘价格 |
| log market cap | `BarraCNE5.SIZE` | `A/1d_BarraCNE5/BarraCNE5.SIZE` | 直接可用 | 优先使用 Barra 标准化暴露 |
| log market cap | `sub_BarraCNE5.LNCAP` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.LNCAP` | 直接可用 | Barra 子因子口径 |
| 非线性市值 | `BarraCNE5.SIZENL` | `A/1d_BarraCNE5/BarraCNE5.SIZENL` | 直接可用 | 判断小盘/极端市值方向 |
| 非线性市值 | `sub_BarraCNE5.NLSIZE` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.NLSIZE` | 直接可用 | Barra 子因子口径 |

### 1.2 行业

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 申万一级 | `SwIndMask.SW2021_L1` | `A/1d_SwIndMask/SwIndMask.SW2021_L1` | 直接可用，需 one-hot | 行业编码 `1-31`；回归前转 one-hot |
| 申万二级 | `SwIndMask.SW2021_L2` | `A/1d_SwIndMask/SwIndMask.SW2021_L2` | 直接可用，需 one-hot | 行业编码最高约 `124`；回归前转 one-hot |
| 申万三级 | `SwIndMask.SW2021_L3` | `A/1d_SwIndMask/SwIndMask.SW2021_L3` | 直接可用，需 one-hot | 行业编码最高约 `258/260`；回归前转 one-hot |
| Barra 行业 | `BarraCNE5.INDUSTRY` | `A/1d_BarraCNE5/BarraCNE5.INDUSTRY` | 直接可用，需 one-hot | Barra 行业编码大致 `1-31`；不能当连续变量回归 |
| 中信一级/二级 | 当前未扫到 | 无 | 当前未扫到 | 需要另找行业目录或代码表 |
| 证监会行业 | 当前未扫到 | 无 | 当前未扫到 | 需要另找行业目录或代码表 |

### 1.3 Beta

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 个股对指数 beta | `BarraCNE5.BETA` | `A/1d_BarraCNE5/BarraCNE5.BETA` | 直接可用 | 优先使用 |
| 个股对指数 beta 子因子 | `sub_BarraCNE5.BETA` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.BETA` | 直接可用 | 可做细分解释 |
| rolling beta | `DailyKline.pct_chg` + 指数收益 | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 需要先构造指数收益，再 rolling 回归 |

### 1.4 波动率

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 残差波动/波动率 | `BarraCNE5.RESVOL` | `A/1d_BarraCNE5/BarraCNE5.RESVOL` | 直接可用 | 优先使用 |
| 历史 sigma | `sub_BarraCNE5.HSIGMA` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.HSIGMA` | 直接可用 | 子因子 |
| 日收益波动 | `sub_BarraCNE5.DASTD` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.DASTD` | 直接可用 | 子因子 |
| 20/60/120日 realized vol | `DailyKline.pct_chg` | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | rolling std |
| 20/60/120日 realized vol | `DailyKline.close_hfq` | `A/1d_DailyKline/DailyKline.close_hfq` | 需再处理 | 先算收益，再 rolling std |

### 1.5 流动性

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| Barra 流动性 | `BarraCNE5.LIQUIDTY` | `A/1d_BarraCNE5/BarraCNE5.LIQUIDTY` | 直接可用 | 文件名为 `LIQUIDTY`，不是 `LIQUIDITY` |
| 短期换手 | `sub_BarraCNE5.STOA` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.STOA` | 直接可用 | Barra 流动性子项 |
| 月度换手 | `sub_BarraCNE5.STOM` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.STOM` | 直接可用 | Barra 流动性子项 |
| 季度换手 | `sub_BarraCNE5.STOQ` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.STOQ` | 直接可用 | Barra 流动性子项 |
| 成交额 | `DailyKline.amount` | `A/1d_DailyKline/DailyKline.amount` | 直接可用 | 可做 ADV |
| 成交量 | `DailyKline.vol` | `A/1d_DailyKline/DailyKline.vol` | 直接可用 | 可做 rolling volume |
| Amihud illiquidity | `DailyKline.pct_chg` + `DailyKline.amount` | `A/1d_DailyKline/*` | 需再处理 | 典型近似：`abs(ret) / amount` |

### 1.6 动量与反转

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| Barra 动量 | `BarraCNE5.MOMENTUM` | `A/1d_BarraCNE5/BarraCNE5.MOMENTUM` | 直接可用 | 优先使用 |
| 相对强弱/长期动量 | `sub_BarraCNE5.RSTR` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.RSTR` | 直接可用 | 子因子 |
| 20/60/120/250日收益 | `DailyKline.close_hfq` | `A/1d_DailyKline/DailyKline.close_hfq` | 需再处理 | 需要 rolling return，且排除最近若干日 |
| 20/60/120/250日收益 | `DailyKline.pct_chg` | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 用日收益复合得到 |
| 1/5/10日反转 | `DailyKline.pct_chg` | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 历史短期收益取负或直接作为暴露 |

### 1.7 估值、成长、质量

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 账面市值比 | `BarraCNE5.BTOP` | `A/1d_BarraCNE5/BarraCNE5.BTOP` | 直接可用 | Barra 估值暴露 |
| 盈利收益率 | `BarraCNE5.EARNYILD` | `A/1d_BarraCNE5/BarraCNE5.EARNYILD` | 直接可用 | Barra 估值/盈利暴露 |
| 成长 | `BarraCNE5.GROWTH` | `A/1d_BarraCNE5/BarraCNE5.GROWTH` | 直接可用 | Barra 成长暴露 |
| 杠杆 | `BarraCNE5.LEVERAGE` | `A/1d_BarraCNE5/BarraCNE5.LEVERAGE` | 直接可用 | Barra 杠杆暴露 |
| 子因子 BTOP | `sub_BarraCNE5.BTOP` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.BTOP` | 直接可用 | 估值子项 |
| 子因子 ETOP | `sub_BarraCNE5.ETOP` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.ETOP` | 直接可用 | 盈利收益率子项 |
| 子因子 CETOP | `sub_BarraCNE5.CETOP` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.CETOP` | 直接可用 | 现金盈利收益率，具体口径需确认 |
| 子因子 EPFWD | `sub_BarraCNE5.EPFWD` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.EPFWD` | 直接可用 | forward earnings yield |
| 子因子成长 | `sub_BarraCNE5.EGRO` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.EGRO` | 直接可用 | 成长子项，具体口径需确认 |
| 子因子成长 | `sub_BarraCNE5.SGRO` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.SGRO` | 直接可用 | 成长子项，具体口径需确认 |
| 子因子成长 | `sub_BarraCNE5.EGRLF` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.EGRLF` | 直接可用 | 成长子项，具体口径需确认 |
| 子因子成长 | `sub_BarraCNE5.EGRSF` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.EGRSF` | 直接可用 | 成长子项，具体口径需确认 |
| 子因子杠杆 | `sub_BarraCNE5.BLEV` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.BLEV` | 直接可用 | 杠杆子项 |
| 子因子杠杆 | `sub_BarraCNE5.MLEV` | `A/1d_sub_BarraCNE5/sub_BarraCNE5.MLEV` | 直接可用 | 杠杆子项 |
| PE 预测 | `DailyAvgFcst.year_1_pe/year_2_pe/year_3_pe` | `A/1d_DailyAvgFcst/*_pe` | 直接可用，需确认 | 分析师预测口径，不一定是静态 PE |
| EPS 预测 | `DailyAvgFcst.year_1_eps/year_2_eps/year_3_eps` | `A/1d_DailyAvgFcst/*_eps` | 直接可用，需确认 | 预测盈利 |
| 净利润预测 | `DailyAvgFcst.year_1_np/year_2_np/year_3_np` | `A/1d_DailyAvgFcst/*_np` | 直接可用，需确认 | 预测净利润 |
| 营业收入预测 | `DailyAvgFcst.year_1_op_rt/year_2_op_rt/year_3_op_rt` | `A/1d_DailyAvgFcst/*_op_rt` | 直接可用，需确认 | 预测收入 |
| 营业利润预测 | `DailyAvgFcst.year_1_op_pr/year_2_op_pr/year_3_op_pr` | `A/1d_DailyAvgFcst/*_op_pr` | 直接可用，需确认 | 预测营业利润 |
| ROE 预测 | `DailyAvgFcst.year_1_roe/year_2_roe/year_3_roe` | `A/1d_DailyAvgFcst/*_roe` | 直接可用，需确认 | 质量/盈利能力代理 |
| 研发预测 | `DailyAvgFcst.year_1_rd/year_2_rd/year_3_rd` | `A/1d_DailyAvgFcst/*_rd` | 直接可用，需确认 | 成长/研发代理 |
| 目标价预测 | `DailyAvgFcst.year_1_tp/year_2_tp/year_3_tp` | `A/1d_DailyAvgFcst/*_tp` | 直接可用，需确认 | 分析师目标价 |
| EV/EBITDA 预测 | `DailyAvgFcst.year_1_ev_ebitda/year_2_ev_ebitda/year_3_ev_ebitda` | `A/1d_DailyAvgFcst/*_ev_ebitda` | 直接可用，需确认 | 估值代理 |
| PB/PS/PCF/股息率 | 当前未扫到明确字段 | 无 | 当前未扫到 | 可能需要从财报与价格再加工 |
| ROA/毛利率/资产负债率/现金流质量 | `F/balance_sheet`，`F/income`，`F/cash_flow` | `F/balance_sheet`；`F/income`；`F/cash_flow` | 需再处理 | 需要展开财报字段后计算 |

## 2. Barra/CNE5 优先映射

| Barra/CNE5 暴露 | 映射字段/文件 | 具体位置 | 状态 | 用途 |
|---|---|---|---|---|
| `SIZE` | `BarraCNE5.SIZE` | `A/1d_BarraCNE5/BarraCNE5.SIZE` | 直接可用 | 市值 |
| `BETA` | `BarraCNE5.BETA` | `A/1d_BarraCNE5/BarraCNE5.BETA` | 直接可用 | 市场敏感度 |
| `MOMENTUM` | `BarraCNE5.MOMENTUM` | `A/1d_BarraCNE5/BarraCNE5.MOMENTUM` | 直接可用 | 动量 |
| `RESVOL` | `BarraCNE5.RESVOL` | `A/1d_BarraCNE5/BarraCNE5.RESVOL` | 直接可用 | 残差波动 |
| `LIQUIDITY` | `BarraCNE5.LIQUIDTY` | `A/1d_BarraCNE5/BarraCNE5.LIQUIDTY` | 直接可用 | 流动性；注意文件名拼写 |
| `EARNYILD` | `BarraCNE5.EARNYILD` | `A/1d_BarraCNE5/BarraCNE5.EARNYILD` | 直接可用 | 盈利收益率/估值 |
| `GROWTH` | `BarraCNE5.GROWTH` | `A/1d_BarraCNE5/BarraCNE5.GROWTH` | 直接可用 | 成长 |
| `LEVERAGE` | `BarraCNE5.LEVERAGE` | `A/1d_BarraCNE5/BarraCNE5.LEVERAGE` | 直接可用 | 杠杆 |
| `BTOP` | `BarraCNE5.BTOP` | `A/1d_BarraCNE5/BarraCNE5.BTOP` | 直接可用 | 账面市值比 |
| `SIZENL` | `BarraCNE5.SIZENL` | `A/1d_BarraCNE5/BarraCNE5.SIZENL` | 直接可用 | 非线性市值 |
| CNE5 因子收益 | `CNE5FactorRet.parquet` | `B/BarraCNE5_RET/CNE5FactorRet.parquet` | 直接可用 | 已确认为 `2539 x 42`；10 个风格因子 + country + 31 个行业因子收益 |
| CNE5 特异收益 | `CNE5SpecRet.parquet` | `B/BarraCNE5_RET/CNE5SpecRet.parquet` | 直接可用 | 已确认为 `2539 x 5642` 的 `date x stock` 个股特异收益/残差收益矩阵 |
| CNE5 协方差 | `BarraCNE5_COV/*.parquet` | `B/BarraCNE5_COV` | 直接可用 | 已确认为每天一个 `42 x 42` 协方差矩阵 |

## 3. 第二优先级：交易成本 / 流动性类

| 需要字段 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| ADV 20/60 | `DailyKline.amount` | `A/1d_DailyKline/DailyKline.amount` | 需再处理 | rolling mean |
| EWM 成交额 | `TradeUniverse.ewm_amt` | `A/1d_TradeUniverse/TradeUniverse.ewm_amt` | 直接可用 | 可作为 ADV 近似 |
| 大流动性股票池 | `TradeUniverse.TOP1500` | `A/1d_TradeUniverse/TradeUniverse.TOP1500` | 直接可用 | 流动性过滤 |
| 大流动性股票池 | `TradeUniverse.TOP3000` | `A/1d_TradeUniverse/TradeUniverse.TOP3000` | 直接可用 | 流动性过滤 |
| 日成交量 | `DailyKline.vol` | `A/1d_DailyKline/DailyKline.vol` | 直接可用 | rolling volume 可再处理 |
| 日成交额 | `DailyKline.amount` | `A/1d_DailyKline/DailyKline.amount` | 直接可用 | 比成交量更适合容量分析 |
| 分钟成交额 | `Grid1mTradeBar.amount` | `A/1m_Grid1mTradeBar/Grid1mTradeBar.amount` | 直接可用 | 分钟级流动性 |
| 5分钟成交额 | `Intv5mTradeBar.amount` | `A/5m_Intv5mTradeBar/Intv5mTradeBar.amount` | 直接可用 | 5分钟流动性 |
| 分钟成交量 | `Grid1mBar.volume`，`Grid1mTradeBar.volume` | `A/1m_Grid1mBar/*`；`A/1m_Grid1mTradeBar/*` | 直接可用 | 分钟流动性 |
| 5分钟成交量 | `Intv5mBar.volume`，`Intv5mTradeBar.volume` | `A/5m_Intv5mBar/*`；`A/5m_Intv5mTradeBar/*` | 直接可用 | 5分钟流动性 |
| 换手率 | `sub_BarraCNE5.STOA/STOM/STOQ` | `A/1d_sub_BarraCNE5/*` | 直接可用 | Barra 流动性子项 |
| 分钟 turnover | `Grid1mBar.turnover` | `A/1m_Grid1mBar/Grid1mBar.turnover` | 需确认 | 需确认具体单位 |
| 5分钟 turnover | `Intv5mBar.turnover` | `A/5m_Intv5mBar/Intv5mBar.turnover` | 需确认 | 需确认具体单位 |
| 回测 VWAP | `IntraVwap.VwapBegin30` | `A/1d_IntraVwap/IntraVwap.VwapBegin30` | 直接可用 | 当前回测撮合价：开盘 30 分钟 VWAP |
| 其他开盘 VWAP | `IntraVwap.VwapBegin15/60/120` | `A/1d_IntraVwap/*` | 直接可用 | 可做成交价敏感性 |
| 收盘段 VWAP | `IntraVwap.VwapEnd15/30/60/120` | `A/1d_IntraVwap/*` | 直接可用 | 可做收盘撮合口径 |
| 开盘 TWAP | `IntraTwap.TwapBegin15/30/60/120` | `A/1d_IntraTwap/*` | 直接可用 | 可做 TWAP 交易口径 |
| 收盘 TWAP | `IntraTwap.TwapEnd15/30/60/120` | `A/1d_IntraTwap/*` | 直接可用 | 可做 TWAP 交易口径 |
| bid-ask spread 近似 | `first_weighted_ask_prc` - `first_weighted_bid_prc` | `A/1m_Grid1mBar/*` 或 `A/5m_Intv5mBar/*` | 需再处理 | 没有现成 spread 字段 |
| 未停牌正向 mask | `StockMask2.SuspendStock` | `A/1d_StockMask2/StockMask2.SuspendStock` | 直接可用 | `1.0=可用/未停牌`，`NaN=剔除` |
| 涨跌停/交易限制正向 mask | `StockMask2.LimitMask` | `A/1d_StockMask2/StockMask2.LimitMask` | 直接可用 | `1.0=可交易`，`NaN=受限`；不区分涨停/跌停方向 |
| 非 ST 风险正向 mask | `StockMask2.STStock` | `A/1d_StockMask2/StockMask2.STStock` | 直接可用 | `1.0=可用/非 ST 风险`，不是 ST 标记 |
| 非新股正向 mask | `StockMask2.NoNewStockMask` | `A/1d_StockMask2/StockMask2.NoNewStockMask` | 直接可用 | `1.0=不是需剔除新股` |
| 上市天数 | `StockMask2.StockListedDays` | `A/1d_StockMask2/StockMask2.StockListedDays` | 直接可用 | 数据质量/新股异常 |

## 4. 第三优先级：指数 / benchmark 暴露

| 需要数据 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 中证1000成分权重 | `IndexWeight.000852.SH` | `A/1d_IndexWeight/IndexWeight.000852.SH` | 直接可用 | 约 1000 只成分，权重和约 100；使用时除以 100 |
| 中证500成分权重 | `IndexWeight.000905.SH` | `A/1d_IndexWeight/IndexWeight.000905.SH` | 直接可用 | 500 只成分，权重和约 100；使用时除以 100 |
| 800 成分指数权重 | `IndexWeight.000906.SH` | `A/1d_IndexWeight/IndexWeight.000906.SH` | 直接可用 | 约 800 只成分，权重和约 100；使用时除以 100 |
| 沪深300成分权重 | `IndexWeight.399300.SZ` | `A/1d_IndexWeight/IndexWeight.399300.SZ` | 直接可用 | 约 300 只成分，权重和约 100；使用时除以 100 |
| 2000 成分宽基权重 | `IndexWeight.399303.SZ` | `A/1d_IndexWeight/IndexWeight.399303.SZ` | 直接可用，名称需确认 | 约 2000 只成分，权重和约 100；正式指数名称仍需确认 |
| 指数日收益 | 指数权重 + `DailyKline.pct_chg` | `A/1d_IndexWeight/*` + `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 先将原始权重 `/100`，再用成分权重聚合 |
| 指数成交额 | 指数权重 + `DailyKline.amount` | `A/1d_IndexWeight/*` + `A/1d_DailyKline/DailyKline.amount` | 需再处理 | 用成分权重或成分求和近似 |
| 指数波动率 | 指数收益 rolling std | 派生字段 | 需再处理 | 由指数收益计算 |

## 5. 第四优先级：市场环境 / regime 因子

| 需要 regime | 当前可用映射 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 全A日收益 | `DailyKline.pct_chg` 横截面等权/市值加权聚合 | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 可用 base universe 聚合 |
| 沪深300/中证500/中证1000日收益 | `IndexWeight.*` + `DailyKline.pct_chg` | `A/1d_IndexWeight/*` + `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 用权重聚合 |
| 指数 realized vol | 聚合指数收益 rolling std | 派生字段 | 需再处理 | 高波/低波 regime |
| 全市场成交额 | `DailyKline.amount` 横截面求和/分位数 | `A/1d_DailyKline/DailyKline.amount` | 需再处理 | 流动性 regime |
| 涨跌家数 | `DailyKline.pct_chg > 0/<0` 横截面计数 | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | 市场宽度 |
| 交易受限数量 | `StockMask2.LimitMask` 横截面 `NaN` 计数 | `A/1d_StockMask2/StockMask2.LimitMask` | 需再处理 | 可统计涨跌停/交易受限总量，但不能区分涨停和跌停 |
| 横截面离散度 | `DailyKline.pct_chg` 横截面 std | `A/1d_DailyKline/DailyKline.pct_chg` | 需再处理 | alpha 机会多少 |
| 小盘-大盘收益差 | `IndexWeight.000852.SH` / `000905.SH` / `399300.SZ` 聚合收益差 | `A/1d_IndexWeight/*` | 需再处理 | 风格轮动 |
| 成长-价值收益差 | `CNE5FactorRet.parquet` 中的 `GROWTH`、`BTOP/EARNYILD` 风格收益 | `B/BarraCNE5_RET/CNE5FactorRet.parquet` | 直接可用，需读取 | 用 Barra 风格收益构造 |
| 融资融券余额 | 当前未扫到 | 无 | 当前未扫到 | 需要另找数据源 |
| 北向资金 | 当前未扫到 | 无 | 当前未扫到 | 需要另找数据源 |

## 6. 第五优先级：数据质量 / 可交易性因子

| 需要数据 | 映射字段/文件 | 具体位置 | 状态 | 说明 |
|---|---|---|---|---|
| 因子缺失率 | 原始 alpha `X_t` 的 `nan` 比例 | `XROOT/<factor_name>` | 需再处理 | 必须在 `nan_to_num` 之前统计 |
| 股票有效天数 | `W` 或 `BaseUnivMask` 跨日累计 | `A/1d_StockMask2/StockMask2.BaseUnivMask` | 需再处理 | 判断样本稳定性 |
| 上市天数 | `StockMask2.StockListedDays` | `A/1d_StockMask2/StockMask2.StockListedDays` | 直接可用 | 排除新股异常 |
| 新股过滤 | `StockMask2.NoNewStockMask` | `A/1d_StockMask2/StockMask2.NoNewStockMask` | 直接可用 | 避免新股异常 |
| 停牌/不可用频率 | `StockMask2.SuspendStock` 跨日 `NaN` 频率 | `A/1d_StockMask2/StockMask2.SuspendStock` | 需再处理 | 该字段是正向 mask，统计 `NaN` 才是剔除频率 |
| 涨跌停/交易受限频率 | `StockMask2.LimitMask` 跨日 `NaN` 频率 | `A/1d_StockMask2/StockMask2.LimitMask` | 需再处理 | 该字段是正向 mask，且不区分 up/down |
| 非 ST 风险过滤 | `StockMask2.STStock` | `A/1d_StockMask2/StockMask2.STStock` | 直接可用 | 正向 mask：`1.0=可用/非 ST 风险` |
| 财报发布日期 | 当前未扫到明确字段 | 无 | 当前未扫到 | 需要另找财报发布日字段 |
| 异常成交标记 | 当前未扫到专门 flag | 无 | 当前未扫到 | 可用成交额/收益极值间接构造 |

## 7. 建议第一版进入风险暴露矩阵 `B_t` 的字段

| 模块 | 建议字段 |
|---|---|
| Barra 风格 | `SIZE`，`SIZENL`，`BETA`，`MOMENTUM`，`RESVOL`，`LIQUIDTY`，`EARNYILD`，`GROWTH`，`LEVERAGE`，`BTOP` |
| 行业 | `SwIndMask.SW2021_L1` 或 `BarraCNE5.INDUSTRY`，先转 one-hot 后进入回归 |
| 指数暴露 | `IndexWeight.000852.SH`，`IndexWeight.000905.SH`，`IndexWeight.399300.SZ`；原始权重需 `/100` |
| 流动性/成本 | `DailyKline.amount` rolling ADV，`TradeUniverse.ewm_amt`，`sub_BarraCNE5.STOA/STOM/STOQ` |
| 可交易性 | `BaseUnivMask`，`LimitMask`，`NoNewStockMask`，`SuspendStock`，`STStock`，`StockListedDays`；除上市天数外均为正向 mask |
| 数据质量 | alpha 缺失率、停牌频率、涨跌停频率、有效样本天数 |

## 8. 明确不确定或缺失的映射

| 数据需求 | 当前结论 | 处理建议 |
|---|---|---|
| PB/PS/PCF/股息率 | 当前扫描未看到明确字段 | 从 `FdmCache` 财报表 + 价格/市值再算，或继续扫描估值专用目录 |
| ROA/毛利率/现金流质量/资产负债率 | FdmCache 结构已确认，但字段名仍未解析 | `.nc` 基本形态为 `股票 x 财报期数 x 财务字段`；第一版暂不依赖 |
| 中信行业/证监会行业 | 当前未扫到 | 若必须使用，需要扩大行业目录扫描 |
| 指数日行情 | 当前只看到指数权重，没看到直接指数收益/成交额/波动率 | 可以用指数权重 `/100` + 个股行情聚合，或另找指数行情缓存 |
| bid-ask spread | 没有现成 spread 字段 | 用分钟级 weighted ask/bid 价格近似 |
| 融资融券余额 | 当前未扫到 | 需要外部或其他缓存数据 |
| 北向资金 | 当前未扫到 | 需要外部或其他缓存数据 |
| 财报发布日期 | 当前未扫到 | FdmCache 已确认为 `.nc` 财务表结构，但尚未解析出公告日字段 |
| 异常成交标记 | 当前未扫到专门 flag | 可用极端收益、极端成交额、停牌/涨跌停组合间接构造 |
| baseline alpha | 不在因子缓存中 | 从 baseline 模型输出目录读取 `alpha_history.pt` |
| 回测持仓/换手/成本 | 不在因子缓存中 | 从回测输出目录读取 `daily_pnl.csv` |

## 9. 使用建议

第一版 Spectral Ridge / shrink-white 诊断不建议一开始塞入所有可疑字段。建议先做三层解释：

1. **纯风险暴露解释**：`BarraCNE5` 十大风格 + 行业。
2. **交易可行性解释**：ADV、换手、停牌、涨跌停、ST、新股、上市天数。
3. **benchmark/regime 解释**：500/1000/300 权重暴露 + 聚合指数收益/成交额/波动率。

如果某个谱方向 `s_{t,j}` 对上述变量的横截面回归 `R^2` 很高，则它更像风险/流动性/数据结构方向，不应被当成独立 alpha 方向直接放大。若 residual 后仍有稳定 IC/ICIR，才更可能是真正有增量的 alpha 方向。

实际落地时建议固定以下约束：

1. **读取约束**：`.ares` 必须按 `meta.npy` 指定的 `float64`、日期分片、股票维度读取；不能直接按 `float32` 裸读。
2. **日期约束**：所有计算先取 `alpha_dates ∩ label_dates ∩ mask_dates ∩ barra_dates`，尤其 5日 label 最后日期短于 Barra/mask。
3. **mask 约束**：`StockMask2.*` 除 `StockListedDays` 外都是正向 mask，使用 `isfinite(mask)` 判断通过。
4. **行业约束**：行业字段必须转 one-hot；不能直接把行业 code 当连续变量进入线性回归。
5. **指数约束**：指数权重是百分比制，使用前必须 `/100`。
6. **Barra 约束**：风格暴露已标准化，可直接进 `B_t`；但在有效样本内仍建议 winsorize 或重标准化。
7. **FdmCache 约束**：财务 `.nc` 结构已确认，但字段名未解析前，不放进第一版风险解释矩阵。

## 10. 当前仍未完全解决的点

| 项目 | 当前状态 | 后续动作 |
|---|---|---|
| `399303.SZ` 正式指数名称 | 成分数和权重结构确认，约 2000 成分宽基，但正式名称未确认 | 查指数代码表或缓存元数据 |
| `LimitMask` 涨停/跌停方向 | 确认为正向可交易 mask，但不区分 up/down | 若要区分方向，需要结合日收益、涨跌停价或行情规则派生 |
| FdmCache 财务字段名 | `.nc` 结构确认，字段名未解析 | 读取坐标/属性/元数据，建立字段字典 |
| bid-ask spread | 仍未确认直接字段 | 用 weighted ask/bid price 派生 proxy |
| alpha 缺失率 | 还未读取 `/root/ml_data/ZsimPool` 或 `XROOT` 的具体 alpha 值 | 单独统计原始 alpha 在 `nan_to_num` 前的缺失率 |
| baseline alpha | 不属于因子缓存 | 从 baseline 输出目录读取 `alpha_history.pt` |
| 成本/换手 | `daily_pnl.csv` 字段已确认 | 从候选模型与 baseline 的回测输出目录对齐读取 |
