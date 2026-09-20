# opt1 默认参数

本文记录当前 `comb2` 正常工作流使用的
`config.py::DEFAULT_OPTIMIZER_CONFIG`。它对应完整的默认优化器配置，
不是 `runEval --simple` 使用的简化配置。

## 使用规则

- `<strategy><optimizer>` 中显式写出的属性覆盖默认值，未写出的属性继承默认值。
- `config.eg.old.xml` 是完整的 XML 样例；可以复制其中的 optimizer 属性，再按实验修改日期、路径和输出设置。
- `opt1` 使用股票权重约束，`opt2` 使用金额约束并额外处理 T+1 锁定和日内累计换手。
- 优化器要求 `Mosek==11.0.25` 和有效的 `MOSEKLM_LICENSE_FILE`。

## 完整默认值

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `type` | `opt1` | 优化器模式。 |
| `lambda0` | `0.5` | 相对基准方差惩罚系数；为 0 时不构造方差项。 |
| `shrinkage` | `0.5` | 协方差向对角方差收缩比例，范围 `[0, 1]`。 |
| `ret_days` | `60` | 方差和收益有效性检查的历史交易日数。 |
| `ret_delay` | `1` | 风险收益窗口相对求解日的交易日延迟。 |
| `ret_method` | `2` | `1` 使用个股收益；`2` 使用相对基准收益。 |
| `benchmark` | `000905.SH` | 风险、行业和相对方差使用的基准指数。 |
| `benchmark_delay` | `1` | 基准指数权重的交易日延迟。 |
| `target_size` | `100000000` | opt1 容量约束使用的目标股票 book，单位为元。 |
| `maxtvr` | `0.4` | 全天累计换手额度。 |
| `max_weight` | `0.0075` | 可买股票的单票权重上限。 |
| `maxtrd` | `0.0` | 单票交易额约束；为 0 时不添加。 |
| `maxpos` | `0.0` | 单票持仓额约束；为 0 时不添加。 |
| `liquidity_delay` | `1` | 流动性约束使用的交易日延迟。 |
| `lambda_slp` | `0.0` | 滑点成本惩罚系数；为 0 时不添加。 |
| `slippage_delay` | `1` | 滑点和对应收盘价的交易日延迟。 |
| `min_participation_ratio` | `0.07` | 有效持股数相对候选股票数的最低比例。 |
| `parti_penalty` | `0.0` | 持仓集中度惩罚系数；为 0 时不添加。 |
| `trim_threshold` | `0.00001` | 求解前后的小仓位裁剪阈值。 |
| `long_ratio` | `0.5` | 当前截面保留为正 alpha 的做多比例。 |
| `min_valid_instruments` | `200` | 启动求解所需的最少候选股票数。 |
| `min_return_obs` | `20` | 历史窗口所需的最少有效收益观测数。 |
| `soft_univ_penalty` | `0.00025` | soft universe 越界惩罚。 |
| `soft_risk_penalty` | `0.00004` | soft risk 越界惩罚。 |
| `soft_group_penalty` | `0.0002` | soft industry group 越界惩罚。 |
| `num_mosek_threads` | `1` | MOSEK 求解线程数。 |
| `max_time` | `30.0` | 单日最大求解秒数；0 表示不限制。 |
| `post_trim_renorm` | `false` | 求解后裁剪小仓位时不重新归一化。 |

### 默认约束列表

```
univ_list="ZZ500:0.18:0.70,1|AshareST:0.00:0.00,1|AshareSH:0.00:0.60,1|AshareSZ:0.00:0.60,1|NONETOP3000:0.00:0.17,1|AshareCYB:0.10:0.30,1"

soft_univ_list="ZZ1800:0.74:0.85:0.70,1|ZZ1800:0.79:0.85:0.45,1|ZZ1800:0.83:0.89:0.25,1|ZZ500:0.28:0.50:2.0,1|ZZ500:0.30:0.50:0.2,1|ZZ500:0.32:0.50:0.05,1|AshareSH:0.00:0.55:0.50,1|AshareCYB:0.10:0.25:1.00,1|AshareSZ:0.00:0.55:0.50,1|HS300:0.10:0.30:1.00,1|NONETOP3000:0.00:0.15:1.00,1"

risk_list="cap:-0.40:0.30,1,4|cap:-0.20:0.21,1,2|returns120:-0.14:0.14,1|vola_30:-0.30:0.30,1|vola_5:-0.30:0.30,1|close:-0.10:0.10,1|BarraCNE5.BETA:-0.20:0.30,1|BarraCNE5.GROWTH:-0.15:0.20,1|BarraCNE5.BTOP:-0.15:0.20,1|BarraCNE5.LEVERAGE:-0.30:0.30,1|BarraCNE5.RESVOL:-0.30:0.30,1"

soft_risk_list="cap:-0.02:0.07:2.0,1,4|returns120:-0.08:0.08:5.0,1|close:-0.05:0.05:1.0,1|BarraCNE5.BETA:0.00:0.04:1.7,1|BarraCNE5.GROWTH:-0.02:0.06:1.3,1|BarraCNE5.BTOP:-0.02:0.07:1.3,1|BarraCNE5.EARNYILD:-0.03:0.03:2.0,1"

group_list="WindIndustry.sw1:-0.065:0.065,1"
soft_group_list="WindIndustry.sw1:-0.05:0.05,1|WindIndustry.sw3:-0.012:0.012,1"
```

风险约束中的 method 定义如下：

- method `1`：在有效股票截面上直接做 z-score；
- method `2`：截面 rank 后居中；
- method `4`：正值取自然对数后做截面 z-score。

默认列表使用 method `4`、`2`，省略 method 时按 `2` 解析。

## `runEval --simple`

只有显式传入 `--simple` 时才切换到简化 profile。其它未覆盖参数仍继承完整默认值。

| 参数 | 完整默认值 | simple 值 |
|---|---:|---:|
| `maxtvr` | `0.4` | `0.08` |
| `max_weight` | `0.0075` | `0.008` |
| `min_participation_ratio` | `0.07` | `0.1` |
| `parti_penalty` | `0.0` | `0.05` |
| `univ_list` | 完整列表 | 空 |
| `soft_univ_list` | 完整列表 | 空 |
| `risk_list` | 完整列表 | 空 |
| `soft_risk_list` | 完整列表 | 空 |
| `group_list` | 完整列表 | 空 |
| `soft_group_list` | 完整列表 | 空 |

## 成交价配置

成交价仍通过现有的 XML 配置指定来源和字段，例如：

```
<backtest execution_price="execution:execution" />
```

本版本不提供额外的 direct daily `--execution-price` 或
`vwap_begin30` 选择开关。研究员应在 loader 中声明并处理自己的
`source:column` 成交价来源。
