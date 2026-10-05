# comb2-organize

combo2 将研究员的 Memmap 数据、源级降维、数组模型、训练、预测、opt2 成交回测和评估接到同一流程。每天一个或多个 sample_times 使用相同接口。

发布包名为 combo2，版本由 VERSION 管理；发布记录见 RELEASE.md。

## 安装方式

源码开发使用 Python 3.13，在仓库根目录安装：

```bash
python -m pip install -e ".[dev]"
```

项目依赖、命令和包映射由 `pyproject.toml` 管理；受保护 wheel 构建使用同一份依赖和命令定义。发布包不包含 Optuna 框架及其依赖；`optuna_framework/` 仅保留为源码研究工具。

当前仓库内置运行所需源码：

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

### 默认优化器与 simple 模式

默认 opt1 参数的逐项说明见 [opt1_parameters.md](opt1_parameters.md)。

当前默认配置使用简化 daily-VA optimizer 参数：`maxtvr=0.08`、`max_weight=0.008`、`min_participation_ratio=0.1`、`parti_penalty=0.05`，并保留完整的 hard/soft universe、risk 和 industry group 列表。

直接 daily VA 评估默认使用上述参数；显式传入 `runEval ... --simple` 时，保留上述四项数值并清空各 hard/soft 列表。默认和 simple 的评估结果会写入不同目录。

`config.eg.old.xml` 保留了旧版完整 optimizer 样例，供迁移和口径对比使用。它不会被自动加载；使用时请复制其中的 optimizer 属性，并按实际实验修改路径、日期和输出目录。

## 实验名称与输出隔离

新实验在 XML 根节点设置 `Name`（大小写敏感）：

```xml
<config Name="experiment_a">
  <constants cache_path="/data/Cache" output_root="output" />
  <!-- strategy、combo、backtest 等配置保持原接口 -->
</config>
```

`output_root` 是实验集合目录，加载配置后得到的有效输出目录为 `output/experiment_a/`：

```text
output/experiment_a/
  experiment_a.parquet
  experiment_a.alpha_history.pt
  experiment_a.train.log
  experiment_a.daily_ic.csv
  experiment_a.ic_by_time.csv
  experiment_a.perf_metrics.csv   # 启用监控时
  checkpoints/<snaptime>/<date>/
  backtest/
  eval_report/
  live/experiment_a_<date>_<time>.csv
```

`Name` 必须以字母或数字开头，最多 128 字符，只允许字母、数字、下划线、点和短横线，不能以点结尾。不同 Name 隔离实验输出；同一输出根目录和 Name 表示同一实验，重跑可能覆盖结果或复用检查点，新的参数实验应使用新 Name。Name 不替代 sample_times；snaptime 继续用于检查点子目录。

未设置 Name 的旧 XML 保留原输出目录和 `alpha.parquet` 等文件名。显式设置空 Name 会报错。配置驱动评估只读取当前配置指定的信号文件，不再递归选择其他目录中最新的 alpha。显式指定 monitor.output_path、--report-dir、--plot-output 或 --eval-dir 时使用该路径，调用方需保证其唯一性。

## 使用

```bash
combo-hello-world -y
runCombo config.xml
runEval config.xml
runEval myposition.parquet target.parquet
runEval myposition.parquet target.parquet --simple --ti 093000
comboOpt1 signal.parquet [--config config.xml]
```

在 ResearchLoader.data_requirements 中声明数据与 delay，在 process_source 中对分钟数据按时点降维。通过 model_input_sources、model_target 选择 X、Y；配置 cache_path 时默认交集使用 delay=1 的 BaseUnivMask、NoNewStockMask、LimitMask，model_validity_source 可显式替换默认筛选。Python 数据路径相对研究员文件解析。

数据集返回 (idx, ds, ti, x, y, w)。x 是 [tsDays, stock, feature] 普通张量，窗口取过去 tsDays 个交易日的同一时点快照。模型实现 fit、predict(x_window, di=..., ti=...)、save、load。

完整配置、索引示例、扩展接口及优化器参数见 [config.human](config.human)。示例见 [eg-torch](eg-torch) 和 [eg-lgbm](eg-lgbm)。

## 回测和评估

execution_price 显式指定 source:column。日内使用 opt2，执行层保留 T+1 锁定、真实持仓和累计换手，每天结算一次。原始成交价、涨跌停和停牌状态共同决定可交易池；不可交易旧持仓冻结，策略若返回池外订单会立即失败。`StockMask2.StockListedDays` 从有效变为缺失时，已有持仓在优化前按零值核销并写入 `settlements.csv`；临时停牌仍沿用最后估值。默认策略的风险、行业和相对方差使用可配置 benchmark，默认 000905.SH；ZZ500 股票池约束和回测报告评价基准独立。

实验信号 `<Name>.parquet`（旧配置为 `alpha.parquet`）使用 (date,time) 索引。`runEval` 分为两部分：`runEval alpha1.parquet alpha2.parquet run/read` 的双 parquet 日频分组回测与 PnL/VA 是主流程，实现在 `runEval.py`；其余 config overall、`--sim`、`--pnl`、`--va`、`--corr`、`--exposure` 模式实现在 `evals/comb_eval/run_eval_other.py`，保留原 CLI 兼容。`--skip-exposure` 和 `--skip-deciles` 只影响 overall。overall 中的十档统计是原始 target 的分组均值，不是各档实际成交 PnL。

`comboOpt1 signal.parquet [--config config.xml]` 用 opt1 回测单个信号，预处理与双 parquet VA 的端点相同，输出逐年 ZZ500 超额、IR、换手和冻结日数；不传 --config 时使用默认配置。回测口径由 `<backtest fixbs>` 选择：默认 `true`（1.1.5 起）为固定 book（等于 cash，不预留现金）与零碎股数，仿照 pysim CalcSimple；`false` 为复利 book（`总资产 * reserve_cash`）与整手成交，即 1.1.4 及以前的默认口径，复现旧结果时需显式设置 `<backtest fixbs="false">`。卖出在两种口径下均按目标为零全部卖出、否则按金额（整手时四舍五入）成交。细节见 config.human。

双 parquet VA 的所有 PnL 路径（0.00 target、各混合权重、1.00 信号）使用同一组日期，不一致时报错。未传 `--start` 时，起点是两个信号都有非零值的第一个共同日；传入 `--start`/`--end` 时严格使用该窗口，便于不同种子或变体在同一路径上比较：信号开头最多允许 5 个全零日（当天只用另一个信号），超过即报错；窗口内缺少任一交易日也报错，不会悄悄缩短路径。

## 内存和监控

<combo><loader compression="fp4" /></combo> 配置训练特征和预测缓冲区的编码。来源数据仅在当前读取块中复用，作用域结束后释放；训练数据集保存滚动窗口各时点的 X/Y/W，取样时切片、解码并执行窗口变换；训练窗口在增长（不足 `max_train_days`）和滑动阶段都跨训练保留并原地追加新日期，首次训练即按最终窗口长度和全部股票列预留存储；预测按时点复用滚动缓冲区。none/fp4/fp8 使用现有张量 Codec；成交价格和原始目标不经过特征压缩。需按训练天数、时点数、股票数、特征数、模型和临时读取块评估内存；源数据更新后需重建数据集。

monitor 可记录读取、训练、预测和回测耗时及内存。runCombo 在各主要阶段、回测循环每月末和每次训练开始/结束输出一行 `[STAGE|时刻]`，附累计耗时。长任务须按 cgroup 可用资源评估峰值。
