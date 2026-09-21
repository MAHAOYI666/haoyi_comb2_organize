# comb2-organize

combo2 将研究员的 Memmap 数据、源级降维、数组模型、训练、预测、opt2 成交回测和评估接到同一流程。每天一个或多个 sample_times 使用相同接口。

发布包名为 combo2，版本由 VERSION 管理；发布记录见 RELEASE.md。

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

### 默认优化器与 simple 模式

默认 opt1 参数的逐项说明见 [opt1_parameters.md](opt1_parameters.md)。

当前默认配置恢复为旧版 optimizer 口径：`maxtvr=0.4`、`max_weight=0.0075`、`min_participation_ratio=0.07`、`parti_penalty=0.0`，并保留完整的 hard/soft universe、risk 和 industry group 列表。

直接 daily VA 评估默认使用上述旧配置；只有显式传入 `runEval ... --simple` 时，才使用简化配置（`maxtvr=0.08`、`max_weight=0.008`、`min_participation_ratio=0.1`、`parti_penalty=0.05`，且各 hard/soft 列表为空）。simple 和 old 的评估结果会写入不同目录。

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

## 回测和评估

execution_price 显式指定 source:column。日内使用 opt2，执行层保留 T+1 锁定、真实持仓和累计换手，每天结算一次。原始成交价、涨跌停和停牌状态共同决定可交易池；不可交易旧持仓冻结，策略若返回池外订单会立即失败。`StockMask2.StockListedDays` 从有效变为缺失时，已有持仓在优化前按零值核销并写入 `settlements.csv`；临时停牌仍沿用最后估值。默认策略的风险、行业和相对方差使用可配置 benchmark，默认 000905.SH；ZZ500 股票池约束和回测报告评价基准独立。

alpha.parquet 使用 (date,time) 索引。runEval 使用研究员原始目标计算 IC，并读取实际成交日报计算 PnL。--skip-exposure 和 --skip-deciles 控制额外分析。独立的 --sim/--pnl/--corr/--va/--exposure 模式继续接受本地表格。

## 内存和监控

<combo><loader compression="fp4" cacheDays="64" /></combo> 配置特征编码和降维后来源缓存。`cacheDays` 按每个来源保留交易日数，每个日期包含全部配置 ti；`load_chunk_days` 只控制一次读取和处理的原始块大小。来源分块处理后释放原始数组，保留区外的降维结果也在当前工作块结束后释放；训练数据集一次性保存各时点的特征和标签，取样时切片、解码并执行窗口变换；预测按时点复用滚动缓冲区。none/fp4/fp8 使用现有张量 Codec；成交价格和原始目标不经过特征压缩。训练存储不受 `cacheDays` 限制，需按训练天数、时点数、股票数、特征数和来源缓存评估内存；源数据更新后需重建数据集。

monitor 可记录读取、训练、预测和回测耗时及内存。长任务须按 cgroup 可用资源评估峰值。
