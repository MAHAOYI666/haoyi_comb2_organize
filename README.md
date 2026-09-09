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

## 使用

```bash
combo-hello-world -y
runCombo config.xml
runEval config.xml
```

在 ResearchLoader.data_requirements 中声明数据与 delay，在 process_source 中对分钟数据按时点降维。通过 model_input_sources、model_target 和 model_validity_source 选择 X、Y 及有效性字段。Python 数据路径相对研究员文件解析。

数据集返回 (idx, ds, ti, x, y, w)。x 是 [tsDays, stock, feature] 普通张量，窗口取过去 tsDays 个交易日的同一时点快照。模型实现 fit、predict(x_window, di=..., ti=...)、save、load。

完整配置、索引示例、扩展接口及优化器参数见 [config.human](config.human)。示例见 [eg-torch](eg-torch) 和 [eg-lgbm](eg-lgbm)。

## 回测和评估

execution_price 显式指定 source:column。日内使用 opt2，执行层保留 T+1 锁定、真实持仓和累计换手，每天结算一次。原始成交价、涨跌停和停牌状态共同决定可交易池；不可交易旧持仓冻结，策略若返回池外订单会立即失败。`StockMask2.StockListedDays` 从有效变为缺失时，已有持仓在优化前按零值核销并写入 `settlements.csv`；临时停牌仍沿用最后估值。默认策略的风险、行业和相对方差使用可配置 benchmark，默认 000905.SH；ZZ500 股票池约束和回测报告评价基准独立。

alpha.parquet 使用 (date,time) 索引。runEval 使用研究员原始目标计算 IC，并读取实际成交日报计算 PnL。--skip-exposure 和 --skip-deciles 控制额外分析。独立的 --sim/--pnl/--corr/--va/--exposure 模式继续接受本地表格。

## 内存和监控

<combo><loader compression="fp4" registry_cache_days="64" /></combo> 配置特征缓存。来源分块处理后释放原始数组，数据集按需组装窗口。none/fp4/fp8 使用现有张量 Codec；成交价格和原始目标不经过特征压缩。

monitor 可记录读取、训练、预测和回测耗时及内存。长任务须按 cgroup 可用资源评估峰值。
