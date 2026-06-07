# optuna_framework 使用指南

`optuna_framework` 是给 comb2 模型做自动调参、稳定性复跑和样本外验证的 Optuna 工作流。它从一个能直接运行的 baseline XML 出发，为 baseline、每个 Optuna trial、每个 seed 检查和最终 holdout 验证分别渲染独立的 `config.xml`，再调用 `runCombo.py` 完成训练、回测、指标解析和报告汇总。

这个框架本身只负责“改 XML 参数并评估结果”。真正模型能不能跑，取决于 baseline XML 指向的 comb2 模型是否能被 `runCombo.py` 正常执行。

所有命令请在有完整因子运行环境的远程实例、仓库根目录下执行。下面统一用 `python` 表示当前运行环境的 Python 入口。

## 搜索和验证策略

完整流程分四步：

1. Baseline：用 baseline XML 里的原始参数跑 full window，得到对照指标和 hard filter。
2. Phase A：用 Optuna 在配置的搜索空间里调参，目标是在 scoring window 上最大化 `sharpe_idx`。
3. Phase B：从 Phase A 的 top 候选里挑参数组，用多个 seed 复跑 tuning/scoring window，筛掉不稳定参数。
4. Phase C：把 Phase B 存活候选放到 full window 和 holdout 年份上复验，决定是否接受。

Phase A 正式搜索使用：

- `TPESampler(n_startup_trials=24, multivariate=True, group=True, seed=42)`
- `MedianPruner(n_startup_trials=24, n_warmup_steps=2, interval_steps=1)`
- `direction="maximize"`
- sqlite storage 持久化 study，重复运行会继续已有结果
- study 为空时，会先 enqueue baseline 参数，保证 baseline 参数也在 trial 结果里

Phase A 的 objective 逻辑：

1. Optuna 采样一组参数。
2. 框架把参数写入一份独立 XML。
3. 调用 `runCombo.py <rendered_config.xml>`。
4. 读取该 run 的 `output/backtest/pnl_summary.csv`；如果 scoring window 小于 run window，则从 `daily_pnl.csv` 精确切片计算 scoring-window 指标。
5. 用 scoring-window `sharpe_idx` 作为目标值。
6. 如果触发 baseline hard filter，则 objective 记为 `-10.0`。

hard filter 来自 baseline：

```text
min_sharpe_threshold = baseline_tuning_period_sharpe_idx - 0.3
max_dd_threshold = baseline_tuning_period_dd_li * 1.3
```

也就是说，Phase A 是先在 tuning run 内，以 scoring window 的 `sharpe_idx` 为主目标，并用 baseline 的回撤和夏普门槛过滤明显较差的 trial。

## 模型接口依赖

Optuna 框架不绑定某个具体模型文件。只要你的 baseline XML 能正常跑 `runCombo.py`，并且 `<search_space>` 中声明的参数存在于 baseline XML 的 `combo.model` 或 `combo.runtime` 中，就可以用这个框架调参。

但是 `runCombo.py/comb2` 对模型文件有固定接口要求。baseline XML 中：

```xml
<combo>
  <paths model_path="model_hybrid_tcn.py" />
</combo>
```

`model_path` 指向的 Python 文件必须定义：

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

类名必须是 `ResearchModel`。框架会把 `<combo><model ... />` 里的参数整理成 `config` 字典传给 `ResearchModel(config)`。

注意：

- Optuna 调 `section="model"` 的参数时，本质是改 XML 的 `<combo><model ... />` 属性。
- 这些参数会进入 `ResearchModel(config)`，但如果模型代码没有读取某个 key，调这个参数就不会产生实际效果。
- Optuna 调 `section="runtime"` 的参数时，本质是改 XML 的 `<combo><runtime ... />` 属性，影响 comb2 运行逻辑。
- Optuna 不会检查模型代码是否真的使用了某个参数，只会检查该参数是否存在于 baseline XML 中。

## 1. 配置实验

主配置文件：

```text
optuna_framework/config.xml
```

最常需要改的是 `<study>`、`<baseline>`、`<windows>`、`<search_space>`、`<phase_b>` 和 `<phase_c>`。

### 1.1 实验名和 trial 数

```xml
<study
  name="study_hybrid_tcn_detailed"
  optuna_name="hybrid_tcn"
  n_trials="60"
/>
```

- `name`：实验名，默认也是输出目录名。
- `optuna_name`：Optuna sqlite 内部 study 名。
- `n_trials`：`run_study.py` 未指定 `--n-trials` 时的 Phase A trial 数。

默认输出目录：

```text
optuna_runs/<study.name>/
```

也可以运行命令时用 `--study-root` 覆盖。

### 1.2 Baseline XML

```xml
<baseline config_path="eg-torch/config_hybrid_tcn.xml" />
```

这个 XML 必须能直接跑通 `runCombo.py`。所有 trial config 都从它复制并打补丁。

### 1.3 训练和评分窗口

```xml
<windows
  tuning_run_start_ds="20200102"
  tuning_run_end_ds="20231229"
  full_run_start_ds="20200102"
  full_run_end_ds="20240628"
  scoring_start_ds="20210104"
  scoring_end_ds="20231229"
/>
```

- Phase A/B 跑 `tuning_run_start_ds` 到 `tuning_run_end_ds`。
- Baseline/Phase C 跑 `full_run_start_ds` 到 `full_run_end_ds`。
- Objective 使用 `scoring_start_ds` 到 `scoring_end_ds`。
- `loader.data_start_ds` 仍来自 baseline XML，不由这里控制。
- scoring window 可以比 run window 窄；这时框架会从 `daily_pnl.csv` 精确切片计算指标。

## 2. 设置搜索空间

在 `<search_space>` 中声明要调的参数：

```xml
<search_space>
  <param section="model" name="lr" type="float" low="1e-7" high="1e-4" log="true" />
  <param section="model" name="hiddenSize" type="categorical" choices="256,384,512,768" />
</search_space>
```

规则：

- `section="model"` 对应 baseline XML 的 `<combo><model ... />`。
- `section="runtime"` 对应 baseline XML 的 `<combo><runtime ... />`。
- `name` 必须是 baseline XML 里已有的属性名。
- `float` / `int` 需要 `low` 和 `high`。
- `float` / `int` 可选 `log="true"`。
- `int` 可选 `step`。
- `categorical` 需要 `choices`，逗号分隔。

当前默认搜索空间如下：

| 参数 | 类型 | 搜索范围 | baseline |
| --- | --- | --- | --- |
| `lr` | float | `1e-7` 到 `1e-4`，log | `2e-6` |
| `weight_decay` | float | `1e-8` 到 `1e-3`，log | `1e-6` |
| `dropout` | float | `0.2` 到 `0.6` | `0.5` |
| `hiddenSize` | categorical | `256,384,512,768` | `512` |
| `fcSize` | categorical | `128,256,512` | `256` |
| `scheduler_gamma` | float | `0.3` 到 `0.9` | `0.5` |
| `tcnChannels` | categorical | `64,128,256` | `128` |
| `tcnLayers` | categorical | `1,2,3` | `1` |
| `tcnKernelSize` | categorical | `2,3,5` | `3` |
| `tcnDropout` | float | `0.2` 到 `0.6` | `0.5` |
| `tcnDilationBase` | categorical | `1,2` | `1` |

## 3. 校验配置

每次改完 `optuna_framework/config.xml` 或 baseline XML，先运行：

```bash
python optuna_framework/scripts/validate_config.py --config optuna_framework/config.xml
```

输出：

```text
optuna_runs/<study.name>/config_plan.txt
```

重点看：

- `status: OK`
- `recognized tunable params`
- `unrecognized params` 为空
- `invalid params` 为空
- 每个参数的 `baseline_value`

正式运行脚本默认会检查这个 plan 的 hash。配置变了但没重新 validate，会报 `config plan hash mismatch`。

## 4. Dry Run

正式跑之前建议检查渲染结果和命令：

```bash
python optuna_framework/scripts/dry_run_render.py --config optuna_framework/config.xml
python optuna_framework/scripts/run_baseline.py --config optuna_framework/config.xml --dry-run
python optuna_framework/scripts/run_study.py --config optuna_framework/config.xml --dry-run --n-trials 2
python optuna_framework/scripts/run_phase_b.py --config optuna_framework/config.xml --dry-run
python optuna_framework/scripts/run_phase_c.py --config optuna_framework/config.xml --dry-run
python optuna_framework/scripts/export_report.py --config optuna_framework/config.xml --dry-run
```

`dry_run_render.py` 会确认渲染后的 XML 只改了允许字段，例如运行窗口、输出路径、checkpoint 路径、`snaptime` 和 fixed overrides。

## 5. 推荐正式运行顺序

```bash
python optuna_framework/scripts/validate_config.py --config optuna_framework/config.xml
python optuna_framework/scripts/run_baseline.py --config optuna_framework/config.xml
python optuna_framework/scripts/run_study.py --config optuna_framework/config.xml --n-trials 60
python optuna_framework/scripts/run_phase_b.py --config optuna_framework/config.xml
python optuna_framework/scripts/run_phase_c.py --config optuna_framework/config.xml
python optuna_framework/scripts/export_report.py --config optuna_framework/config.xml
```

`run_smoke.py` 是可选检查，可以按需插在 baseline 之后、Phase A 之前，但默认不要求跑。

## 6. 跑 Baseline

```bash
python optuna_framework/scripts/run_baseline.py --config optuna_framework/config.xml
```

这一步会跑 full window baseline，并输出：

```text
optuna_runs/<study.name>/baseline/full_run/
optuna_runs/<study.name>/baseline/baseline_thresholds.json
```

重点看 `baseline_thresholds.json`：

- `tuning_period`：Phase A/B scoring window 的 baseline 指标。
- `full_period`：full window 和分年度 baseline 指标。
- `by_segment`：holdout 年份 baseline 指标。
- `hard_filter`：Phase A 淘汰差 trial 的门槛。

失败时看：

```text
optuna_runs/<study.name>/baseline/full_run/run.stderr.log
optuna_runs/<study.name>/baseline/full_run/run.stdout.log
```

## 7. 可选：跑 Smoke

Smoke 是可选步骤，不是正式流程的必需环节。它的作用是用少量 trial 提前检查完整链路是否能跑通。

一般情况下可以跳过 smoke，直接跑 Phase A。因为 Phase A 如果遇到配置、环境或模型问题，也会在对应 trial 失败时退出或记录错误；这时直接看 trial 目录下的日志即可。

如果仍然想先小规模检查，可以运行：

```bash
python optuna_framework/scripts/run_smoke.py --config optuna_framework/config.xml --n-trials 10
```

Smoke 使用独立的 `study_smoke.db`，不会污染正式 Phase A 的 `study.db`。Smoke 搜索策略使用：

- `TPESampler(n_startup_trials=2, multivariate=True, group=True, seed=42)`
- `MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1)`

## 8. 跑 Phase A

```bash
python optuna_framework/scripts/run_study.py --config optuna_framework/config.xml --n-trials 60
```

Phase A 跑完后，就已经可以看到一批值得实验或复跑的候选参数。最重要的文件是：

```text
optuna_runs/<study.name>/reports/top10.csv
```

`top10.csv` 会按 objective 从高到低列出候选 trial，其中：

- `value` 是 Optuna objective，也就是 scoring window 上经过 hard filter 后的目标值。
- `param_*` 列是该 trial 的具体参数。
- `value = -10.0` 表示触发 hard filter，不应作为候选。
- 如果只是想快速拿一组参数去手工实验，Phase A 跑完后可以先从 `top10.csv` 里选靠前的非 `-10.0` 参数。

不过，Phase A 只说明这些参数在当前 scoring window 上表现好。真正决定是否稳健，仍建议继续跑 Phase B 多 seed 检查和 Phase C full-window/holdout 验证。

每个 trial 的目录结构：

```text
optuna_runs/<study.name>/trials/trial_00000/
  config.xml
  params.json
  resolved_meta.json
  trial_meta.json
  run.stdout.log
  run.stderr.log
  output/backtest/pnl_summary.csv
  output/backtest/daily_pnl.csv
  checkpoints/
```

其他常用报告：

```text
optuna_runs/<study.name>/reports/trials.csv
optuna_runs/<study.name>/reports/scoring_metrics.csv
optuna_runs/<study.name>/reports/optimization_history.html
optuna_runs/<study.name>/reports/parallel_coordinate.html
optuna_runs/<study.name>/reports/param_importance.html
```

如果想清理被 hard filter 淘汰 trial 的重型产物：

```bash
python optuna_framework/scripts/run_study.py --config optuna_framework/config.xml --n-trials 60 --cleanup-bad-trials
```

## 9. 跑 Phase B

```bash
python optuna_framework/scripts/run_phase_b.py --config optuna_framework/config.xml
```

Phase B 配置：

```xml
<phase_b
  seeds="42,43,44"
  candidate_scan_top_n="6"
  candidate_limit="3"
  sharpe_margin="0.15"
  dd_multiplier="1.15"
  sharpe_std_max="0.15"
/>
```

它会从 `reports/top10.csv` 读取前 `candidate_scan_top_n` 行，去重后最多跑 `candidate_limit` 组参数，并对每组参数用多个 seed 复跑。

输出：

```text
optuna_runs/<study.name>/phase_b/results.json
optuna_runs/<study.name>/phase_b/survivors.json
```

淘汰规则：

- 任一 seed 的 scoring-window `sharpe_idx < baseline_sharpe_idx - sharpe_margin`
- 任一 seed 的 scoring-window `dd_li > baseline_dd_li * dd_multiplier`
- 多 seed 的 scoring-window `sharpe_idx` 标准差大于 `sharpe_std_max`

重点看：

- `results.json`：每个候选是否被淘汰，以及淘汰原因。
- `survivors.json`：进入 Phase C 的候选。

## 10. 跑 Phase C

```bash
python optuna_framework/scripts/run_phase_c.py --config optuna_framework/config.xml
```

Phase C 配置：

```xml
<phase_c
  holdout_years="2020,2024"
  tuning_years="2021,2022,2023"
  holdout_sharpe_margin="0.3"
  full_dd_multiplier="1.3"
  min_tuning_years_better="2"
  full_sharpe_std_max="0.25"
/>
```

Phase C 读取 `phase_b/survivors.json`，在 full window 上复跑。输出：

```text
optuna_runs/<study.name>/phase_c/results.json
```

重点看：

- `accepted: true`：通过最终验证。
- `accepted: false`：没有通过。
- `reasons`：失败原因。
- `seed_results`：每个 seed 的 full period 和分年度指标。

接受规则：

- holdout 年份夏普不能低于对应 baseline 太多。
- full window `dd_li` 不能超过 baseline 太多。
- 每个 seed 至少有 `min_tuning_years_better` 个 tuning 年份 beat baseline。
- 多 seed full-period 夏普标准差必须小于 `full_sharpe_std_max`。

## 11. 导出报告

```bash
python optuna_framework/scripts/export_report.py --config optuna_framework/config.xml
```

输出：

```text
optuna_runs/<study.name>/REPORT.md
```

报告会汇总：

- 实验配置和窗口
- baseline 指标
- Phase A trial 状态和 top10
- Phase B 淘汰情况
- Phase C 最终接受情况
- 推荐候选参数；如果没有通过 Phase C，会显示 `no improvement found`

## 12. 常用参数

- `--config`：指定 Optuna XML 配置文件。
- `--study-root`：覆盖默认输出目录，适合临时实验或对比实验。
- `--dry-run`：打印计划和命令，不正式调用 `runCombo.py`。
- `--n-trials`：覆盖 Phase A 或 smoke 的 trial 数。
- `--cleanup-bad-trials`：Phase A 中对触发 hard filter 的 trial 清理 checkpoint 和 alpha history 等重型产物。
- `--skip-plan-check`：绕过已保存 plan hash 检查；只建议在明确知道配置变化影响时手动使用。

可选的同 seed 稳定性检查：

```bash
python optuna_framework/scripts/seed_sanity_check.py --config optuna_framework/config.xml --seed 42
```

## 13. 常见问题

`config plan is missing`

先运行：

```bash
python optuna_framework/scripts/validate_config.py --config optuna_framework/config.xml
```

`config plan hash mismatch`

说明 `config.xml` 或 baseline XML 变了。重新 validate，并检查新的 `config_plan.txt`。

`baseline_thresholds.json not found`

先运行 baseline：

```bash
python optuna_framework/scripts/run_baseline.py --config optuna_framework/config.xml
```

`Phase A top10.csv not found or empty`

先完成 `run_study.py`，并确认有有效 trial。全是 `-10.0` 通常说明搜索空间或 hard filter 需要复查。

`Phase B survivors not found`

先完成 `run_phase_b.py`，并确认 `survivors.json` 是否为空。

`runCombo failed`

查看对应 run 目录下：

```text
run.stderr.log
run.stdout.log
```

`scoring-window metrics` 解析失败

当 scoring window 与 run window 不一致时，框架需要从 `daily_pnl.csv` 切片计算指标。确认 `daily_pnl.csv` 存在，并且包含计算所需的日期、收益或 PnL 字段。
