# comb2 训练框架 Profiling 复现说明

> Historical note: profiling instrumentation was temporary and has been reverted from the current working tree. This file preserves the historical report and output paths; rerunning the commands requires restoring the profiling files.

本文记录 profiling 的实验环境、执行命令、输出格式、结果和限制，供后续复现及性能 A/B 实验使用。

## 1. 实验范围

配置文件：

~~~text
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/config.month.20200521-20200630-lr2e-6-tmp.xml
~~~

输出目录：

~~~text
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/
~~~

测量范围：

- dataset 初始化：特征、标签、mask、缓存和内存快照；
- 数据加载和预处理：source read、DataRegistry 操作、feature/target preprocessing；
- fit：DataLoader、CPU 到 GPU、forward、loss、backward、梯度裁剪和 optimizer step；
- 其他：setup、checkpoint、alpha convert、alpha analysis 和 rolling 外层流程。

本次使用 --skip-backtest。环境没有可用的 MOSEK license，所以结果不包含 backtest。

## 2. Historical Profiling 代码

相关文件：

~~~text
vendor/profile_monitor.py
tools/profile_combo.py
tests/test_profile_monitor.py
runCombo.py
vendor/comb2/comb2/DataRegistry.py
~~~

功能：

- profile_monitor.py：记录 wall time、CPU time、RSS、GPU memory 和 I/O，输出 events.jsonl、event_summary.csv、perf_metrics.csv，支持嵌套事件和受限 torch.profiler；
- profile_combo.py：加载 XML，允许覆盖日期、训练窗口、epoch、batch size、Torch threads，并生成 manifest.json；
- runCombo.py：安装外层、loader、dataset、model 和训练内部 hooks；
- DataRegistry.py：记录 cache lookup、source read、source process、apply ops；
- test_profile_monitor.py：monitor 单元测试。

## 3. 实验环境

### 3.1 主机与资源

~~~text
SSH host: sxh522gpu1
Repository: /home/mengkang/autodl/comb2_organize
Working directory: /home/mengkang/autodl/comb2_organize
KF compute class: gpu-5090
Observed worker: mkli2-0

CPU: 16
Memory: 64Gi
GPU: 1
Timeout: 10800 seconds
Torch intra-op threads: 16
Torch inter-op threads: 1
~~~

所有 GPU 或长时间实验都通过 kf-submit 提交。

### 3.2 Python 与 PyTorch

~~~text
Python: /home/mengkang/.local/bin/python3
Python version: 3.13.11
PyTorch: 2.9.1+cu128
PyTorch CUDA runtime: 12.8
~~~

SSH 节点的 CUDA 状态不作为 GPU 实验依据，实际 GPU 信息以 KF worker 为准。可重新采集：

~~~bash
kf-submit submit \
  --name comb2-profile-env \
  --cpu 1 \
  --memory 2Gi \
  --gpu 1 \
  --compute-class gpu-5090 \
  --timeout-seconds 600 \
  --working-directory-relative autodl/comb2_organize \
  -- /home/mengkang/.local/bin/python3 -c \
  'import platform,sys,torch; print("python",sys.version); print("platform",platform.platform()); print("torch",torch.__version__); print("cuda",torch.version.cuda); print("cuda_available",torch.cuda.is_available()); print("device",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); print("device_count",torch.cuda.device_count())'
~~~

### 3.3 关键有效配置

~~~text
strategy: 20200521 - 20200630
data_start_ds: 20170101
cacheDays: 64
dtype: torch.float16
compression: none
tsDays: 10
trainDelay: 2
max_train_days: 2000
sample_times: [93000]
device: cuda
epochs: 15
batch_size: 5
lr: 2e-6
dropout: 0.3
grad_clip: 10.0
~~~

研究 loader 和 model：

~~~text
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/Model.py
~~~

## 4. Historical 复现命令

### 4.1 静态检查

~~~bash
ssh sxh522gpu1
cd /home/mengkang/autodl/comb2_organize

/home/mengkang/.local/bin/python3 -m py_compile \
  runCombo.py \
  vendor/profile_monitor.py \
  vendor/comb2/comb2/DataRegistry.py \
  tools/profile_combo.py \
  tests/test_profile_monitor.py

/home/mengkang/.local/bin/python3 -m pytest -q tests/test_profile_monitor.py
~~~

本次单元测试结果：2 passed。

### 4.2 完整 profiling

~~~bash
kf-submit submit \
  --name comb2-profile-full-v2 \
  --cpu 16 \
  --memory 64Gi \
  --gpu 1 \
  --compute-class gpu-5090 \
  --timeout-seconds 10800 \
  --working-directory-relative autodl/comb2_organize \
  -- /home/mengkang/.local/bin/python3 tools/profile_combo.py \
  --config /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/config.month.20200521-20200630-lr2e-6-tmp.xml \
  --output-root /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2 \
  --torch-threads 16 \
  --torch-interop-threads 1 \
  --sync-cuda \
  --skip-backtest
~~~

复现时使用新的 output root。注意：

- --working-directory-relative autodl/comb2_organize 不可省略；
- -- 前是 kf-submit 参数，-- 后是实际 Python 命令；
- 必须使用 /home/mengkang/.local/bin/python3；
- --sync-cuda 用于阶段归因，会增加同步开销；
- 吞吐 benchmark 应去掉 --sync-cuda；
- --skip-backtest 用于绕过 MOSEK license 问题。

### 4.3 短 Torch profiler

~~~bash
kf-submit submit \
  --name comb2-profile-deep \
  --cpu 8 \
  --memory 32Gi \
  --gpu 1 \
  --compute-class gpu-5090 \
  --timeout-seconds 3600 \
  --working-directory-relative autodl/comb2_organize \
  -- /home/mengkang/.local/bin/python3 tools/profile_combo.py \
  --config /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/config.month.20200521-20200630-lr2e-6-tmp.xml \
  --output-root /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-deep-20260921 \
  --start-ds 20200630 \
  --end-ds 20200630 \
  --max-train-days 64 \
  --epochs 1 \
  --batch-size 5 \
  --torch-threads 8 \
  --torch-interop-threads 1 \
  --sync-cuda \
  --skip-backtest \
  --torch-profile \
  --torch-profile-wait 1 \
  --torch-profile-warmup 1 \
  --torch-profile-active 4
~~~

该短实验只采集有限的 wait、warmup、active window，避免全量 trace 过大。

## 5. 输出文件

完整结果目录：

~~~text
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2/
~~~

主要文件：

~~~text
manifest.json
profile.run.log
perf_metrics.csv
profile_overview.csv
profile_overview.txt
source_read_top20.csv
profile_details/events.jsonl
profile_details/event_summary.csv
~~~

用途：

- manifest.json：命令行参数、effective config、Python 路径和 repository；
- events.jsonl：事件明细、时间、资源和上下文；
- event_summary.csv：按 event 汇总 count、wall time、CPU time 和峰值内存；
- profile_overview.csv/txt：常用外层事件；
- source_read_top20.csv：source 读取热点；
- profile_details/torch_trace/*.json：短实验 Torch trace。

查看汇总：

~~~bash
column -s, -t < /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2/profile_overview.csv | less -S
column -s, -t < /home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2/source_read_top20.csv | less -S
~~~

## 6. 已完成实验结果

完整 KF run id：

~~~text
718099da-a0cc-5081-9dc7-5d7767cecd78
~~~

完整配置遍历 27 个日期，实际触发 2 次训练。单位为 wall seconds：

| Event | 次数 | 总耗时 |
|---|---:|---:|
| combine | 27 | 909.512 |
| combo_train | 2 | 872.683 |
| fit | 2 | 548.397 |
| detail_dataset_init | 2 | 320.130 |
| alpha_analysis | 1 | 2.261 |

平均一个训练目标日：

~~~text
dataset_init: 160 s，约 2 分 40 秒
fit:          274 s，约 4 分 34 秒
合计:         434–436 s，约 7 分 15 秒
~~~

完整运行的 2 次训练合计约 872.7 秒，外层 rolling 流程约 909.5 秒。backtest 被显式跳过。

### 6.1 Dataset 初始化热点

detail_dataset_init 总计 320.130 秒：

| Event | 次数 | 总耗时 | 说明 |
|---|---:|---:|---|
| detail_gen_feature | 1701 | 131.961 | 特征生成 |
| data_build_raw_feature | 1703 | 85.509 | 原始因子读取、拼接 |
| data_preprocess_features | 1701 | 46.836 | z-score、截断、NaN、dtype |
| detail_gen_target | 1656 | 88.583 | 标签生成 |
| data_gen_raw_target | 1683 | 81.368 | VWAP、Barra、mask、残差目标 |
| data_source_read | 59540 | 48.537 | DataRegistry source read |
| data_source_process | 59540 | 9.090 | source process |
| data_apply_ops | 59540 | 8.800 | DataRegistry 操作链 |
| detail_build_validinsts | 2 | 5.748 | 有效股票集合 |
| detail_gen_base_universe_mask | 3357 | 5.266 | 基础股票池 mask |
| detail_gen_valid_mask | 1701 | 2.643 | 有效样本 mask |
| data_preprocess_target | 1656 | 2.141 | 标签预处理 |

这些事件存在嵌套关系，不能直接相加。gen_feature 包含 raw feature 和 feature preprocessing；gen_target 包含 raw target、mask 和 target preprocessing。

data_gen_raw_target 还会读取复权 VWAP、base/limit mask、Barra 风格因子、行业、市值和上市天数，构建设计矩阵并执行加权残差化。进一步细分时应测量 source history、design matrix 和 torch.linalg.lstsq/pinv。

### 6.2 Fit 热点

两次 fit 共处理 4980 个 batch：

| Event | 次数 | 总耗时 | 说明 |
|---|---:|---:|---|
| detail_dataloader_next | 5010 | 84.419 | next(iterator) |
| detail_dataset_getitem | 24840 | 4.622 | Dataset 单样本读取 |
| detail_batch_to_device | 4980 | 28.852 | CPU 到 GPU 搬运 |
| detail_forward | 4980 | 189.960 | batch forward |
| detail_model_forward | 4998 | 185.172 | 模型 forward |
| detail_model_attn | 4998 | 152.938 | Attention |
| detail_model_ts_slope | 4998 | 10.832 | 时间序列 slope |
| detail_loss | 4980 | 7.524 | loss 外层流程 |
| detail_loss_forward | 4980 | 2.904 | ICLoss.forward |
| detail_backward | 4980 | 184.178 | backward |
| detail_clip_grad | 4980 | 3.472 | 梯度裁剪 |
| detail_optimizer_step | 4980 | 3.828 | optimizer step |
| detail_zero_grad | 4980 | 1.502 | 清空梯度 |
| detail_loss_to_cpu | 4980 | 1.203 | loss 转 CPU 标量 |

当前外部 Model.py 的 DataLoader 基线：

~~~text
num_workers=0
prefetch_factor=disabled
persistent_workers=disabled
pin_memory=True on CUDA
non_blocking=True for tensor.to()
~~~

所以当前 detail_dataloader_next 是主进程中 iterator、dataset、default collate 和 pin-memory 的总耗时，不是 worker queue 等待。

## 7. 连续/滚动训练解释

当前 ComboBase.Train 每个训练目标日都会重新创建 ComboTrainDataset 和 research model：

- ComboDataLoader 和 DataRegistry 在进程内复用；
- fit 后调用 warm_next_training_cache，为下一个窗口预热部分 source cache；
- ComboTrainDataset 的 X/Y/W 快照不会直接复用；
- 特征生成、特征预处理和标签残差化仍会重新执行；
- 每个目标日的 model 仍然重新 fit 15 个 epoch。

因此连续训练目前主要只能减少部分 raw source read/cache miss，不能消除约 160 秒的完整 dataset initialization，也不能消除约 274 秒的 fit。明显优化需要增量式 dataset，或复用 model/optimizer 状态并减少后续 epoch。

同一个 dataset 内连续训练多个 epoch 时，dataset initialization 只发生一次；当前单次 fit 已包含 15 个 epoch。

## 8. 解释和限制

### 8.1 Wall time 与 CPU time

event_summary.csv 同时记录 wall time 和 CPU cumulative time。CPU time 可能大于 wall time，因为 Torch CPU thread pool、内存操作和其他线程并行工作。CPU cumulative time 用于判断 CPU 工作量，不应直接当作单核 wall time。

### 8.2 sync_cuda

完整 profiling 使用 --sync-cuda，便于 GPU 异步操作归因，但会增加同步和观测开销。因此阶段排序可参考，最终吞吐应使用不带 --sync-cuda 的实验，GPU operator 归因使用短 Torch profiler。

### 8.3 DataLoader prefetch

当前基线没有设置 num_workers、prefetch_factor 或 persistent_workers。后续 A/B 建议：

~~~text
num_workers: 0, 2, 4, 8
prefetch_factor: 2, 4
persistent_workers: true when num_workers > 0
~~~

当前主进程峰值 RSS 约 10.6 GiB，增加 worker 可能增加内存，建议从 2 或 4 个 worker 开始。

### 8.4 可重复性

应保持以下内容一致：

- KF compute class 和 GPU 类型；
- CPU、内存、Torch thread 数；
- XML 配置和外部 Model.py；
- cache 路径和数据版本；
- 日期范围、训练窗口、epoch、batch size；
- 是否使用 sync_cuda；
- profiling hooks 的代码版本。

硬件负载和 cache 状态会造成绝对秒数波动。复现时优先比较事件计数、阶段比例和热点排序。

## 9. 后续实验

### 9.1 DataLoader A/B

修改外部 Model.py 后，单独用 kf-submit 测每组参数：

~~~python
DataLoader(
    dataset,
    batch_size=self.batch_size,
    shuffle=True,
    generator=self._dataloader_generator(),
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)
~~~

记录 fit wall time、detail_dataloader_next、detail_dataset_getitem、detail_batch_to_device、GPU utilization、主进程/worker RSS，以及 fork、pickle 或 OOM 问题。吞吐实验去掉 --sync-cuda。

### 9.2 Dataset 初始化进一步细分

建议把 data_gen_raw_target 拆为：

- label/VWAP/adjustment source history；
- mask source history；
- Barra source history；
- valid mask；
- design matrix 和 industry dummy；
- weighted least squares；
- residual 写回；
- codec allocation 和 encode_into。

## 10. 结果位置

~~~text
完整 profiling:
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2/

短 Torch trace:
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-deep-20260921/

manifest:
/home/mengkang/autodl/0919.virtual.combo/mlpresiduallabelcp/tmp/profile-full-20260921-v2/manifest.json
~~~

manifest.json 保存 effective config、输出路径、Python 路径和 profiling 参数。当前 profiling 代码仍有 working tree 修改，复现时必须同时保留这些文件和 git diff，不能只依赖 Git commit。
