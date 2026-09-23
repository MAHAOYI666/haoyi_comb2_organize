# comb2 Framework Performance Test Plan

This plan intentionally separates correctness, compatibility, and performance.
Do not run the full plan on the login host; use a KF worker with explicitly
requested CPU and memory, and use a GPU worker only for end-to-end runs.

## 1. Static and focused correctness checks

1. Compile every changed Python file with the production Python interpreter.
2. Update tests that refer to the removed source-cache API, then run the focused
   suites for codec, memmap, DataRegistry, Dataset, prediction buffering,
   checkpointing, and backtest behavior.
3. Add tests for `DataRegistry.get_day_many` covering working-cache hits and
   misses, aliases, multiple names, and calls inside and outside a cache scope.
4. Add z-score parity tests for float16/float32, every supported axis, all-NaN,
   constant, finite, NaN, and infinity inputs.
5. Add rank parity tests on CPU/GPU for every supported dtype and axis, ties,
   all-invalid rows, NaN/Inf mixtures, empty axes, and `pct=True`.
6. Add `ComboBuffer` tests for partial fill, exact fill, one and multiple wraps,
   every codec, and clear/refill.
7. Add model-rotation tests for smoothing enabled and disabled, checkpoint save
   and reload, training failure, and preservation of prediction feature buffers.
8. Add memmap tests for single-block and cross-block slices, empty slices,
   non-unit steps, read-only behavior, and 2-D/3-D readers.
9. Add monitor tests for close-time flush and exceptions.
10. Add a backtest setup test proving that `get_close` is called once in
   `execution_price=provided` mode and that values match the old behavior.
11. Add valid-instrument construction tests for framework/default validity,
    research-defined validity, multiple sample times, chunk boundaries, and
    parity of selected instruments and source read counts.
12. Add a backtest finalize test proving PnL summary and drawing share one
    benchmark load while producing the same summary and plots.
14. Add incremental-dataset parity tests for one-day and multi-day rolls,
    none/fp8/fp4 codecs, multiple sample times, and `ret_days` 1 and greater.
15. Verify that changing valid instruments, changing window size, rolling
    backwards, and non-overlapping windows select the release-then-rebuild path.
16. Verify that fixed caller-supplied instruments remain fixed and that a
    Dataset with a custom instrument selection uses full rebuilds.
17. Inject failures before and after the in-place shift and prove the framework
    discards a possibly partial incremental snapshot instead of reusing it.
18. Compare incremental validity-history masks and selected instruments against
    a full rebuild across membership changes, multiple times, and chunk edges.
19. Check Dataset storage accounting against X/Y/W/validity tensor shapes and
    codec element sizes. Confirm there is no automatic memory-based mode switch.
20. Rebuild the 1.1.0 protected wheel from current sources, install it in a
    clean environment, and run the demo configuration through the installed
    `runCombo`.

## 2. Framework microbenchmarks

Run on a fixed CPU worker with cold and warm repetitions:

1. `zscore` on representative `[stock, feature]` tensors.
2. `rank` versus the NumPy/Python baseline on representative CPU and GPU
   tensors, including transfer and synchronization cost.
3. `get_day_many` versus repeated one-day `get_data` calls for 1, 32, 128,
   and the production number of sources.
4. `MemmapArray` slices contained in one block and spanning two blocks.
5. `ComboBuffer.get` before and after wrap for none/fp8/fp4 codecs.
6. PerfMonitor with 1k, 10k, and 100k events for flush sizes 1/64/256/1024.
7. Backtest initialization with `execution_price=provided`.
8. Valid-instrument discovery with daily reads versus chunked prefetch for
   representative training windows and one/multiple sample times.
9. One-day target generation versus the former stack/tensordot path.
10. Backtest finalization with summary only and summary plus plotting, including
    benchmark load counts and wall time.
11. Rolling Dataset full rebuild versus in-place update for 1/5/20 added days;
    record generated days, source reads, bytes copied, and update wall time.
12. Measure Dataset storage bytes, steady RSS between training dates, rebuild
    peak RSS, and fit peak RSS against the original processed-cache baseline.
    Confirm that source tensors are not retained alongside X/Y/W and that old
    and new X/Y/W never coexist. Require steady RSS below
    32 GB on the intended production configuration; report peak RSS separately
    and verify it stays within the worker's available memory.
13. Measure validity-mask bytes and scan time separately; expect each update to
    read only newly entered dates in the default Dataset implementation.

Record wall time, CPU time, peak RSS, bytes read, and allocation counts where
available. Require identical outputs before accepting a speed result.

## 3. Dataset construction A/B

Use the same XML, source data version, cache state, worker resources, dates,
and random seed for baseline and candidate.

Measure dataset initialization, feature generation, target generation, source
reads, op application, peak RSS, and cache hit/miss counts. Test cold cache,
warm cache, first training, and the next rolling training. Compare X/Y/W,
valid instruments, feature names, and load statistics exactly or within the
declared dtype tolerance.

## 4. Prediction and rolling-training A/B

Run enough consecutive dates to cross several ring-buffer wraps and at least
two training dates. Compare every emitted alpha, checkpoint date, current/old
model selection, and cache-read trace. Measure daily feature-load time,
prediction-window assembly, bytes copied, checkpoint time, and total combine
time.

## 5. Backtest A/B

Use identical saved alpha so model training is outside this measurement.
Measure setup, per-day optimizer, execution accounting, CSV output, finalize,
and alpha analysis separately. Compare orders, fills, holdings, cash, PnL,
turnover, settlements, and all output files. Profile the Python per-stock loop
and daily CSV appends before deciding whether to vectorize or buffer them.

## 6. End-to-end acceptance run

Run the historical profiling window first without CUDA synchronization for
throughput, then a short diagnostic run with synchronization. Finally run a
representative multi-training rolling window and compare against the baseline.

Acceptance gates:

- no correctness or compatibility regression;
- steady RSS below 32 GB on the intended production configuration, with
  rebuild and fit peaks within the worker's available memory;
- faster dataset initialization and prediction-window assembly;
- no slower end-to-end run outside normal repeat variance;
- monitor-disabled production speed and monitor-enabled diagnostic overhead
  both reported separately.

## 7. Deferred optimizations after measurement

Only after the above results, evaluate block-vectorized feature/target
generation, block-owned cache storage, vectorized backtest execution, buffered
CSV output, and reuse of the MOSEK model structure. Each requires its own
correctness and performance A/B.

## 8. Deferred runEval refactor checks

Do not run these on the login host. Use small fixed tables on a suitable worker
before trying any production-size evaluation.

1. Verify `runEval -h` and all existing CLI routes: direct daily `run`/`read`,
   `--sim`, `--pnl`, `--va`, config overall, `--corr`, and `--exposure`.
2. Compare output text, CSV/parquet schemas, error codes, and error messages
   against the pre-refactor baseline for each route.
3. Check that direct daily `run` and `read` use identical manifests, PnL/VA
   tables, and IC summaries before and after the split; include `--simple`,
   `--ti`, date filters, and multiple weights.
4. Check config overall separately: its decile table must retain target-mean
   semantics and must not be described as actual decile backtest PnL.
5. Build the protected wheel and invoke the installed `runEval` entry point to
   confirm the separated module is bundled.
6. Measure RSS of the daily path with representative `--worker` values. Since
   weight backtests run in separate processes, include aggregate worker RSS
   when assessing the 32 GB steady-memory limit; report run peaks separately.
