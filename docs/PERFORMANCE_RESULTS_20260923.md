# comb2 framework worker-test results (2026-09-23)

All execution below used `kf-submit` workers; the login host was used only
for editing, submission, and reading outputs. No new `runEval` mode was added.
These are framework tests, not a model-quality comparison.

## Correctness

- Full suite of the current worktree: 123 passed, 11 skipped, 14 warnings
  in 54.31 seconds (16 CPU, 32 GiB; run
  `13cb1db7-f703-5ea4-9346-cd8a87c708dd`). The installed combo2 1.0.5 and
  1.0.7 wheels were uninstalled first; their regular `vendor` package had
  shadowed the repository `vendor/perf_monitor.py` in all earlier runs.
- End-to-end `runCombo` on the research model, 2020-05-21 to 2020-06-30,
  500-day FP4 window, one epoch (16 CPU, 48 GiB, 1 GPU;
  `9de3c4d1-22c7-5348-af87-69aed02ab656`): 3m15s total; first training
  Dataset 42.4 s and fit 46 s, second training rolled in place in 1.8 s and
  fit 72 s. `MOSEKLM_LICENSE_FILE` must point to the repository `mosek.lic`.
- Full suite after the rank and backtest regression tests: 124 passed,
  11 skipped, 14 warnings in 854.93 seconds (4 CPU, 12 GiB; run
  `9dc4ffcd-338f-587f-980b-393c9ea988be`). The prior suite with only the
  rank test passed 122/11 in 108.15 seconds on the same requested resources
  (`da9dea8e-3ee9-5c3c-87ea-49fc44a9e9b1`); the large wall-time variance
  needs investigation before treating suite duration as a performance result.
- Tests referencing the removed `cacheDays`, `processed_cache`, retention,
  and warm-next-training APIs were migrated to the Dataset-cache contract.
- A fixed-instrument synthetic rolling Dataset matched a cold rebuild exactly
  for none/fp8/fp4 codecs and two sample times
  (run `96d5f725-d13b-575b-9dad-6a9b1797456c`).

## Memory and performance

| Workload | Result | Worker |
| --- | --- | --- |
| FP4 storage allocation, 2000 days, 1000 factors, 5642 codes | X 5.255 GiB; X/Y/W 5.286 GiB; process peak RSS 5.812 GiB. Storage-only, not a full training run. | 2 CPU, 16 GiB; `7e14cf3f-9b4b-54a8-bbf2-9560c92fb93e` |
| Real research loader, 500 days, 689 factors, 3249 selected instruments, ending 2020-05-19 | Initial Dataset build 63.28 s; storage 0.529 GiB; steady RSS 2.283 GiB; process peak 2.660 GiB. | 8 CPU, 32 GiB; `228c1364-bb86-5453-a2dc-8a75de1dc8ec` |
| Same real window, next day with one new selected instrument | Incremental roll declined; old Dataset released first; fallback full rebuild 44.52 s. RSS after release 1.10 GiB; after rebuild 1.624 GiB; process peak 2.642 GiB. | 8 CPU, 32 GiB; `87a97228-2f45-5710-85c5-1dc08c4d68fa` |
| Validity-mask-only scan, 500 training days (491 sampled days), 20 consecutive transitions | 9/20 transitions (45%) retained identical selected instruments and were eligible for in-place rolling. | 4 CPU, 8 GiB; `22013c98-7923-53e0-a4e5-0f4b5bdf18d2` |
| Eligible real transition, 2020-05-21 to 2020-05-22 | In-place roll 0.31 s versus 37.41 s cold rebuild (121.75x for this one transition). Sampled X/W are exact; Y differences are within declared FP16 tolerance. Roll/cold Y max absolute difference 0.000122; two cold rebuilds differ by up to 0.000244. | 8 CPU, 32 GiB; `7e02d7ff-988d-519f-9015-1ef7c0016aa4` |
| Complete Dataset storage parity on the eligible 500-day transition | Encoded FP4 X exactly equal; W exactly equal; all 1,596,241 Y entries within `rtol=0.005, atol=0.0002` (11,548 not bitwise equal). Roll 0.42 s versus cold rebuild 50.90 s for this run. | 8 CPU, 32 GiB; `224db71e-3c55-5fdb-b2fc-f8575521ad21` |
| Same 500-day Dataset followed by a representative two-batch GPU fit (batch size 5, one epoch setting) | Dataset steady CPU RSS 2.411 GiB, sampled CPU peak 2.732 GiB. After fit: CPU RSS 3.270 GiB, sampled CPU peak 3.488 GiB; GPU peak allocated 8.004 GiB, reserved 8.535 GiB. This is not a full epoch. | 8 CPU, 32 GiB, 1 GPU; `796cbeeb-957f-59a1-ab94-11034c8fe6a0` |
| Full one-epoch GPU fit on the real 500-day Dataset (batch size 5) | Dataset build 50.48 s, fit 85.61 s; sampled CPU peak RSS 3.995 GiB; GPU peak allocated 8.004 GiB, reserved 8.535 GiB. | 8 CPU, 32 GiB, 1 GPU; `55bf6302-ddde-568e-9284-71a29e7d1172` |
| Real prediction-window buffering over 10 consecutive dates | Reused windows and cold windows decoded exactly equal; reuse 5.9222 s total versus cold 14.7990 s (2.50x); process peak RSS 2.123 GiB. | 8 CPU, 16 GiB; `4cebb554-4056-5b8a-8b48-ac97c2a8572f` |

The initial strict bitwise check failed because the research target uses
floating-point least squares and Y is stored as FP16. The three checked
sample positions had exact X/W equality. Roll/cold Y was within
`rtol=0.005, atol=0.0002`; cold/cold Y also differed and had a larger
maximum absolute difference. This supports numerical tolerance for the
sampled targets. The later whole-Dataset storage comparison passed with
the same tolerance, exact encoded X, and exact W.
The adjacent-day scan above predates instrument remapping and does not match
the default training calendar; see the next section for the monthly results.

The 2000-day real-data attempt ending 2025-12-31 failed because the historical
`label` source ends at 2025-01-27; the user chose a 500-day real-data window
instead. The 2000-day storage-only measurement above remains valid as a shape
allocation measurement.

## Scope and remaining gates

The 500-day Dataset and full one-epoch fit CPU RSS are below the 32 GiB
threshold. The fit does not cover data-loader subprocesses (the research fit
uses the default single-process DataLoader), backtest workers, or aggregate
process RSS. The prior historical profiling report's
dataset timing uses a different window and cannot serve as a controlled A/B
baseline.

The first 24-case `rank`/`zscore` A/B found a CPU float16 long-axis
`rank` regression: direct float16 `torch.arange` generated 1,408 different
positions compared with integer arange cast to float16. The operator now
generates positions as integers before casting. Its focused regression test
passed (`24769741-7401-5859-bf2d-85d92de61d28`). All 24 CPU/GPU,
float16/float32, axis 0/1, rank/rank_pct/zscore comparisons passed after
the fix (`5dad519b-8feb-567b-9f7e-58d8d0b2c006`). On the representative
5642 x 689 tensor, CPU rank was 4.35-9.57x faster; GPU rank was 722-1237x
faster because the previous implementation round-tripped through CPU. CPU
zscore was 1.90-3.80x faster and GPU zscore 1.82-1.88x faster. These are
operator microbenchmarks, not whole-training speedups.

An optional verbose diagnostic of the slow 854.93-second suite run
(`baaef07b-5ef4-5e6f-b173-8cb2a7687942`) failed after its 900-second
limit with a Job Gateway HTTP 503 for its log. The slowdown did not recur:
the current 123-pass suite finished in 54.31 seconds.

## Default isTrainDay rolling and instrument remapping

The default `ComboBase.isTrainDay` trains once per month (last trading day of
the month's last trading week), so consecutive training windows move by
14-25 trading days rather than one day.

| Check | Result | Worker |
| --- | --- | --- |
| Selected-instrument stability, 500-day window, default training days 2019-01 to 2024-12 | 72 training days, 71 transitions, 0 with identical selected instruments. Added per transition 3-81 (mean 29.3), removed mean 2.9. Without remapping every monthly training rebuilds. | 4 CPU, 8 GiB; `bdcdc488-f9b6-5a60-ad85-05dbcf81f31e` |
| Features of added instruments on retained days, 9 transitions (3-81 added) | 0 nonzero raw or FP4-encoded values on all retained days, including the last tsDays-1 days; W True 0 and Y nonzero 0 there. Control: every added instrument has nonzero features on the new tail days. | 32 CPU, 64 GiB; `d64d5201-f1af-5d63-a120-bc33953cd2cc` |

`ComboTrainDataset` now reserves `column_slack` (default 10%) spare
instrument columns. With the default instrument selection, a roll whose
selection changes shifts and reorders columns in place: removed instruments
are dropped, added instruments get encoded-zero X, zero Y and False W on the
retained days, and only the new tail days are read. A selection larger than
the reserved capacity releases and rebuilds. This relies on the data contract
that features are zero outside the base universe and targets are invalid
there, which the check above confirmed for the research data.

Real 500-day FP4 rolls against cold rebuilds (8 CPU, 32 GiB):

| Transition | Added / removed | Instruments | Shift + remap | Tail read | Roll total | Cold rebuild | Parity | Peak RSS | Worker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20200731 to 20200828 (5% slack) | 21 / 0 | 3297 to 3318 | 0.55 s | 1.58 s | 2.14 s | 44.79 s | X, W exact; Y 12,379/1,629,138 inexact, all within tolerance | 3.30 GiB | `01e1e072-8abb-52ac-a8f9-338a9be0e23d` |
| 20200828 to 20200930 (5% slack) | 81 / 0 | 3318 to 3399 | 0.30 s | 1.97 s | 2.27 s | 47.39 s | X, W exact; Y within tolerance | 3.46 GiB | `ebefd907-4dcf-51f8-8b38-5366c6c5ed00` |
| 20200828 to 20200930 (10% slack) | 81 / 0 | 3318 to 3399 | 0.45 s | 2.01 s | 2.46 s | 46.98 s | X, W exact; Y 12,635/1,668,909 inexact, all within tolerance | 2.70 GiB during roll | `0d16f705-047b-5ac1-842d-50450cda1e60` |
| 20230825 to 20230928 (5% slack) | 51 / 3 | 4708 to 4756 | 0.33 s | 2.34 s | 2.67 s | 56.65 s | X, W exact; Y within tolerance | 3.80 GiB | `adfb5105-1a54-5e05-ac82-fad2bc5db9f3` |

Storage size is unchanged by a roll (for example 0.594 GiB before and after
with 10% slack). Y inexact counts are of the same order as two cold rebuilds
of the same window, which come from the research target's least squares.
Peak RSS in the 5% rows includes the later cold comparison builds.

Instrument counts on default training days grow from 3143 (20191129) to 4895
(20241231) (`94c7a2e5-eee4-5c93-863d-131f33cb9c20`; its log retained the
last 62 training days). Simulating capacity over those 61 transitions,
10% slack needs 4 capacity rebuilds and 5% needs 8; the rest roll in place.
A training run takes roughly 21 minutes (15 epochs at the measured 85.6 s per
epoch), so the saved 45-55 s per monthly training is about 4% of it.

## Backtest A/B on saved alpha

`tools/benchmark_backtest_ab.py` replays one saved alpha and one set of
execution prices through `DailyBacktest` built from each source tree. The
alpha came from the current code on the research model, 2020-05-21 to
2020-06-30 (27 trading days, 18 with positions after the first training;
`d138ab82-f43b-5dad-ad89-3c720b640863`). Baseline is `82c62b8` (1.0.8);
candidate is the current master. Both ran in one 8 CPU, 16 GiB worker
(`4493e430-a346-58f5-8ef6-95f452379a90`).

| Tree | Init | 27 steps | Finalize | Peak RSS |
| --- | --- | --- | --- | --- |
| 1.0.8 | 0.29 s | 21.57 s | 0.73 s | 0.827 GiB |
| current | 0.26 s | 20.25 s | 0.71 s | 0.830 GiB |

All nine output files are byte-identical: holdings, position, yield,
`daily_pnl.csv`, `executions.csv`, `pnl_summary.csv` and three plots. The
backtest changes (single close load, cached benchmark) do not change results;
their time saving is within run-to-run variance because the daily optimizer
dominates. No delisting settlement occurred in this window, so
`settlements.csv` was not exercised, and parallel backtest processes were not
part of this configuration.

## Not measured

A production-scale multi-training run (2000-day window, 15 epochs, several
monthly trainings) was not run. Edge-case tests and microbenchmarks for
`DataRegistry.get_day_many`, `ComboBuffer` wraps, model rotation, cleanup
after training failures, and output parity of every `runEval` route were not
run beyond the existing test suite.
