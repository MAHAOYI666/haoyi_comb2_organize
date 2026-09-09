# Runtime data flow

config.py resolves runtime paths and loader settings. ResearchLoader declares DataItem sources in Python. Relative source paths are anchored to the defining Python module, including when Optuna renders the XML elsewhere.

DataRegistry reads Memmap blocks into LoadedSource, applies the declared trading-day delay once, calls process_source, and validates [date, stock, field] results. dates/codes must be retained; fields may be derived by source processing. No raw bar axis survives processing. Source ops and declared dependencies operate on logical dates after reduction.

Processed caches are bounded and keyed by logical date and sampling time. Live refresh invalidates current-date processed results and feature caches and reopens raw readers. Raw prices are not converted to the model's low-precision dtype.

ComboDataLoader concatenates explicitly selected sources into one tensor, applies feature preprocessing, and uses the existing Codec for the bounded feature cache. ComboTrainDataset stores coordinates and training stock indices, loads same-time daily windows on demand, and returns (idx, ds, ti, x, y, w). It reserves retDays-1 endpoint rows for default forward component aggregation; researcher target indexing is explicit.

ComboBase retains prediction-before-training scheduling, isTrainDay, checkpoint IO and model smoothing. All prediction calls include di/ti, and all alpha keys are (date,time). runCombo passes explicitly selected execution prices to the existing backtest. DailyBacktest preserves state between intraday calls, checks T+1 at execution, accumulates turnover, and settles once at the last configured time.

runEval uses the same researcher raw target and validity hooks for IC. PnL comes from actual execution daily records. Aggregate-only evaluation utilities remain available for standalone matrices.

Verification: tests/test_cube_feature_groups.py covers real Memmap reduction, delay, real-time mutation and array-model training. tests/test_backtest_snap_ti.py contains the actual-cache, actual-Torch and actual-MOSEK two-day intraday pipeline including report generation. The codec tests exercise the unchanged tensor codecs.
