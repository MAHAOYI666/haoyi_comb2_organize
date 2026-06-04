from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optuna_framework.metrics_parser import WindowMetrics
from optuna_framework.scripts import run_study
from optuna_framework.search_space import ConfigDrivenAdapter
from optuna_framework.study_config import load_study_config
from optuna_framework.study_utils import create_study, maybe_enqueue_baseline, optimize_study, write_study_reports


FIXTURES = Path(__file__).parent / "fixtures"


def test_mocked_smoke_trial_writes_reports(tmp_path, monkeypatch) -> None:
    pytest.importorskip("optuna")
    config = load_study_config(study_root_override=tmp_path / "study")
    threshold_dir = config.study_root / "baseline"
    threshold_dir.mkdir(parents=True)
    (threshold_dir / "baseline_thresholds.json").write_text(
        (FIXTURES / "baseline_thresholds.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_run_inference(_run_paths):
        return WindowMetrics(
            sharpe_idx=10.0,
            dd_li=0.01,
            li_ret=0.1,
            ret=0.1,
            pnl=1.0,
            days=600,
            row_label="20210104-20231229",
            all_rows=["20210104-20231229"],
            run_start_ds=config.tuning_run_window[0],
            run_end_ds=config.tuning_run_window[1],
            score_start_ds=config.scoring_window[0],
            score_end_ds=config.scoring_window[1],
        )

    monkeypatch.setattr(run_study, "run_inference", fake_run_inference)
    study = create_study(config.optuna_name + "_mocked", run_study.storage_url(config.study_root, "mocked.db"), smoke=True)
    maybe_enqueue_baseline(study, ConfigDrivenAdapter(config).baseline_params())

    optimize_study(study, run_study.make_objective(config), 1)
    write_study_reports(study, config.study_root)

    trials = pd.read_csv(config.study_root / "reports" / "trials.csv")
    scoring = pd.read_csv(config.study_root / "reports" / "scoring_metrics.csv")
    assert len(trials) == 1
    assert float(trials.loc[0, "value"]) == 10.0
    assert len(scoring) == 1
    assert float(scoring.loc[0, "sharpe_idx"]) == 10.0
