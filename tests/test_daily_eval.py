from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evals.comb_eval.daily_eval import (
    DailyEvaluationResult,
    WeightResult,
    _normalize_daily_frame,
    _apply_excess_returns,
    _read_daily_eval_artifacts,
    normalize_position_sides,
    resolve_daily_eval_dir,
    _write_daily_eval_artifacts,
    blend_positions,
    build_va_table,
    format_daily_evaluation,
    _strategy_config,
)


def test_blend_positions_keeps_the_requested_endpoints_with_nan_values():
    index = pd.Index([20240102], name="date")
    columns = pd.Index(["000001", "000002", "000003"], name="code")
    target = pd.DataFrame([[1.0, np.nan, 3.0]], index=index, columns=columns)
    myposition = pd.DataFrame([[np.nan, 2.0, 4.0]], index=index, columns=columns)

    target_only = blend_positions(target, myposition, 0.0)
    test_only = blend_positions(target, myposition, 1.0)

    np.testing.assert_allclose(
        target_only.to_numpy(), target.to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        test_only.to_numpy(), myposition.to_numpy(), equal_nan=True
    )


def test_va_table_uses_target_as_zero_weight_benchmark():
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    base = pd.Series(0.01, index=dates)
    blended = pd.Series(0.014, index=dates)
    test = pd.Series(0.03, index=dates)
    results = {
        0.0: WeightResult(0.0, base),
        0.2: WeightResult(0.2, blended),
        1.0: WeightResult(1.0, test),
    }

    table = build_va_table(results, weights=(0.0, 0.2, 1.0))

    assert table.loc["2024", "0.00"] == pytest.approx(250.0)
    assert table.loc["2024", "0.20"] == pytest.approx(100.0)
    assert table.loc["2024", "1.00"] == pytest.approx(750.0)


def test_normalize_daily_frame_selects_the_093000_sample():
    index = pd.MultiIndex.from_tuples(
        [(20240102, 100000), (20240102, 93000)], names=["date", "time"]
    )
    frame = pd.DataFrame([[1.0], [2.0]], index=index, columns=["1"])

    normalized = _normalize_daily_frame(frame, label="alpha")

    assert list(normalized.index) == [20240102]
    assert list(normalized.columns) == ["000001"]
    assert normalized.iloc[0, 0] == 2.0


def test_normalize_position_sides_uses_independent_long_short_l1_sums():
    index = pd.Index([20240102], name="date")
    columns = pd.Index(["000001", "000002", "000003", "000004", "000005"])
    frame = pd.DataFrame([[2.0, -3.0, 1.0, -1.0, np.nan]], index=index, columns=columns)

    normalized = normalize_position_sides(frame)

    np.testing.assert_allclose(
        normalized.iloc[0, :4].to_numpy(),
        [2.0 / 3.0, -3.0 / 4.0, 1.0 / 3.0, -1.0 / 4.0],
    )
    assert normalized.iloc[0, 0] + normalized.iloc[0, 2] == pytest.approx(1.0)
    assert -(normalized.iloc[0, 1] + normalized.iloc[0, 3]) == pytest.approx(1.0)
    assert np.isnan(normalized.iloc[0, 4])


def test_daily_eval_directory_separates_long_ratio_variants(tmp_path):
    myposition = tmp_path / "myposition.parquet"
    target = tmp_path / "target.parquet"

    ratio_33 = resolve_daily_eval_dir(myposition, target, long_ratio=0.33)
    ratio_50 = resolve_daily_eval_dir(myposition, target, long_ratio=0.50)
    simple_50 = resolve_daily_eval_dir(
        myposition, target, long_ratio=0.50, simple=True
    )

    assert ratio_33 != ratio_50
    assert ratio_33.name.endswith("_old_long_short_l1_v1_lr0.330000")
    assert ratio_50.name.endswith("_old_long_short_l1_v1_lr0.500000")
    assert simple_50 != ratio_50
    assert simple_50.name.endswith("_simple_long_short_l1_v1_lr0.500000")


def test_daily_eval_profile_switches_optimizer_defaults():
    old = _strategy_config(20240102, 20240105)
    simple = _strategy_config(20240102, 20240105, simple=True)

    assert old["optimizer"]["maxtvr"] == 0.4
    assert old["optimizer"]["max_weight"] == 0.0075
    assert old["optimizer"]["min_participation_ratio"] == 0.07
    assert old["optimizer"]["parti_penalty"] == 0.0
    assert old["optimizer"]["univ_list"]
    assert old["optimizer"]["risk_list"]
    assert simple["optimizer"]["maxtvr"] == 0.08
    assert simple["optimizer"]["max_weight"] == 0.008
    assert simple["optimizer"]["min_participation_ratio"] == 0.1
    assert simple["optimizer"]["parti_penalty"] == 0.05
    assert simple["optimizer"]["univ_list"] == ""
    assert simple["optimizer"]["risk_list"] == ""

    result = DailyEvaluationResult(
        myposition_path=None,
        target_path=None,
        cache_path=None,
        start_ds=20240102,
        end_ds=20240105,
        workers=8,
        va_table=pd.DataFrame(
            {"0.00": [-15.898056], "0.01": [0.013052], "1.00": [-15.844998]},
            index=["full"],
        ),
        ic_table=pd.DataFrame({"2025": ["nan%"]}, index=["IC_20d_Filter"]),
    )

    text = format_daily_evaluation(result, include_header=False)

    assert "-15.90" in text
    assert "0.01" in text
    assert "-15.898056" not in text
    assert "nan%" in text


def test_daily_eval_artifacts_support_read_mode(tmp_path):
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    results = {
        0.0: WeightResult(0.0, pd.Series([0.01, 0.02], index=dates)),
        1.0: WeightResult(1.0, pd.Series([0.03, 0.04], index=dates)),
    }
    myposition_path = tmp_path / "myposition.parquet"
    target_path = tmp_path / "target.parquet"
    cache_path = tmp_path / "Cache"

    _write_daily_eval_artifacts(
        tmp_path / "eval",
        results,
        myposition_path=myposition_path,
        target_path=target_path,
        cache_path=cache_path,
        start_ds=20240102,
        end_ds=20240103,
        workers=8,
        long_ratio=0.33,
    )
    loaded, manifest = _read_daily_eval_artifacts(
        tmp_path / "eval",
        myposition_path=myposition_path,
        target_path=target_path,
    )

    assert manifest["workers"] == 8
    assert manifest["long_ratio"] == pytest.approx(0.33)
    assert manifest["simple"] is False
    np.testing.assert_allclose(loaded[1.0].daily_returns.to_numpy(), [0.03, 0.04])


def test_excess_returns_match_li_ret_and_preserve_va_deltas():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    raw = {
        0.0: WeightResult(0.0, pd.Series([0.02, 0.03], index=dates)),
        0.2: WeightResult(0.2, pd.Series([0.021, 0.031], index=dates)),
        1.0: WeightResult(1.0, pd.Series([0.04, 0.05], index=dates)),
    }
    benchmark = pd.Series([0.0, 0.01], index=dates)

    excess = _apply_excess_returns(raw, benchmark)

    np.testing.assert_allclose(excess[0.0].daily_returns.to_numpy(), [0.02, 0.02])
    np.testing.assert_allclose(excess[0.2].daily_returns.to_numpy(), [0.021, 0.021])
    np.testing.assert_allclose(excess[1.0].daily_returns.to_numpy(), [0.04, 0.04])
