from __future__ import annotations

import numpy as np
import pandas as pd

from comb_eval.report_details import build_target_details, execution_details


def test_execution_periods_include_first_loss_and_prior_year_equity():
    daily = pd.DataFrame({
        "total_asset": [90., 99., 89.1], "pnl": [-10., 9., -9.9],
        "trade_cost": [1., 1., 1.], "reserve_cash": [9., 9., 9.],
        "tvr": [.1, .2, .3], "long_num": [10, 12, 11],
    }, index=[20201230, 20201231, 20210104])
    details, summary = execution_details(daily)
    np.testing.assert_allclose(details["return"], [-.1, .1, -.1])
    np.testing.assert_allclose(details.drawdown_pct, [-10., -1., -10.9])
    assert np.isclose(summary.loc["2021", "return_pct"], -10.)
    assert np.isclose(summary.loc["2021", "max_drawdown_pct"], -10.)
    assert np.isclose(summary.loc["ALL", "return_pct"], -10.9)


def test_target_layers_preserve_mask_and_do_not_split_ties():
    index = pd.MultiIndex.from_tuples([(20200102, 100000), (20200103, 100000)], names=["date", "time"])
    alpha = pd.DataFrame([np.arange(21), np.zeros(21)], index=index)
    target = np.arange(21, dtype=float)
    target[-1] = 1e9  # Excluded target must not influence ranks, layer means or universe.
    mask = np.ones(21, dtype=bool)
    mask[-1] = False
    layers, diagnostics = build_target_details(alpha, {key: (target, mask) for key in index})
    first = diagnostics.loc[index[0]]
    assert np.isclose(first.rank_ic, 1.)
    assert np.isclose(first.layer_ic, 1.)
    assert np.isclose(first.universe_target, 9.5)
    assert np.isclose(first.q10_target, 18.5)
    assert np.isclose(first.q10_minus_q1, 18.)
    assert layers.loc[index[0], "count"].sum() == 20
    assert not diagnostics.loc[index[1], "complete_deciles"]
    assert layers.loc[index[1], "target_mean"].isna().all()
    assert diagnostics.loc[index[1], "q10_minus_universe"] != diagnostics.loc[index[1], "q10_minus_universe"]


def test_rank_ic_ranks_only_the_common_valid_universe():
    index = pd.MultiIndex.from_tuples([(20200102, 100000)], names=["date", "time"])
    alpha = pd.DataFrame([[3., 1., 2., np.nan]], index=index)
    target = np.array([30., 10., 20., -100.])
    _, diagnostic = build_target_details(alpha, {index[0]: (target, np.ones(4, dtype=bool))})
    assert np.isclose(diagnostic.iloc[0].rank_ic, 1.)
    assert diagnostic.iloc[0].valid_count == 3
    assert not diagnostic.iloc[0].complete_deciles
