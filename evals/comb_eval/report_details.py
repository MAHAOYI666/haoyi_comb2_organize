"""Detailed tables and plots for config evaluation using its native target semantics."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def execution_details(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve actual settlement NAV and use prior equity for each subperiod."""
    daily = daily.sort_index().copy()
    if daily.empty or not daily.index.is_unique:
        raise ValueError("Execution dates must be nonempty and unique")
    initial = float(daily.total_asset.iloc[0] - daily.pnl.iloc[0])
    previous = daily.total_asset.shift(1)
    previous.iloc[0] = initial
    if (previous <= 0).any() or not np.isfinite(previous).all():
        raise ValueError("Execution returns require positive finite prior equity")
    daily["previous_asset"] = previous
    daily["return"] = daily.total_asset / previous - 1
    daily["nav"] = daily.total_asset / initial
    daily["drawdown_pct"] = (daily.total_asset / daily.total_asset.cummax().clip(lower=initial) - 1) * 100
    daily["cash_pct"] = daily.reserve_cash / daily.total_asset * 100
    years = pd.to_datetime(daily.index.astype(str)).year
    rows = []
    for period, group in [("ALL", daily), *[(str(y), g) for y, g in daily.groupby(years)]]:
        opening = group.previous_asset.iloc[0]
        ratio = group.total_asset.iloc[-1] / opening
        ret = group["return"]
        peak = group.total_asset.cummax().clip(lower=opening)
        rows.append({
            "period": period, "start": group.index[0], "end": group.index[-1],
            "days": len(group), "pnl": group.pnl.sum(), "return_pct": (ratio - 1) * 100,
            "annualized_return_pct": (ratio ** (250 / len(group)) - 1) * 100 if ratio > 0 else np.nan,
            "daily_ir": ret.mean() / ret.std() if ret.std() > 0 else np.nan,
            "sharpe": ret.mean() / ret.std() * np.sqrt(250) if ret.std() > 0 else np.nan,
            "max_drawdown_pct": min(0.0, (group.total_asset / peak - 1).min()) * 100,
            "win_rate_pct": ret.gt(0).mean() * 100, "trade_cost": group.trade_cost.sum(),
            "turnover_mean_pct": group.tvr.mean() * 100, "holdings_mean": group.long_num.mean(),
            "cash_mean_pct": group.cash_pct.mean(),
        })
    return daily, pd.DataFrame(rows).set_index("period")


def _corr(left, right) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def build_target_details(alpha: pd.DataFrame, sample_inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use precisely the raw targets and valid masks used by calculate_alpha_ic."""
    layers, diagnostics = [], []
    for (ds, ti), row in alpha.iterrows():
        target, mask = sample_inputs[(int(ds), int(ti))]
        values = row.to_numpy(dtype=float)
        valid = mask & np.isfinite(values) & np.isfinite(target)
        x, y = values[valid], np.asarray(target, dtype=float)[valid]
        buckets = np.asarray(pd.qcut(x, 10, labels=False, duplicates="drop")) if len(x) else np.array([])
        complete = np.array_equal(np.unique(buckets[np.isfinite(buckets)]), np.arange(10))
        means = []
        for decile in range(10):
            selected = y[buckets == decile] if complete else np.array([])
            mean = float(selected.mean()) if len(selected) else np.nan
            means.append(mean)
            layers.append((ds, ti, decile + 1, mean, len(selected)))
        universe = float(y.mean()) if len(y) else np.nan
        diagnostics.append((ds, ti, _corr(pd.Series(x).rank(), pd.Series(y).rank()),
                            _corr(np.arange(1, 11), means) if complete else np.nan,
                            universe, means[-1], means[-1] - universe,
                            means[-1] - means[0], complete, len(y)))
    return (
        pd.DataFrame(layers, columns=["date", "time", "decile", "target_mean", "count"]).set_index(["date", "time", "decile"]),
        pd.DataFrame(diagnostics, columns=["date", "time", "rank_ic", "layer_ic", "universe_target", "q10_target",
                                          "q10_minus_universe", "q10_minus_q1", "complete_deciles", "valid_count"]).set_index(["date", "time"]),
    )


def _dates(index):
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values("date")
    return index if isinstance(index, pd.DatetimeIndex) else pd.to_datetime(index.astype(str))


def _time(value):
    value = f"{int(value):06d}"
    return f"{value[:2]}:{value[2:4]}"


def render_config_report(artifacts, *, alpha, ic, daily, pnl_periods, decile_daily,
                         target_diagnostics, exposure_daily, exposure_summary, cap_daily, messages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    report_dir = artifacts.report_dir
    plots = report_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    model = artifacts.output_root.name
    palette = ["#2563eb", "#e66b36", "#0d9488", "#9333ea", "#ca8a04"]
    panels = []

    def add(name, title, draw, dates=False, height=4):
        panels.append((name, title, draw, dates, height))

    def lines(ax, frame, columns, *, level="time", rolling=1, scale=1, cumulative=False):
        for ti, group in frame.groupby(level=level):
            for column in columns:
                series = group[column].astype(float)
                if cumulative:
                    series = series.cumsum()
                if rolling > 1:
                    series = series.rolling(rolling, min_periods=1).mean()
                ax.plot(_dates(group.index), series * scale, label=f"{_time(ti)} {column}", linewidth=1.35)
        ax.legend(loc="best", ncols=3, fontsize=8)
        ax.axhline(0, color="#64748b", linewidth=.7)

    def overview(ax):
        ax.axis("off")
        row = pnl_periods.loc["ALL"]
        means = ic.groupby(level="time").ic.mean()
        text = [
            f"{model} | {_dates(alpha.index).min():%Y-%m-%d} to {_dates(alpha.index).max():%Y-%m-%d}",
            f"{len(alpha):,} signal samples | {alpha.shape[1]:,} instruments | {len(daily):,} execution days",
            f"Total return {row.return_pct:.2f}% | Annualized {row.annualized_return_pct:.2f}% | Sharpe {row.sharpe:.3f}",
            f"Max drawdown {row.max_drawdown_pct:.2f}% | Fees {row.trade_cost / 1e6:.3f} million | Mean turnover {row.turnover_mean_pct:.2f}%",
            "Mean raw-target IC: " + ", ".join(f"{_time(t)} = {v:.5f}" for t, v in means.items()),
            "Targets: ResearchLoader.gen_raw_target + public evaluation mask; no training-target substitution.",
            "Decile curves are cumulative raw-target diagnostics, before costs and execution constraints.",
            "Barra / CAP use the existing same-date alignment. IC IR is mean / sample SD, without annualization.",
        ]
        for i, line in enumerate(text):
            ax.text(.01, .94 - i * .115, line, transform=ax.transAxes, va="top", fontsize=11 if i < 5 else 9)
    add("00_overview", "Evaluation summary", overview, height=3.3)

    def ic_plot(ax):
        lines(ax, ic, [c for c in ("ic", "rank_ic", "layer_ic") if c in ic], rolling=20)
        ax.set_ylabel("20-sample rolling mean")
    add("01_ic", "Raw-target IC, Rank IC and layer monotonicity", ic_plot, True)

    yearly_rows = []
    for ti, group in ic.groupby(level="time"):
        for period, frame in [("ALL", group), *[(str(y), g) for y, g in group.groupby(_dates(group.index).year)]]:
            for col in [c for c in ("ic", "rank_ic", "layer_ic") if c in frame]:
                series = frame[col].dropna()
                std = series.std()
                yearly_rows.append((ti, period, col, len(series), series.mean(), std,
                                    series.mean() / std if std > 0 else np.nan, series.gt(0).mean()))
    yearly = pd.DataFrame(yearly_rows, columns=["time", "period", "metric", "count", "mean", "std", "ir", "positive_fraction"])
    yearly.to_csv(report_dir / "ic_by_period.csv", index=False)

    def ic_years(ax):
        table = yearly[yearly.period.ne("ALL")].pivot(index="period", columns=["time", "metric"], values="mean")
        table.columns = [f"{_time(t)} {c}" for t, c in table.columns]
        table.plot.bar(ax=ax, rot=0, color=palette)
        ax.set_ylabel("Mean correlation"); ax.set_xlabel("")
        ax.legend(fontsize=8, ncols=3)
    add("02_ic_by_year", f"IC by calendar year (coverage through {_dates(alpha.index).max():%Y-%m-%d})", ic_years)

    def ic_hist(ax):
        for ti, group in ic.groupby(level="time"):
            ax.hist(group.ic.dropna(), bins=45, alpha=.55, label=f"{_time(ti)} IC")
            ax.axvline(group.ic.mean(), color="#e66b36", linestyle="--")
        ax.set_xlabel("Raw-target IC"); ax.set_ylabel("Samples"); ax.legend()
    add("03_ic_distribution", "Distribution of raw-target IC", ic_hist)

    xdaily = _dates(daily.index)
    def nav(ax):
        ax.plot(xdaily, daily.nav, color=palette[0], label="Actual execution NAV (net of fees)")
        ax.axhline(1, color="#64748b", linewidth=.7)
        ax.set_ylabel("NAV / initial asset"); ax.legend()
    add("04_execution_nav", "Actual execution NAV", nav, True)

    def drawdown(ax):
        ax.fill_between(xdaily, daily.drawdown_pct, 0, color="#dc654e", alpha=.65)
        ax.set_ylabel("Drawdown (%)")
    add("05_execution_drawdown", "Actual execution drawdown", drawdown, True)

    def pnl_years(ax):
        table = pnl_periods.drop(index="ALL")
        ax.bar(table.index, table.return_pct, color=palette[0])
        for i, value in enumerate(table.return_pct):
            ax.annotate(f"{value:.2f}%", (i, value), xytext=(0, 5 if value >= 0 else -13), textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=.2); ax.set_ylabel("Period return (%)")
    add("06_execution_by_year", f"Actual return by calendar year (coverage through {xdaily.max():%Y-%m-%d})", pnl_years)

    def costs(ax):
        ax.plot(xdaily, daily.tvr.rolling(20, min_periods=1).mean() * 100, label="Turnover 20d", color=palette[0])
        ax.set_ylabel("Turnover (%)")
        right = ax.twinx()
        right.plot(xdaily, daily.trade_cost.cumsum() / 1e6, label="Cumulative fees", color=palette[1])
        right.set_ylabel("Cumulative fees (million)")
        ax.legend(loc="upper left"); right.legend(loc="upper right")
    add("07_execution_costs", "Actual turnover and transaction costs", costs, True)

    def holdings(ax):
        ax.plot(xdaily, daily.long_num, label="Long positions", color=palette[0])
        ax.set_ylabel("Holdings count")
        right = ax.twinx(); right.plot(xdaily, daily.cash_pct, color=palette[1], label="Cash / NAV", alpha=.6)
        right.set_ylabel("Cash (%)"); ax.legend(loc="upper left"); right.legend(loc="upper right")
    add("08_execution_holdings", "Actual holdings and cash", holdings, True)

    finite = np.isfinite(alpha.to_numpy(dtype=float))
    coverage = pd.DataFrame({"finite_count": finite.sum(axis=1), "nonzero_count": (finite & alpha.ne(0).to_numpy()).sum(axis=1)}, index=alpha.index)
    coverage["evaluation_valid_count"] = ic["count"]
    coverage.to_csv(report_dir / "signal_coverage.csv")
    def coverage_plot(ax):
        lines(ax, coverage, list(coverage.columns)); ax.set_ylabel("Instrument count")
    add("09_signal_coverage", "Signal coverage and evaluation universe", coverage_plot, True)

    quantiles = alpha.replace([np.inf, -np.inf], np.nan).quantile([.01, .1, .25, .5, .75, .9, .99], axis=1).T
    quantiles.columns = [f"q{int(q * 100):02d}" for q in quantiles.columns]
    quantiles.to_csv(report_dir / "signal_quantiles.csv")
    def quantiles_plot(ax):
        lines(ax, quantiles, list(quantiles.columns)); ax.set_ylabel("Signal value")
    add("10_signal_quantiles", "Cross-sectional signal distribution over time", quantiles_plot, True)

    if decile_daily is not None:
        def layer_means(ax):
            table = decile_daily.groupby(["time", "decile"]).target_mean.mean().unstack(0)
            table.columns = [_time(t) for t in table.columns]
            table.plot.bar(ax=ax, rot=0, color=palette)
            ax.set_ylabel("Mean raw target"); ax.set_xlabel("Signal decile: Q1 low to Q10 high")
        add("11_decile_means", "Ten signal deciles: mean raw target", layer_means)

        def layer_curves(ax):
            for ti, group in decile_daily.groupby(level="time"):
                table = group.droplevel("time").target_mean.unstack("decile")
                for decile in table.columns:
                    ax.plot(_dates(table.index), table[decile].cumsum(), color=plt.cm.coolwarm((decile - 1) / 9),
                            label=f"{_time(ti)} Q{decile}", linewidth=1.8 if decile in (1, 10) else .9)
            ax.set_ylabel("Cumulative raw target"); ax.legend(ncols=5, fontsize=8)
        add("12_decile_curves", "Ten-decile cumulative target (before costs; diagnostic)", layer_curves, True)

        def top_excess(ax):
            lines(ax, target_diagnostics, ["q10_minus_universe", "q10_minus_q1"], cumulative=True)
            ax.set_ylabel("Cumulative raw-target difference")
        add("13_top_decile_excess", "Top-decile target excess over universe and bottom decile", top_excess, True)

    if exposure_summary is not None:
        def exposure_ranges(ax):
            count = len(exposure_summary)
            for i, ((ti, style), row) in enumerate(exposure_summary.iterrows()):
                ax.plot([row["min"], row["max"]], [i, i], color="#94a3b8", linewidth=2.5)
                ax.scatter(row["mean"], i, color=palette[0], s=25, zorder=3)
            ax.set_yticks(range(count), [f"{_time(t)} {s}" for t, s in exposure_summary.index], fontsize=8)
            ax.axvline(0, color="#475569", linewidth=.8)
            ax.set_xlabel("Cross-sectional correlation: min / mean / max")
            ax.margins(x=.1)
        add("14_barra_ranges", "All Barra styles: exposure ranges", exposure_ranges, height=max(4, .24 * len(exposure_summary) + 1.4))

        def exposure_time(ax):
            blocks, labels = [], []
            for ti, group in exposure_daily.groupby(level="time"):
                group = group.droplevel("time").copy()
                group.index = _dates(group.index)
                monthly = group.resample("MS").mean()
                blocks.append(monthly.T)
                labels.extend([f"{_time(ti)} {c}" for c in monthly.columns])
            table = pd.concat(blocks)
            limit = max(.01, float(np.nanmax(np.abs(table.to_numpy()))))
            heat = ax.imshow(table.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ticks = np.linspace(0, len(table.columns) - 1, min(9, len(table.columns)), dtype=int)
            ax.set_xticks(ticks, [f"{table.columns[i]:%Y-%m}" for i in ticks], fontsize=8)
            ax.figure.colorbar(heat, ax=ax, fraction=.02, pad=.02, label="Monthly mean correlation")
        add("15_barra_monthly", "All Barra styles: monthly exposure", exposure_time, height=max(4, .24 * len(exposure_summary) + 1.4))

    if cap_daily is not None:
        def cap_plot(ax):
            lines(ax, cap_daily, ["cap_corr"], rolling=20); ax.set_ylabel("CAP correlation, 20-sample mean")
        add("16_cap_correlation", "CAP correlation: neutral weights versus ranked market cap", cap_plot, True)

    def decorate(ax, title, is_date):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        if ax.has_data():
            if not ax.images:
                ax.grid(axis="y", color="#e2e8f0", linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if is_date:
                locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#334155",
             "text.color": "#0f172a", "axes.prop_cycle": plt.cycler(color=palette), "savefig.facecolor": "white"}
    with plt.rc_context(style):
        for name, title, draw, is_date, height in panels:
            fig, ax = plt.subplots(figsize=(13, height), layout="constrained")
            draw(ax); decorate(ax, f"{model} | {title}", is_date)
            fig.savefig(plots / f"{name}.png", dpi=180)
            plt.close(fig)
        heights = [p[4] for p in panels]
        fig, axes = plt.subplots(len(panels), 1, figsize=(15, sum(heights)),
                                 gridspec_kw={"height_ratios": heights}, layout="constrained")
        for ax, (_, title, draw, is_date, _) in zip(axes, panels):
            draw(ax); decorate(ax, f"{model} | {title}", is_date)
        artifacts.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts.plot_path, dpi=125)
        plt.close(fig)

    image_names = [f"plots/{p[0]}.png" for p in panels]
    sections = "\n".join(f'<section><h2>{html.escape(p[1])}</h2><img src="{name}" loading="lazy"></section>' for p, name in zip(panels, image_names))
    tables = " ".join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(report_dir.glob("*.csv")))
    (report_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(model)} evaluation</title><style>body{{font:16px system-ui;background:#f1f5f9;color:#0f172a;max-width:1320px;margin:30px auto;padding:20px}}'
        'section{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}h2{font-size:20px}a{display:inline-block;margin:8px}</style>'
        f'<h1>{html.escape(model)} | Complete runEval report</h1>'
        + "".join(f"<p>{html.escape(message)}</p>" for message in messages)
        + f'<h2>Data tables</h2>{tables}{sections}', encoding="utf-8")
    manifest_path = report_dir / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"plot_files": image_names, "plot_count": len(image_names),
                     "html_report": str(report_dir / "index.html"),
                     "tables": [p.name for p in sorted(report_dir.glob("*.csv"))],
                     "sample_count": len(alpha), "instrument_count": alpha.shape[1],
                     "execution_days": len(daily), "deciles_enabled": decile_daily is not None,
                     "exposure_enabled": exposure_summary is not None})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
