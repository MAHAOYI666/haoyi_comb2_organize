# overfit_diagnostics_non_train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# CONFIG：你只需要改这里
# =============================================================================

CONFIG = {
    # 总输出目录：脚本会在下面分 01/02/03... 独立输出
    "OUT_ROOT": str(REPO_ROOT / "optuna_framework" / "diagnostic_results" / "overfit_diagnostics"),

    # baseline 模型输出目录
    # 目录下应包含：
    #   backtest/daily_pnl.csv
    #   alpha.parquet 可选
    #   daily_ic 可选
    "BASELINE_DIR": "",

    # 候选模型输出目录，可以填多个
    # key 是你给它起的名字，value 是模型输出目录
    "CANDIDATE_DIRS": {
        # "best_trial": "/root/autodl-tmp/haoyi_comb2_organize/eg-torch/output-torch-xxx",
    },

    # baseline 不同平滑参数 / banding 参数下的回测输出目录
    # 注意：这些目录必须已经跑好回测。本脚本不会重跑。
    # 用于做“换手率 vs 净 Sharpe”的 baseline 前沿曲线
    "BASELINE_FRONTIER_DIRS": {
        # "baseline_smooth_0.00": "/root/.../output-baseline-smooth-0.00",
        # "baseline_smooth_0.05": "/root/.../output-baseline-smooth-0.05",
        # "baseline_smooth_0.10": "/root/.../output-baseline-smooth-0.10",
    },

    # 所有 Optuna trial 的输出目录。
    # 用于 Reality Check / DSR。
    # 如果你现在只有 best trial，可以先不填。
    "TRIAL_DIRS": {
        # "trial_000": "/root/.../output-trial-000",
        # "trial_001": "/root/.../output-trial-001",
    },

    # Optuna trials dataframe，可选。
    # 用于参数平台分析 plateau check。
    # 支持 columns 里有 value / sharpe_idx / net_sharpe_idx 作为目标列；
    # 参数列建议是 params_xxx 或 param_xxx。
    "OPTUNA_TRIALS_CSV": "",

    # 如果你 Phase B 对 best config 跑了多个 seed，把这些 seed 输出目录填进来。
    # 用于检查 best-baseline 是否落在 2 * sigma_seed 训练噪声内。
    "PHASEB_SEED_DIRS": {
        # "seed_0": "/root/.../output-best-seed-0",
        # "seed_1": "/root/.../output-best-seed-1",
        # "seed_2": "/root/.../output-best-seed-2",
    },

    # fee sweep 倍数。
    # 1.0 表示当前 daily_pnl.csv 里的真实费率。
    "FEE_MULTIPLIERS": [0.0, 0.5, 1.0, 2.0],

    # block bootstrap 设置
    "BOOTSTRAP_BLOCK_LEN": 20,
    "BOOTSTRAP_N": 5000,
    "BOOTSTRAP_RANDOM_SEED": 20260610,

    # Sharpe 年化天数
    "ANNUAL_DAYS": 252,

    # turnover-driven 标签阈值
    # 如果 Δ净 Sharpe > 0 且 abs(Δ毛 Sharpe) 小于该阈值，就倾向标记为 turnover_driven
    "TURNOVER_GROSS_DELTA_ABS_THRESHOLD": 0.05,

    # plateau check 近邻数量
    "PLATEAU_K_NEIGHBORS": 10,

    # plateau check 目标列。留空则自动猜。
    "PLATEAU_METRIC_COL": "",

    # daily_pnl 里优先使用哪个列作为收益序列。
    # 建议保持 AUTO。
    # AUTO 优先级：pnl -> li_ret -> net_li_ret -> ret -> return
    "RETURN_COL": "AUTO",

    # daily_pnl 里成本列。
    # AUTO 优先级：trade_cost -> cost -> fee
    "COST_COL": "AUTO",

    # daily_pnl 里换手列。
    # AUTO 优先级：tvr -> turnover -> turn_over
    "TVR_COL": "AUTO",

    # daily_pnl 里持仓数量列。
    # AUTO 优先级：long_num -> n_long -> hold_num
    "LONG_NUM_COL": "AUTO",
}


# =============================================================================
# 基础工具
# =============================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def path_or_none(x: str | Path | None) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return Path(s)


def model_name(path: str | Path) -> str:
    return Path(str(path).rstrip("/")).name


def ensure_output_dirs(out_root: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_root,
        "p0_attr": out_root / "01_gross_net_attribution",
        "fee": out_root / "02_fee_sweep",
        "frontier": out_root / "03_turnover_frontier",
        "bootstrap": out_root / "04_paired_block_bootstrap",
        "reality": out_root / "05_reality_check_and_dsr",
        "plateau": out_root / "06_plateau_check",
        "logs": out_root / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def choose_col(df: pd.DataFrame, config_value: str, candidates: List[str], label: str) -> Optional[str]:
    if config_value and config_value != "AUTO":
        if config_value not in df.columns:
            raise KeyError(f"CONFIG 指定的 {label}={config_value} 不在 daily_pnl columns 中: {list(df.columns)}")
        return config_value

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def find_daily_pnl(model_dir: Path) -> Path:
    path = model_dir / "backtest" / "daily_pnl.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find daily_pnl.csv: {path}")
    return path


def load_daily_pnl(model_dir: str | Path) -> pd.DataFrame:
    model_dir = Path(model_dir)
    pnl_path = find_daily_pnl(model_dir)

    df = pd.read_csv(pnl_path)

    if "date" not in df.columns:
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col.lower() in {"index", "datetime", "trade_date"}:
            df = df.rename(columns={first_col: "date"})
        else:
            df.insert(0, "date", np.arange(len(df)))

    ret_col = choose_col(
        df,
        CONFIG["RETURN_COL"],
        ["pnl", "li_ret", "net_li_ret", "ret", "return"],
        "RETURN_COL",
    )
    if ret_col is None:
        raise KeyError(
            f"无法自动识别收益列。daily_pnl columns={list(df.columns)}。"
            f"请在 CONFIG['RETURN_COL'] 里手动指定。"
        )

    cost_col = choose_col(
        df,
        CONFIG["COST_COL"],
        ["trade_cost", "cost", "fee"],
        "COST_COL",
    )

    tvr_col = choose_col(
        df,
        CONFIG["TVR_COL"],
        ["tvr", "turnover", "turn_over"],
        "TVR_COL",
    )

    long_col = choose_col(
        df,
        CONFIG["LONG_NUM_COL"],
        ["long_num", "n_long", "hold_num"],
        "LONG_NUM_COL",
    )

    out = pd.DataFrame()
    out["date"] = df["date"].astype(str)
    out["net"] = pd.to_numeric(df[ret_col], errors="coerce")

    if cost_col is not None:
        cost = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)

        # 统一成本为正数。如果原文件中成本是负数，这里转成正的扣费金额。
        nonzero = cost[cost != 0]
        if len(nonzero) > 0 and nonzero.median() < 0:
            cost = -cost

        out["cost"] = cost
    else:
        out["cost"] = 0.0

    out["gross"] = out["net"] + out["cost"]

    if tvr_col is not None:
        out["tvr"] = pd.to_numeric(df[tvr_col], errors="coerce")
    else:
        out["tvr"] = np.nan

    if long_col is not None:
        out["long_num"] = pd.to_numeric(df[long_col], errors="coerce")
    else:
        out["long_num"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["net", "gross"])
    out = out.sort_values("date")
    out = out.set_index("date", drop=False)

    return out


def align_two(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = a.index.intersection(b.index).sort_values()
    return a.loc[idx].copy(), b.loc[idx].copy()


def sharpe(x: pd.Series | np.ndarray, annualize: bool = True) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    std = arr.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    s = arr.mean() / std
    if annualize:
        s *= math.sqrt(CONFIG["ANNUAL_DAYS"])
    return float(s)


def perf_summary(df: pd.DataFrame, prefix: str = "") -> dict:
    net = df["net"]
    gross = df["gross"]
    cost = df["cost"]

    return {
        f"{prefix}n_days": int(len(df)),
        f"{prefix}net_sharpe": sharpe(net),
        f"{prefix}gross_sharpe": sharpe(gross),
        f"{prefix}net_mean": float(net.mean()),
        f"{prefix}gross_mean": float(gross.mean()),
        f"{prefix}net_sum": float(net.sum()),
        f"{prefix}gross_sum": float(gross.sum()),
        f"{prefix}trade_cost_sum": float(cost.sum()),
        f"{prefix}trade_cost_mean": float(cost.mean()),
        f"{prefix}avg_tvr": float(df["tvr"].mean()) if "tvr" in df else np.nan,
        f"{prefix}avg_long_num": float(df["long_num"].mean()) if "long_num" in df else np.nan,
    }


# =============================================================================
# 01 毛/净分解
# =============================================================================

def run_gross_net_attribution(
    baseline_dir: Path,
    candidate_dirs: Dict[str, Path],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    baseline_raw = load_daily_pnl(baseline_dir)

    for cand_key, cand_dir in candidate_dirs.items():
        try:
            cand_raw = load_daily_pnl(cand_dir)

            cand, base = align_two(cand_raw, baseline_raw)

            cand_perf = perf_summary(cand, "candidate_")
            base_perf = perf_summary(base, "baseline_")

            delta_net_sharpe = cand_perf["candidate_net_sharpe"] - base_perf["baseline_net_sharpe"]
            delta_gross_sharpe = cand_perf["candidate_gross_sharpe"] - base_perf["baseline_gross_sharpe"]

            # Sharpe 是非线性的，所以这里是报告意义上的分解：
            # Δ净Sharpe = Δ毛Sharpe + 成本贡献
            cost_contribution = delta_net_sharpe - delta_gross_sharpe

            turnover_driven = (
                delta_net_sharpe > 0
                and abs(delta_gross_sharpe) <= CONFIG["TURNOVER_GROSS_DELTA_ABS_THRESHOLD"]
            )

            row = {
                "candidate": cand_key,
                "candidate_dir": str(cand_dir),
                "baseline_dir": str(baseline_dir),
                "common_days": int(len(cand)),
                **cand_perf,
                **base_perf,
                "delta_net_sharpe": float(delta_net_sharpe),
                "delta_gross_sharpe": float(delta_gross_sharpe),
                "cost_contribution_to_delta_net_sharpe": float(cost_contribution),
                "turnover_driven_flag": bool(turnover_driven),
                "status": "ok",
            }

            rows.append(row)

            detail_path = out_dir / f"{safe_name(cand_key)}__gross_net_detail.csv"
            pd.DataFrame({
                "date": cand.index,
                "candidate_net": cand["net"].values,
                "candidate_gross": cand["gross"].values,
                "candidate_cost": cand["cost"].values,
                "candidate_tvr": cand["tvr"].values,
                "baseline_net": base["net"].values,
                "baseline_gross": base["gross"].values,
                "baseline_cost": base["cost"].values,
                "baseline_tvr": base["tvr"].values,
            }).to_csv(detail_path, index=False)

        except Exception as e:
            rows.append({
                "candidate": cand_key,
                "candidate_dir": str(cand_dir),
                "baseline_dir": str(baseline_dir),
                "status": f"error: {repr(e)}",
            })

            with open(out_dir / f"{safe_name(cand_key)}__ERROR.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "gross_net_attribution_summary.csv", index=False)

    with open(out_dir / "gross_net_attribution_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary.to_string(index=False))

    return summary


# =============================================================================
# 02 fee sweep
# =============================================================================

def recompute_net_with_fee(df: pd.DataFrame, fee_mult: float) -> pd.Series:
    # 当前 net = gross - 1.0 * cost
    # fee_mult=0 表示零费率，fee_mult=2 表示当前费率的 2 倍
    return df["gross"] - fee_mult * df["cost"]


def run_fee_sweep(
    baseline_dir: Path,
    candidate_dirs: Dict[str, Path],
    out_dir: Path,
) -> pd.DataFrame:
    baseline_raw = load_daily_pnl(baseline_dir)
    fee_mults = CONFIG["FEE_MULTIPLIERS"]

    all_rows = []

    for cand_key, cand_dir in candidate_dirs.items():
        try:
            cand_raw = load_daily_pnl(cand_dir)
            cand, base = align_two(cand_raw, baseline_raw)

            rows = []
            for m in fee_mults:
                cand_net_m = recompute_net_with_fee(cand, m)
                base_net_m = recompute_net_with_fee(base, m)

                rows.append({
                    "candidate": cand_key,
                    "fee_multiplier": float(m),
                    "candidate_sharpe": sharpe(cand_net_m),
                    "baseline_sharpe": sharpe(base_net_m),
                    "delta_sharpe": sharpe(cand_net_m) - sharpe(base_net_m),
                    "candidate_sum": float(cand_net_m.sum()),
                    "baseline_sum": float(base_net_m.sum()),
                    "delta_sum": float(cand_net_m.sum() - base_net_m.sum()),
                    "common_days": int(len(cand)),
                })

            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f"{safe_name(cand_key)}__fee_sweep.csv", index=False)
            all_rows.extend(rows)

            plt.figure(figsize=(9, 5))
            plt.plot(df["fee_multiplier"], df["candidate_sharpe"], marker="o", label=cand_key)
            plt.plot(df["fee_multiplier"], df["baseline_sharpe"], marker="o", label="baseline")
            plt.xlabel("fee multiplier")
            plt.ylabel("net Sharpe")
            plt.title(f"Fee sweep: {cand_key} vs baseline")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{safe_name(cand_key)}__fee_sweep.png", dpi=160)
            plt.close()

        except Exception as e:
            all_rows.append({
                "candidate": cand_key,
                "status": f"error: {repr(e)}",
            })
            with open(out_dir / f"{safe_name(cand_key)}__ERROR.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    out = pd.DataFrame(all_rows)
    out.to_csv(out_dir / "fee_sweep_all.csv", index=False)
    return out


# =============================================================================
# 03 baseline 换手前沿 vs candidate 点
# =============================================================================

def run_turnover_frontier(
    baseline_dir: Path,
    candidate_dirs: Dict[str, Path],
    frontier_dirs: Dict[str, Path],
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    rows = []

    # 原 baseline 也放进前沿
    all_frontier_dirs = {"baseline_original": baseline_dir}
    all_frontier_dirs.update(frontier_dirs)

    for key, d in all_frontier_dirs.items():
        try:
            pnl = load_daily_pnl(d)
            row = {
                "type": "baseline_frontier",
                "name": key,
                "dir": str(d),
                "net_sharpe": sharpe(pnl["net"]),
                "gross_sharpe": sharpe(pnl["gross"]),
                "avg_tvr": float(pnl["tvr"].mean()),
                "trade_cost_sum": float(pnl["cost"].sum()),
                "n_days": int(len(pnl)),
                "status": "ok",
            }
            rows.append(row)
        except Exception as e:
            rows.append({
                "type": "baseline_frontier",
                "name": key,
                "dir": str(d),
                "status": f"error: {repr(e)}",
            })

    for key, d in candidate_dirs.items():
        try:
            pnl = load_daily_pnl(d)
            row = {
                "type": "candidate",
                "name": key,
                "dir": str(d),
                "net_sharpe": sharpe(pnl["net"]),
                "gross_sharpe": sharpe(pnl["gross"]),
                "avg_tvr": float(pnl["tvr"].mean()),
                "trade_cost_sum": float(pnl["cost"].sum()),
                "n_days": int(len(pnl)),
                "status": "ok",
            }
            rows.append(row)
        except Exception as e:
            rows.append({
                "type": "candidate",
                "name": key,
                "dir": str(d),
                "status": f"error: {repr(e)}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "turnover_frontier_points.csv", index=False)

    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return df

    baseline_points = ok[ok["type"].eq("baseline_frontier")].copy()
    candidate_points = ok[ok["type"].eq("candidate")].copy()

    if not baseline_points.empty:
        baseline_points = baseline_points.sort_values("avg_tvr")

    # 对每个 candidate，找同等或更低换手 baseline 中最好的点
    eval_rows = []
    for _, c in candidate_points.iterrows():
        same_or_lower = baseline_points[baseline_points["avg_tvr"] <= c["avg_tvr"]]
        if same_or_lower.empty:
            best_frontier_sharpe = np.nan
            best_frontier_name = None
            above_frontier = None
            delta_vs_frontier = np.nan
        else:
            best_idx = same_or_lower["net_sharpe"].idxmax()
            best = same_or_lower.loc[best_idx]
            best_frontier_sharpe = best["net_sharpe"]
            best_frontier_name = best["name"]
            delta_vs_frontier = c["net_sharpe"] - best_frontier_sharpe
            above_frontier = bool(delta_vs_frontier > 0)

        eval_rows.append({
            "candidate": c["name"],
            "candidate_avg_tvr": c["avg_tvr"],
            "candidate_net_sharpe": c["net_sharpe"],
            "best_baseline_frontier_name_at_lower_or_equal_tvr": best_frontier_name,
            "best_baseline_frontier_sharpe_at_lower_or_equal_tvr": best_frontier_sharpe,
            "delta_vs_frontier": delta_vs_frontier,
            "above_frontier_flag": above_frontier,
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "candidate_vs_baseline_frontier_eval.csv", index=False)

    plt.figure(figsize=(9, 6))

    if not baseline_points.empty:
        plt.plot(
            baseline_points["avg_tvr"],
            baseline_points["net_sharpe"],
            marker="o",
            label="baseline frontier",
        )

        for _, r in baseline_points.iterrows():
            plt.annotate(str(r["name"]), (r["avg_tvr"], r["net_sharpe"]), fontsize=8)

    if not candidate_points.empty:
        plt.scatter(
            candidate_points["avg_tvr"],
            candidate_points["net_sharpe"],
            marker="x",
            s=80,
            label="candidate",
        )

        for _, r in candidate_points.iterrows():
            plt.annotate(str(r["name"]), (r["avg_tvr"], r["net_sharpe"]), fontsize=8)

    plt.xlabel("average TVR")
    plt.ylabel("net Sharpe")
    plt.title("Turnover frontier: baseline smoothing vs candidate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "turnover_frontier.png", dpi=170)
    plt.close()

    return df


# =============================================================================
# 04 配对 block bootstrap：检验 Sharpe 差异
# =============================================================================

def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)

    idx = []
    for s in starts:
        block = (s + np.arange(block_len)) % n
        idx.extend(block.tolist())

    return np.asarray(idx[:n], dtype=int)


def paired_block_bootstrap_delta_sharpe(
    candidate_net: np.ndarray,
    baseline_net: np.ndarray,
    block_len: int,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    candidate_net = np.asarray(candidate_net, dtype=float)
    baseline_net = np.asarray(baseline_net, dtype=float)

    mask = np.isfinite(candidate_net) & np.isfinite(baseline_net)
    c = candidate_net[mask]
    b = baseline_net[mask]

    n = len(c)
    obs = sharpe(c) - sharpe(b)

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        boots[i] = sharpe(c[idx]) - sharpe(b[idx])

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    # 这是 bootstrap 分布下 “差异 <= 0” 的比例，作为直观显著性参考。
    # 严格 p-value 可进一步用 centered bootstrap 做。
    prob_delta_le_0 = float(np.mean(boots <= 0))
    prob_delta_ge_0 = float(np.mean(boots >= 0))

    return {
        "n_days": int(n),
        "observed_delta_sharpe": float(obs),
        "ci_2p5": float(ci_low),
        "ci_97p5": float(ci_high),
        "prob_bootstrap_delta_le_0": prob_delta_le_0,
        "prob_bootstrap_delta_ge_0": prob_delta_ge_0,
        "bootstrap_values": boots,
    }


def run_paired_bootstrap(
    baseline_dir: Path,
    candidate_dirs: Dict[str, Path],
    out_dir: Path,
) -> pd.DataFrame:
    baseline_raw = load_daily_pnl(baseline_dir)
    rows = []

    for cand_key, cand_dir in candidate_dirs.items():
        try:
            cand_raw = load_daily_pnl(cand_dir)
            cand, base = align_two(cand_raw, baseline_raw)

            res = paired_block_bootstrap_delta_sharpe(
                cand["net"].values,
                base["net"].values,
                block_len=int(CONFIG["BOOTSTRAP_BLOCK_LEN"]),
                n_boot=int(CONFIG["BOOTSTRAP_N"]),
                seed=int(CONFIG["BOOTSTRAP_RANDOM_SEED"]),
            )

            boot_values = res.pop("bootstrap_values")

            boot_df = pd.DataFrame({"delta_sharpe": boot_values})
            boot_df.to_csv(out_dir / f"{safe_name(cand_key)}__bootstrap_distribution.csv", index=False)

            summary = {
                "candidate": cand_key,
                "candidate_dir": str(cand_dir),
                "baseline_dir": str(baseline_dir),
                **res,
                "significant_if_ci_excludes_0": bool(res["ci_2p5"] > 0 or res["ci_97p5"] < 0),
                "status": "ok",
            }
            rows.append(summary)

            write_json(out_dir / f"{safe_name(cand_key)}__bootstrap_summary.json", summary)

            plt.figure(figsize=(9, 5))
            plt.hist(boot_values, bins=60)
            plt.axvline(0, linestyle="--", label="0")
            plt.axvline(res["observed_delta_sharpe"], linestyle="-", label="observed")
            plt.xlabel("bootstrap ΔSharpe")
            plt.ylabel("count")
            plt.title(f"Paired block bootstrap ΔSharpe: {cand_key} vs baseline")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{safe_name(cand_key)}__bootstrap_delta_sharpe.png", dpi=160)
            plt.close()

        except Exception as e:
            rows.append({
                "candidate": cand_key,
                "status": f"error: {repr(e)}",
            })
            with open(out_dir / f"{safe_name(cand_key)}__ERROR.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "paired_block_bootstrap_summary.csv", index=False)
    return out


# =============================================================================
# 05 简化 Reality Check + 近似 Deflated Sharpe
# =============================================================================

def load_net_series_for_many(dirs: Dict[str, Path]) -> pd.DataFrame:
    series = {}
    for key, d in dirs.items():
        pnl = load_daily_pnl(d)
        series[key] = pnl["net"]
    df = pd.concat(series, axis=1, join="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def run_reality_check_and_dsr(
    baseline_dir: Path,
    trial_dirs: Dict[str, Path],
    out_dir: Path,
) -> Optional[dict]:
    if not trial_dirs:
        msg = {
            "status": "skipped",
            "reason": "CONFIG['TRIAL_DIRS'] 为空；没有 trial 输出目录，无法做 Reality Check / DSR。",
        }
        write_json(out_dir / "reality_check_and_dsr_summary.json", msg)
        return msg

    try:
        baseline = load_daily_pnl(baseline_dir)["net"].rename("baseline")
        trial_df = load_net_series_for_many(trial_dirs)

        all_df = pd.concat([baseline, trial_df], axis=1, join="inner")
        all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        if len(all_df) < 50:
            raise ValueError(f"共同日期太少: {len(all_df)}")

        base = all_df["baseline"].values.astype(float)
        trial_names = list(trial_df.columns)
        trials = all_df[trial_names].values.astype(float)

        baseline_sharpe = sharpe(base)
        trial_sharpes = np.array([sharpe(trials[:, j]) for j in range(trials.shape[1])])
        delta_sharpes = trial_sharpes - baseline_sharpe

        best_j = int(np.nanargmax(delta_sharpes))
        best_trial = trial_names[best_j]
        observed_best_delta = float(delta_sharpes[best_j])

        # White Reality Check 简化版：
        # 用 trial-baseline 的差异序列，去均值后作为“无信息差异”。
        # bootstrap 后看 N 个 trial 的 max ΔSharpe 零分布。
        D = trials - base[:, None]
        D_centered = D - np.nanmean(D, axis=0, keepdims=True)

        n = len(base)
        block_len = int(CONFIG["BOOTSTRAP_BLOCK_LEN"])
        n_boot = int(CONFIG["BOOTSTRAP_N"])
        rng = np.random.default_rng(int(CONFIG["BOOTSTRAP_RANDOM_SEED"]))

        null_max = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = circular_block_indices(n, block_len, rng)
            base_sample = base[idx]
            pseudo_trials = base_sample[:, None] + D_centered[idx, :]

            base_s = sharpe(base_sample)
            pseudo_s = np.array([sharpe(pseudo_trials[:, j]) for j in range(pseudo_trials.shape[1])])
            null_max[i] = np.nanmax(pseudo_s - base_s)

        reality_p_value = float(np.mean(null_max >= observed_best_delta))

        pd.DataFrame({
            "null_max_delta_sharpe": null_max,
        }).to_csv(out_dir / "reality_check_null_max_distribution.csv", index=False)

        trial_summary = pd.DataFrame({
            "trial": trial_names,
            "trial_sharpe": trial_sharpes,
            "baseline_sharpe": baseline_sharpe,
            "delta_sharpe": delta_sharpes,
        }).sort_values("delta_sharpe", ascending=False)

        trial_summary.to_csv(out_dir / "trial_sharpe_summary.csv", index=False)

        # 近似 DSR：
        # 用 daily Sharpe，避免年化尺度影响公式。
        nd = NormalDist()

        baseline_sr_daily = sharpe(base, annualize=False)
        trial_sr_daily = np.array([sharpe(trials[:, j], annualize=False) for j in range(trials.shape[1])])
        delta_sr_daily = trial_sr_daily - baseline_sr_daily

        n_trials = len(trial_names)
        best_delta_daily = float(np.nanmax(delta_sr_daily))
        mean_delta_daily = float(np.nanmean(delta_sr_daily))
        std_delta_daily = float(np.nanstd(delta_sr_daily, ddof=1)) if n_trials > 1 else np.nan

        dsr_prob = np.nan
        sr_star = np.nan

        if n_trials > 1 and np.isfinite(std_delta_daily) and std_delta_daily > 0:
            euler_gamma = 0.5772156649015329

            # 期望最大 Sharpe 的近似阈值
            q1 = nd.inv_cdf(1.0 - 1.0 / n_trials)
            q2 = nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
            sr_star = mean_delta_daily + std_delta_daily * ((1.0 - euler_gamma) * q1 + euler_gamma * q2)

            # 用 best trial 的差异收益序列估计偏度、峰度
            best_diff = D[:, best_j]
            best_diff = best_diff[np.isfinite(best_diff)]

            if len(best_diff) > 10 and best_diff.std(ddof=1) > 0:
                z = (best_diff - best_diff.mean()) / best_diff.std(ddof=1)
                skew = float(np.mean(z ** 3))
                kurt = float(np.mean(z ** 4))

                denom = math.sqrt(
                    max(
                        1e-12,
                        1.0 - skew * best_delta_daily + ((kurt - 1.0) / 4.0) * best_delta_daily ** 2,
                    )
                )

                stat = (best_delta_daily - sr_star) * math.sqrt(len(best_diff) - 1) / denom
                dsr_prob = float(nd.cdf(stat))

        result = {
            "status": "ok",
            "n_common_days": int(len(all_df)),
            "n_trials": int(n_trials),
            "baseline_sharpe": float(baseline_sharpe),
            "best_trial": best_trial,
            "best_trial_sharpe": float(trial_sharpes[best_j]),
            "observed_best_delta_sharpe": observed_best_delta,
            "reality_check_p_value_simplified": reality_p_value,
            "dsr_approx_probability": dsr_prob,
            "dsr_approx_sr_star_daily_delta": sr_star,
            "note": (
                "Reality Check 是简化版 White Reality Check；"
                "DSR 是基于 daily Sharpe 差异的近似口径。"
                "如果你们框架 sharpe_idx 定义不同，应以框架口径复核。"
            ),
        }

        write_json(out_dir / "reality_check_and_dsr_summary.json", result)

        plt.figure(figsize=(9, 5))
        plt.hist(null_max, bins=60)
        plt.axvline(observed_best_delta, linestyle="-", label="observed best ΔSharpe")
        plt.xlabel("null max ΔSharpe across trials")
        plt.ylabel("count")
        plt.title("Simplified Reality Check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "reality_check_null_max_distribution.png", dpi=160)
        plt.close()

        return result

    except Exception as e:
        result = {
            "status": f"error: {repr(e)}",
            "traceback": traceback.format_exc(),
        }
        write_json(out_dir / "reality_check_and_dsr_summary.json", result)
        return result


# =============================================================================
# 06 plateau check + seed sigma
# =============================================================================

def infer_metric_col(df: pd.DataFrame) -> str:
    explicit = CONFIG["PLATEAU_METRIC_COL"]
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"PLATEAU_METRIC_COL={explicit} 不在 columns 中")
        return explicit

    candidates = [
        "value",
        "sharpe_idx",
        "net_sharpe_idx",
        "gross_sharpe_idx",
        "objective",
        "score",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise KeyError("无法自动识别 plateau metric 列；请手动设置 CONFIG['PLATEAU_METRIC_COL']")
    return numeric_cols[0]


def infer_param_cols(df: pd.DataFrame, metric_col: str) -> List[str]:
    prefixes = ("params_", "param_")
    cols = []
    for c in df.columns:
        if c == metric_col:
            continue
        if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if cols:
        return cols

    # 如果没有 params_ 前缀，就退化为：所有数值列中排除明显不是参数的列
    exclude = {
        metric_col,
        "number",
        "trial",
        "datetime_start",
        "datetime_complete",
        "duration",
        "state",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in exclude]
    return cols


def run_plateau_check(
    baseline_dir: Path,
    optuna_trials_csv: Optional[Path],
    phaseb_seed_dirs: Dict[str, Path],
    out_dir: Path,
) -> dict:
    result = {
        "status": "started",
        "plateau_available": False,
        "seed_sigma_available": False,
    }

    # -------- plateau check --------
    if optuna_trials_csv is not None and optuna_trials_csv.exists():
        try:
            df = pd.read_csv(optuna_trials_csv)

            metric_col = infer_metric_col(df)
            param_cols = infer_param_cols(df, metric_col)

            work = df.dropna(subset=[metric_col]).copy()

            if not param_cols:
                raise ValueError(
                    "没有找到可用于 KNN 的数值参数列。"
                    "建议 Optuna csv 中参数列命名为 params_xxx 或 param_xxx。"
                )

            work = work.dropna(subset=param_cols).copy()

            best_idx = work[metric_col].idxmax()
            best_row = work.loc[best_idx]

            X = work[param_cols].astype(float).values
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0, ddof=1)
            sd[sd == 0] = 1.0

            Z = (X - mu) / sd
            best_z = (best_row[param_cols].astype(float).values - mu) / sd

            dist = np.sqrt(np.nanmean((Z - best_z[None, :]) ** 2, axis=1))
            work["_knn_dist_to_best"] = dist

            k = int(CONFIG["PLATEAU_K_NEIGHBORS"])
            neighbors = work.sort_values("_knn_dist_to_best").head(k).copy()
            neighbors.to_csv(out_dir / "plateau_knn_neighbors.csv", index=False)

            metric_values = neighbors[metric_col].astype(float)

            result.update({
                "plateau_available": True,
                "optuna_trials_csv": str(optuna_trials_csv),
                "metric_col": metric_col,
                "param_cols": param_cols,
                "n_trials_usable": int(len(work)),
                "k_neighbors": int(len(neighbors)),
                "best_metric": float(best_row[metric_col]),
                "neighbor_mean_metric": float(metric_values.mean()),
                "neighbor_std_metric": float(metric_values.std(ddof=1)),
                "best_minus_neighbor_mean": float(best_row[metric_col] - metric_values.mean()),
                "argmax_spike_flag": bool(
                    len(metric_values) > 1
                    and (best_row[metric_col] - metric_values.mean()) > 2.0 * metric_values.std(ddof=1)
                ),
            })

        except Exception as e:
            result.update({
                "plateau_available": False,
                "plateau_error": repr(e),
                "plateau_traceback": traceback.format_exc(),
            })

    else:
        result.update({
            "plateau_available": False,
            "plateau_reason": "OPTUNA_TRIALS_CSV 为空或文件不存在，跳过参数平台分析。",
        })

    # -------- seed sigma check --------
    if phaseb_seed_dirs:
        try:
            baseline_pnl = load_daily_pnl(baseline_dir)
            baseline_s = sharpe(baseline_pnl["net"])

            seed_rows = []
            for seed_name, seed_dir in phaseb_seed_dirs.items():
                pnl = load_daily_pnl(seed_dir)
                seed_rows.append({
                    "seed": seed_name,
                    "dir": str(seed_dir),
                    "net_sharpe": sharpe(pnl["net"]),
                    "gross_sharpe": sharpe(pnl["gross"]),
                    "avg_tvr": float(pnl["tvr"].mean()),
                    "trade_cost_sum": float(pnl["cost"].sum()),
                    "n_days": int(len(pnl)),
                })

            seed_df = pd.DataFrame(seed_rows)
            seed_df.to_csv(out_dir / "phaseb_seed_metrics.csv", index=False)

            seed_sigma = float(seed_df["net_sharpe"].std(ddof=1)) if len(seed_df) > 1 else np.nan
            seed_mean = float(seed_df["net_sharpe"].mean())
            seed_best = float(seed_df["net_sharpe"].max())
            best_minus_baseline = seed_best - baseline_s

            flag_in_noise_band = (
                np.isfinite(seed_sigma)
                and best_minus_baseline < 2.0 * seed_sigma
            )

            result.update({
                "seed_sigma_available": True,
                "baseline_net_sharpe": float(baseline_s),
                "phaseb_seed_mean_net_sharpe": seed_mean,
                "phaseb_seed_best_net_sharpe": seed_best,
                "phaseb_seed_sigma_net_sharpe": seed_sigma,
                "phaseb_best_minus_baseline": float(best_minus_baseline),
                "best_minus_baseline_less_than_2sigma_seed_flag": bool(flag_in_noise_band),
            })

        except Exception as e:
            result.update({
                "seed_sigma_available": False,
                "seed_sigma_error": repr(e),
                "seed_sigma_traceback": traceback.format_exc(),
            })

    else:
        result.update({
            "seed_sigma_available": False,
            "seed_sigma_reason": "PHASEB_SEED_DIRS 为空，跳过 seed sigma 门槛检查。",
        })

    result["status"] = "ok"
    write_json(out_dir / "plateau_and_seed_sigma_summary.json", result)
    return result


# =============================================================================
# 总入口
# =============================================================================

def validate_config() -> Tuple[Path, Path, Dict[str, Path], Dict[str, Path], Dict[str, Path], Optional[Path], Dict[str, Path]]:
    out_root = Path(CONFIG["OUT_ROOT"])

    baseline_dir = path_or_none(CONFIG["BASELINE_DIR"])
    if baseline_dir is None:
        raise ValueError("请先填写 CONFIG['BASELINE_DIR']")

    candidate_dirs = {
        k: Path(v)
        for k, v in CONFIG["CANDIDATE_DIRS"].items()
        if str(v).strip()
    }

    frontier_dirs = {
        k: Path(v)
        for k, v in CONFIG["BASELINE_FRONTIER_DIRS"].items()
        if str(v).strip()
    }

    trial_dirs = {
        k: Path(v)
        for k, v in CONFIG["TRIAL_DIRS"].items()
        if str(v).strip()
    }

    optuna_trials_csv = path_or_none(CONFIG["OPTUNA_TRIALS_CSV"])

    seed_dirs = {
        k: Path(v)
        for k, v in CONFIG["PHASEB_SEED_DIRS"].items()
        if str(v).strip()
    }

    return out_root, baseline_dir, candidate_dirs, frontier_dirs, trial_dirs, optuna_trials_csv, seed_dirs


def main() -> None:
    out_root, baseline_dir, candidate_dirs, frontier_dirs, trial_dirs, optuna_trials_csv, seed_dirs = validate_config()
    dirs = ensure_output_dirs(out_root)

    master_log = {
        "out_root": str(out_root),
        "baseline_dir": str(baseline_dir),
        "candidate_dirs": {k: str(v) for k, v in candidate_dirs.items()},
        "baseline_frontier_dirs": {k: str(v) for k, v in frontier_dirs.items()},
        "trial_dirs_count": len(trial_dirs),
        "optuna_trials_csv": str(optuna_trials_csv) if optuna_trials_csv else "",
        "phaseb_seed_dirs": {k: str(v) for k, v in seed_dirs.items()},
    }

    print("=" * 120)
    print("Non-training overfit diagnostics")
    print("=" * 120)
    print(json.dumps(master_log, ensure_ascii=False, indent=2))

    write_json(dirs["root"] / "run_config_resolved.json", master_log)

    # 01 毛/净分解
    if candidate_dirs:
        print("\n[01] Running gross/net attribution...")
        attr_df = run_gross_net_attribution(
            baseline_dir=baseline_dir,
            candidate_dirs=candidate_dirs,
            out_dir=dirs["p0_attr"],
        )
        print(attr_df.to_string(index=False))
    else:
        print("\n[01] skipped: CANDIDATE_DIRS 为空")

    # 02 fee sweep
    if candidate_dirs:
        print("\n[02] Running fee sweep...")
        fee_df = run_fee_sweep(
            baseline_dir=baseline_dir,
            candidate_dirs=candidate_dirs,
            out_dir=dirs["fee"],
        )
        print(fee_df.to_string(index=False))
    else:
        print("\n[02] skipped: CANDIDATE_DIRS 为空")

    # 03 turnover frontier
    print("\n[03] Running turnover frontier...")
    frontier_df = run_turnover_frontier(
        baseline_dir=baseline_dir,
        candidate_dirs=candidate_dirs,
        frontier_dirs=frontier_dirs,
        out_dir=dirs["frontier"],
    )
    if frontier_df is not None:
        print(frontier_df.to_string(index=False))

    # 04 paired block bootstrap
    if candidate_dirs:
        print("\n[04] Running paired block bootstrap...")
        boot_df = run_paired_bootstrap(
            baseline_dir=baseline_dir,
            candidate_dirs=candidate_dirs,
            out_dir=dirs["bootstrap"],
        )
        print(boot_df.to_string(index=False))
    else:
        print("\n[04] skipped: CANDIDATE_DIRS 为空")

    # 05 Reality Check + DSR
    print("\n[05] Running simplified Reality Check + approximate DSR...")
    reality_result = run_reality_check_and_dsr(
        baseline_dir=baseline_dir,
        trial_dirs=trial_dirs,
        out_dir=dirs["reality"],
    )
    print(json.dumps(reality_result, ensure_ascii=False, indent=2, cls=NpEncoder))

    # 06 plateau check + seed sigma
    print("\n[06] Running plateau check + seed sigma...")
    plateau_result = run_plateau_check(
        baseline_dir=baseline_dir,
        optuna_trials_csv=optuna_trials_csv,
        phaseb_seed_dirs=seed_dirs,
        out_dir=dirs["plateau"],
    )
    print(json.dumps(plateau_result, ensure_ascii=False, indent=2, cls=NpEncoder))

    print("\nAll done.")
    print("Output root:", out_root)


if __name__ == "__main__":
    main()
