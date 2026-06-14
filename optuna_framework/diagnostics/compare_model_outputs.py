# -*- coding: utf-8 -*-
"""Compare a candidate combo output directory against a baseline output directory.

This is the repository-integrated version of the pasted one-off comparison script.
It keeps the original comparison logic while making the script directly runnable
and placing default outputs under ``optuna_framework/diagnostic_results``.
"""

from __future__ import annotations

import argparse
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "optuna_framework" / "diagnostic_results" / "model_output_compare"

# Original pasted defaults. They can be overridden by CLI arguments.
BASE = "/root/autodl-tmp/haoyi_comb2_organize/eg-torch/output-torch"
CANDIDATES = [
    "/root/autodl-tmp/haoyi_comb2_organize/eg-torch/output-torch-tcn-only-ablation",
]


def model_name(path: str | Path) -> str:
    return Path(str(path).rstrip("/")).name


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def find_position_file(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    files = sorted((model_dir / "backtest").glob("AlphaStrategy_*_position.csv"))
    if not files:
        raise FileNotFoundError(f"Cannot find AlphaStrategy_*_position.csv under {model_dir / 'backtest'}")
    return files[-1]


def load_alpha(model_dir: str | Path) -> pd.DataFrame:
    path = Path(model_dir) / "alpha.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing alpha file: {path}")
    return pd.read_parquet(path)


def load_position(model_dir: str | Path) -> pd.DataFrame:
    path = find_position_file(model_dir)
    return pd.read_csv(path, index_col=0)


def load_pnl(model_dir: str | Path) -> pd.DataFrame:
    path = Path(model_dir) / "backtest" / "daily_pnl.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pnl file: {path}")
    return pd.read_csv(path)


def load_ic(model_dir: str | Path) -> pd.DataFrame:
    path = Path(model_dir) / "daily_ic"
    if not path.exists():
        raise FileNotFoundError(f"Missing IC file: {path}")
    ic = pd.read_csv(path, sep="\t", index_col=0, na_values=["NAN"])
    ic.index = ic.index.astype(int)
    return ic


def compare_matrix(x, y, name, x_name, y_name):
    dates = x.index.intersection(y.index).sort_values()
    cols = x.columns.intersection(y.columns)

    rows = []
    for d in dates:
        a = x.loc[d, cols].astype(float)
        b = y.loc[d, cols].astype(float)

        valid = a.notna() & b.notna() & (a != 0) & (b != 0)

        pa = a > 0
        pb = b > 0
        union = pa | pb

        ta = set(a.dropna().nlargest(500).index)
        tb = set(b.dropna().nlargest(500).index)
        top_union = ta | tb

        rows.append(
            {
                "date": d,
                "corr": a[valid].corr(b[valid]) if valid.sum() >= 1000 else np.nan,
                "n_valid": int(valid.sum()),
                "positive_jaccard": float((pa & pb).sum() / union.sum()) if union.sum() else np.nan,
                "top500_jaccard": len(ta & tb) / len(top_union) if top_union else np.nan,
                f"{x_name}_positive": int(pa.sum()),
                f"{y_name}_positive": int(pb.sum()),
                f"{x_name}_nonzero": int((a.fillna(0) != 0).sum()),
                f"{y_name}_nonzero": int((b.fillna(0) != 0).sum()),
            }
        )

    out = pd.DataFrame(rows)

    print(f"\n== {name} compare: {x_name}  vs  {y_name} ==")
    print(f"common dates: {len(dates)}")
    print(f"common instruments: {len(cols)}")

    if out.empty:
        print("No overlapping dates.")
        return out

    print(out.describe().to_string())

    print("\nlowest top500_jaccard days:")
    print(out.sort_values("top500_jaccard").head(10).to_string(index=False))

    return out


def ic_stats(df):
    n = df["ic"].notna().sum()
    ic_std = df["ic"].std(ddof=1)
    rankic_std = df["rankic"].std(ddof=1)

    return pd.Series(
        {
            "ic": df["ic"].mean(),
            "rankic": df["rankic"].mean(),
            "5dic": df["5dic"].mean(),
            "ic_ir": df["ic"].mean() / ic_std * np.sqrt(252) if ic_std and not pd.isna(ic_std) else np.nan,
            "rankic_ir": df["rankic"].mean() / rankic_std * np.sqrt(252) if rankic_std and not pd.isna(rankic_std) else np.nan,
            "ic_win": (df["ic"] > 0).sum() / n if n else np.nan,
            "coverage": df["coverage"].mean(),
            "n": int(n),
        }
    )


def yearly_ic(ic):
    yr = ic.index // 10000
    rows = {int(y): ic_stats(ic[yr == y]) for y in sorted(set(yr))}
    rows["period"] = ic_stats(ic)
    return pd.DataFrame(rows).T


def compare_one(candidate: str, base: str, out_dir: Path) -> dict:
    candidate_name = model_name(candidate)
    base_name = model_name(base)

    txt_path = out_dir / f"{safe_name(candidate_name)}__vs__{safe_name(base_name)}.txt"

    summary = {
        "candidate": candidate_name,
        "baseline": base_name,
        "txt_path": str(txt_path),
        "alpha_corr_mean": np.nan,
        "alpha_top500_jaccard_mean": np.nan,
        "position_corr_mean": np.nan,
        "position_top500_jaccard_mean": np.nan,
        "pnl_corr": np.nan,
        "candidate_sum_pnl": np.nan,
        "baseline_sum_pnl": np.nan,
        "candidate_ic_period": np.nan,
        "baseline_ic_period": np.nan,
        "candidate_rankic_period": np.nan,
        "baseline_rankic_period": np.nan,
        "status": "ok",
    }

    with open(txt_path, "w", encoding="utf-8") as f, redirect_stdout(f):
        print("=" * 120)
        print("BASELINE")
        print("name:", base_name)
        print("path:", base)
        print("-" * 120)
        print("CANDIDATE")
        print("name:", candidate_name)
        print("path:", candidate)
        print("=" * 120)

        print("\nPosition file:")
        print(f"{base_name}:", find_position_file(base))
        print(f"{candidate_name}:", find_position_file(candidate))

        # ========== alpha 对比 ==========
        alpha_candidate = load_alpha(candidate)
        alpha_base = load_alpha(base)
        alpha_cmp = compare_matrix(
            alpha_candidate,
            alpha_base,
            "alpha",
            candidate_name,
            base_name,
        )

        if not alpha_cmp.empty:
            summary["alpha_corr_mean"] = float(alpha_cmp["corr"].mean())
            summary["alpha_top500_jaccard_mean"] = float(alpha_cmp["top500_jaccard"].mean())

        # ========== position 对比 ==========
        pos_candidate = load_position(candidate)
        pos_base = load_position(base)
        pos_cmp = compare_matrix(
            pos_candidate,
            pos_base,
            "position",
            candidate_name,
            base_name,
        )

        if not pos_cmp.empty:
            summary["position_corr_mean"] = float(pos_cmp["corr"].mean())
            summary["position_top500_jaccard_mean"] = float(pos_cmp["top500_jaccard"].mean())

        # ========== pnl 对比 ==========
        pnl_candidate = load_pnl(candidate)
        pnl_base = load_pnl(base)
        p = pnl_candidate.merge(pnl_base, on="date", suffixes=("_candidate", "_base"))

        print(f"\n== pnl compare: {candidate_name}  vs  {base_name} ==")
        print("common pnl dates:", len(p))

        if not p.empty:
            pnl_corr = p["pnl_candidate"].corr(p["pnl_base"])
            candidate_sum_pnl = p["pnl_candidate"].sum()
            base_sum_pnl = p["pnl_base"].sum()

            summary["pnl_corr"] = float(pnl_corr)
            summary["candidate_sum_pnl"] = float(candidate_sum_pnl)
            summary["baseline_sum_pnl"] = float(base_sum_pnl)

            print("pnl corr:", pnl_corr)
            print(f"sum pnl {candidate_name}:", candidate_sum_pnl)
            print(f"sum pnl {base_name}:", base_sum_pnl)
            print(f"trade_cost {candidate_name}:", p["trade_cost_candidate"].sum())
            print(f"trade_cost {base_name}:", p["trade_cost_base"].sum())
            print(f"avg tvr {candidate_name}:", p["tvr_candidate"].mean())
            print(f"avg tvr {base_name}:", p["tvr_base"].mean())
            print(f"avg long_num {candidate_name}:", p["long_num_candidate"].mean())
            print(f"avg long_num {base_name}:", p["long_num_base"].mean())
        else:
            print("No overlapping pnl dates.")

        # ========== IC 对比 ==========
        ic_candidate = load_ic(candidate)
        ic_base = load_ic(base)

        dates = ic_candidate.index.intersection(ic_base.index).sort_values()
        ic_candidate = ic_candidate.loc[dates]
        ic_base = ic_base.loc[dates]

        yc = yearly_ic(ic_candidate)
        yb = yearly_ic(ic_base)

        metrics = ["ic", "rankic", "5dic", "ic_ir", "rankic_ir", "ic_win", "coverage", "n"]

        cmp = pd.concat(
            {
                m: pd.DataFrame(
                    {
                        candidate_name: yc[m],
                        base_name: yb[m],
                    }
                )
                for m in metrics
            },
            axis=1,
        )

        print(f"\n== IC compare: {candidate_name}  vs  {base_name} ==")
        print("common IC dates:", len(dates))
        print(cmp.to_string(float_format=lambda v: f"{v:.4f}"))

        if "period" in yc.index and "period" in yb.index:
            summary["candidate_ic_period"] = float(yc.loc["period", "ic"])
            summary["baseline_ic_period"] = float(yb.loc["period", "ic"])
            summary["candidate_rankic_period"] = float(yc.loc["period", "rankic"])
            summary["baseline_rankic_period"] = float(yb.loc["period", "rankic"])

        print("\nSaved txt:", txt_path)

    return summary


def run_compare(base: str, candidates: list[str], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for candidate in candidates:
        candidate_name = model_name(candidate)
        try:
            print(f"[RUN] {candidate_name} vs {model_name(base)}")
            s = compare_one(candidate, base, out_dir)
            all_summaries.append(s)
            print(f"[DONE] {candidate_name}")
        except Exception as e:
            txt_path = out_dir / f"{safe_name(candidate_name)}__vs__{safe_name(model_name(base))}__ERROR.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"ERROR comparing {candidate_name} vs {model_name(base)}\n")
                f.write(f"candidate: {candidate}\n")
                f.write(f"baseline: {base}\n\n")
                f.write(traceback.format_exc())

            all_summaries.append(
                {
                    "candidate": candidate_name,
                    "baseline": model_name(base),
                    "txt_path": str(txt_path),
                    "alpha_corr_mean": np.nan,
                    "alpha_top500_jaccard_mean": np.nan,
                    "position_corr_mean": np.nan,
                    "position_top500_jaccard_mean": np.nan,
                    "pnl_corr": np.nan,
                    "candidate_sum_pnl": np.nan,
                    "baseline_sum_pnl": np.nan,
                    "candidate_ic_period": np.nan,
                    "baseline_ic_period": np.nan,
                    "candidate_rankic_period": np.nan,
                    "baseline_rankic_period": np.nan,
                    "status": f"error: {repr(e)}",
                }
            )
            print(f"[ERROR] {candidate_name}: {repr(e)}")
            print(f"        see {txt_path}")

    summary_df = pd.DataFrame(all_summaries)

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Baseline: {model_name(base)}\n")
        f.write(f"Baseline path: {base}\n")
        f.write("=" * 120 + "\n\n")
        f.write(summary_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        f.write("\n")

    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nAll done.")
    print("TXT summary:", summary_txt)
    print("CSV summary:", summary_csv)
    print("Output dir:", out_dir)

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate combo outputs against a baseline output directory.")
    parser.add_argument("--base", default=BASE, help="Baseline output directory.")
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        default=None,
        help="Candidate output directory. Can be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for comparison reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = args.candidates if args.candidates is not None else list(CANDIDATES)
    run_compare(args.base, candidates, Path(args.out_dir))


if __name__ == "__main__":
    main()
