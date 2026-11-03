#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annual-observation bootstrap tests for SAME-CLOSE S&P 500 / NDX strategies.

ONE complete calendar year is ONE bootstrap observation. No rolling-ten-year
windows, no daily i.i.d. resampling, and absolutely no next-close inputs.

Exactly THREE one-sided tests per index:
  1  DMA200 fixed 3x > buy-and-hold 1x: mean annual return
  2  Filtered fixed 3x > unfiltered fixed 3x: mean annual return
  3  Filtered dynamic leverage > filtered fixed 3x: pooled Sharpe ratio
     (the original fourth comparison, retained as the third displayed row).

No DMA200-vs-3x-buy-and-hold test is run.

Uses four-file daily/annual output from DMA200_SameClose_Daily_Annual_Only_Colab.py;\nolder per-strategy same-close daily CSVs also work.
Each bootstrap draw selects complete years WITH replacement and applies the SAME
selection to both strategies and to the risk-free series. Year-block length is
1 by default; --year-block 3 is an optional serial-dependence sensitivity test.

Outputs annual_samples.csv, annual_bootstrap_pvalues.csv, bootstrap_summary.md.
The Markdown summary contains exactly the three requested rows.
One-sided H1: strategy A outperforms B. The centered-bootstrap p-values are exploratory, do not correct for strategy
selection/data-mining, and do not imply a live-tradable same-close fill.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
TESTS = (
    ("01_DMA200_vs_1x_buyhold", "dma_fixed3", "buyhold1", "annual_return"),
    ("02_sell_filter_vs_unfiltered", "dma_filtered_fixed3", "dma_fixed3", "annual_return"),
    ("03_dynamic_vs_filtered_fixed3", "dma_filtered_dynamic", "dma_filtered_fixed3", "pooled_sharpe"),
)


def read_csv(root: Path, fname: str) -> pd.DataFrame:
    f = root / fname
    if not f.is_file():
        raise FileNotFoundError(f"Missing {f}. Use the output folder from the SAME-CLOSE main backtest.")
    return pd.read_csv(f)




def collect_consolidated_data(root: Path, index: str):
    """Read <INDEX>_daily.csv from the four-file main backtest layout."""
    label=f"{index}_daily.csv"
    df=read_csv(root,label)
    required={"date","execution","dma200","index_total_return","cash_return",
              "all_in_correction_annual","dynamic_signal_ready"}
    names=("dma_unfiltered_3x","dma_filtered_3x","dma_filtered_dynamic")
    for name in names:
        required.update({f"{name}_return",f"{name}_held_leverage_for_return",
                         f"{name}_target_leverage_at_close"})
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if not df.execution.eq("same_close").all():
        raise ValueError(f"{label}: not same_close")
    df["date"]=pd.to_datetime(df.date)
    df=df.set_index("date").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{label}: duplicated dates")
    for name in names:
        held=df[f"{name}_held_leverage_for_return"].to_numpy(dtype=float)
        target=df[f"{name}_target_leverage_at_close"].to_numpy(dtype=float)
        if not np.allclose(held[1:],target[:-1],atol=1e-10,rtol=0,equal_nan=True):
            raise ValueError(f"{label}: target/held same-close consistency failure for {name}")
    ready=df.dynamic_signal_ready.astype(str).str.lower().eq("true")
    ready_years=df.index.year[ready.to_numpy()]
    if len(ready_years)==0:
        raise ValueError(f"{label}: no dynamic-signal-ready years")
    dynamic_first_year=int(ready_years.min())
    data=pd.DataFrame(index=df.index)
    data["cash"]=df.cash_return
    data["buyhold1"]=df.index_total_return
    data["dma_fixed3"]=df.dma_unfiltered_3x_return
    data["dma_filtered_fixed3"]=df.dma_filtered_3x_return
    data["dma_filtered_dynamic"]=df.dma_filtered_dynamic_return
    data["dma200"]=df.dma200
    good=data.notna().all(axis=1)
    for col in ("buyhold1","dma_fixed3","dma_filtered_fixed3","dma_filtered_dynamic"):
        good &= data[col]>-1
    data=data.loc[good].drop(columns=["dma200"])
    data["year"]=data.index.year
    full_years=[]
    for y,group in data.groupby("year",sort=True):
        if (len(group)>=220 and group.index.min().month==1 and
                group.index.min().day<=10 and group.index.max().month==12 and
                group.index.max().day>=22):
            full_years.append(int(y))
    data=data.loc[data.year.isin(full_years)].copy()
    if len(full_years)<8:
        raise ValueError(f"{label}: fewer than 8 complete years")
    return data,dynamic_first_year

def collect_data(root: Path, index: str, filter_name: str, leverage: str):
    if (root / f"{index}_daily.csv").is_file():
        return collect_consolidated_data(root,index)
    parts = []
    for filt, lev in (("none", "fixed3"), (filter_name, "fixed3"), (filter_name, leverage)):
        label = f"{index}_{filt}_{lev}_cash_daily.csv"
        d = read_csv(root, label)
        required = {
            "date", "execution", "dma200", "index_total_return", "cash_return",
            "strategy_return", "all_in_correction_annual",
            "signal_target_leverage_at_close", "held_leverage_for_return",
        }
        missing = required - set(d.columns)
        if missing:
            raise ValueError(f"{label}: missing columns: {sorted(missing)}")
        if not d["execution"].eq("same_close").all():
            raise ValueError(f"{label}: NOT same_close. Refusing to use delayed execution data.")
        held = d.held_leverage_for_return.to_numpy(dtype=float)
        targets = d.signal_target_leverage_at_close.to_numpy(dtype=float)
        if not np.allclose(held[1:], targets[:-1], atol=1e-10, rtol=0, equal_nan=True):
            raise ValueError(f"{label}: held exposure does not match preceding-close signal")
        d["date"] = pd.to_datetime(d.date)
        d = d.set_index("date").sort_index()
        if d.index.has_duplicates:
            raise ValueError(f"{label}: duplicate dates")
        parts.append(d)
    base, filtered, dynamic = parts
    if not (base.index.equals(filtered.index) and base.index.equals(dynamic.index)):
        raise ValueError(f"{index}: dates differ among the three strategies")
    for d in (filtered, dynamic):
        for field in ("cash_return", "index_total_return", "all_in_correction_annual"):
            if not np.allclose(base[field], d[field], atol=1e-12, rtol=1e-9, equal_nan=True):
                raise ValueError(f"{index}: {field} disagrees across model outputs")

    data = pd.DataFrame(index=base.index)
    data["cash"] = base.cash_return
    data["buyhold1"] = base.index_total_return
    data["dma_fixed3"] = base.strategy_return
    data["dma_filtered_fixed3"] = filtered.strategy_return
    data["dma_filtered_dynamic"] = dynamic.strategy_return
    data["dma200"] = base.dma200

    # Calendar-year bootstrap must not splice across missing dates or put partial
    # first/last years into the same sampling pool as complete years.
    good = data.notna().all(axis=1)
    for col in ("buyhold1", "dma_fixed3",
                "dma_filtered_fixed3", "dma_filtered_dynamic"):
        good &= data[col] > -1
    data = data.loc[good].drop(columns=["dma200"])
    source = read_csv(root, f"{index}_return_sources_audit.csv")
    if "date" not in source or source.empty:
        raise ValueError(f"{index}: source audit has no usable dates")
    # After the partial first source year, two further COMPLETE calendar years
    # are used to form the dynamic signal; matches the parent backtest's convention.
    dynamic_first_year = int(pd.to_datetime(source.date.iloc[0]).year) + 3
    data["year"] = data.index.year

    full_years = []
    for y, group in data.groupby("year", sort=True):
        # Market-holiday tolerance: at least 220 observations, first by Jan 10,
        # last on or after Dec 22. Incomplete 2026 YTD is excluded.
        if (len(group) >= 220 and group.index.min().month == 1
                and group.index.min().day <= 10
                and group.index.max().month == 12
                and group.index.max().day >= 22):
            full_years.append(int(y))
    data = data.loc[data.year.isin(full_years)].copy()
    if len(full_years) < 8:
        raise ValueError(f"{index}: fewer than eight complete years")
    return data, dynamic_first_year


def sharpe_from_sums(sums: np.ndarray, sums_sq: np.ndarray, counts: np.ndarray):
    mean = sums / counts
    var = np.maximum((sums_sq - sums * sums / counts) / (counts - 1), 1e-20)
    return np.sqrt(TRADING_DAYS) * mean / np.sqrt(var)


def one_sided_pvalue(obs: float, boot: np.ndarray):
    """Centered-null bootstrap: H0 effect=0. One-sided H1 A > B."""
    n = len(boot)
    zero_centered = boot - obs
    p1 = (1 + np.count_nonzero(zero_centered >= obs)) / (n + 1)
    lower_95 = np.quantile(boot, 0.05)
    return float(p1), float(lower_95)


def year_bootstrap_samples(n: int, repeats: int, rng: np.random.Generator, year_block: int):
    """Sample pairs of contiguous complete YEARS, with circular wrap for >1."""
    if year_block < 1 or year_block > n:
        raise ValueError("--year-block must be between 1 and the number of sample years")
    nb = (n + year_block - 1) // year_block
    starts = rng.integers(0, n, size=(repeats, nb), dtype=np.int32)
    offsets = np.arange(year_block, dtype=np.int32)
    return ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(repeats, -1)[:, :n]


def holm(pvals):
    result = np.zeros(len(pvals), dtype=float)
    order = np.argsort(pvals)
    last = 0.0
    for i, j in enumerate(order):
        last = max(last, min(1.0, (len(pvals) - i) * pvals[j]))
        result[j] = last
    return result


def process_index(data: pd.DataFrame, index: str, dynamic_first_year: int,
                  reps: int, year_block: int, rng: np.random.Generator):
    years = sorted(data.year.unique())
    yearly = []
    for y in years:
        g = data.loc[data.year == y]
        r = {"index": index, "year": int(y), "trading_days": len(g),
             "dynamic_signal_ready": bool(y >= dynamic_first_year)}
        for col in ("buyhold1", "dma_fixed3",
                    "dma_filtered_fixed3", "dma_filtered_dynamic"):
            values = g[col].to_numpy(dtype=float)
            excess = values - g.cash.to_numpy(dtype=float)
            r[f"{col}_annual_return"] = float(np.expm1(np.log1p(values).sum()))
            sd = float(excess.std(ddof=1))
            r[f"{col}_annual_sharpe"] = (
                float(np.sqrt(TRADING_DAYS) * excess.mean() / sd) if sd > 1e-14 else np.nan)
            r[f"{col}_excess_sum"] = float(excess.sum())
            r[f"{col}_excess_sumsq"] = float(np.dot(excess, excess))
        yearly.append(r)
    annual = pd.DataFrame(yearly)
    output = []
    for name, A, B, metric in TESTS:
        subset = annual.loc[annual.year >= dynamic_first_year].copy() if name.startswith("03_") else annual
        if len(subset) < 8:
            raise ValueError(f"{index} {name}: not enough complete years for Bootstrap")
        sel = year_bootstrap_samples(len(subset), reps, rng, year_block)
        if metric == "annual_return":
            a = subset[f"{A}_annual_return"].to_numpy(dtype=float)
            b = subset[f"{B}_annual_return"].to_numpy(dtype=float)
            delta_year = a - b
            observed = float(delta_year.mean())
            sampled = delta_year[sel].mean(axis=1)
            a_metric, b_metric = float(a.mean()), float(b.mean())
        else:
            cnt = subset.trading_days.to_numpy(dtype=float)
            sa = subset[f"{A}_excess_sum"].to_numpy(dtype=float)
            sb = subset[f"{B}_excess_sum"].to_numpy(dtype=float)
            ssa = subset[f"{A}_excess_sumsq"].to_numpy(dtype=float)
            ssb = subset[f"{B}_excess_sumsq"].to_numpy(dtype=float)
            a_metric = float(sharpe_from_sums(sa.sum(), ssa.sum(), cnt.sum()))
            b_metric = float(sharpe_from_sums(sb.sum(), ssb.sum(), cnt.sum()))
            observed = a_metric - b_metric
            draw_a = sharpe_from_sums(sa[sel].sum(axis=1), ssa[sel].sum(axis=1), cnt[sel].sum(axis=1))
            draw_b = sharpe_from_sums(sb[sel].sum(axis=1), ssb[sel].sum(axis=1), cnt[sel].sum(axis=1))
            sampled = draw_a - draw_b
        p1, lower_95 = one_sided_pvalue(observed, sampled)
        output.append({
            "index": index, "test": name, "metric": metric,
            "strategy_A": A, "strategy_B": B,
            "first_year": int(subset.year.iloc[0]), "last_year": int(subset.year.iloc[-1]),
            "n_annual_samples": len(subset), "year_block": year_block,
            "bootstrap_reps": reps, "mean_or_sharpe_A": a_metric,
            "mean_or_sharpe_B": b_metric, "observed_difference_A_minus_B": observed,
            "one_sided_95_lower_bound": lower_95,
            "p_one_sided_A_better": p1,
            "execution": "same_close",
        })
    p = np.asarray([x["p_one_sided_A_better"] for x in output], dtype=float)
    for x, adjusted in zip(output, holm(p)):
        x["p_one_sided_Holm_3_tests"] = float(adjusted)
    return annual, pd.DataFrame(output)



def write_readme_table(pvalues: pd.DataFrame, out_file: Path, reps: int, year_block: int) -> None:
    """Write the three requested rows as a GitHub-friendly English table."""
    labels = (
        ("01_DMA200_vs_1x_buyhold", "DMA200 fixed 3x > 1x buy-and-hold: mean annual return"),
        ("02_sell_filter_vs_unfiltered", "Filtered fixed 3x > unfiltered fixed 3x: mean annual return"),
        ("03_dynamic_vs_filtered_fixed3", "Filtered dynamic leverage > filtered fixed 3x: Sharpe ratio"),
    )
    rows = [
        f"### One-Sided p-Values ({reps:,} Annual Bootstrap Resamples; {year_block}-Year Blocks)",
        "",
        "**One-sided alternative:** left-hand strategy > right-hand strategy.",
        "",
        "| Comparison | S&P 500 | NDX |",
        "|---|---:|---:|",
    ]
    for test_id, title in labels:
        cells = []
        for idx in ("SP500", "NDX"):
            matches = pvalues.loc[(pvalues["test"] == test_id) & (pvalues["index"] == idx)]
            cells.append(f"{matches.iloc[0]['p_one_sided_A_better']:.5f}" if len(matches) else "—")
        rows.append(f"| {title} | {cells[0]} | {cells[1]} |")
    rows.extend(("", "Raw one-sided p-values; no multiple-testing correction applied to this table.",
                 "2026 is excluded from the complete-calendar-year bootstrap. Same-close simulated execution.", ""))
    out_file.write_text("\n".join(rows), encoding="utf-8")


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--results", default="/content/dma_daily_annual_results",
                    help="Four-file same-close daily/annual output folder")
    pa.add_argument("--out", default="/content/annual_bootstrap_results")
    pa.add_argument("--reps", type=int, default=10000)
    pa.add_argument("--year-block", type=int, default=1,
                    help="Resample individual years (1); optional 3/5-year dependence check")
    pa.add_argument("--indices", choices=("SP500", "NDX", "both"), default="both")
    pa.add_argument("--filter", default="sp_downside")
    pa.add_argument("--leverage", default="blend_1y_2y")
    pa.add_argument("--seed", type=int, default=20261103)
    pa.add_argument("--download", action="store_true", help="Colab: download the two CSVs and Markdown summary")
    args = pa.parse_args()
    if args.reps < 200:
        raise ValueError("At least 200 bootstrap replicates required")
    source = Path(args.results).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {source}")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    annual_parts, pvalue_parts = [], []
    rng = np.random.default_rng(args.seed)
    for idx in (["SP500", "NDX"] if args.indices == "both" else [args.indices]):
        d, first_dynamic_year = collect_data(source, idx, args.filter, args.leverage)
        ann, pv = process_index(d, idx, first_dynamic_year, args.reps, args.year_block, rng)
        annual_parts.append(ann)
        pvalue_parts.append(pv)
        print(f"{idx}: {len(ann)} complete calendar-year samples; dynamic test starts {first_dynamic_year}", flush=True)
    annual = pd.concat(annual_parts, ignore_index=True)
    pvals = pd.concat(pvalue_parts, ignore_index=True)
    annual_path = out / "annual_samples.csv"
    pvalues_path = out / "annual_bootstrap_pvalues.csv"
    annual.to_csv(annual_path, index=False)
    pvals.to_csv(pvalues_path, index=False)
    summary_path = out / "bootstrap_summary.md"
    write_readme_table(pvals, summary_path, args.reps, args.year_block)
    print("\nBOOTSTRAP P-VALUES (SAME-CLOSE, ANNUAL SAMPLING; THREE TESTS)")
    print(pvals[["index", "test", "n_annual_samples", "observed_difference_A_minus_B",
                 "p_one_sided_A_better", "p_one_sided_Holm_3_tests"]].to_string(index=False))
    print(f"\nFiles: {annual_path}, {pvalues_path}, {summary_path} (no ZIP)")
    if args.download:
        try:
            from google.colab import files
            files.download(str(annual_path))
            files.download(str(pvalues_path))
            files.download(str(summary_path))
        except ImportError:
            print("Outside Colab: download the two CSV files manually.")


if __name__ == "__main__":
    main()
