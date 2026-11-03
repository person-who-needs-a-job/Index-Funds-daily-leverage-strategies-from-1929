#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize all S&P 500 and Nasdaq-100 strategies from the FOUR consolidated CSV files.

Inputs (nothing else needed):
    SP500_daily.csv, SP500_annual.csv, NDX_daily.csv, NDX_annual.csv

Outputs:
    strategy_summary.csv  -- full-history metrics for all 8 index/strategy combinations
    strategy_summary.md   -- concise, GitHub-ready Markdown tables

No market downloads, new backtests, bootstrap tests, or statistical p-values.
All monetary returns are NET of the costs already applied in the supplied daily CSVs.

Google Colab:
    %run /content/DMA200_All_Strategies_Summary_Colab.py \
        --results /content/dma_daily_annual_results \
        --out /content/strategy_summary_results
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.2425
STRATEGIES = (
    ('buyhold_1x', '1x Buy & Hold'),
    ('dma_unfiltered_3x', 'DMA200 Fixed 3x (Unfiltered)'),
    ('dma_filtered_3x', 'DMA200 Fixed 3x + Sell Filter'),
    ('dma_filtered_dynamic', 'DMA200 Dynamic Leverage + Sell Filter'),
)


def read_inputs(directory: Path, index: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dpath = directory / f'{index}_daily.csv'
    apath = directory / f'{index}_annual.csv'
    if not dpath.is_file() or not apath.is_file():
        raise FileNotFoundError(
            f'Missing {index} CSVs in {directory}. Required: '
            f'{dpath.name} and {apath.name}'
        )
    daily = pd.read_csv(dpath, parse_dates=['date'])
    annual = pd.read_csv(apath)
    needed_daily = {'date', 'execution', 'cash_return'}
    needed_annual = {'year', 'complete_year'}
    for key, _ in STRATEGIES:
        needed_daily.add(f'{key}_return')
        needed_annual.add(f'{key}_annual_return')
        if key != 'buyhold_1x':
            needed_daily.add(f'{key}_held_leverage_for_return')
    if needed_daily - set(daily.columns):
        raise ValueError(f'{dpath.name}: missing columns {sorted(needed_daily - set(daily.columns))}')
    if needed_annual - set(annual.columns):
        raise ValueError(f'{apath.name}: missing columns {sorted(needed_annual - set(annual.columns))}')
    if daily.empty or annual.empty:
        raise ValueError(f'Empty input file for {index}')
    if daily.date.isna().any() or daily.date.duplicated().any() or not daily.date.is_monotonic_increasing:
        raise ValueError(f'{dpath.name}: dates must be unique, valid, and sorted')
    if not daily.execution.astype(str).eq('same_close').all():
        raise ValueError(f'{dpath.name}: strategy execution must be same_close')
    if annual.year.duplicated().any():
        raise ValueError(f'{apath.name}: duplicated calendar years')
    annual['complete_year'] = annual.complete_year.astype(str).str.lower().isin(['true', '1'])
    # Cross-check the annual CSV against the actual daily return stream.
    actual_years = set(daily.date.dt.year.unique())
    stored_years = set(annual.year.astype(int))
    if actual_years != stored_years:
        raise ValueError(f'{index}: annual and daily CSVs have different calendar-year coverage')
    for key, _ in STRATEGIES:
        col = f'{key}_return'
        values = daily[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= -1):
            raise ValueError(f'{index}/{key}: invalid daily returns')
        for year, subset in daily.groupby(daily.date.dt.year):
            actual = float(np.expm1(np.log1p(subset[col].to_numpy(dtype=float)).sum()))
            reported = float(annual.loc[annual.year == year, f'{key}_annual_return'].iloc[0])
            if not np.isclose(actual, reported, rtol=1e-8, atol=1e-8):
                raise ValueError(f'{index}/{key}/{year}: annual return differs from daily compounded returns')
    return daily, annual


def make_summary_row(index: str, key: str, title: str,
                     daily: pd.DataFrame, annual: pd.DataFrame) -> dict:
    r = daily[f'{key}_return'].to_numpy(dtype=float)
    rf = daily['cash_return'].to_numpy(dtype=float)
    start = daily.date.iloc[0]
    end = daily.date.iloc[-1]
    years = float((end - start).days / CALENDAR_DAYS_PER_YEAR)
    if years <= 0:
        raise ValueError(f'{index}: insufficient date coverage')
    log_growth = float(np.log1p(r).sum())
    cumulative_multiple = float(np.exp(log_growth))
    cagr = float(np.expm1(log_growth / years))
    wealth = np.r_[1., np.exp(np.cumsum(np.log1p(r)))]
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1.0
    lowest = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[lowest])
    max_dd_date = str(daily.date.iloc[lowest - 1].date()) if lowest > 0 else str(start.date())
    vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    excess = r - rf
    excess_sd = float(np.std(excess, ddof=1))
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(excess) / excess_sd) if excess_sd > 0 else float('nan')
    downside = np.minimum(excess, 0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(excess) / downside_dev) if downside_dev > 0 else float('nan')
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float('nan')

    if key == 'buyhold_1x':
        lev = np.ones(len(daily), dtype=float)
        transaction_cost = 0.0
    else:
        lev = daily[f'{key}_held_leverage_for_return'].to_numpy(dtype=float)
        if not np.isfinite(lev).all() or np.any(lev < 0):
            raise ValueError(f'{index}/{key}: invalid leverage')
        cost_col = f'{key}_trade_cost'
        transaction_cost = float(daily[cost_col].sum()) if cost_col in daily else float('nan')
    invested = lev > 0
    invested_fraction = float(invested.mean())
    avg_lev_invested = float(lev[invested].mean()) if invested.any() else 0.0

    complete = annual.loc[annual.complete_year].copy()
    yearly = complete[f'{key}_annual_return'].astype(float)
    if len(complete) > 0:
        best_i = yearly.idxmax()
        worst_i = yearly.idxmin()
        best_year = int(complete.loc[best_i, 'year'])
        worst_year = int(complete.loc[worst_i, 'year'])
        best_return = float(yearly.loc[best_i])
        worst_return = float(yearly.loc[worst_i])
        positive_year_fraction = float((yearly > 0).mean())
        avg_year_return = float(yearly.mean())
        median_year_return = float(yearly.median())
    else:
        best_year = worst_year = None
        best_return = worst_return = positive_year_fraction = avg_year_return = median_year_return = float('nan')

    return {
        'index': index,
        'strategy': title,
        'strategy_key': key,
        'start_date': str(start.date()),
        'end_date': str(end.date()),
        'trading_days': len(daily),
        'elapsed_years': round(years, 6),
        'complete_calendar_years': len(complete),
        'total_return': cumulative_multiple - 1,
        'growth_multiple': cumulative_multiple,
        'cagr': cagr,
        'annual_volatility': vol,
        'sharpe_excess_cash': sharpe,
        'sortino_excess_cash': sortino,
        'maximum_drawdown': max_dd,
        'maximum_drawdown_date': max_dd_date,
        'calmar': calmar,
        'mean_leverage_all_days': float(np.mean(lev)),
        'mean_leverage_invested': avg_lev_invested,
        'invested_fraction': invested_fraction,
        'cash_fraction': 1 - invested_fraction,
        'mean_complete_year_return': avg_year_return,
        'median_complete_year_return': median_year_return,
        'positive_complete_year_fraction': positive_year_fraction,
        'best_complete_year': best_year,
        'best_complete_year_return': best_return,
        'worst_complete_year': worst_year,
        'worst_complete_year_return': worst_return,
        'sum_simple_transaction_cost': transaction_cost,
    }


def p(x: float, digits: int = 2) -> str:
    return 'N/A' if not np.isfinite(x) else f'{100*x:.{digits}f}%'


def n(x: float, digits: int = 2) -> str:
    return 'N/A' if not np.isfinite(x) else f'{x:.{digits}f}'


def write_markdown(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        '# Historical DMA200 Strategy Performance Summary',
        '',
        'Computed **only from** the consolidated `SP500_daily.csv`, `SP500_annual.csv`, '
        '`NDX_daily.csv`, and `NDX_annual.csv` files. No bootstrap tests or p-values.',
        '',
    ]
    for index, name in [('SP500', 'S&P 500'), ('NDX', 'Nasdaq-100 (NDX)')]:
        sub = summary.loc[summary['index'].eq(index)]
        if sub.empty:
            continue
        first = sub.iloc[0]
        lines.extend([
            f'## {name}: {first.start_date} to {first.end_date}',
            '',
            f'Includes all {int(first.trading_days):,} trading days ({first.elapsed_years:.2f} elapsed years), '
            f'plus {int(first.complete_calendar_years)} complete calendar years for annual statistics.',
            '',
            '| Strategy | CAGR | Sharpe | Annual vol. | Max drawdown | Mean leverage (all days) | Mean leverage (invested) | Invested days |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for row in sub.itertuples(index=False):
            lines.append(
                f'| {row.strategy} | {p(row.cagr)} | {n(row.sharpe_excess_cash, 3)} | '
                f'{p(row.annual_volatility)} | {p(row.maximum_drawdown)} | '
                f'{n(row.mean_leverage_all_days)}x | {n(row.mean_leverage_invested)}x | '
                f'{p(row.invested_fraction)} |'
            )
        lines.extend([
            '',
            '| Strategy | Mean annual return* | Positive years* | Best year* | Worst year* | Sortino | Calmar |',
            '|---|---:|---:|---|---|---:|---:|',
        ])
        for row in sub.itertuples(index=False):
            lines.append(
                f'| {row.strategy} | {p(row.mean_complete_year_return)} | '
                f'{p(row.positive_complete_year_fraction)} | '
                f'{row.best_complete_year} ({p(row.best_complete_year_return)}) | '
                f'{row.worst_complete_year} ({p(row.worst_complete_year_return)}) | '
                f'{n(row.sortino_excess_cash, 3)} | {n(row.calmar, 3)} |'
            )
        lines.append('')
    lines.extend([
        '*Annual-return statistics use only complete calendar years. The CAGR, Sharpe, drawdown and leverage '
        'metrics use the **entire available daily record**, including any partial start/end years.*',
        '',
        '**Definitions:** CAGR uses actual elapsed calendar time; Sharpe is annualized mean daily return '
        'in excess of the modeled cash return divided by daily excess-return volatility (×√252). '
        'Sortino uses downside deviation of daily excess returns; maximum drawdown uses daily compounded '
        'wealth with a starting wealth of 1. Average leverage counts cash days as 0x; 1x buy-and-hold '
        'is treated as continuously invested at 1x.',
        '',
        '**Caveats:** Returns are synthetic, pre-tax, and already incorporate the fee/financing/trading-cost '
        'assumptions of the supplied CSVs. Execution is the idealized same-close convention. '
        'Different indexes have different historical starting dates; this is not a same-period comparison.',
        '',
    ])
    output.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize all 4 strategies per index from daily/annual CSVs; no bootstrap.')
    parser.add_argument('--results', default='/content/dma_daily_annual_results',
                        help='Directory containing the 4 consolidated daily/annual CSV files')
    parser.add_argument('--out', default='/content/strategy_summary_results',
                        help='Output directory for summary CSV and Markdown')
    args = parser.parse_args()
    source = Path(args.results).expanduser()
    if not source.is_dir():
        raise NotADirectoryError(f'Missing daily/annual results folder: {source}')
    rows = []
    for index in ('SP500', 'NDX'):
        daily, annual = read_inputs(source, index)
        for key, title in STRATEGIES:
            rows.append(make_summary_row(index, key, title, daily, annual))
    summary = pd.DataFrame(rows)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'strategy_summary.csv'
    md_path = out / 'strategy_summary.md'
    summary.to_csv(csv_path, index=False, float_format='%.10g')
    write_markdown(summary, md_path)
    for index, group in summary.groupby('index', sort=False):
        print(f'\n{index}: {group.start_date.iloc[0]} through {group.end_date.iloc[0]}')
        display = group[['strategy', 'cagr', 'sharpe_excess_cash', 'maximum_drawdown',
                         'mean_leverage_all_days', 'mean_leverage_invested', 'invested_fraction']].copy()
        display.columns = ['Strategy','CAGR','Sharpe','Max DD','Avg Lev (All)','Avg Lev (Invested)','Time Invested']
        for col in ('CAGR','Max DD','Time Invested'):
            display[col] = display[col].map(lambda x: f'{x:.2%}')
        for col in ('Sharpe','Avg Lev (All)','Avg Lev (Invested)'):
            display[col] = display[col].map(lambda x: f'{x:.3f}')
        print(display.to_string(index=False))
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    print('No statistical tests were performed. No original input files were modified.')


if __name__ == '__main__':
    main()
