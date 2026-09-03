"""
S&P 500 synthetic 3x: monthly 200DMA +/-1% band backtest (1950-present)

Standalone Google Colab script.  Upload it and run:
    %run /content/sp500_3x_monthly_200dma_band_1950_colab.py

Monthly rule
------------
* Observe the S&P 500 close and its 200-day simple moving average only on the
  final trading session of each calendar month.
* If price >= 200DMA * 1.01, target 3x.
* If price <= 200DMA * 0.99, target cash.
* Inside the +/-1% band, keep the previous target.
* Implement the month-end decision on the next trading session.  Therefore
  there is at most one exposure change per calendar month; there is no
  intramonth reaction.

Comparisons
-----------
1. S&P 500 total-return buy and hold (1x)
2. Fixed synthetic daily-reset 3x
3. Daily-evaluated 200DMA +/-1% band 3x
4. Monthly-evaluated 200DMA +/-1% band 3x
5. Monthly-evaluated 200DMA +/-1% band 1x

Data and cost model
-------------------
* Real S&P 500 daily closing prices: Yahoo Chart API.
* Historical monthly dividend yield: Multpl; DataHub is a backup.
* Daily risk-free return: Kenneth French data; Yahoo ^IRX fills newest dates.
* Synthetic daily-reset 3x return:
      3 * S&P total return - 2 * funding return - daily expense ratio
* Funding rate = T-bill benchmark + assumed swap spread.
* Trading costs are applied whenever fund/market weight changes.

Important limitation: public daily realized dealer swap spreads do not exist
back to 1950.  SWAP_SPREAD_BPS is an explicit assumption, not real swap data.
Multpl dividends are accrued from a monthly yield and are not exact daily
ex-dividend cash flows.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path


# -----------------------------------------------------------------------------
# 0. Install/import packages inside a normal Colab .py file
# -----------------------------------------------------------------------------

REQUIRED_PACKAGES = ["pandas", "numpy", "matplotlib", "requests", "lxml"]


def install_if_missing() -> None:
    import importlib.util

    missing = [p for p in REQUIRED_PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            *missing,
        ])


install_if_missing()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# 1. User settings
# -----------------------------------------------------------------------------

START_DATE = "1950-01-01"
END_DATE = None                    # None = latest available observation
INITIAL_CAPITAL = 10_000.0

DMA_DAYS = 200
ENTRY_BUFFER = 0.01               # +1%
EXIT_BUFFER = 0.01                # -1%

SWAP_SPREAD_BPS = 40.0            # assumption, not an observed 1950+ series
LETF_EXPENSE_RATIO = 0.0095        # 0.95% per year
TRADE_COST_BPS = 5.0              # one-way, charged on weight changed

AUTO_DOWNLOAD_ZIP = True
OUTPUT_DIR = Path("/content") if Path("/content").exists() else Path.cwd()

YAHOO_SP500_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
    "https://query2.finance.yahoo.com/v8/finance/chart/%5EGSPC",
]
YAHOO_IRX_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX",
    "https://query2.finance.yahoo.com/v8/finance/chart/%5EIRX",
]
MULTPL_DIVIDEND_YIELD = "https://www.multpl.com/s-p-500-dividend-yield/table/by-month"
DATAHUB_SP500_CSV = "https://datahub.io/core/s-and-p-500/_r/-/data/data.csv"
FRENCH_DAILY_FACTORS_ZIP = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


# -----------------------------------------------------------------------------
# 2. Downloads and parsing
# -----------------------------------------------------------------------------

def get_bytes(url: str, label: str, attempts: int = 2, timeout: int = 20) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 ColabBacktest/1.0"}
    last_error = None
    print(f"  -> {label}...", flush=True)
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            print(f"     completed: {label}", flush=True)
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                print(f"     retry {attempt + 2}/{attempts}: {label}", flush=True)
                time.sleep(1.5)
    raise RuntimeError(f"Download failed: {label}\nURL: {url}\n{last_error}")


def download_yahoo_close(
    urls: list[str],
    label: str,
    series_name: str,
    min_observations: int,
) -> pd.Series:
    warmup_start = pd.Timestamp(START_DATE) - pd.Timedelta(days=450)
    end = (pd.Timestamp(END_DATE) if END_DATE else pd.Timestamp.today()) + pd.Timedelta(days=1)
    query = (
        f"?period1={int(warmup_start.timestamp())}"
        f"&period2={int(end.timestamp())}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )
    errors = []
    for server, base_url in enumerate(urls, start=1):
        try:
            raw = get_bytes(base_url + query, f"Yahoo {label} (server {server})")
            payload = json.loads(raw.decode("utf-8"))
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            results = chart.get("result") or []
            if not results:
                raise RuntimeError("Yahoo JSON contains no result")
            result = results[0]
            timestamps = result.get("timestamp") or []
            quotes = result.get("indicators", {}).get("quote") or []
            closes = quotes[0].get("close", []) if quotes else []
            if len(timestamps) != len(closes) or not timestamps:
                raise RuntimeError("Yahoo timestamps and closes are missing or mismatched")
            dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            values = pd.to_numeric(pd.Series(closes), errors="coerce")
            series = pd.Series(values.to_numpy(), index=dates, name=series_name).dropna()
            series = series[series > 0]
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if len(series) < min_observations:
                raise RuntimeError(f"Yahoo returned only {len(series):,} observations")
            return series
        except Exception as exc:
            errors.append(f"server {server}: {exc}")
            print(f"     server {server} failed; switching server...", flush=True)
    raise RuntimeError(f"Both Yahoo servers failed for {label}:\n" + "\n".join(errors))


def download_sp500_price() -> pd.Series:
    return download_yahoo_close(
        YAHOO_SP500_URLS,
        "S&P 500 daily closing prices",
        "sp500_price",
        5_000,
    )


def download_irx_yield_optional() -> pd.Series:
    try:
        return download_yahoo_close(
            YAHOO_IRX_URLS,
            "13-week Treasury bill yield (^IRX)",
            "irx_yield_percent",
            2_000,
        )
    except Exception as exc:
        print(f"     ^IRX unavailable; French RF will be used alone: {exc}", flush=True)
        return pd.Series(dtype=float, name="irx_yield_percent")


def download_french_daily_rf() -> pd.Series:
    raw = get_bytes(FRENCH_DAILY_FACTORS_ZIP, "Kenneth French daily RF")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("Kenneth French ZIP contains no CSV")
        text = archive.read(csv_names[0]).decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_row = next(
        (i for i, line in enumerate(lines) if "Mkt-RF" in line and "RF" in line),
        None,
    )
    if header_row is None:
        raise RuntimeError("Kenneth French daily CSV header was not found")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_row:])))
    date_col = frame.columns[0]
    date_text = frame[date_col].astype(str).str.strip()
    valid = date_text.str.fullmatch(r"\d{8}")
    dates = pd.to_datetime(date_text.loc[valid], format="%Y%m%d", errors="coerce")
    # French RF is percentage per trading day: 0.01 means 0.01%.
    values = pd.to_numeric(frame.loc[valid, "RF"], errors="coerce") / 100.0
    series = pd.Series(values.to_numpy(), index=dates, name="daily_rf_return").dropna()
    if series.empty:
        raise RuntimeError("Kenneth French RF series is empty")
    return series[~series.index.duplicated(keep="last")].sort_index()


def parse_multpl_dividend_yield(raw: bytes) -> pd.Series:
    tables = pd.read_html(io.StringIO(raw.decode("utf-8", errors="replace")))
    for table in tables:
        table.columns = [str(c).strip() for c in table.columns]
        if {"Date", "Value"}.issubset(table.columns):
            dates = pd.to_datetime(table["Date"], errors="coerce")
            values = pd.to_numeric(
                table["Value"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            ) / 100.0
            series = pd.Series(values.to_numpy(), index=dates, name="dividend_yield").dropna()
            series = series[(series >= 0) & (series <= 0.25)]
            if len(series) >= 100:
                return series[~series.index.duplicated(keep="last")].sort_index()
    raise RuntimeError("No usable Multpl Date/Value table found")


def download_datahub_dividend_yield() -> pd.Series:
    raw = get_bytes(DATAHUB_SP500_CSV, "DataHub dividend backup")
    frame = pd.read_csv(io.BytesIO(raw))
    required = {"Date", "SP500", "Dividend"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Unexpected DataHub schema")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    price = pd.to_numeric(frame["SP500"], errors="coerce")
    dividend = pd.to_numeric(frame["Dividend"], errors="coerce")
    series = pd.Series((dividend / price).to_numpy(), index=dates, name="dividend_yield")
    series = series.dropna()
    series = series[(series > 0) & (series <= 0.25)]
    if series.empty:
        raise RuntimeError("DataHub has no usable dividend observations")
    return series[~series.index.duplicated(keep="last")].sort_index()


def download_dividend_yield() -> tuple[pd.Series, str]:
    try:
        raw = get_bytes(MULTPL_DIVIDEND_YIELD, "Multpl S&P 500 dividend yield")
        return parse_multpl_dividend_yield(raw), "Multpl monthly dividend yield"
    except Exception as exc:
        print(f"     Multpl unavailable: {exc}", flush=True)
        return download_datahub_dividend_yield(), "DataHub dividend backup"


# -----------------------------------------------------------------------------
# 3. Signals and portfolio returns
# -----------------------------------------------------------------------------

def annual_percent_to_daily_return(rate_percent: pd.Series) -> pd.Series:
    rate = rate_percent.astype(float) / 100.0
    return np.power(1.0 + rate.clip(lower=-0.99), 1.0 / 252.0) - 1.0


def synthetic_3x_return(market_return: pd.Series, funding_return: pd.Series) -> pd.Series:
    expense_daily = (1.0 + LETF_EXPENSE_RATIO) ** (1.0 / 252.0) - 1.0
    result = 3.0 * market_return - 2.0 * funding_return - expense_daily
    result.name = "synthetic_3x_fund_return_before_trading_cost"
    return result


def hysteresis_band_decision(price_vs_dma: pd.Series) -> pd.Series:
    """Daily-evaluated 0/1 state with a true +/-1% hold band."""
    state = 0.0
    values = np.zeros(len(price_vs_dma), dtype=float)
    for i, distance in enumerate(price_vs_dma):
        if np.isfinite(distance):
            if state == 0.0 and distance >= ENTRY_BUFFER:
                state = 1.0
            elif state == 1.0 and distance <= -EXIT_BUFFER:
                state = 0.0
        values[i] = state
    return pd.Series(values, index=price_vs_dma.index, name="daily_band_decision")


def monthly_hysteresis_band_decision(
    price_vs_dma: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Evaluate only at calendar month-end; hold the state on all other days."""
    month = pd.Series(price_vs_dma.index.to_period("M"), index=price_vs_dma.index)
    # Require a following observed session in a different month.  This prevents
    # the latest, still-incomplete month from being falsely treated as complete.
    next_month = month.shift(-1)
    is_month_end = month.ne(next_month) & next_month.notna()
    state = 0.0
    daily_state = np.zeros(len(price_vs_dma), dtype=float)
    rows = []

    for i, date in enumerate(price_vs_dma.index):
        previous_state = state
        action = "NO_MONTHLY_DECISION"
        if bool(is_month_end.iloc[i]) and np.isfinite(price_vs_dma.iloc[i]):
            distance = float(price_vs_dma.iloc[i])
            if distance >= ENTRY_BUFFER:
                state = 1.0
            elif distance <= -EXIT_BUFFER:
                state = 0.0
            if state > previous_state:
                action = "BUY_ON_NEXT_SESSION"
            elif state < previous_state:
                action = "SELL_ON_NEXT_SESSION"
            else:
                action = "HOLD"
            execution_date = (
                price_vs_dma.index[i + 1] if i + 1 < len(price_vs_dma) else pd.NaT
            )
            rows.append({
                "decision_date": date,
                "execution_date": execution_date,
                "price_vs_dma": distance,
                "previous_signal": previous_state,
                "new_signal": state,
                "target_3x_exposure": 3.0 * state,
                "action": action,
            })
        daily_state[i] = state

    decision = pd.Series(
        daily_state,
        index=price_vs_dma.index,
        name="monthly_band_decision",
    )
    return decision, is_month_end.astype(bool), pd.DataFrame(rows)


def leveraged_strategy(
    name: str,
    effective_exposure: pd.Series,
    fund_return: pd.Series,
    cash_return: pd.Series,
) -> pd.DataFrame:
    exposure = effective_exposure.clip(0.0, 3.0).fillna(0.0)
    fund_weight = exposure / 3.0
    gross_return = fund_weight * fund_return + (1.0 - fund_weight) * cash_return
    turnover = fund_weight.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(fund_weight.iloc[0])
    trading_cost = turnover.fillna(0.0) * TRADE_COST_BPS / 10_000.0
    net_return = gross_return - trading_cost
    wiped = net_return <= -1.0
    if wiped.any():
        first = wiped.idxmax()
        net_return.loc[first] = -1.0
        net_return.loc[net_return.index > first] = 0.0
    wealth = INITIAL_CAPITAL * (1.0 + net_return.fillna(0.0)).cumprod()
    return pd.DataFrame({
        f"{name}_exposure": exposure,
        f"{name}_return": net_return,
        f"{name}_wealth": wealth,
        f"{name}_turnover": turnover.fillna(0.0),
        f"{name}_trading_cost": trading_cost,
    })


def unleveraged_strategy(
    name: str,
    market_weight: pd.Series,
    market_return: pd.Series,
    cash_return: pd.Series,
) -> pd.DataFrame:
    weight = market_weight.clip(0.0, 1.0).fillna(0.0)
    gross_return = weight * market_return + (1.0 - weight) * cash_return
    turnover = weight.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(weight.iloc[0])
    cost = turnover.fillna(0.0) * TRADE_COST_BPS / 10_000.0
    net_return = gross_return - cost
    wealth = INITIAL_CAPITAL * (1.0 + net_return.fillna(0.0)).cumprod()
    return pd.DataFrame({
        f"{name}_exposure": weight,
        f"{name}_return": net_return,
        f"{name}_wealth": wealth,
        f"{name}_turnover": turnover.fillna(0.0),
        f"{name}_trading_cost": cost,
    })


# -----------------------------------------------------------------------------
# 4. Reporting
# -----------------------------------------------------------------------------

def performance_row(
    strategy: str,
    returns: pd.Series,
    exposure: pd.Series,
    cash_return: pd.Series,
) -> dict:
    returns = returns.dropna()
    years = (returns.index[-1] - returns.index[0]).days / 365.2425
    wealth = INITIAL_CAPITAL * (1.0 + returns.fillna(0.0)).cumprod()
    final_value = float(wealth.iloc[-1])
    cagr = (final_value / INITIAL_CAPITAL) ** (1.0 / years) - 1.0
    volatility = returns.std(ddof=1) * math.sqrt(252.0)
    excess = returns - cash_return.reindex(returns.index).fillna(0.0)
    sharpe = (
        excess.mean() / excess.std(ddof=1) * math.sqrt(252.0)
        if excess.std(ddof=1) > 0 else np.nan
    )
    running_peak = wealth.cummax().clip(lower=INITIAL_CAPITAL)
    max_drawdown = (wealth / running_peak - 1.0).min()
    aligned_exposure = exposure.reindex(returns.index).fillna(0.0)
    changes = aligned_exposure.diff().abs().gt(1e-12)
    return {
        "Strategy": strategy,
        "Start": returns.index[0].date(),
        "End": returns.index[-1].date(),
        "Total Return": final_value / INITIAL_CAPITAL - 1.0,
        "CAGR": cagr,
        "Annual Volatility": volatility,
        "Sharpe vs T-Bill": sharpe,
        "Max Drawdown": max_drawdown,
        "Final $10k": final_value,
        "Time in Market": aligned_exposure.gt(0).mean(),
        "Average Gross Exposure": aligned_exposure.mean(),
        "Exposure Changes": int(changes.sum()),
    }


def make_summary(
    backtest: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    cash_return: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame([
        performance_row(
            display,
            backtest[return_col],
            backtest[exposure_col],
            cash_return,
        )
        for display, return_col, exposure_col in specs
    ])


def make_decade_summary(
    backtest: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    cash_return: pd.Series,
) -> pd.DataFrame:
    rows = []
    first_decade = backtest.index.min().year // 10 * 10
    last_decade = backtest.index.max().year // 10 * 10
    for decade in range(first_decade, last_decade + 1, 10):
        period = backtest[
            (backtest.index.year >= decade) & (backtest.index.year <= decade + 9)
        ]
        if len(period) < 2:
            continue
        for display, return_col, exposure_col in specs:
            row = performance_row(
                display,
                period[return_col],
                period[exposure_col],
                cash_return.reindex(period.index).fillna(0.0),
            )
            row["Decade"] = f"{decade}s"
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[["Decade"] + [c for c in frame.columns if c != "Decade"]]


def formatted_table(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    for col in [
        "Total Return",
        "CAGR",
        "Annual Volatility",
        "Max Drawdown",
        "Time in Market",
    ]:
        shown[col] = shown[col].map(lambda x: f"{x:.2%}")
    shown["Sharpe vs T-Bill"] = shown["Sharpe vs T-Bill"].map(lambda x: f"{x:.3f}")
    shown["Final $10k"] = shown["Final $10k"].map(lambda x: f"${x:,.2f}")
    shown["Average Gross Exposure"] = shown["Average Gross Exposure"].map(
        lambda x: f"{x:.3f}"
    )
    return shown


# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Downloading S&P 500, dividends, and T-bill data...", flush=True)
    price_full = download_sp500_price()
    french_rf_full = download_french_daily_rf()
    irx_yield_full = download_irx_yield_optional()
    monthly_dividend_yield, dividend_source = download_dividend_yield()

    price = price_full.loc[:END_DATE] if END_DATE else price_full.copy()
    trading_days = price.index

    dividend_yield = monthly_dividend_yield.reindex(trading_days, method="ffill")
    if dividend_yield.loc[pd.Timestamp(START_DATE):].isna().any():
        raise RuntimeError("Dividend yield has missing observations in the backtest period")
    dividend_daily = np.power(1.0 + dividend_yield, 1.0 / 252.0) - 1.0
    price_return = price.pct_change()
    market_return = (1.0 + price_return) * (1.0 + dividend_daily) - 1.0
    market_return.name = "sp500_total_return"

    irx_rate = irx_yield_full.reindex(trading_days).ffill()
    irx_daily = annual_percent_to_daily_return(irx_rate)
    cash_return = french_rf_full.reindex(trading_days).combine_first(irx_daily)
    cash_return = cash_return.ffill()
    if cash_return.loc[pd.Timestamp(START_DATE):].isna().any():
        raise RuntimeError("Risk-free return has missing observations in the backtest period")
    cash_annual_percent = (
        np.power(1.0 + cash_return.clip(lower=-0.99), 252.0) - 1.0
    ) * 100.0
    funding_annual_percent = cash_annual_percent + SWAP_SPREAD_BPS / 100.0
    funding_return = annual_percent_to_daily_return(funding_annual_percent)
    fund_3x_return = synthetic_3x_return(market_return, funding_return)

    daily = pd.DataFrame(index=trading_days)
    daily.index.name = "date"
    daily["sp500_price"] = price
    daily["annual_dividend_yield"] = dividend_yield
    daily["sp500_price_return"] = price_return
    daily["sp500_total_return"] = market_return
    daily["cash_return"] = cash_return
    daily["cash_annual_rate_percent"] = cash_annual_percent
    daily["assumed_funding_rate_percent"] = funding_annual_percent
    daily["synthetic_3x_fund_return_before_trading_cost"] = fund_3x_return
    daily["dma200"] = price.rolling(DMA_DAYS, min_periods=DMA_DAYS).mean()
    daily["price_vs_dma"] = price / daily["dma200"] - 1.0

    daily_signal = hysteresis_band_decision(daily["price_vs_dma"])
    monthly_signal, month_end_flag, monthly_decisions = monthly_hysteresis_band_decision(
        daily["price_vs_dma"]
    )
    daily["month_end_decision_day"] = month_end_flag
    daily["daily_band_decision"] = daily_signal
    daily["monthly_band_decision"] = monthly_signal

    # Decision after close, execution on the next trading session.
    daily["fixed_3x_exposure"] = 3.0
    daily["daily_band_3x_exposure"] = (3.0 * daily_signal).shift(1).fillna(0.0)
    daily["monthly_band_3x_exposure"] = (3.0 * monthly_signal).shift(1).fillna(0.0)
    daily["monthly_band_1x_weight"] = monthly_signal.shift(1).fillna(0.0)

    backtest = daily.loc[pd.Timestamp(START_DATE):].copy()
    if END_DATE is not None:
        backtest = backtest.loc[:pd.Timestamp(END_DATE)]
    backtest = backtest.dropna(subset=["sp500_total_return", "dma200"])

    market_r = backtest["sp500_total_return"]
    cash_r = backtest["cash_return"]
    fund_r = backtest["synthetic_3x_fund_return_before_trading_cost"]

    backtest["buy_hold_1x_exposure"] = 1.0
    backtest["buy_hold_1x_return"] = market_r
    backtest["buy_hold_1x_wealth"] = INITIAL_CAPITAL * (1.0 + market_r).cumprod()

    strategies = [
        ("fixed_3x", "fixed_3x_exposure"),
        ("daily_band_3x", "daily_band_3x_exposure"),
        ("monthly_band_3x", "monthly_band_3x_exposure"),
    ]
    for key, exposure_col in strategies:
        result = leveraged_strategy(key, backtest[exposure_col], fund_r, cash_r)
        # The exposure column already exists in `backtest` and is deliberately
        # passed into the strategy engine.  Join only the newly calculated
        # return/wealth/turnover/cost columns to avoid a duplicate-column error.
        result = result.drop(columns=[f"{key}_exposure"])
        backtest = backtest.join(result)

    monthly_1x = unleveraged_strategy(
        "monthly_band_1x",
        backtest["monthly_band_1x_weight"],
        market_r,
        cash_r,
    )
    backtest = backtest.join(monthly_1x)

    specs = [
        ("S&P 500 Total Return 1x B&H", "buy_hold_1x_return", "buy_hold_1x_exposure"),
        ("Fixed Synthetic 3x", "fixed_3x_return", "fixed_3x_exposure"),
        (
            "Daily 200DMA +/-1% Band 3x",
            "daily_band_3x_return",
            "daily_band_3x_exposure",
        ),
        (
            "Monthly 200DMA +/-1% Band 3x",
            "monthly_band_3x_return",
            "monthly_band_3x_exposure",
        ),
        (
            "Monthly 200DMA +/-1% Band 1x",
            "monthly_band_1x_return",
            "monthly_band_1x_weight",
        ),
    ]
    summary = make_summary(backtest, specs, cash_r)
    decade_summary = make_decade_summary(backtest, specs, cash_r)

    # Keep only decisions inside the requested backtest dates.
    monthly_decisions = monthly_decisions[
        pd.to_datetime(monthly_decisions["decision_date"]) >= pd.Timestamp(START_DATE)
    ].copy()
    if END_DATE is not None:
        monthly_decisions = monthly_decisions[
            pd.to_datetime(monthly_decisions["decision_date"]) <= pd.Timestamp(END_DATE)
        ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "sp500_monthly_200dma_summary.csv"
    decade_path = OUTPUT_DIR / "sp500_monthly_200dma_decade_summary.csv"
    decisions_path = OUTPUT_DIR / "sp500_monthly_200dma_decisions.csv"
    daily_path = OUTPUT_DIR / "sp500_monthly_200dma_daily_audit.csv"
    chart_path = OUTPUT_DIR / "sp500_monthly_200dma_equity_curve.png"
    zip_path = OUTPUT_DIR / "sp500_monthly_200dma_all_outputs.zip"

    summary.to_csv(summary_path, index=False)
    decade_summary.to_csv(decade_path, index=False)
    monthly_decisions.to_csv(decisions_path, index=False)
    backtest.to_csv(daily_path, index_label="date")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    wealth_columns = [
        ("buy_hold_1x_wealth", "S&P 500 1x B&H"),
        ("fixed_3x_wealth", "Fixed synthetic 3x"),
        ("daily_band_3x_wealth", "Daily 200DMA +/-1% 3x"),
        ("monthly_band_3x_wealth", "Monthly 200DMA +/-1% 3x"),
        ("monthly_band_1x_wealth", "Monthly 200DMA +/-1% 1x"),
    ]
    for column, label in wealth_columns:
        axes[0].plot(backtest.index, backtest[column], label=label, lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_title("S&P 500: Monthly vs Daily 200DMA +/-1% Band")
    axes[0].set_ylabel("Portfolio value, log scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9, ncol=2)

    axes[1].step(
        backtest.index,
        backtest["monthly_band_3x_exposure"],
        where="post",
        label="Monthly strategy exposure",
        lw=1.2,
    )
    axes[1].scatter(
        backtest.index[backtest["month_end_decision_day"]],
        backtest.loc[backtest["month_end_decision_day"], "monthly_band_3x_exposure"],
        s=5,
        alpha=0.5,
        label="Month-end evaluation",
    )
    axes[1].set_ylim(-0.1, 3.2)
    axes[1].set_ylabel("Exposure")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.show()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [summary_path, decade_path, decisions_path, daily_path, chart_path]:
            archive.write(path, arcname=path.name)

    print("\n=== FULL-SAMPLE SUMMARY ===")
    print(formatted_table(summary).to_string(index=False))
    print("\n=== RESULTS BY CALENDAR DECADE ===")
    print(formatted_table(decade_summary).to_string(index=False))
    print("\n=== MONTHLY RULE AUDIT ===")
    print(f"Backtest: {backtest.index.min().date()} to {backtest.index.max().date()}")
    print(f"Month-end evaluations: {len(monthly_decisions):,}")
    print(
        "Actual monthly 3x exposure changes: "
        f"{backtest['monthly_band_3x_exposure'].diff().abs().gt(1e-12).sum():,}"
    )
    print("Signals are decided at month-end close and executed next session.")
    print("No intramonth trades are permitted.")
    print(f"Dividend source: {dividend_source}")
    print(f"Assumed swap spread: {SWAP_SPREAD_BPS:.1f} bps/year")
    print(f"LETF expense ratio: {LETF_EXPENSE_RATIO:.2%}/year")
    print(f"Trading cost: {TRADE_COST_BPS:.1f} bps one-way")
    print("\nCreated:")
    for path in [summary_path, decade_path, decisions_path, daily_path, chart_path, zip_path]:
        print(f"  {path}")

    if AUTO_DOWNLOAD_ZIP:
        try:
            from google.colab import files

            files.download(str(zip_path))
        except Exception:
            pass


if __name__ == "__main__":
    main()
