# DMA200 Leveraged Strategies: S&P 500 (1929+) and Nasdaq-100 (1985+)

Historical research on a **200-day moving-average (DMA200 ±1%)** trading rule, an **S&P 500 downside-volatility sell filter**, and **dynamic leverage (1x–3x)**. The S&P 500 and Nasdaq-100 are tested using **their own historical coverage**, rather than forcing both series to start in the same year.

> **Execution caveat:** All simulations assume **same-close execution**: a signal calculated using the official close is assumed to be filled at that same close. This is a deliberately idealized backtest assumption and is not a guaranteed executable live-trading strategy.

## Project Files

**The research package contains exactly 10 files, excluding this `README.md`:** two Python scripts, one historical interest-rate input, four backtest CSVs, and three bootstrap output files.

```text
.
├── README.md
├── DMA200_SameClose_Daily_Annual_Only_Colab.py
├── Annual_Bootstrap_3Tests_Colab_20261103.py
├── sp500_repaired_rate_audit.csv
├── dma_daily_annual_results/
│   ├── SP500_daily.csv
│   ├── SP500_annual.csv
│   ├── NDX_daily.csv
│   └── NDX_annual.csv
└── annual_bootstrap_results/
    ├── annual_bootstrap_pvalues.csv
    ├── annual_samples.csv
    └── bootstrap_summary.md
```

| File | Purpose |
|---|---|
| `DMA200_SameClose_Daily_Annual_Only_Colab.py` | Runs the historical backtest; writes **only four consolidated CSVs**. |
| `Annual_Bootstrap_3Tests_Colab_20261103.py` | Runs **three one-sided, calendar-year bootstrap tests** per index. |
| `sp500_repaired_rate_audit.csv` | Historical cash and financing-rate input. |
| `SP500_daily.csv`, `NDX_daily.csv` | Daily index/strategy returns, target and held leverage, trade costs, and cash returns. |
| `SP500_annual.csv`, `NDX_annual.csv` | Annual compounded returns, Sharpe ratios, within-year drawdowns, average leverage, and full-year indicators. |
| `annual_bootstrap_pvalues.csv` | Original and Holm-adjusted **one-sided** p-values, observed effects, and 95% one-sided lower bounds. |
| `annual_samples.csv` | Complete calendar-year samples and statistics used in the bootstrap. |
| `bootstrap_summary.md` | A GitHub-ready Markdown table of the three bootstrap comparisons. |

No per-strategy CSVs, correction-calibration audit CSVs, rolling-window outputs, or ZIP archives are required in this repository layout.

## Historical Coverage

| Index | Underlying history | Effective strategy backtest | Complete years used for annual-return bootstrap | Complete years used for dynamic-leverage Sharpe bootstrap |
|---|---|---|---|---|
| **S&P 500** | 1929 onward, with earlier prices for indicator warm-up | **1929-01-02 to 2026-09-25** | **1929–2025 (97 years)** | **1930–2025 (96 years)** |
| **Nasdaq-100 (NDX)** | **1985 onward** | **1986-07-16 to 2026-09-25** | **1987–2025 (39 years)** | **1988–2025 (38 years)** |

The 200-trading-day average requires warm-up observations. Consequently, **1985 is the NDX data starting year, not its first complete backtest year**. The 2026 daily observations are retained, while the incomplete 2026 calendar year is excluded from bootstrap testing. The archived end date is limited by the historical-rate input; reruns may differ if underlying data are revised.

## Trading Rules

### 1. DMA200 ±1% Entry and Exit

The moving average is the **simple average of the underlying price index's previous 200 trading-day closing values, including the current close**.

- **Buy:** When out of the market, enter if `Close >= 1.01 × DMA200`.
- **Sell:** When invested, exit if `Close <= 0.99 × DMA200`, unless the sell filter blocks the exit.
- **Otherwise:** Retain the existing invested/flat state.
- **While flat:** Hold modeled **interest-bearing cash** (0x index exposure).

For NDX, the primary DMA200 trading signal is based on **NDX prices**, not S&P 500 prices.

### 2. S&P 500 Downside-Volatility Sell Filter

This filter acts **only when the ordinary DMA200 exit would otherwise trigger**. The exit is deferred when **all three** S&P 500 conditions hold:

1. Year-to-date S&P 500 **price return** is at least **−5%**.
2. The S&P 500 price crossed its DMA200 **fewer than five times** during the previous **60 trading days**.
3. Annualized **20-day downside volatility** is below `0.25 / sqrt(2)` (approximately **17.68%**).

Both the S&P 500 and NDX filtered strategies use this **same S&P 500-based sell filter**. Their ordinary entry/exit signals remain index-specific.

### 3. Fixed and Dynamic Leverage

The fixed-leverage variant uses **3x whenever invested**. The dynamic variant (`blend_1y_2y`) determines its leverage using a trailing 20-trading-day average annualized financing rate and returns from **completed previous calendar years**.

| Trailing financing-rate average | Starting leverage |
|---|---:|
| Below 3% | 3x |
| 3% to 6% | 2x |
| Above 6% | 1x |

The historical-return signal combines the **previous calendar year's total return** and the **annualized return over the previous two complete years**:

```text
signal = (previous_1_year_return + previous_2_year_annualized_return) / 2
```

- If the signal is **negative**, add **1x** to starting leverage.
- If the signal is **above 15%**, subtract **0.5x**.
- Otherwise, do not adjust starting leverage.
- Clip invested leverage to **1x–3x**; set actual index exposure to **0x while in cash**.

### Four Return Series in Each Daily / Annual CSV

| Column prefix | Meaning |
|---|---|
| `buyhold_1x` | Unleveraged underlying index total-return buy-and-hold. |
| `dma_unfiltered_3x` | DMA200 ±1%, fixed 3x, **without** the sell filter. |
| `dma_filtered_3x` | DMA200 ±1%, fixed 3x, **with** the sell filter. |
| `dma_filtered_dynamic` | DMA200 ±1%, **dynamic 1x–3x**, with the same sell filter. |

In the daily files, the return columns end in `_return`. Each active strategy also has `_target_leverage_at_close`, `_held_leverage_for_return`, and `_trade_cost` columns. **Cash days count as 0x** when calculating average leverage across *all* trading days.

## Full-History Backtest Results

The tables below are computed from the four consolidated backtest CSVs, using **each index's complete available backtest**, ending **September 25, 2026**. They are **synthetic, pre-tax results**, net of the model's financing, calibrated ETF adjustment, and simplified trading costs. Because the indexes have different starting dates, these are **not matched-period index comparisons**.

### S&P 500 — 1929-01-02 to 2026-09-25

| Metric | Dynamic leverage + filter | Fixed 3x + filter |
|---|---:|---:|
| Average leverage while invested | **2.22x** | 3.00x |
| Average leverage across **all** trading days, including cash | **1.55x** | 2.09x |
| CAGR | 19.00% | **20.68%** |
| Annualized excess-return Sharpe ratio | **0.6171** | 0.6061 |
| Maximum drawdown | **−87.68%** | −90.16% |

### Nasdaq-100 — 1986-07-16 to 2026-09-25

| Metric | Dynamic leverage + filter | Fixed 3x + filter |
|---|---:|---:|
| Average leverage while invested | **2.16x** | 3.00x |
| Average leverage across **all** trading days, including cash | **1.72x** | 2.39x |
| CAGR | 24.56% | **24.63%** |
| Annualized excess-return Sharpe ratio | **0.6815** | 0.6212 |
| Maximum drawdown | **−82.92%** | −93.91% |

The fixed-3x strategy has a slightly higher full-period CAGR in both markets. Dynamic leverage has a higher historical Sharpe ratio and smaller maximum drawdown over the shown full-history periods; **statistical significance of the Sharpe difference is assessed separately below**.

## Annual Bootstrap: Three One-Sided Tests

**One complete calendar year = one bootstrap observation.** The script samples entire years with replacement, using the **same resampled years for both strategies in each comparison**. The default one-year block size does not independently resample individual trading days or create rolling ten-year samples.

**Alternative hypothesis for each comparison:** `H1: left-hand strategy > right-hand strategy` (against `H0: difference <= 0`).

| Comparison | S&P 500 one-sided p-value | NDX one-sided p-value |
|---|---:|---:|
| **DMA200 fixed 3x > 1x buy-and-hold** — mean annual compounded return | **0.000100** | **0.029349** |
| **Filtered fixed 3x > unfiltered fixed 3x** — mean annual compounded return | **0.000100** | **0.048198** |
| **Filtered dynamic leverage > filtered fixed 3x** — pooled Sharpe ratio | 0.419429 | 0.108195 |

These are **raw one-sided p-values** from the illustrated archived run with **20,000 resamples, one-year blocks**, and the script's fixed seed. They are not the Holm-adjusted p-values. In particular, the script **does not test DMA200 fixed 3x against 3x buy-and-hold**.

| Comparison (same row order) | S&P 500 Holm-adjusted p | NDX Holm-adjusted p |
|---|---:|---:|
| DMA200 vs 1x buy-and-hold | **0.000300** | 0.088046 |
| Filtered vs unfiltered fixed 3x | **0.000300** | 0.096395 |
| Filtered dynamic vs filtered fixed 3x (Sharpe) | 0.419429 | 0.108195 |

- **Return tests:** Compare the average of each year's **compounded annual return difference**, *not* the full-period CAGR difference.
- **Sharpe test:** Sample **whole years of daily excess-return data**, then recompute the **pooled annualized Sharpe difference** for each bootstrap draw. The two strategies both use the sell filter; **only leverage differs**.
- **Null distribution:** A centered bootstrap tests a zero-effect null. `annual_bootstrap_pvalues.csv` reports the observed difference, raw p-value, one-sided 95% lower bound, and Holm-adjusted p-value (for the three within-index tests).
- **Serial dependence:** Adjacent years may be dependent. The option `--year-block 3` resamples contiguous three-year blocks as a sensitivity check.
- **Scope of evidence:** The historical strategy and parameter-search process is **not** fully corrected for data snooping. Statistical evidence from these reconstructed historical records does not establish live-trading performance.

## Costs, Data, and Methodology

| Item | Historical source or implementation |
|---|---|
| S&P 500 prices | Yahoo Finance `^GSPC`; historical dividend yields from Multpl, with DataHub/Shiller fallback. |
| S&P 500 total return | Reconstructed using historical dividend-yield estimates rather than exact historical dividend event dates. |
| Nasdaq-100 prices | Yahoo Finance `^NDX`; history begins in 1985. |
| NDX total return | Historical dividend-yield estimates before 1999; FRED `NASDAQXNDX` when available thereafter, with QQQ dividend-based proxy fallback. |
| Cash and financing rates | `sp500_repaired_rate_audit.csv` (`cash_new` and `rate_annual_investment`). |
| Leveraged ETF calibration | Observed adjusted historical **UPRO/TQQQ** returns are used to estimate an index-specific **all-in correction** to a synthetic 3x daily-return model. This includes ETF fund expenses; **do not deduct the management fee again**. |
| Transaction costs | Simplified modeled charge of `0.05% × abs(change in target leverage) / 3` on leveraged transitions. |
| Taxes and real-world execution | **Not included:** capital-gains tax and actual, potentially variable execution slippage, market impact, and broker-specific terms. |

**Historical limitation:** UPRO and TQQQ were launched much later than 1929/1985. Applying a correction calibrated using their subsequent trading histories to earlier decades is **retrospective calibration**, not a point-in-time investable ETF backtest. Early dividend data and some NDX dividend estimates are also approximations.

### Same-Close Accounting

At close `t`, the strategy observes the close, DMA200, and sell-filter data, and assumes it can execute the new target at **that same close**. The position held during the return ending at close `t` was determined at the preceding close. Trading costs for changes made at close `t` are booked at `t`. The resulting returns are therefore **idealized** rather than directly tradable at a known final closing price.

## Reproduce in Google Colab

Upload these **three input files** to `/content/`:

```text
DMA200_SameClose_Daily_Annual_Only_Colab.py
Annual_Bootstrap_3Tests_Colab_20261103.py
sp500_repaired_rate_audit.csv
```

### Step 1 — Backtest (creates four CSVs)

```python
%run /content/DMA200_SameClose_Daily_Annual_Only_Colab.py --rates /content/sp500_repaired_rate_audit.csv --out /content/dma_daily_annual_results
```

This step downloads the necessary public historical series and saves only:

```text
dma_daily_annual_results/
    SP500_daily.csv
    SP500_annual.csv
    NDX_daily.csv
    NDX_annual.csv
```

### Step 2 — Annual, One-Sided Bootstrap (creates three files)

```python
%run /content/Annual_Bootstrap_3Tests_Colab_20261103.py --results /content/dma_daily_annual_results --out /content/annual_bootstrap_results --reps 20000 --year-block 1
```

Output:

```text
annual_bootstrap_results/
    annual_bootstrap_pvalues.csv
    annual_samples.csv
    bootstrap_summary.md
```

To download the bootstrap outputs directly in Colab, append `--download` to Step 2. For a three-year-block sensitivity test, change `--year-block 1` to `--year-block 3` (preferably saving the results to a **separate** output folder if preserving both sets).

The bootstrap script reads **the four-file consolidated format directly**. You do not need to generate separate daily CSVs for individual strategies.

## Disclaimer

This repository documents **historical, educational quantitative research**, not investment advice. Daily leveraged exposure can produce extreme losses, and synthetic historical returns depend on reconstruction choices, costs, simulated execution, and leverage assumptions. Past performance does not predict future results.

## AI Assistance
This project was developed with assistance from OpenAI Codex and ChatGPT.
