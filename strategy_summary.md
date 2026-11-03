# Historical DMA200 Strategy Performance Summary

Computed **only from** the consolidated `SP500_daily.csv`, `SP500_annual.csv`, `NDX_daily.csv`, and `NDX_annual.csv` files. No bootstrap tests or p-values.

## S&P 500: 1929-01-02 to 2026-09-25

Includes all 24,550 trading days (97.73 elapsed years), plus 97 complete calendar years for annual statistics.

| Strategy | CAGR | Sharpe | Annual vol. | Max drawdown | Mean leverage (all days) | Mean leverage (invested) | Invested days |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x Buy & Hold | 9.84% | 0.416 | 18.94% | -83.85% | 1.00x | 1.00x | 100.00% |
| DMA200 Fixed 3x (Unfiltered) | 17.94% | 0.547 | 36.44% | -90.16% | 2.02x | 3.00x | 67.47% |
| DMA200 Fixed 3x + Sell Filter | 20.68% | 0.606 | 37.02% | -90.16% | 2.09x | 3.00x | 69.78% |
| DMA200 Dynamic Leverage + Sell Filter | 19.00% | 0.617 | 30.43% | -87.68% | 1.55x | 2.22x | 69.78% |

| Strategy | Mean annual return* | Positive years* | Best year* | Worst year* | Sortino | Calmar |
|---|---:|---:|---|---|---:|---:|
| 1x Buy & Hold | 11.62% | 73.20% | 1954 (52.44%) | 1931 (-43.63%) | 0.590 | 0.117 |
| DMA200 Fixed 3x (Unfiltered) | 26.80% | 68.04% | 1954 (229.42%) | 1932 (-69.75%) | 0.770 | 0.199 |
| DMA200 Fixed 3x + Sell Filter | 29.80% | 72.16% | 1954 (229.42%) | 1932 (-69.75%) | 0.856 | 0.229 |
| DMA200 Dynamic Leverage + Sell Filter | 25.66% | 73.20% | 1954 (228.46%) | 1932 (-69.75%) | 0.876 | 0.217 |

## Nasdaq-100 (NDX): 1986-07-16 to 2026-09-25

Includes all 10,127 trading days (40.20 elapsed years), plus 39 complete calendar years for annual statistics.

| Strategy | CAGR | Sharpe | Annual vol. | Max drawdown | Mean leverage (all days) | Mean leverage (invested) | Invested days |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x Buy & Hold | 14.87% | 0.544 | 26.01% | -82.90% | 1.00x | 1.00x | 100.00% |
| DMA200 Fixed 3x (Unfiltered) | 20.41% | 0.560 | 54.26% | -90.77% | 2.26x | 3.00x | 75.21% |
| DMA200 Fixed 3x + Sell Filter | 24.63% | 0.621 | 55.96% | -93.91% | 2.39x | 3.00x | 79.80% |
| DMA200 Dynamic Leverage + Sell Filter | 24.56% | 0.681 | 38.88% | -82.92% | 1.72x | 2.16x | 79.80% |

| Strategy | Mean annual return* | Positive years* | Best year* | Worst year* | Sortino | Calmar |
|---|---:|---:|---|---|---:|---:|
| 1x Buy & Hold | 19.25% | 82.05% | 1999 (101.96%) | 2008 (-41.68%) | 0.780 | 0.179 |
| DMA200 Fixed 3x (Unfiltered) | 40.53% | 61.54% | 1999 (422.38%) | 2000 (-65.18%) | 0.786 | 0.225 |
| DMA200 Fixed 3x + Sell Filter | 46.13% | 66.67% | 1999 (422.38%) | 2000 (-68.20%) | 0.875 | 0.262 |
| DMA200 Dynamic Leverage + Sell Filter | 35.36% | 66.67% | 1999 (167.10%) | 2002 (-33.58%) | 0.960 | 0.296 |

*Annual-return statistics use only complete calendar years. The CAGR, Sharpe, drawdown and leverage metrics use the **entire available daily record**, including any partial start/end years.*

**Definitions:** CAGR uses actual elapsed calendar time; Sharpe is annualized mean daily return in excess of the modeled cash return divided by daily excess-return volatility (×√252). Sortino uses downside deviation of daily excess returns; maximum drawdown uses daily compounded wealth with a starting wealth of 1. Average leverage counts cash days as 0x; 1x buy-and-hold is treated as continuously invested at 1x.

**Caveats:** Returns are synthetic, pre-tax, and already incorporate the fee/financing/trading-cost assumptions of the supplied CSVs. Execution is the idealized same-close convention. Different indexes have different historical starting dates; this is not a same-period comparison.
