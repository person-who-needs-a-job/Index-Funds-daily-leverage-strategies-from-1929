# QuantConnect LEAN: Synthetic SP500 DMA200 triple sell-filter from 1929.
# Copy the ENTIRE file into main.py (Ctrl+A, then paste).
# This is synthetic simulated equity, NOT QuantConnect trading orders.
# The custom chart and runtime statistics report modeled performance;
# platform default portfolio CAGR reflects idle starting cash.

from AlgorithmImports import *
from collections import deque
from datetime import datetime, timedelta
from bisect import bisect_right
from io import StringIO
import csv
import math

# Public S&P 500 price index, 1927-12-30 through 2019-12-23.
OLD_URL = (
    "https://raw.githubusercontent.com/fja05680/"
    "dow-sp500-100-years/master/SP500.csv"
)
# FRED SP500 fills the post-2019 period (FRED carries ~10 years).
NEW_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
SHILLER_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500/master/data/data.csv"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


class LegacySPX(PythonData):
    def get_source(self, config, date, is_live_mode):
        return SubscriptionDataSource(OLD_URL, SubscriptionTransportMedium.REMOTE_FILE)

    def reader(self, config, line, date, is_live_mode):
        if not line or not line[0].isdigit():
            return None
        try:
            a = line.strip().split(",")
            x = LegacySPX()
            x.symbol = config.symbol
            x.time = datetime.strptime(a[0], "%Y-%m-%d")
            x.end_time = x.time + timedelta(days=1)
            x.value = float(a[4])  # price-index CLOSE; not total return
            return x if x.value > 0 else None
        except (ValueError, IndexError):
            return None


class RecentSPX(PythonData):
    def get_source(self, config, date, is_live_mode):
        return SubscriptionDataSource(NEW_URL, SubscriptionTransportMedium.REMOTE_FILE)

    def reader(self, config, line, date, is_live_mode):
        if not line or not line[0].isdigit():
            return None
        try:
            a = line.strip().split(",")
            x = RecentSPX()
            x.symbol = config.symbol
            x.time = datetime.strptime(a[0], "%Y-%m-%d")
            x.end_time = x.time + timedelta(days=1)
            x.value = float(a[1])
            return x if x.value > 0 else None
        except (ValueError, IndexError):
            return None  # missing / holiday rows marked '.'


# Helpers are module-level functions (not class methods), to avoid
# issues caused by accidentally truncating/indent-changing a paste.
def load_rates_from_fred(algo):
    # NBER short US rate: 1920-1934, monthly.
    # Treasury bill: 1934-1954, monthly.
    # Effective fed funds: 1954 onward, daily.
    # These are approximations to the old repaired-rate file, NOT identical.
    rates = {}
    for series, monthly in [
        ("M1329AUSM193NNBR", True),
        ("TB3MS", True),
        ("DFF", False),
    ]:
        raw = algo.download(FRED_URL + series)
        if not raw or len(raw) < 50:
            raise RuntimeError("FRED rate data unavailable: " + series)
        for row in csv.DictReader(StringIO(raw.lstrip("\ufeff"))):
            key = row.get("DATE") or row.get("observation_date")
            val = row.get(series)
            if not key or not val or val.strip() in (".", ""):
                continue
            try:
                d = datetime.strptime(key[:10], "%Y-%m-%d")
                r = float(val) / 100.0
            except ValueError:
                continue
            if monthly:
                # Monthly averages are observable only after month end.
                d = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
            # Newer series overrides older series on an identical date.
            rates[d] = r
    dates = sorted(rates)
    if not dates or dates[0] > algo.START_TRADING:
        raise RuntimeError("No usable rate history for 1929")
    return dates, [rates[d] for d in dates]


def load_dividend_yields(algo):
    # Shiller monthly 'Dividend' is an annualized index-point amount.
    # Use previous completed month's dividend/price yield, daily accrual.
    values = {}
    raw = algo.download(SHILLER_URL)
    if not raw or len(raw) < 200:
        raise RuntimeError("Monthly Shiller dividend data unavailable")
    for row in csv.DictReader(StringIO(raw.lstrip("\ufeff"))):
        try:
            d = datetime.strptime(row["Date"][:10], "%Y-%m-%d")
            p = float(row["SP500"])
            div = float(row["Dividend"])
            if p <= 0 or div <= 0:
                continue  # post-2023 zero means MISSING, not zero dividend
            eff = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
            values[eff] = div / p
        except (KeyError, ValueError):
            continue
    dates = sorted(values)
    if not dates or dates[0] > algo.START_TRADING:
        raise RuntimeError("No Shiller dividends for 1929")
    return dates, [values[d] for d in dates]


class DMA200Since1929(QCAlgorithm):
    # This is a HISTORICAL SYNTHETIC P&L study, NOT tradable securities.
    # Use CUSTOM CHART / runtime statistics, not built-in Portfolio CAGR.
    COST_PER_3X_SWITCH = 0.0005  # 5 basis points per 0x<->3x switch
    ETF_RESIDUAL_ANNUAL = 0.016  # illustrative; replace with calibrated cost
    START_TRADING = datetime(1929, 1, 2)
    LEGACY_LAST = datetime(2019, 12, 23)

    def initialize(self):
        # 1928 is used only to warm up the 200-day average and filter.
        self.set_start_date(1928, 1, 1)
        self.set_end_date(2026, 11, 4)
        self.set_cash(100000)

        self.old = self.add_data(
            LegacySPX, "HIST_SPX", Resolution.DAILY,
            fill_forward=False
        ).symbol
        self.new = self.add_data(
            RecentSPX, "FRED_SPX", Resolution.DAILY,
            fill_forward=False
        ).symbol

        # Macro inputs are retrieved before running any synthetic daily returns.
        self.rate_dates, self.rate_values = load_rates_from_fred(self)
        self.div_dates, self.div_yields = load_dividend_yields(self)

        self.prices = deque(maxlen=200)
        self.crosses = deque(maxlen=60)
        self.downsq = deque(maxlen=20)
        self.last_price = None
        self.last_gap = None
        self.last_date = None
        self.last_year_close = None
        self.year = None
        self.is_in = False
        self.last_target = 0

        self.nav = 100000.0
        self.benchmark_nav = 100000.0
        self.peak = self.nav
        self.max_dd = 0.0
        self.rets = []
        self.excess = []
        self.days = 0
        self.trades = 0
        self.ignored = 0
        self.first_day = None
        self.last_day = None

        self.debug("1929 SYNTHETIC research: read CUSTOM equity chart. "
                   "QC built-in Portfolio statistics will show cash, not model NAV.")



    @staticmethod
    def lookup(dates, values, d):
        i = bisect_right(dates, d) - 1
        return values[i] if i >= 0 else None

    def on_data(self, data: Slice):
        legacy = data.get(self.old)
        recent = data.get(self.new)

        # Prefer the original 1927-2019 daily file; use FRED after its end.
        record = None
        if legacy is not None and legacy.time <= self.LEGACY_LAST:
            record = legacy
        elif recent is not None and recent.time > self.LEGACY_LAST:
            record = recent
        if record is None:
            return

        d = record.time
        p = float(record.value)
        if p <= 0 or (self.last_date is not None and d <= self.last_date):
            return

        if self.year is None:
            self.year = d.year
        elif d.year != self.year:
            self.last_year_close = self.last_price
            self.year = d.year

        pxret = None
        if self.last_price is not None:
            pxret = p / self.last_price - 1.0
            self.downsq.append(min(pxret, 0.0)**2)

        self.prices.append(p)
        self.last_price = p
        self.last_date = d
        if len(self.prices) < 200 or pxret is None:
            return

        dma = sum(self.prices) / 200.0
        gap = p - dma
        crossed = int(self.last_gap is not None and gap * self.last_gap < 0)
        self.crosses.append(crossed)
        self.last_gap = gap
        if len(self.downsq) < 20:
            return

        ytd = None if self.last_year_close is None else p / self.last_year_close - 1.0
        downside = math.sqrt(sum(self.downsq) * 252.0 / 20.0)

        # 1928 signals establish the position held into January 1929;
        # 1928 returns are excluded from the performance window.
        if d < self.START_TRADING:
            self.update_signal(p, dma, ytd, downside)
            self.last_target = 3 if self.is_in else 0
            return

        rate = self.lookup(self.rate_dates, self.rate_values, d)
        dividend_yield = self.lookup(self.div_dates, self.div_yields, d)
        if rate is None or dividend_yield is None:
            raise RuntimeError("Missing historical rate/dividend at " + str(d))

        # PREVIOUS CLOSE target earns today's close-to-close return.
        total_ret = pxret + dividend_yield / 252.0
        rf = rate / 252.0
        if self.last_target == 3:
            daily = 3.0 * total_ret - 2.0 * rf - self.ETF_RESIDUAL_ANNUAL / 252.0
        else:
            daily = rf
        self.benchmark_nav *= 1.0 + total_ret

        self.update_signal(p, dma, ytd, downside)
        target = 3 if self.is_in else 0
        if target != self.last_target:
            daily -= self.COST_PER_3X_SWITCH
            self.trades += 1
            if self.trades <= 12:
                self.debug(f"{d:%Y-%m-%d}: target {self.last_target}x -> {target}x; "
                           f"SPX={p:.2f} DMA200={dma:.2f}")
        self.last_target = target

        if daily <= -1.0:
            raise RuntimeError("Synthetic 3x NAV is insolvent at " + str(d))
        self.nav *= (1.0 + daily)
        self.peak = max(self.peak, self.nav)
        self.max_dd = min(self.max_dd, self.nav / self.peak - 1.0)
        self.rets.append(daily)
        self.excess.append(daily - rf)
        self.days += 1
        self.last_day = d
        if self.first_day is None:
            self.first_day = d

        # About one point per month avoids free-tier chart quotas.
        if self.days % 21 == 0:
            self.plot("1929 Synthetic Equity (log10)", "DMA200 Filter 3x", math.log10(self.nav / 100000.0))
            self.plot("1929 Synthetic Equity (log10)", "SP500 BuyHold 1x", math.log10(self.benchmark_nav / 100000.0))
            self.plot("1929 Synthetic Drawdown", "Drawdown %", 100.0 * (self.nav / self.peak - 1.0))

    def update_signal(self, p, dma, ytd, downside):
        if not self.is_in and p >= dma * 1.01:
            self.is_in = True
        elif self.is_in and p <= dma * 0.99:
            skip_sell = (
                ytd is not None and ytd >= -0.05
                and len(self.crosses) == 60
                and sum(self.crosses) < 5
                and downside < 0.25 / math.sqrt(2.0)
            )
            if skip_sell:
                self.ignored += 1
            else:
                self.is_in = False

    def on_end_of_algorithm(self):
        if self.first_day is None or self.last_day is None:
            self.debug("ERROR: No 1929+ price observations processed. Check RemoteFile downloads.")
            return
        years = (self.last_day - self.first_day).days / 365.2425
        cagr = (self.nav / 100000.0)**(1.0 / years) - 1.0
        bcagr = (self.benchmark_nav / 100000.0)**(1.0 / years) - 1.0
        n = len(self.excess)
        mean = sum(self.excess) / n
        std = math.sqrt(sum((v - mean)**2 for v in self.excess) / (n - 1)) if n > 1 else 0
        sharpe = math.sqrt(252.0) * mean / std if std > 0 else float("nan")
        self.set_runtime_statistic("Synthetic CAGR", f"{cagr:.2%}")
        self.set_runtime_statistic("Synthetic Sharpe", f"{sharpe:.3f}")
        self.set_runtime_statistic("Synthetic MaxDD", f"{self.max_dd:.2%}")
        self.set_runtime_statistic("1x benchmark CAGR", f"{bcagr:.2%}")
        self.set_runtime_statistic("Switches", str(self.trades))
        self.debug(f"SYNTHETIC {self.first_day:%Y-%m-%d} to {self.last_day:%Y-%m-%d}: "
                   f"CAGR={cagr:.2%}, Sharpe={sharpe:.3f}, "
                   f"MaxDD={self.max_dd:.2%}, BenchmarkCAGR={bcagr:.2%}, "
                   f"Switches={self.trades}, IgnoredSellDays={self.ignored}")
        self.debug("NOTE: built-in QC portfolio statistics are NOT synthetic model statistics.")
