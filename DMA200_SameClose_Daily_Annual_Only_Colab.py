#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY+ANNUAL ONLY S&P 500 (1929+) and Nasdaq-100 (1985+) SAME-CLOSE research backtest.

Single-file, Google Colab compatible. Only additional local input is
sp500_repaired_rate_audit.csv (repaired historical financing and cash rates).

Legacy methods used verbatim:
  * S&P 500 ^GSPC prices from Yahoo v8 chart API; Multpl monthly dividend-yield
    data, with DataHub/Shiller fallback, spread smoothly over trading days.
  * Nasdaq-100 ^NDX prices from Yahoo v8 chart API; SEC/Nasdaq annual historical
    dividend-yield table before XNDX; thereafter FRED total-return index if
    available, otherwise QQQ dividend-adjustment proxy.
  * Real UPRO and TQQQ adjusted NAV/price returns calibrate an index-specific
    annual all-in synthetic tracking/funding correction. No fixed 1.6% term.
  * ONLY same-close trading: close[t] signals the TARGET at close[t]; that target
    earns returns over close[t] -> close[t+1]. Transaction cost at close[t].


This version produces EXACTLY four files: SP500_daily.csv, SP500_annual.csv,
NDX_daily.csv, NDX_annual.csv. Each file includes 1x buy-and-hold, unfiltered
DMA200 fixed 3x, filtered DMA200 fixed 3x, and filtered DMA200 dynamic leverage.
All intermediate dividend/correction audits stay in a temporary directory.

To reuse an older same-close backtest without any downloads, pass:
  --from-existing /content/total_return_backtest_results

For annual THREE-test bootstrap on these consolidated files, use the accompanying
DMA200_Annual_OneSided_3Tests_DailyAnnual_Colab.py.

WARNING: exact same-close fills from a signal using the final official close are
idealized and not directly executable without additional assumptions.
Pre-ETF correction extrapolation is EX-POST, not a point-in-time historical cost.
Daily dividend reconstructions for early history are approximations.
"""
from __future__ import annotations
import argparse
import tempfile
import io
import json
import time
import requests
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

START_DATE='1928-01-01'
END_DATE=None
NDX_LAUNCH_DATE='1985-01-31'
YAHOO_SP500_URLS=[
  'https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC',
  'https://query2.finance.yahoo.com/v8/finance/chart/%5EGSPC',
]
YAHOO_NDX_URLS=[
  'https://query1.finance.yahoo.com/v8/finance/chart/%5ENDX',
  'https://query2.finance.yahoo.com/v8/finance/chart/%5ENDX',
]
MULTPL_DIVIDEND_YIELD='https://www.multpl.com/s-p-500-dividend-yield/table/by-month'
DATAHUB_SP500_CSV='https://datahub.io/core/s-and-p-500/_r/-/data/data.csv'
PRE_XNDX_ANNUAL_DIVIDEND_YIELD_PCT={
    1985:0.00,1986:0.33,1987:0.41,1988:0.47,1989:0.91,
    1990:1.07,1991:0.53,1992:0.55,1993:0.52,1994:0.46,
    1995:0.26,1996:0.11,1997:0.13,1998:0.07,1999:0.03,
}

# Exact historical acquisition methods transcribed from user's original scripts.

def get_bytes(url: str, label: str, attempts: int = 3, timeout: int = 20) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 ColabBacktest/1.0"}
    last_error = None
    print(f"  -> {label}...", flush=True)
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            print(f"     completed: {label}", flush=True)
            return r.content
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                print(f"     retry {attempt + 2}/{attempts}: {label}", flush=True)
                time.sleep(1.0)
    raise RuntimeError(f"Download failed: {label}\nURL: {url}\n{last_error}")


def download_sp500_price() -> pd.Series:
    # Request enough pre-1928 history to warm up DMA200.
    warmup = pd.Timestamp(START_DATE) - pd.Timedelta(days=800)
    ending = (pd.Timestamp(END_DATE) if END_DATE else pd.Timestamp.today().normalize()) + pd.Timedelta(days=1)
    query = (
        f"?period1={int(warmup.timestamp())}"
        f"&period2={int(ending.timestamp())}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )

    errors = []
    for n, base in enumerate(YAHOO_SP500_URLS, start=1):
        try:
            raw = get_bytes(base + query, f"Yahoo S&P 500 daily close (server {n})")
            payload = json.loads(raw.decode("utf-8"))
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise RuntimeError("Yahoo returned no chart result")

            ts = result.get("timestamp") or []
            quotes = result.get("indicators", {}).get("quote") or []
            closes = quotes[0].get("close", []) if quotes else []
            if not ts or len(ts) != len(closes):
                raise RuntimeError("Yahoo timestamps/closes missing or mismatched")

            dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
            vals = pd.to_numeric(pd.Series(closes), errors="coerce")
            s = pd.Series(vals.to_numpy(), index=dates, name="sp500_price").dropna()
            s = s[s > 0]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) < 5_000:
                raise RuntimeError(f"Only {len(s):,} usable observations returned")
            return s
        except Exception as exc:
            errors.append(f"server {n}: {exc}")
            print(f"     server {n} failed; switching server...", flush=True)

    raise RuntimeError("Both Yahoo servers failed:\n" + "\n".join(errors))


def parse_multpl_dividend_yield(raw: bytes) -> pd.Series:
    tables = pd.read_html(io.StringIO(raw.decode("utf-8", errors="replace")))
    for table in tables:
        table.columns = [str(c).strip() for c in table.columns]
        if {"Date", "Value"}.issubset(table.columns):
            dates = pd.to_datetime(table["Date"], errors="coerce")
            vals = table["Value"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            vals = pd.to_numeric(vals, errors="coerce") / 100.0
            s = pd.Series(vals.to_numpy(), index=dates, name="dividend_yield").dropna()
            s = s[(s >= 0) & (s <= 0.25)]
            if len(s) >= 100:
                return s[~s.index.duplicated(keep="last")].sort_index()
    raise RuntimeError("No usable Multpl dividend table found")


def download_datahub_dividend_yield() -> pd.Series:
    raw = get_bytes(DATAHUB_SP500_CSV, "DataHub/Shiller dividend backup")
    frame = pd.read_csv(io.BytesIO(raw))
    required = {"Date", "SP500", "Dividend"}
    if not required.issubset(frame.columns):
        raise RuntimeError("DataHub dividend file has unexpected schema")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    prices = pd.to_numeric(frame["SP500"], errors="coerce")
    dividends = pd.to_numeric(frame["Dividend"], errors="coerce")
    yields = dividends / prices
    s = pd.Series(yields.to_numpy(), index=dates, name="dividend_yield").dropna()
    s = s[(s > 0) & (s <= 0.25)]
    if s.empty:
        raise RuntimeError("DataHub backup contains no usable dividend observations")
    return s[~s.index.duplicated(keep="last")].sort_index()


def download_dividend_yield() -> tuple[pd.Series, str]:
    try:
        raw = get_bytes(MULTPL_DIVIDEND_YIELD, "Multpl S&P 500 dividend yield")
        return parse_multpl_dividend_yield(raw), "Multpl monthly dividend yield"
    except Exception as exc:
        print(f"     Multpl unavailable: {exc}", flush=True)
        return download_datahub_dividend_yield(), "DataHub/Shiller dividend-yield backup"


def download_ndx_price() -> pd.Series:
    start = pd.Timestamp(NDX_LAUNCH_DATE)
    ending = (pd.Timestamp(END_DATE) if END_DATE else pd.Timestamp.today().normalize()) + pd.Timedelta(days=1)
    query = (
        f"?period1={int(start.timestamp())}"
        f"&period2={int(ending.timestamp())}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )

    errors = []
    for n, base in enumerate(YAHOO_NDX_URLS, start=1):
        try:
            raw = get_bytes(base + query, f"Yahoo Nasdaq-100 daily close (server {n})")
            payload = json.loads(raw.decode("utf-8"))
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise RuntimeError("Yahoo returned no chart result")

            ts = result.get("timestamp") or []
            quotes = result.get("indicators", {}).get("quote") or []
            closes = quotes[0].get("close", []) if quotes else []
            if not ts or len(ts) != len(closes):
                raise RuntimeError("Yahoo timestamps/closes missing or mismatched")

            dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
            vals = pd.to_numeric(pd.Series(closes), errors="coerce")
            s = pd.Series(vals.to_numpy(), index=dates, name="ndx_price").dropna()
            s = s[s > 0]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) < 5_000:
                raise RuntimeError(f"Only {len(s):,} usable NDX observations returned")
            return s.loc[pd.Timestamp(NDX_LAUNCH_DATE):]
        except Exception as exc:
            errors.append(f"server {n}: {exc}")
            print(f"     server {n} failed; switching server...", flush=True)

    raise RuntimeError("Both Yahoo NDX servers failed:\n" + "\n".join(errors))



TD=252
COST=0.0005
FEE3=0.0095
# No fixed 1.6% drag: estimate the all-in 3x correction independently for UPRO/TQQQ.
# FEE3 is an *illustrative accounting split* only; the measured all-in term
# already includes the actual ETF's fees (avoid subtracting FEE3 twice).
DSVOL_THRESHOLD=.25/np.sqrt(2)
FILTERS=('none','own_vol25','own_downside','sp_downside',
         'both_downside','sp_downside_own_guard')
LEVERAGES=('fixed3','rate_only','prior_1y','prior_2y','blend_1y_2y',
           'rolling_252','rolling_504')
DEFENSES=('cash','sp500','treasury','rotation')


def fetch_yahoo(symbol,start,output,adjusted=False,optional=False):
    """Yahoo CHART API, same v8 query1/query2 JSON method as old 1928/1985 scripts.

    For ETFs/price proxies the adjustedclose field is the dividend/split-adjusted
    price. For ^GSPC and ^NDX, the original old scripts are used instead.
    """
    import json
    import urllib.parse
    start_dt=pd.Timestamp(start)
    end_dt=pd.Timestamp.today().normalize()+pd.Timedelta(days=2)
    symbol_encoded=urllib.parse.quote(symbol,safe='')
    query=(f'?period1={int(start_dt.timestamp())}'
           f'&period2={int(end_dt.timestamp())}'
           '&interval=1d&events=history&includeAdjustedClose=true')
    errors=[]
    for server in ('query1.finance.yahoo.com','query2.finance.yahoo.com'):
        try:
            url=f'https://{server}/v8/finance/chart/{symbol_encoded}{query}'
            raw=get_bytes(url,f'Yahoo {symbol} ({server})',attempts=2,timeout=18)
            payload=json.loads(raw.decode('utf-8'))
            chart=payload.get('chart',{})
            if chart.get('error'):raise RuntimeError(str(chart['error']))
            obj=(chart.get('result') or [None])[0]
            if not obj:raise RuntimeError('Yahoo result missing')
            ts=obj.get('timestamp') or []
            ind=obj.get('indicators',{})
            q=(ind.get('quote') or [{}])[0]
            a=(ind.get('adjclose') or [{}])[0]
            vals=a.get('adjclose',[]) if adjusted else q.get('close',[])
            if not ts or len(ts)!=len(vals):
                raise ValueError(f'Yahoo {symbol} {"adjusted" if adjusted else "close"} missing/misaligned')
            dates=pd.to_datetime(ts,unit='s',utc=True).tz_convert(None).normalize()
            s=pd.Series(pd.to_numeric(pd.Series(vals),errors='coerce').to_numpy(),index=dates,dtype=float)
            s=s.loc[(s>0)&s.notna()]
            s=s.loc[~s.index.duplicated(keep='last')].sort_index()
            if len(s)<100:raise ValueError(f'Only {len(s)} valid quotes')
            s.name=symbol
            safe=symbol.strip('^').replace('/','_')
            s.to_csv(output/f'yahoo_{safe}_{"adjusted" if adjusted else "price"}.csv',header=True,index_label='date')
            print(f'Yahoo chart API {symbol}: {s.index[0].date()}—{s.index[-1].date()} ({len(s):,} rows)')
            return s
        except Exception as exc:
            errors.append(f'{server}: {str(exc)[:200]}')
    if optional:
        print(f'NOTICE: Yahoo {symbol} unavailable, optional; {" | ".join(errors)}')
        return pd.Series(dtype=float,name=symbol)
    raise RuntimeError(f'Could not fetch real Yahoo {symbol} adjusted/close data: {errors}')


def load_legacy_sources(out):
    """Use the original scripts' actual downloader and dividend construction."""
    print('Downloading exactly as in the original S&P 500 1928 and NDX 1985 scripts...')
    sp_price=download_sp500_price()
    # Historical Yahoo price, independent of ^SP500TR availability.
    sp_monthly,sp_div_source=download_dividend_yield()
    sp_yield=sp_monthly.reindex(sp_price.index,method='ffill')
    if sp_yield.isna().any():
        raise RuntimeError('Historical S&P 500 dividend-yield series does not cover full price history')
    sp_daily_div=(1+sp_yield).pow(1/252)-1
    sp_ret=(1+pct(sp_price))*(1+sp_daily_div)-1
    sp_data=pd.DataFrame({'price':sp_price,'daily_return':sp_ret,
                          'source':f'LEGACY_SP500_DIVIDEND_ESTIMATE_{sp_div_source}',
                          'annual_dividend_yield':sp_yield},index=sp_price.index)
    sp_data.to_csv(out/'SP500_legacy_dividend_audit.csv',index_label='date')
    print(f'S&P500 legacy dividend source: {sp_div_source}')
    ndx_price=download_ndx_price()
    ndx_price=ndx_price.loc[ndx_price.index>=pd.Timestamp('1985-01-31')]
    # Early annual historical Nasdaq disclosures are retained AS IN the old script.
    yrs=pd.Series([PRE_XNDX_ANNUAL_DIVIDEND_YIELD_PCT.get(int(dt.year),np.nan)
                   for dt in ndx_price.index],index=ndx_price.index,dtype=float)
    annual=yrs/100
    dividend_day=(1+annual).pow(1/252)-1
    historical=(1+pct(ndx_price))*(1+dividend_day)-1
    ndx_data=pd.DataFrame({'price':ndx_price,'daily_return':np.nan,
                           'source':'MISSING',
                           'annual_dividend_yield':annual},index=ndx_price.index)
    # Keep original 1999 annual-yield estimate for the few sessions between
    # the official XNDX start and the first available official/QQQ return.
    # Actual ETF/proxy observations below OVERRIDE this estimate as soon as available.
    historical=historical.where(ndx_price.index<pd.Timestamp('2000-01-01'))
    early=historical.notna()
    ndx_data.loc[early,'daily_return']=historical[early]
    ndx_data.loc[early,'source']='LEGACY_NDX_NASDAQ_SEC_ANNUAL_YIELD_ESTIMATE'
    # Direct from original FRED source (short single-attempt timeout; never abort
    # simply because FRED is unavailable, as happened in the user's Colab).
    official=fetch_fred_xndx(out)
    official_ret=pct(official).reindex(ndx_price.index) if len(official) else pd.Series(dtype=float)
    # Fallback QQQ ETF distributions (YAHOO API, no assumed post-1999 zero dividends).
    qraw=fetch_yahoo('QQQ','1999-03-01',out,adjusted=False,optional=True)
    qadj=fetch_yahoo('QQQ','1999-03-01',out,adjusted=True,optional=True)
    qret=None
    if len(qraw)>0 and len(qadj)>0:
        try:qret=ndx_qqq_dividend_proxy(ndx_price,qraw,qadj,out)
        except Exception as exc:print(f'NOTICE: QQQ dividend proxy invalid: {exc}')
    if qret is not None:
        mask=qret.notna() & (ndx_price.index>=pd.Timestamp('1999-03-04'))
        ndx_data.loc[mask,'daily_return']=qret[mask]
        ndx_data.loc[mask,'source']='NDX_QQQ_DISTRIBUTION_PROXY'
    if len(official):
        mask=official_ret.notna() & (ndx_price.index>=pd.Timestamp('1999-03-04'))
        ndx_data.loc[mask,'daily_return']=official_ret[mask]
        ndx_data.loc[mask,'source']='FRED_NASDAQXNDX'
    if ndx_data.loc[ndx_data.index>=pd.Timestamp('1999-03-04'),'daily_return'].isna().any():
        missing=ndx_data.loc[(ndx_data.index>=pd.Timestamp('1999-03-04'))&ndx_data.daily_return.isna()]
        raise RuntimeError(f'NDX 1999+ requires FRED XNDX or QQQ proxy, but {len(missing)} return days missing. '
                           'No price-only returns substituted silently.')
    ndx_data.to_csv(out/'NDX_legacy_dividend_audit.csv',index_label='date')
    return sp_data,ndx_data,official

def fetch_fred_xndx(output):
    """Try two FRED CSV/text endpoints with SHORT timeouts.

    FRED is optional. Yahoo QQQ's distribution-adjustment proxy is available
    separately if FRED cannot be reached from Google Colab.
    """
    import requests
    endpoints = [
        'https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQXNDX',
    ]
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=(4, 8), headers={
                'User-Agent': 'Mozilla/5.0 (compatible; academic historical research)'})
            resp.raise_for_status()
            if url.endswith('.csv'):
                df = pd.read_csv(StringIO(resp.text))
                dt = 'observation_date' if 'observation_date' in df else 'DATE'
                if not {dt, 'NASDAQXNDX'}.issubset(df.columns):
                    raise ValueError('Unexpected FRED CSV schema')
                vals = pd.to_numeric(df['NASDAQXNDX'], errors='coerce')
                s = pd.Series(vals.to_numpy(), index=pd.to_datetime(df[dt]),
                              name='FRED_NASDAQXNDX').dropna()
            else:
                vals=[]
                for line in resp.text.splitlines():
                    m = re.match(r'^\s*(\d{4}-\d{2}-\d{2})\s+([0-9.]+|\.)\s*$', line)
                    if m and m.group(2)!='.':
                        vals.append((pd.Timestamp(m.group(1)),float(m.group(2))))
                s = pd.Series(dict(vals), dtype=float, name='FRED_NASDAQXNDX')
            s=s[s>0].sort_index()
            if len(s)<252: raise ValueError(f'Only {len(s)} official data rows')
            s.to_csv(output/'fred_NASDAQXNDX_raw.csv',header=True,index_label='date')
            print(f'FRED Nasdaq-100 TR: {s.index[0].date()}—{s.index[-1].date()} ({len(s)} rows)')
            return s
        except Exception as exc:
            print(f'NOTICE: FRED attempt failed ({str(exc)[:140]}); moving to next source')
    print('NOTICE: FRED offline; Yahoo QQQ-dividend proxy will be used instead')
    return pd.Series(dtype=float, name='FRED_NASDAQXNDX')


def ndx_qqq_dividend_proxy(ndx_price, qqq_price, qqq_adjusted, output):
    """Estimate NDX total return from NDX PR + QQQ cash distribution factor.

    This is an **approximation**, NOT the official XNDX total-return index:
    QQQ distributions occur on ETF ex-dates, not constituents' ex-dates.
    The dividend factor is inferred from the QQQ AdjClose/Close ratio so
    ETF's price trend and fee differences are NOT substituted for NDX.
    """
    idx=ndx_price.index
    raw=qqq_price.reindex(idx)
    adj=qqq_adjusted.reindex(idx)
    valid=raw.notna() & adj.notna() & raw.shift(1).notna() & adj.shift(1).notna()
    if valid.sum() < 252:
        raise RuntimeError('QQQ dividend proxy cannot be constructed: need >=252 matched days')
    ratio=adj/raw
    dividend_factor=(ratio/ratio.shift(1)).where(valid)
    # No inferred dividend on isolated QQQ missing sessions; carry no premium.
    dividend_factor=dividend_factor.where(dividend_factor.notna(), 1.0)
    daily_ndx_price_return=ndx_price.pct_change(fill_method=None)
    ret=(1+daily_ndx_price_return)*dividend_factor-1
    first=max(qqq_price.index.min(),qqq_adjusted.index.min())
    ret.loc[ret.index < first]=np.nan
    ret.name='NDX_QQQ_DISTRIBUTION_PROXY'
    pd.DataFrame({'ndx_price_return':daily_ndx_price_return,
                  'qqq_raw':raw,'qqq_adjusted':adj,
                  'qqq_dividend_factor':dividend_factor,
                  'ndx_estimated_TR_daily_return':ret}).to_csv(
        output/'NDX_QQQ_dividend_proxy_audit.csv',index_label='date')
    print(f'Yahoo QQQ distributions -> approximate NDX TR: {ret.first_valid_index().date()} onward')
    return ret

def read_repaired_rates(path):
    df=pd.read_csv(path,parse_dates=['date']).sort_values('date')
    needed={'date','cash_new','rate_annual_investment'}
    if not needed.issubset(df):raise ValueError(f'repaired rates require {sorted(needed)}')
    if df.date.duplicated().any():raise ValueError('Duplicate repaired rate dates')
    rate=df.set_index('date')[['cash_new','rate_annual_investment']].astype(float)
    if not np.isfinite(rate.to_numpy()).all():raise ValueError('Missing/invalid repaired rate values')
    print(f'Repaired rates: {rate.index.min().date()}—{rate.index.max().date()}')
    return rate



def calibrate_etf_corrections(sp_data,ndx_data,sp_etf,ndx_etf,rates,output,
                               min_obs=252):
    """Estimate historical daily 3x correction against actual adjusted ETF returns.

    r_expected_before_tracking = 3 * (index daily TOTAL return) - 2 * cash_new
    observed_all_in_correction = r_expected_before_tracking - ETF daily TR
    fee component = illustrative FEE3/252; other component = correction - fee.

    No volatility penalty is added: volatility drag is already captured by
    daily-compounding leveraged returns. The average includes real financing
    spreads, actual fund fees, tracking differences and measurement error.
    """
    rows=[]
    ledgers=[]
    for label,df,prices,ticker in [
            ('SP500',sp_data,sp_etf,'UPRO'),('NDX',ndx_data,ndx_etf,'TQQQ')]:
        x=pd.DataFrame(index=prices.index)
        x['etf_adjusted_price']=prices
        x['etf_total_return']=prices.pct_change(fill_method=None)
        x['index_total_return']=df.daily_return.reindex(x.index)
        x['benchmark_source']=df.source.reindex(x.index)
        x['cash_return']=rates.cash_new.reindex(x.index)
        x=x.dropna(subset=['etf_total_return','index_total_return','cash_return'])
        # ETF trading dates must be supported by observed/marked benchmark TR.
        # Estimated pre-1999 dividend histories never enter the actual ETF fit.
        ok=(x.benchmark_source.isin(['YAHOO_SP500TR','YAHOO_XNDX','FRED_NASDAQXNDX','NDX_QQQ_DISTRIBUTION_PROXY']))
        x=x.loc[ok].copy()
        if len(x)<min_obs:
            raise RuntimeError(f'{ticker}: only {len(x)} matching observed ETF/official-index/rate days; '
                               f'at least {min_obs} required to estimate correction. '
                               'Check Yahoo/FRED access, source dates and repaired CSV coverage.')
        x['predicted_before_cost']=3*x.index_total_return-2*x.cash_return
        x['correction_all_in_daily']=x.predicted_before_cost-x.etf_total_return
        x['correction_after_fee_daily']=x.correction_all_in_daily-FEE3/TD
        x['calibrated_etf']=ticker
        x['benchmark']=label
        # Average daily simple-return difference is used rather than a log gap;
        # unusual tracking days remain visible, not silently removed.
        overall_all_in=TD*x.correction_all_in_daily.mean()
        yearly=x.groupby(x.index.year).agg(
            days=('correction_all_in_daily','count'),
            observed_all_in_annualized=('correction_all_in_daily',lambda y:TD*y.mean()),
            observed_extra_annualized=('correction_after_fee_daily',lambda y:TD*y.mean()))
        yearly['index']=label
        yearly['etf']=ticker
        yearly.index.name='year'
        yearly.to_csv(output/f'{ticker}_correction_by_year.csv')
        x.index.name='date'
        ledger=x.reset_index()
        ledger.to_csv(output/f'{ticker}_correction_daily.csv',index=False)
        rows.append({'index':label,'etf':ticker,'n_matching_days':len(x),
                     'calibration_start':str(x.index.min().date()),
                     'calibration_end':str(x.index.max().date()),
                     'assumed_expense_fee_annual':FEE3,
                     'observed_all_in_correction_annual':overall_all_in,
                     'observed_extra_after_fee_annual':overall_all_in-FEE3,
                     'correction_daily_std':x.correction_all_in_daily.std(ddof=1),
                     'negative_correction_days':int((x.correction_all_in_daily<0).sum()),
                     'note':'EX-POST retrospective estimate; NOT an observed pre-inception rate'})
        ledgers.append(x)
        print(f'{ticker}: {len(x):,} matched ETF days: '
              f'all-in correction={overall_all_in:.4%}/year, '
              f'extra beyond assumed fee={overall_all_in-FEE3:.4%}/year '
              f'(not a constant 1.6%)')
    fit=pd.DataFrame(rows)
    fit.to_csv(output/'etf_correction_calibration.csv',index=False)
    return fit.set_index('index'),ledgers


def pct(series):return series.pct_change(fill_method=None)


def features(price,ret=None):
    p=price.astype(float)
    r=pct(p) if ret is None else ret
    dma=p.rolling(200,min_periods=200).mean()
    close=p.groupby(p.index.year).last().shift(1)
    ytd=p/p.index.year.map(close)-1
    gap=p-dma
    crosses=(gap.mul(gap.shift(1))<0).astype(float).rolling(60,min_periods=60).sum()
    vol=pct(p).rolling(20,min_periods=20).std(ddof=1)*np.sqrt(TD)
    dn=np.sqrt(pct(p).clip(upper=0).pow(2).rolling(20,min_periods=20).mean()*TD)
    return pd.DataFrame({'price':p,'ret':r,'dma':dma,'ytd':ytd,'cross60':crosses,
                         'vol20':vol,'downside20':dn})


def annual_signals(price):
    years=price.index.year
    first=int(years.min());last=int(years.max())
    yearly=price.groupby(years).last()
    annual={year:float(yearly[year]/yearly[year-1]-1)
            for year in range(first+1,last) if year-1 in yearly.index and year in yearly.index}
    one=np.array([annual.get(int(y)-1,np.nan) for y in years])
    two=np.array([( (1+annual[int(y)-2])*(1+annual[int(y)-1]) )**.5-1
                  if int(y)-2 in annual and int(y)-1 in annual else np.nan for y in years])
    return one,two


def leverage_choices(price,rate_average):
    base=np.select([rate_average<.03,rate_average<=.06],[3.,2.],1.)
    base[~np.isfinite(rate_average)]=3.
    one,two=annual_signals(price)
    # Signal uses TOTAL RETURN (not price index), where actual TR observations exist.
    r=pct(price)
    rolling252=np.expm1(np.log1p(r).rolling(252,min_periods=252).sum()).to_numpy()
    rolling504=np.expm1(np.log1p(r).rolling(504,min_periods=504).sum()/2).to_numpy()
    signals={'prior_1y':one,'prior_2y':two,'blend_1y_2y':(one+two)/2,
             'rolling_252':rolling252,'rolling_504':rolling504}
    result={'fixed3':np.full(len(base),3.),'rate_only':base.copy()}
    for name,signal in signals.items():
        bump=np.select([signal<0,signal>.15],[1.,-.5],0.)
        result[name]=np.clip(base+bump,1.,3.)
    return result


def sell_gates(own,sp):
    own_good=own.ytd.ge(-.05)&own.cross60.lt(5)
    sp_good=sp.ytd.ge(-.05)&sp.cross60.lt(5)
    skip={
      'none':np.zeros(len(own),dtype=bool),
      'own_vol25':(own_good&own.vol20.lt(.25)).to_numpy(),
      'own_downside':(own_good&own.downside20.lt(DSVOL_THRESHOLD)).to_numpy(),
      'sp_downside':(sp_good&sp.downside20.lt(DSVOL_THRESHOLD)).to_numpy(),
      'both_downside':(own_good&own.downside20.lt(DSVOL_THRESHOLD)&
                       sp_good&sp.downside20.lt(DSVOL_THRESHOLD)).to_numpy(),
      'sp_downside_own_guard':(sp_good&sp.downside20.lt(DSVOL_THRESHOLD)&
                               own.ytd.ge(-.05)&own.vol20.lt(.30)).to_numpy()
    }
    close=own.price.to_numpy();ma=own.dma.to_numpy()
    ans={}
    for key,flag in skip.items():
        out=np.zeros(len(own),dtype=np.int8);state=0
        for i in range(len(own)):
            if not np.isfinite(ma[i]): state=0
            elif state==0 and close[i]>=ma[i]*1.01:state=1
            elif state==1 and close[i]<=ma[i]*.99 and not flag[i]:state=0
            out[i]=state
        ans[key]=out
    return ans


def defense_choices(sp_price,sp_ret,bond_price,bond_ret,cash,index):
    sp_p=sp_price.reindex(index).ffill(limit=2)
    sp_r=sp_ret.reindex(index).fillna(0)
    sp_tr=(1+sp_r).cumprod()*100
    # Both defensive momentum and DMA use the TOTAL RETURN series.
    sp_tr_ok=sp_tr>=sp_tr.rolling(200,min_periods=200).mean()*1.01
    sp_mom=sp_tr/sp_tr.shift(63)-1
    bond_p=bond_price.reindex(index).ffill(limit=2)
    bond_r=bond_ret.reindex(index).fillna(0)
    bond_tr=(1+bond_r).cumprod()*100
    bond_ma=bond_tr.rolling(200,min_periods=200).mean()
    # Lookbacks must begin AFTER bond inception, not from cash padding.
    bond_live=bond_p.notna().astype(int).rolling(200,min_periods=200).sum().eq(200)
    bond_tr_ok=bond_live&(bond_tr>=bond_ma*1.01)
    bond_mom=bond_tr/bond_tr.shift(63)-1
    cash63=np.expm1(np.log1p(cash).rolling(63,min_periods=63).sum())
    sp_ok=sp_tr_ok&(sp_mom>cash63)
    bond_ok=bond_tr_ok&(bond_mom>cash63)
    return {'cash':np.zeros(len(index),dtype=np.int8),
            'sp500':np.ones(len(index),dtype=np.int8),
            'treasury':np.where(bond_p.notna(),2,0).astype(np.int8),
            'rotation':np.select([sp_ok,bond_ok],[1,2],default=0).astype(np.int8)}


def simulate(own_ret,sp_ret,bond_ret,rf,gate,lev,choice,correction_all_in_annual):
    """ONE execution convention only: signal at close[t], fill at close[t].

    All index return for date t belongs to positions chosen at close[t-1].
    Today's close[t] switch is charged on day t, never earns day-t index return.
    """
    target_lev=np.where(gate>0,lev,0.).astype(float)
    target_def=np.where(gate>0,3,choice).astype(np.int8)
    held_lev=np.r_[0.,target_lev[:-1]]
    held_def=np.r_[0,target_def[:-1]]
    leveraged_fraction=np.clip((held_lev-1.)/2.,0.,1.)
    # Calibrated residual includes actual ETF fees; never deduct fee again.
    primary=held_lev*own_ret-(held_lev-1.)*rf- \
            leveraged_fraction*(correction_all_in_annual/TD)
    defensive=np.select([held_def==1,held_def==2],
                        [sp_ret,bond_ret],default=rf)
    returns=np.where(held_lev>0,primary,defensive)
    previous_target_lev=np.r_[0.,target_lev[:-1]]
    previous_target_def=np.r_[0,target_def[:-1]]
    leveraged_trade_cost=COST*np.abs(target_lev-previous_target_lev)/3.
    defensive_sell=(previous_target_lev==0)&(previous_target_def>0)& \
        (previous_target_def<3)&((target_lev>0)|(target_def!=previous_target_def))
    defensive_buy=(target_lev==0)&(target_def>0)&(target_def<3)& \
        ((previous_target_lev>0)|(target_def!=previous_target_def))
    trading_cost=leveraged_trade_cost+COST*(defensive_buy.astype(float)+defensive_sell.astype(float))
    returns=returns-trading_cost
    if not np.isfinite(returns).all() or (returns<=-1).any():
        raise ValueError('Invalid / insolvent portfolio return')
    return returns,held_lev,held_def,trading_cost,target_lev,target_def


def summary(dates,ret,rf):
    wealth=np.r_[1.,np.cumprod(1+ret)]
    yrs=(dates[-1]-dates[0]).days/365.2425
    excess=ret-rf
    sd=excess.std(ddof=1)
    return dict(start=dates[0].strftime('%Y-%m-%d'),end=dates[-1].strftime('%Y-%m-%d'),
                days=len(ret),cagr=wealth[-1]**(1/yrs)-1,
                sharpe=np.sqrt(TD)*excess.mean()/sd if sd>0 else np.nan,
                max_dd=np.min(wealth/np.maximum.accumulate(wealth)-1),
                annual_vol=ret.std(ddof=1)*np.sqrt(TD),multiple=wealth[-1])





# Consolidated outputs: 4 files total, 2 per index.
STRATEGY_KEYS = (
    ('dma_unfiltered_3x', 'none', 'fixed3'),
    ('dma_filtered_3x', 'sp_downside', 'fixed3'),
    ('dma_filtered_dynamic', 'sp_downside', 'blend_1y_2y'),
)


def make_daily(index_name, streams, dynamic_first_year):
    """Assemble the three identical-date strategy return streams into one audit-friendly CSV."""
    blocks = {}
    for display, filt, leverage in STRATEGY_KEYS:
        data = streams[(filt, leverage)].copy()
        required = ('date', 'execution', 'dma200', 'index_total_return', 'cash_return',
                    'strategy_return', 'all_in_correction_annual',
                    'signal_target_leverage_at_close', 'held_leverage_for_return', 'trade_cost')
        absent = set(required).difference(data.columns)
        if absent:
            raise ValueError(f'{index_name}/{display}: missing {sorted(absent)}')
        if not data.execution.eq('same_close').all():
            raise ValueError(f'{index_name}/{display}: not same_close; refusing incompatible data')
        data['date'] = pd.to_datetime(data.date)
        data = data.set_index('date').sort_index()
        if data.index.has_duplicates:
            raise ValueError(f'{index_name}/{display}: duplicated dates')
        held = data.held_leverage_for_return.to_numpy(dtype=float)
        target = data.signal_target_leverage_at_close.to_numpy(dtype=float)
        if not np.allclose(held[1:], target[:-1], atol=1e-10, rtol=0, equal_nan=True):
            raise ValueError(f'{index_name}/{display}: previous-close holding consistency failed')
        blocks[display] = data
    base = blocks['dma_unfiltered_3x']
    for label, d in blocks.items():
        if not d.index.equals(base.index):
            raise ValueError(f'{index_name}: dates differ: {label}')
        for c in ('index_total_return','cash_return','all_in_correction_annual','dma200'):
            if not np.allclose(d[c], base[c], atol=1e-10, rtol=1e-9, equal_nan=True):
                raise ValueError(f'{index_name}: {c} differs between strategies: {label}')
    out = pd.DataFrame(index=base.index)
    out.index.name='date'
    out['execution']='same_close'
    out['dma200']=base.dma200
    out['index_total_return']=base.index_total_return
    out['cash_return']=base.cash_return
    out['buyhold_1x_return']=base.index_total_return
    out['all_in_correction_annual']=base.all_in_correction_annual
    out['dynamic_signal_ready']=(base.index.year>=dynamic_first_year)
    for label,d in blocks.items():
        out[f'{label}_return'] = d.strategy_return
        out[f'{label}_target_leverage_at_close'] = d.signal_target_leverage_at_close
        out[f'{label}_held_leverage_for_return'] = d.held_leverage_for_return
        out[f'{label}_trade_cost'] = d.trade_cost
    if out.isna().any().any():
        raise ValueError(f'{index_name}: missing daily output values')
    return out.reset_index()


def make_annual(daily):
    daily = daily.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['year']=daily.date.dt.year
    rows=[]
    keys=('buyhold_1x',)+tuple(k for k,_,_ in STRATEGY_KEYS)
    for y, group in daily.groupby('year', sort=True):
        days=len(group)
        first=group.date.iloc[0]
        last=group.date.iloc[-1]
        full=days>=220 and first.month==1 and first.day<=10 and last.month==12 and last.day>=22
        row={'year':int(y), 'start_date':first.strftime('%Y-%m-%d'),
             'end_date':last.strftime('%Y-%m-%d'), 'trading_days':days,
             'complete_year':bool(full),
             'dynamic_signal_ready':bool(group.dynamic_signal_ready.all())}
        rf=group.cash_return.to_numpy(dtype=float)
        for key in keys:
            r=group[f'{key}_return'].to_numpy(dtype=float)
            if np.any(r<=-1) or not np.isfinite(r).all():
                raise ValueError(f'Invalid annual returns in {y} {key}')
            row[f'{key}_annual_return']=float(np.expm1(np.log1p(r).sum()))
            excess=r-rf
            sd=excess.std(ddof=1) if len(excess)>1 else float('nan')
            row[f'{key}_annual_sharpe']=float(np.sqrt(TD)*excess.mean()/sd) if sd>1e-14 else float('nan')
            wealth=np.r_[1.,np.cumprod(1+r)]
            row[f'{key}_annual_max_drawdown']=float((wealth/np.maximum.accumulate(wealth)-1).min())
            if key!='buyhold_1x':
                lev=group[f'{key}_held_leverage_for_return'].to_numpy(dtype=float)
                row[f'{key}_mean_leverage_all_days']=float(lev.mean())
                row[f'{key}_mean_leverage_invested']=float(lev[lev>0].mean()) if (lev>0).any() else 0.
                row[f'{key}_invested_fraction']=float(np.mean(lev>0))
                row[f'{key}_trade_cost_sum']=float(group[f'{key}_trade_cost'].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def read_existing_daily(folder, index_name):
    streams={}
    for _,filt,leverage in STRATEGY_KEYS:
        file=folder/f'{index_name}_{filt}_{leverage}_cash_daily.csv'
        if not file.is_file():
            raise FileNotFoundError(f'{file} is missing; check the older backtest folder')
        streams[(filt,leverage)]=pd.read_csv(file)
    audit=folder/f'{index_name}_return_sources_audit.csv'
    if audit.is_file():
        a=pd.read_csv(audit,usecols=['date'],nrows=1)
        first_source_year=int(pd.to_datetime(a.date.iloc[0]).year)
        dynamic_first_year=first_source_year+3
    else:
        # Original first available historical source year: 1927 / 1985.
        dynamic_first_year={'SP500':1930,'NDX':1988}[index_name]
    return make_daily(index_name,streams,dynamic_first_year)


def run_original_model(rates_path, requested):
    rates=read_repaired_rates(rates_path)
    print('End date limited to the last repaired-rate observation.')
    print('All models use idealized SAME-CLOSE execution.')
    with tempfile.TemporaryDirectory(prefix='dma_sameclose_inputs_') as scratch:
        work=Path(scratch)
        spdata, ndxdata, _=load_legacy_sources(work)
        sp_official=fetch_yahoo('^SP500TR','1988-01-01',work,optional=True)
        if len(sp_official)<252:
            raise RuntimeError('Not enough ^SP500TR data to calibrate actual UPRO correction')
        sp_cal=pd.DataFrame({'daily_return':pct(sp_official),
                             'source':'YAHOO_SP500TR'},index=sp_official.index)
        upro=fetch_yahoo('UPRO','2009-01-01',work,adjusted=True)
        tqqq=fetch_yahoo('TQQQ','2010-01-01',work,adjusted=True)
        correction_table,_=calibrate_etf_corrections(sp_cal,ndxdata,upro,tqqq,rates,work)
    outputs={}
    for name,data,first in [('SP500',spdata,'1929-01-01'),('NDX',ndxdata,'1985-01-01')]:
        if requested!='both' and name!=requested:
            continue
        valid_dma=data.price.rolling(200,min_periods=200).mean().notna()
        good=(data.daily_return.notna()) & valid_dma & (data.index>=first) & (data.index.isin(rates.index))
        idx=data.index[good]
        if len(idx)<252:
            raise RuntimeError(f'{name}: insufficient verified total return data')
        bad=data.loc[(data.index>=idx.min())&(data.index<=idx.max())&(data.index.isin(rates.index)),
                     'daily_return'].isna().sum()
        if bad:
            raise RuntimeError(f'{name}: {bad} missing total-return days')
        if name=='SP500' and idx.min().year>1929:
            raise RuntimeError('Yahoo did not supply 1929 S&P 500 prices')
        if name=='NDX' and idx.min().year>1985:
            print('NOTE: NDX DMA200 warm-up starts the valid strategy in',idx.min().date())
        d=features(data.price.loc[:idx.max()],data.daily_return.loc[:idx.max()]).reindex(idx)
        sp=features(spdata.price.loc[:idx.max()],spdata.daily_return.loc[:idx.max()]).reindex(idx)
        if name=='NDX':
            sp=sp.ffill(limit=2)
        rf=rates.loc[idx,'cash_new']
        if d.ret.isna().any() or sp.ret.isna().any():
            raise ValueError('Unaligned underlying returns')
        trr=data.daily_return.loc[:idx.max()].dropna()
        tr_level=(1+trr).cumprod()*100
        lev_full=leverage_choices(tr_level,
            rates.rate_annual_investment.rolling(20,min_periods=20).mean().reindex(tr_level.index).to_numpy())
        levdf=pd.DataFrame(lev_full,index=tr_level.index).reindex(idx).ffill()
        levdf['fixed3']=3.
        sell=sell_gates(d,sp)
        spret=spdata.daily_return.reindex(idx)
        if spret.isna().any():
            aligned=spdata.price.reindex(idx).ffill(limit=2)
            extra=pct(aligned)
            no_quote=~idx.isin(spdata.index)
            if (spret.isna()&~no_quote).any():
                raise ValueError('S&P 500 total-return price gaps')
            spret=spret.where(spret.notna(),extra)
        spret=spret.fillna(0)
        cash=rf.to_numpy()
        annual_correction=float(correction_table.loc[name,'observed_all_in_correction_annual'])
        cash_defense=np.zeros(len(idx),dtype=np.int8)
        no_bonds=np.zeros(len(idx),dtype=float)
        streams={}
        for key,filt,leverage in STRATEGY_KEYS:
            level=levdf[leverage].to_numpy()
            if not np.isfinite(level).all():
                raise ValueError(f'{name}/{key}: missing leverage observations')
            r,held,defheld,tc,target_lev,target_def=simulate(
                d.ret.to_numpy(),spret.to_numpy(),no_bonds,cash,
                sell[filt],level,cash_defense,annual_correction)
            streams[(filt,leverage)]=pd.DataFrame({
                'date':idx, 'execution':'same_close',
                'dma200':d.dma.to_numpy(), 'index_total_return':d.ret.to_numpy(),
                'cash_return':cash, 'strategy_return':r,
                'all_in_correction_annual':annual_correction,
                'signal_target_leverage_at_close':target_lev,
                'held_leverage_for_return':held, 'trade_cost':tc})
            s=summary(idx,r,cash)
            print(f'{name}/{key}: CAGR={s["cagr"]:.2%}, '
                  f'Sharpe={s["sharpe"]:.3f}, maximum drawdown={s["max_dd"]:.2%}')
        outputs[name]=make_daily(name,streams,int(data.index[0].year)+3)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rates',default='sp500_repaired_rate_audit.csv')
    parser.add_argument('--out',default='/content/dma_daily_annual_results',
                        help='New clean directory for exactly two CSVs per index')
    parser.add_argument('--only',choices=('SP500','NDX','both'),default='both')
    parser.add_argument('--from-existing',type=Path,default=None,
                        help='Reuse 3 strategy CSVs per index from older same-close output; no downloads')
    args=parser.parse_args()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    requested=['SP500','NDX'] if args.only=='both' else [args.only]
    if args.from_existing:
        existing=args.from_existing.expanduser().resolve()
        if not existing.is_dir():
            raise NotADirectoryError(f'Existing daily results folder does not exist: {existing}')
        print(f'Reusing same-close strategy data from {existing}; no price downloads.')
        output={key:read_existing_daily(existing,key) for key in requested}
    else:
        output=run_original_model(args.rates,args.only)
    for name,daily in output.items():
        annual=make_annual(daily)
        daily_path=out/f'{name}_daily.csv'
        annual_path=out/f'{name}_annual.csv'
        daily.to_csv(daily_path,index=False)
        annual.to_csv(annual_path,index=False)
        print(f'{name}: {len(daily):,} trading days, {len(annual)} calendar-year rows '
              f'({int(annual.complete_year.sum())} complete years)')
        print(f'  {daily_path.name} | {annual_path.name}')
    extraneous=[p.name for p in out.iterdir() if p.is_file() and p.name not in
                {f'{name}_{suffix}.csv' for name in requested for suffix in ('daily','annual')}]
    if extraneous:
        print('WARNING: existing unrelated files in the output directory were NOT deleted:',extraneous)
        print('Use a new --out directory for exactly four files.')
    print(f'SUCCESS. Consolidated daily and annual strategy data are in: {out}')
    print('The 2026 annual row is partial and marked complete_year=False.')
    print('SAME-CLOSE is idealized; ETF correction before inception is retrospective.')


if __name__=='__main__':
    main()
