"""데이터 로더 — FRED / yfinance / 합성데이터(오프라인 테스트용).

FRED는 API 키 없이 fredgraph CSV 엔드포인트로 받는다.
실전에서는 ALFRED(빈티지 데이터)로 '당시 발표치'를 써야 수정치 룩어헤드를 피할 수 있다 → TODO
"""
from __future__ import annotations
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)


def _cached(key: str, fetch) -> pd.Series:
    f = CACHE / f"{key.replace('^', '_')}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    s = fetch()
    if s is None or s.dropna().empty:
        raise ValueError(f"{key}: 빈 데이터 (네트워크/티커 확인)")
    s.to_csv(f)
    return s


def load_fred(series_id: str, start: str) -> pd.Series:
    def fetch():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True, na_values=".")
        return df.iloc[:, 0].astype(float).dropna().rename(series_id)
    return _cached(f"fred_{series_id}", fetch)


def load_yf(ticker: str, start: str) -> pd.Series:
    def fetch():
        import yfinance as yf  # pip install yfinance
        px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
        return px.squeeze().rename(ticker)
    return _cached(f"yf_{ticker}", fetch)


def synthetic(name: str, start: str, seed: int = 0) -> pd.Series:
    """네트워크 없이 파이프라인을 검증하기 위한 랜덤워크."""
    idx = pd.bdate_range(start, "2026-10-30")
    rng = np.random.default_rng(abs(hash(name)) % 2**32 + seed)
    if name.startswith("PRICE"):
        r = rng.normal(0.0003, 0.011, len(idx))
        return pd.Series(100 * np.exp(np.cumsum(r)), idx, name=name)
    return pd.Series(np.cumsum(rng.normal(0, 1, len(idx))), idx, name=name)


def load(source: str, ticker: str, start: str) -> pd.Series:
    if source == "fred":
        return load_fred(ticker, start)
    if source == "yf":
        return load_yf(ticker, start)
    return synthetic(ticker, start)
