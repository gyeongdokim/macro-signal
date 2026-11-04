"""시그널 → 포지션 → 성과. 포지션은 t일 종가에 결정, t+1 수익률에 적용."""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import BacktestConfig


def signal_to_position(score: pd.Series, ret: pd.Series, cfg: BacktestConfig) -> pd.Series:
    # 1) 방향/강도: tanh로 부드럽게 매핑 (극단 z에서 포지션 폭주 방지)
    raw = np.tanh(score)
    if cfg.long_only:
        raw = (raw + 1) / 2                     # [-1,1] → [0,1]
    # 2) 변동성 타기팅
    realized = ret.rolling(cfg.vol_window).std() * np.sqrt(252)
    pos = (raw * cfg.target_vol / realized).clip(-cfg.max_leverage, cfg.max_leverage)
    # 3) 리밸런싱 주기에서만 변경
    pos = pos.resample(cfg.rebalance).last().reindex(pos.index).ffill()
    return pos.shift(1)                         # 체결 지연 1일


def run(pos: pd.Series, ret: pd.Series, cost_bps: float) -> pd.DataFrame:
    pos = pos.fillna(0)
    turnover = pos.diff().abs().fillna(0)
    strat = pos * ret - turnover * cost_bps / 1e4
    return pd.DataFrame({"pos": pos, "ret": strat, "bench": ret, "turnover": turnover})


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": cagr / vol if vol else np.nan,
            "MDD": dd, "Calmar": cagr / abs(dd) if dd else np.nan}


def report(bt: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"strategy": stats(bt["ret"]), "buy&hold": stats(bt["bench"])}).T
    df["AvgTurnover(yr)"] = [bt["turnover"].sum() / (len(bt) / 252), 0.0]
    return df.round(3)


def signal_diagnostics(z: pd.DataFrame, fwd_ret: pd.Series) -> pd.DataFrame:
    """개별 시그널 IC(정보계수) — 합성 전에 각 지표가 쓸모 있는지 먼저 확인."""
    rows = {}
    for c in z:
        x = pd.concat([z[c], fwd_ret], axis=1).dropna()
        if len(x) < 252:
            continue
        ic = x.iloc[:, 0].corr(x.iloc[:, 1], method="spearman")
        yearly = x.groupby(x.index.year).apply(lambda g: g.iloc[:, 0].corr(g.iloc[:, 1], method="spearman"))
        rows[c] = {"IC": ic, "IC_IR": yearly.mean() / yearly.std(), "hit_years": (yearly > 0).mean(), "n": len(x)}
    return pd.DataFrame(rows).T.round(3)
