"""원시 지표 → point-in-time 일간 시그널 → 표준화 → 합성."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import SignalSpec


def to_point_in_time(raw: pd.Series, spec: SignalSpec, calendar: pd.DatetimeIndex) -> pd.Series:
    """변환 → 거래일 캘린더로 정렬 → 발표 지연만큼 shift."""
    s = raw.sort_index()
    if spec.transform == "yoy":                 # 월간 지표 전년비의 변화(가속도)
        s = s.pct_change(12).diff()
    s = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
    if spec.transform == "diff":
        s = s.diff(spec.lookback)
    elif spec.transform == "pct":
        s = s.pct_change(spec.lookback)
    # 관측일이 아니라 '공개된 날' 이후에만 사용 가능
    return s.shift(spec.lag_days) * spec.sign


def rolling_z(s: pd.Series, window: int, clip: float) -> pd.Series:
    if window and window > 0:
        mu, sd = s.rolling(window, min_periods=window // 2).mean(), s.rolling(window, min_periods=window // 2).std()
    else:
        mu, sd = s.expanding(252).mean(), s.expanding(252).std()
    return ((s - mu) / sd).clip(-clip, clip)


def combine_equal(z: pd.DataFrame) -> pd.Series:
    """동일가중 합성 — 가장 견고한 베이스라인 (Goyal-Welch-Zafirov 2024 교훈: 단순한 게 OOS에서 잘 버팀)."""
    return z.mean(axis=1, skipna=True)


def combine_ridge(z: pd.DataFrame, fwd_ret: pd.Series, train_days: int, alpha: float,
                  refit: str = "ME") -> pd.Series:
    """Walk-forward Ridge: 매월 말 과거 데이터로만 재학습 → 다음 달 예측.
    fwd_ret는 t+1~t+h 수익률이므로 학습 시 h일 만큼 잘라 라벨 누수를 막는다.
    """
    h = 21
    out = pd.Series(np.nan, z.index)
    refit_dates = z.resample(refit).last().index
    for d in refit_dates:
        train = z.loc[:d].iloc[:-h].tail(train_days)
        y = fwd_ret.reindex(train.index)
        mask = train.notna().all(axis=1) & y.notna()
        if mask.sum() < 252:
            continue
        model = Ridge(alpha=alpha).fit(train[mask], y[mask])
        nxt = z.loc[d:].iloc[1:].head(23)
        nxt = nxt[nxt.notna().all(axis=1)]
        if len(nxt):
            out.loc[nxt.index] = model.predict(nxt)
    return out / out.expanding(252).std()   # 스케일을 z 수준으로
