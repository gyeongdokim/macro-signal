"""실행: python main.py            (실데이터: FRED + yfinance)
       python main.py --offline  (합성데이터로 파이프라인 점검)
"""
import argparse
import pandas as pd

import config as C
from data import load
from signals import to_point_in_time, rolling_z, combine_equal, combine_ridge
from backtest import signal_to_position, run, report, signal_diagnostics


def main(offline: bool):
    px = load("synthetic" if offline else "yf", "PRICE_" + C.TARGET if offline else C.TARGET, C.START)
    cal = px.index
    ret = px.pct_change()
    fwd21 = px.shift(-21) / px - 1                      # 진단/학습용 1개월 선행수익률

    z = {}
    for spec in C.SIGNALS:
        try:
            raw = load("synthetic" if offline else spec.source, spec.ticker, C.START)
        except Exception as e:
            print(f"[skip] {spec.name}: {e}")
            continue
        pit = to_point_in_time(raw, spec, cal)
        z[spec.name] = rolling_z(pit, C.BT.zscore_window, C.BT.zscore_clip)
    z = pd.DataFrame(z)

    print("\n== 개별 시그널 진단 (Spearman IC vs 21D fwd return) ==")
    print(signal_diagnostics(z, fwd21))

    if C.BT.combine == "ridge":
        score = combine_ridge(z, fwd21, C.BT.train_years * 252, C.BT.ridge_alpha)
    else:
        score = combine_equal(z)

    pos = signal_to_position(score, ret, C.BT)
    bt = run(pos, ret, C.BT.cost_bps).dropna()
    bt = bt.loc[score.first_valid_index():]

    print(f"\n== 백테스트 ({C.TARGET}, {bt.index[0].date()} ~ {bt.index[-1].date()}) ==")
    print(report(bt))
    bt.to_csv("backtest_result.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    main(ap.parse_args().offline)
