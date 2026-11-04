"""전략 설정 — 유니버스, 시그널, 백테스트 파라미터를 한 곳에서 관리."""
from dataclasses import dataclass, field

# ── 거래 대상 ────────────────────────────────────────────
# 처음엔 단일 자산(주식 지수) 타이밍으로 시작 → 이후 멀티에셋으로 확장
TARGET = "SPY"            # 예: "SPY", "069500.KS"(KODEX 200), "TLT"
START = "2005-01-01"

# ── 매크로 시그널 정의 ───────────────────────────────────
# source: fred | yf | synthetic
# lag_days: 발표 지연(영업일). 룩어헤드 방지의 핵심 — 관측일이 아니라 '알 수 있었던 날' 기준으로 정렬
# transform: level | diff | pct | yoy
# sign: +1 이면 값↑ → 위험자산 우호, -1 이면 값↑ → 위험자산 비우호
@dataclass
class SignalSpec:
    name: str
    source: str
    ticker: str
    lag_days: int
    transform: str = "level"
    lookback: int = 0          # diff/pct 계산 기간(영업일)
    sign: int = 1

SIGNALS = [
    # 금리·크레딧 (일간, 지연 1일)
    SignalSpec("yield_curve", "fred", "T10Y2Y", 1, "level", sign=+1),
    SignalSpec("hy_spread_chg", "fred", "BAMLH0A0HYM2", 1, "diff", 21, sign=-1),
    SignalSpec("fin_conditions", "fred", "NFCI", 6, "level", sign=-1),  # 주간, 수요일 발표
    # 실물 (주간/월간)
    SignalSpec("jobless_claims", "fred", "ICSA", 5, "pct", 63, sign=-1),
    SignalSpec("cpi_yoy_chg", "fred", "CPIAUCSL", 15, "yoy", sign=-1),   # 인플레 가속 = 비우호
    # 시장 기반
    SignalSpec("vix", "yf", "^VIX", 0, "level", sign=-1),
    SignalSpec("tsmom_12m", "yf", "SPY", 0, "pct", 252, sign=+1),
]

@dataclass
class BacktestConfig:
    zscore_window: int = 756        # 3년 롤링 z-score (expanding이면 0)
    zscore_clip: float = 3.0
    combine: str = "equal"          # equal | ridge
    ridge_alpha: float = 10.0
    train_years: int = 5            # walk-forward 학습 구간 (ridge 사용 시)
    rebalance: str = "W-FRI"        # 리밸런싱 주기
    target_vol: float = 0.15        # 연 목표 변동성
    vol_window: int = 63
    max_leverage: float = 1.5
    long_only: bool = True
    cost_bps: float = 5.0           # 편도 거래비용
    extra: dict = field(default_factory=dict)

BT = BacktestConfig()
