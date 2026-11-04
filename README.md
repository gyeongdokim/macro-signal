# macro-signal — 매크로·시장 신호 기반 타이밍 전략 (뼈대)

여러 경제지표·시장신호를 **point-in-time**으로 정렬 → z-score 표준화 → 합성 점수 → 변동성 타기팅 포지션 → 백테스트.

```
pip install pandas numpy scikit-learn yfinance
python main.py --offline   # 합성데이터로 파이프라인 점검
python main.py             # FRED + yfinance 실데이터
```

| 파일 | 역할 |
|---|---|
| `config.py` | 대상 자산, 시그널 목록(발표지연·변환·부호), 백테스트 파라미터 |
| `data.py` | FRED(키 불필요) / yfinance 로더 + CSV 캐시 |
| `signals.py` | 발표지연 shift, 롤링 z-score, 동일가중 / walk-forward Ridge 합성 |
| `backtest.py` | tanh 매핑 → 변동성 타기팅 → 주간 리밸런싱, 비용, 성과·IC 진단 |

## 설계 원칙 (논문 트렌드에서 가져온 것)
1. **동일가중 베이스라인 먼저** — Goyal·Welch·Zafirov(RFS 2024): 대부분의 단일 예측변수는 OOS에서 실패. 복잡한 모델은 단순 합성을 이겨야 채택.
2. **복잡성은 의심하며 사용** — Kelly·Malamud·Zhou(JF 2024)의 "Virtue of Complexity" vs Nagel(2025) 반론: 고차원 모델 성과의 상당 부분이 사실상 변동성 타이밍/모멘텀일 수 있음 → 반드시 `tsmom`·변동성 타기팅 단독 대비 초과성과로 검증.
3. **레짐 인식** — 매크로 레짐 탐지 후 TAA (arXiv 2503.11499 등). 다음 단계로 HMM/클러스터링 레짐 필터 추가.
4. **텍스트 신호** — LLM 기반 중앙은행·뉴스 감성 점수가 새 시그널 소스로 부상. `SignalSpec`에 source 하나 추가하는 식으로 확장.

## TODO
- [ ] ALFRED 빈티지 데이터로 수정치 룩어헤드 제거
- [ ] 한국 지표: ECOS(한국은행) API — 수출 증가율, 기준금리, 원/달러, 국고채 스프레드 → KODEX 200 타이밍
- [ ] 레짐 필터(HMM 2~3 state), 멀티에셋(주식/채권/금/달러) 배분
- [ ] 거래비용·슬리피지 민감도, 파라미터 안정성(heatmap), 2020·2022 서브샘플 분석
