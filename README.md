# ConShift

`originaldata-2`: Qwen3-4B를 유지하며 vLLM Tensor Parallel 2로 GPU 두 장에서 응답을 생성합니다. 최종 층 표현 추출용 모델은 GPU 1에 함께 상주시킵니다. 설치와 실행은 [RUNBOOK-VLLM.md](RUNBOOK-VLLM.md)를 참고하세요.

```text
conshift/
├── configs/
│   ├── experiment.yaml          # 모델·학습·평가 설정
│   └── sources.json             # 공식 원본 버전·다운로드 URL·무결성 hash
├── src/
│   ├── contracts.py             # 공통 I/O·실험 계약
│   ├── setup.py                 # 공식 원본 다운로드
│   ├── source_audit.py          # 원본·checker 확인
│   ├── evolif_adapter.py        # 원본 trajectory 정규화
│   ├── splitting.py            # group·ID/OOD 분할
│   ├── target.py                # 동결 Target 순차 추론
│   ├── representation.py        # 생성 전 표현 추출·cache
│   ├── checker.py               # 공식 constraint checker
│   ├── labeling.py              # 실제 응답 failure 라벨
│   ├── features.py              # 현재 표현·turn 간 차분
│   ├── probes.py                # P1·T1·ConShift
│   ├── training.py              # 학습·early stopping·재개
│   ├── checkpointing.py         # 원자적 checkpoint 저장
│   ├── baselines.py             # G1·L1
│   ├── metrics.py               # 지표·paired group bootstrap
│   ├── evaluation.py            # ID/OOD 평가·분석·그림
│   ├── privacy_demo.py          # 개인정보 응용 데모
│   └── cli.py                   # 공통 CLI
├── scripts/
│   ├── run_experiment.sh        # 전처리부터 평가까지 순차 실행
│   ├── 01_audit.py
│   ├── 02_split.py
│   ├── 03_collect.py
│   ├── 04_label.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   └── 07_demo.py
├── tests/                       # 코드 검증
├── data/
│   ├── raw/                     # 공식 원본·checker·모델 설정
│   ├── normalized/              # 전처리 시 생성
│   ├── manifests/               # 실행 시 고정되는 데이터·split 정보
│   ├── responses/               # 실제 Target 응답
│   ├── representations/         # 실제 hidden vector
│   └── labeled/                 # 실제 응답의 failure 라벨
├── outputs/
│   ├── checkpoints/             # 실제 학습 checkpoint
│   ├── evaluation/              # 평가 재개 상태
│   ├── predictions/             # 실제 ID/OOD 예측
│   ├── reports/                 # 실제 실행의 통계·지표·분석
│   ├── figures/                 # 실제 평가 그림
│   └── logs/                    # 실제 학습 로그
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── RUNBOOK.md                   # Bash 설치·실행·재개 절차
```
