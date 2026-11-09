# 작업 방식
- `main` 브랜치: 확정된 코드를 보관
- `dev` 브랜치: 미확정된 작업중인 코드를 보관, 개인별 브랜치의 코드를 합칠 때 임시 저장 장소.
- 개인 브랜치: `식별가능한이름이나별명등-날짜-간단한작업내용` 형식으로 브랜치를 만들어 주시면 감사하겠습니다! (날짜 넣는 이유는 브랜치간 선후관계를 파악하기 쉽게 하기 위해)
    - 예시. hc-1109-directory

# 간단한 파일구조 및 실행방법

```text
conshift/
├── configs/data.yaml # [공통] 데이터 및 모델 설정
├── configs/model.yaml # [공통] 데이터 및 모델 설정
├── configs/train.yaml # [공통] 데이터 및 모델 설정
├── src/rules/schema.py # State, Rule, Transition 구조 정의
├── src/rules/engine.py # 규칙 변경 적용 및 Gold Decision 계산
├── src/models/dataset.py # Tokenization, Tensor 및 Label 구성
├── src/models/predictor.py # ModernBERT + Failure/State Heads
├── src/evaluation/metrics.py # AUPRC, AUROC, Brier, ECE, State F1
├── src/evaluation/ood_analysis.py # OOD 성능 및 Representation 분석
├── src/evaluation/failure_guided.py # 실패 위험 기반 테스트 우선순위 평가
├── scripts/04_generate_gold.py # [A] Gold State/Decision 생성
├── scripts/05_collect_failures.py # [A] Target LLM 실행 및 Failure Label 생성
├── train.py # [B] Baseline A / ConShift C 학습
├── train_type_aware.py # [C] 실험 B: Transition Type-aware Baseline
├── evaluate.py # [C] ID/OOD Test 및 모델 간 성능 비교
├── requirements.txt # [공통] 패키지 목록
├── src/rules/ # [A] Privacy Rule 처리
├── src/models/ # [B] Predictor 구현
├── src/evaluation/ # [C] 실험 및 평가
├── tests/ # [각자] 담당 모듈 테스트
├── outputs/ # [공통] Checkpoints, Predictions, Results
└── README.md # [공통] 실행 방법
```
