## 작업 방식

- **중요! PR 만드실 때 base를 dev 브랜치로 해주세요!**
-  다른 브랜치 작업에 의존하는 경우에만 해당 브랜치를 base로 지정해도 됩니다.
-  작업 전에는 `dev`의 최신 변경 사항을 가져와 개인 브랜치에 반영해 주세요.
- `main`에는 확정된 버전만 merge할 예정입니다.

### 브랜치별 역할
## 작업 방식

- **중요! PR 만드실 때 base를 dev 브랜치로 해주세요!**
-  다른 브랜치 작업에 의존하는 경우에만 해당 브랜치를 base로 지정해도 됩니다.
-  작업 전에는 `dev`의 최신 변경 사항을 가져와 개인 브랜치에 반영해 주세요.
- `main`에는 확정된 버전만 merge할 예정입니다.

### 브랜치별 역할
- `main` 브랜치: 확정된 코드를 보관
- `dev` 브랜치: 미확정된 작업중인 코드를 보관, 개인별 브랜치의 코드를 합칠 때 임시 저장 장소.
- 개인 브랜치: `식별가능한이름이나별명등-날짜-간단한작업내용` 형식으로 브랜치를 만들어 주시면 감사하겠습니다! (날짜 넣는 이유는 브랜치간 선후관계를 파악하기 쉽게 하기 위해)
    - 예시. hc-1109-directory


## 담당
- [A]: Rule Engine과 Gold State/Decision 계산
- [B]: 데이터셋 재구성·생성·라벨링·분할 코드, Predictor와 학습
- [C]: 실험과 평가. `[각자]`는 각 담당자가 자신의 모듈을 테스트한다는 뜻.

## 폴더 구조
초기 구조입니다. 변경이 필요한 경우 개인 브랜치에서 변경된 구조로 작업해주세요 dev에 반영하겠습니다

```text
conshift/
├── configs/                         # [공통] 데이터·모델·학습 설정
│   ├── data.yaml                    # [공통] 데이터 경로 및 생성·분할 설정
│   ├── model.yaml                   # [공통] 사용 모델 및 추론 설정
│   └── train.yaml                   # [공통] 학습 Hyperparameter
├── data/
│   ├── raw/                         # CoPriva/CAMBER 원본 보관
│   ├── annotated/                   # 구조화된 Disclosure Rule 및 State
│   ├── generated/                   # Qwen이 생성한 멀티턴 대화
│   ├── labeled/                     # Gold 및 Target Failure Label
│   └── splits/                      # Train/Validation/ID/OOD 데이터
├── prompts/                         # [B] 대화 생성·검증 및 Target 판단 프롬프트
│   ├── generation.txt               # [B] Qwen의 멀티턴 대화 생성용 7-shot 프롬프트
│   ├── validation.txt               # [B] 생성 대화와 구조화된 규칙의 일치 여부 검증
│   └── target_decision.txt          # [B] Target LLM의 ALLOW/DENY/CLARIFY 판단
├── src/
│   ├── rules/                       # [A] Rule Engine 및 Gold Label
│   │   ├── schema.py                # [A] Rule, State, Transition 등 구조 정의
│   │   └── engine.py                # [A] 규칙 변경 적용 및 Gold Decision 계산
│   ├── models/                      # [B] Predictor 및 Training
│   │   ├── dataset.py               # [B] Dataset, Tokenization, Label 구성
│   │   └── predictor.py             # [B] ModernBERT 및 Failure/State Heads
│   └── evaluation/                  # [C] 실험 및 성능 평가
│       ├── metrics.py               # [C] Failure/State 성능 지표
│       ├── ood_analysis.py          # [C] ID/OOD 성능 및 Representation 분석
│       └── failure_guided.py        # [C, 선택 H3] 실패 위험 기반 테스트 우선순위
├── scripts/                          # [B] 데이터셋 재구성 (Gold 계산은 [A])
│   ├── 01_data_setup.py             # [B] CoPriva/CAMBER 다운로드 및 원본 분석
│   ├── 02_prepare.py                # [B] 대화 생성·검증 / [A] Gold State/Decision 계산
│   └── 03_label_and_split.py        # [B] Target Failure Label 수집 및 데이터 분할
├── train.py                          # [B] Baseline A / ConShift C 학습
├── train_type_aware.py               # [C, 선택] Transition Type-aware 실험 B
├── evaluate.py                       # [C] ID/OOD 평가 및 모델 비교
├── tests/                            # [각자] 담당 모듈 테스트
├── outputs/
│   ├── checkpoints/                 # 학습 모델
│   ├── predictions/                 # 예측 결과
│   └── reports/                     # 평가 지표 및 그래프
├── requirements.txt                  # [공통] Python 의존성
└── README.md                         # [공통] 연구 설명 및 실행 방법
```

각 파일의 자세한 역할은 각 파일 docstring에 있습니다
