# ConShift 실험 실행

프로젝트 루트에서 실행합니다. Target은 동결한 `Qwen/Qwen3-4B-Instruct-2507`이며 CUDA를 사용할 수 있는 NVIDIA GPU가 필요합니다. Probe와 baseline만 학습하며 Target은 fine-tuning하지 않습니다. `system_prompt: null`로 별도의 system 메시지를 넣지 않습니다. 새 사용자 대화나 가상 실험 결과는 생성하지 않습니다.

## 환경과 의존성

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader -d data/raw/nltk_data -e punkt punkt_tab
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

CUDA 출력이 `True`인지 확인합니다. CPU용 PyTorch라면 해당 GPU/CUDA 환경과 호환되는 빌드를 설치해야 합니다. 모델 weight를 받을 디스크 공간과 추론용 GPU 메모리도 필요합니다. NLTK 리소스는 공식 문장 수 checker에 사용합니다.

## 전체 실행

```bash
bash scripts/run_experiment.sh configs/experiment.yaml
```

공식 원본 다운로드·확인 → 원본 정규화 → group split 고정 → train pilot → 전체 Target 수집 → 라벨링 → 다섯 모델 학습 → ID/OOD 평가 순서로 실행합니다. 동일 run의 완료 cache와 checkpoint를 재사용합니다. 오류나 선행 조건 미충족으로 종료하면 원인을 해결한 뒤 같은 명령으로 재개합니다.

실제 Target 응답이 확보되기 전에는 실패율이나 양·음성 표본 수가 결정되지 않습니다. 클래스 coverage가 기준에 미달하면 해당 학습·평가를 차단하며, 이를 맞추려고 원본·seed·라벨을 변경하지 않습니다.

## 단계별 실행

```bash
# 공식 원본 다운로드, hash 검사, trajectory 정규화
python scripts/01_audit.py --config configs/experiment.yaml --fetch

# 실제 응답을 보기 전에 group split과 OOD 조합 고정
python scripts/02_split.py --config configs/experiment.yaml

# Train의 20개 trajectory로 pilot; 양성이 없으면 최대 50개로 확장
python scripts/03_collect.py --config configs/experiment.yaml --pilot --resume

# Pilot 조건 통과 후 전체 trajectory 수집
python scripts/03_collect.py --config configs/experiment.yaml --resume

# 공식 checker로 실제 응답 라벨링
python scripts/04_label.py --config configs/experiment.yaml

# P1 / T1 / ConShift / G1 / L1 학습
python scripts/05_train.py --config configs/experiment.yaml --resume

# 같은 ID/OOD cohort에서 평가·분석·그림 생성
python scripts/06_evaluate.py --config configs/experiment.yaml --resume
```

단계 진행 상태와 차단 사유는 실행 시 `outputs/reports/progress.json`에 기록합니다. 원본이 이미 있으면 audit의 `--fetch`를 생략할 수 있습니다.

## 입력과 버전 고정

| 경로 | 용도 |
|---|---|
| `configs/experiment.yaml` | Target revision·생성 설정·학습 설정·평가 설정·입출력 경로 |
| `configs/sources.json` | 공식 EvolIF 파일·checker 코드·Target 설정의 URL·revision·SHA-256 |
| `data/raw/official_data/dialog_v0.1/` | 공식 원본 trajectory JSONL |
| `data/raw/official_code/` | 고정된 공식 checker와 대화 경계 근거 |
| `data/raw/target_metadata/` | 고정 모델의 config·tokenizer 설정 |
| `data/normalized/turns.jsonl` | 원본 `user_query_verified`를 그대로 유지한 정규화 레코드 |
| `data/manifests/` | 실행 시 생성되는 source·split·collection lock |

Dataset revision은 `47115ae2af4830948f3f15697221a1acca6078a7`, checker repository commit은 `00be379c882c15353d43dbabe0bf91ee4152b6c8`, model/tokenizer revision은 `cdbee75f17c01a7cc42f958dc650907174af0554`입니다. 버전 고정은 데이터·checker 변경 및 서로 다른 cache의 혼용을 방지합니다. GitHub commit이나 hash는 성능 수치가 아닙니다.

각 파일은 실제 공식 원본에서 받습니다. Source manifest와 split이 고정된 이후에는 임의로 삭제하거나 바꾸지 않습니다. Dataset attribution은 Qi Jia et al., EvolIF, CC BY-NC 4.0이며 원문을 재작성하지 않습니다. 공식 출처: [데이터](https://huggingface.co/datasets/KikiNLP/EvolIF), [코드](https://github.com/JiaQiSJTU/EvolIF), [논문](https://aclanthology.org/2026.acl-long.433/).

Git으로 전달되는 `data/`에는 폴더 유지용 `.gitkeep`만 포함됩니다. 원본 JSONL과 checker는 첫 실행의 `--fetch`에서 고정 버전으로 자동 다운로드합니다. 로컬에 이미 있는 원본은 hash가 일치하면 재사용합니다. 모델 weight도 지정한 revision으로 내려받으므로 최초 실행에는 인터넷 연결이 필요합니다.

## 응답 수집과 라벨

각 trajectory에서 원본 사용자 발화와 동일 Target의 실제 이전 응답으로 history를 구성합니다. 현재 응답 생성 전에 마지막 non-padding prompt token의 최종 반환 hidden state를 추출합니다. Prompt 및 token hash, 모델·tokenizer·template·dtype·코드·패키지 버전이 cache와 일치해야 재사용합니다.

기술 오류는 같은 설정으로 한 번 재시도합니다. Prompt 길이+최대 생성 길이 1024가 8192를 초과하면 자르지 않고 해당 turn과 후속 의존 turn을 제외합니다. 생성된 응답의 truncation은 그 turn의 라벨을 제외하고 history는 유지합니다.

공식 deterministic checker 9종을 사용합니다. 모든 제약 PASS이면 0, UNKNOWN 없이 FAIL이 하나 이상이면 1입니다. Judge가 필요한 emotion/reader_age/style, checker 오류, 활성 제약 없음, truncation은 주 cohort에서 제외합니다. 첫 turn과 현재 또는 직전 표현이 누락된 turn도 제외합니다. 원본의 `instruction_success`·`topic_success`는 Target failure label로 사용하지 않습니다.

공식 EvolIF의 interactive termination과 달리 고정 원본 trajectory를 재생하는 실험 프로토콜입니다. 공식 leaderboard 수치와 직접 비교하지 않습니다.

## Split과 최소 coverage

동일·파생 trajectory를 group으로 묶고, OOD 조합을 먼저 holdout한 뒤 seed=42로 train/validation/ID test를 70/15/15 비율로 정수 배정합니다. OOD 후보는 서로 다른 instruction ID 2개 이상, 최소 20개 group에서 관측되는 조합입니다. Holdout 이후 각 구성 ID가 train에 남는 후보 중 group 수가 가장 많은 조합 하나를 선택하고 동률이면 사전순으로 정합니다. OOD 조합이 나타나는 group 전체를 격리하며, 평가에는 해당 조합을 실제로 가진 적격 turn만 포함합니다.

`outputs/reports/label_coverage.json`에서 다음 조건을 검사합니다.

| Split | 양성·음성 turn 각각 | 해당 class 포함 group 각각 |
|---|---:|---:|
| Train | 30 이상 | 10 이상 |
| Validation / ID test / OOD | 10 이상 | 5 이상 |

전체 수집 완료와 ID 기준을 충족해야 학습합니다. OOD만 부족하면 OOD 평가를 제외하고 ID 실험을 유지합니다. 최소 기준이 통계적 표본 충분성을 보장하지는 않습니다.

## 학습과 중단 후 재개

| 항목 | 설정 |
|---|---|
| Seed | 42 |
| Optimizer / learning rate | Adam / 0.001 |
| Batch size / 최대 epoch | 64 / 100 |
| Patience | 10 epoch |
| Loss | BCEWithLogitsLoss + head weight에만 L2=0.0001 |
| Checkpoint 선택 | Validation AP 최대 → 동률이면 Brier 최소 → 동률이면 이른 epoch |
| F1 threshold | Validation에서 선택하고 ID/OOD에 고정 |

P1은 현재 normalized 표현, T1은 현재 표현과 직전 turn 대비 차분, ConShift는 cosine distance로 차분 기여를 조절하는 학습 가능한 gate를 사용합니다. Gate a/c는 0에서 시작합니다. 새로운 auxiliary task·state head·operation head는 없습니다. G1은 train-only TF-IDF+LR, L1은 train-only 길이 StandardScaler+LR입니다.

`outputs/checkpoints/`에 각 probe의 `*.last.pt`, `*.best.pt`, 최종 `p1.pt`·`t1.pt`·`conshift.pt`를 저장합니다. 매 epoch `last`에 optimizer·epoch·patience·best state·난수·shuffle 상태까지 저장하며 `best`는 최적 validation 모델입니다. G1/L1은 joblib와 metadata를 저장합니다.

`05_train.py --resume`은 완료 모델을 재학습하지 않고 중단된 동일 run만 이어갑니다. Liblinear는 내부 epoch checkpoint API가 없으므로 fit 도중 중단된 baseline만 같은 고정 설정으로 다시 fit합니다. 설정·데이터·feature가 바뀌면 재개를 차단합니다.

## 평가와 결과 저장

`06_evaluate.py --resume`은 `outputs/evaluation/`의 완료 모델 예측을 재사용합니다. Paired group bootstrap은 50 replicate마다 RNG와 진행 상태를 저장하고 같은 난수 순서로 재개합니다.

실제로 평가한 후 다음 결과를 생성합니다.

- `outputs/predictions/`: 동일 cohort의 ID/OOD raw prediction JSONL
- `outputs/reports/`: AP·AUROC·F1·Brier·ECE·prevalence, paired group bootstrap 2,000회·95% CI, latency, parameter count, gate·turn/shift 분석
- `outputs/figures/`: PR·ROC·reliability·gate-distance 그림
- `outputs/logs/`: 실제 epoch별 학습 이력

표현 추출 forward 비용과 생성 비용을 모두 기록합니다. Subgroup cutpoint는 train에서만 구합니다. Single-class AP/AUROC는 null로 기록하고 OOD 제외 사유를 명시합니다. 실행하지 않은 checkpoint·prediction·metric·결과표를 미리 생성하지 않습니다. 생성물은 Git에서 제외하고 폴더 구조만 `.gitkeep`으로 유지합니다.

## 개인정보 응용 데모

```bash
python scripts/07_demo.py --config configs/experiment.yaml --input /path/to/synthetic_privacy_case.json
```

실제 학습된 ConShift checkpoint가 필요합니다. 입력은 사용자가 제공한 합성 identifier·발화로 구성하며 `synthetic_identifiers_confirmed: true`, `turns`, turn별 `query`, `forbidden_identifiers`(value, 선택적 kind=phone), `policy_description`을 포함합니다. 첫 turn risk는 N/A이고, 이후 실제 생성 전 risk를 먼저 출력한 뒤 응답과 별도 identifier checker 결과를 표시합니다. 이는 일반 제약 predictor의 응용 시연이며 개인정보 유출 확률이나 방지 성능이 아닙니다.
