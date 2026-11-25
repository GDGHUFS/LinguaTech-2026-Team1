# ConShift 실험 실행

현재 실행 설정은 **입력(history 포함) 최대 56K + 출력 최대 4K = 총 context 60K**입니다. 저장 경로는 `input56k-output4k-vllm-tp2`이며 설치·실행은 [현재 vLLM 런북](RUNBOOK-VLLM.md)을 따릅니다. 아래 과거 56K/1K 설정의 포함률과 실행 시간 추정은 현재 설정에 적용하지 않습니다.

> **`originaldata-2` 브랜치의 설치·GPU 구성·실행은 [GPU 두 장 / vLLM 실행 문서](RUNBOOK-VLLM.md)를 따릅니다.** 아래 단일 GPU 추론 구현과 모델 추천은 이전 브랜치의 기록입니다. 현재 모델은 Qwen3-4B 그대로이며 기본 데이터·결과 경로는 `input56k-output4k-vllm-tp2`입니다. 아래 단계별 명령도 현재 YAML 경로를 사용합니다.

프로젝트 루트에서 실행합니다. Target은 동결한 `Qwen/Qwen3-4B-Instruct-2507`이며 CUDA를 사용할 수 있는 NVIDIA GPU가 필요합니다. Probe와 baseline만 학습하며 Target은 fine-tuning하지 않습니다. `system_prompt: null`로 별도의 system 메시지를 넣지 않습니다. 새 사용자 대화나 가상 실험 결과는 생성하지 않습니다.

기본 설정은 context **57,344 (56K)**, 최소 AP 개선량 0.001인 새 run입니다. 데이터는 `data/runs/context56k/`, 결과는 `outputs/runs/context56k/`에 저장하여 기존 8K 및 131K pilot과 분리합니다. 이번 실행은 전체 turn의 일부 제외를 허용하는 조건이며 모든 대화의 100% 보존 실험이 아닙니다. 매 응답에 1,024 token을 잡은 보수적 예산에서는 7,517/7,620 turn(98.65%), 144/150 trajectory(96%)를 포함할 수 있습니다. 실제 응답 길이와 제외율은 실행 후 보고합니다. 기존 cache·checkpoint를 새 경로로 복사해서 재개하거나 진행 중인 pilot의 설정을 중간에 바꾸지 않습니다.

현재 추가된 효율화는 동결 Target·cache 재사용·CPU probe 학습·L2 탐색 등입니다. vLLM, prefix 재사용, 모든 층 hidden state·전체 logits 반환 제거는 아직 구현하지 않았습니다. Context 상한을 131K에서 56K로 줄여도 실제 입력에 상한까지 padding하지 않으므로 시간·메모리가 비례해서 줄지는 않습니다. 56K를 넘는 후반 turn이 생략될 때 그만큼 줄어들며, 모든 실제 prompt가 이미 56K 이하면 처리량 차이가 거의 없을 수 있습니다. 기존 이틀 추정은 실측으로 다시 계산합니다.

## 환경과 의존성

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader -d data/runs/context56k/raw/nltk_data -e punkt punkt_tab
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

CUDA 출력이 `True`인지 확인합니다. CPU용 PyTorch라면 해당 GPU/CUDA 환경과 호환되는 빌드를 설치해야 합니다. 모델 weight를 받을 디스크 공간과 추론용 GPU 메모리도 필요합니다. NLTK 리소스는 공식 문장 수 checker에 사용합니다.

## GPU별 Target 모델 선택

`target.gpu_memory_fraction: 0.98`을 적용합니다. 모델을 GPU에 올리기 전에 현재 CUDA 장치의 PyTorch caching allocator 상한을 전체 visible GPU 메모리의 98%로 설정합니다. 남은 free memory의 98%라는 뜻은 아니며 다른 프로세스·CUDA 등 allocator 밖의 사용량까지 제한하지 않습니다. 2% 여유를 두는 보조 제한일 뿐, 필요 메모리가 상한을 넘으면 OOM이 발생합니다. 입력을 자동으로 줄이거나 모델을 자동 교체하지 않습니다. 설정은 수집 fingerprint에 포함됩니다.

**GPU의 실제 가용 VRAM에 맞춰 실험 시작 전에 Target 모델과 추론 방식을 선택해야 합니다. A100 80GB와 L40S를 같은 조건으로 취급하지 않습니다.** 이 프로젝트의 큰 GPU 비용은 작은 probe 학습이 아니라 Target의 긴 대화 응답 생성·표현 추출에서 발생합니다.

**전체 대화를 보존하면서 9B~14B급 Target으로 실험을 원활하게 진행하려면 전용 A100 80GB 또는 그 이상의 가용 VRAM을 갖춘 장비를 강력히 권장합니다. L40S를 동일한 대체 장비로 잡지 마세요.** L40S에서는 메모리 최적화·모델 선택·긴 입력 검증에 더 많은 제약이 생깁니다. GPU가 아직 확정되지 않았다면 장비 확보 계획은 A100 80GB 이상을 기준으로 세우고, L40S만 확보되면 아래 제한된 후보부터 검증합니다. 이 권장은 메모리 여유에 관한 것이며 A100 40GB는 포함하지 않습니다. 더 큰 VRAM도 현재 코드의 메모리 낭비나 모든 모델의 실행 성공을 보장하지 않습니다.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

L40S의 공식 물리 메모리는 48GB이며 A100은 40GB/80GB 제품을 구분해야 합니다. “L40S 40GB”라고 전달받았다면 GPU 이름과 실제 할당량·다른 프로세스 사용량을 위 명령으로 확인합니다. A100 80GB도 분할 또는 공유 환경에서는 실제 가용량이 적을 수 있습니다. 사양 출처: [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/), [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/).

| 실제 GPU / 가용 메모리 | Target 선택과 실행 기준 |
|---|---|
| A100 80GB, 충분한 전용 VRAM | 아래 추천 순서에 따라 12B~14B급 또는 효율적인 9B를 검토합니다. 추론 메모리 최적화와 긴 입력 검증 후 전체 수집합니다. 현재 등록된 4B 설정만으로 추천 후보가 구현된 것은 아닙니다. |
| L40S 48GB, 충분한 전용 VRAM | 아래 9B 후보부터 메모리 최적화 및 긴 입력 검증을 진행합니다. 현재 구현 그대로 전체 장기 대화 수집을 수행하는 기본 장비로 권하지 않습니다. 실패하면 더 작은 모델로 별도 실험을 구성하거나 A100 80GB 이상을 사용합니다. |
| L40S 등 실제 사용 가능 VRAM이 약 40GB 이하 | **현행 4B BF16 구현으로 약 77K 길이의 대화까지 처리하는 용도에는 메모리가 부족한 환경으로 취급합니다. 짧은 pilot이 실행돼도 전체 수집은 진행하지 않습니다.** 메모리 최적화 또는 모델 변경과 긴 입력 검증이 선행돼야 합니다. |

**L40S는 이번 장기 context 작업에서 A100 80GB보다 메모리 여유가 확실히 부족합니다. “4B weight가 올라간다”는 이유만으로 전체 파이프라인이 가능하다고 판단하면 안 됩니다.** 현재 `src/representation.py`는 `output_hidden_states=True`로 모든 층의 전체 token 표현을 반환하고, CausalLM forward의 전체 token logits도 반환합니다. 약 76K token에서 BF16 hidden state와 logits, 4B weight만으로도 합계 약 42GiB 수준이라는 단순 추산이 나옵니다. 이는 실측 peak가 아니며 attention·임시 tensor·CUDA allocator 등의 추가 메모리를 포함하지 않습니다. 따라서 40GB급은 현행 구현에 부족하고, 48GB도 성공을 보장할 여유가 아닙니다. 추론 메모리 최적화가 우선이며 모델 축소만으로 해결된다고 가정하지 않습니다.

### 모델 추천 순서

**Weight 전체가 GPU에 올라가는 것만으로 모델을 채택하지 않습니다. Weight 로딩 후 실행 중 추가되는 KV cache·중간 activation·표현 추출 tensor·임시 작업 공간까지 처리하고도 메모리 여유가 충분해야 합니다. 부족하면 모델 크기를 낮춥니다. 상한 0.98은 이 추가 메모리를 자동 확보하거나 OOM을 방지하지 않습니다.** 가장 긴 입력에서 peak memory를 측정하고 한도에 겨우 맞는 후보는 채택하지 않습니다.

현재 Target은 동결하므로 Target gradient·optimizer state는 만들지 않습니다. 큰 추가 GPU 메모리는 Target 추론·표현 추출에서 발생하며 작은 probe는 현재 CPU에서 학습합니다. 향후 Target fine-tuning까지 수행한다면 gradient·optimizer state·학습 activation 메모리까지 추가 산정해야 합니다. Weight와 실행 단계의 추가 메모리 합계가 장비 예산에 맞지 않으면 작은 모델로 내려갑니다. A100에서도 20B 이상은 선택적 상위 실험이며, 기본은 12B급입니다.

아래 순서는 이번 장문 제약 실패 예측 실험의 크기·context·메모리 구조·도입 부담을 고려한 검토 순서이며, 언어 성능의 실측 순위나 GPU 호환 보증이 아닙니다. 모든 후보는 표현 추출 최적화와 실제 긴 입력 검증을 전제로 합니다. 현재 기본 실행 모델은 여전히 4B이며 다음 후보로 자동 변경되지 않습니다.

| GPU | Primary 1xB candidates | Smaller fallback | Optional 20B+ experiment |
|---|---|---|---|
| A100 80GB+ | Gemma 4 12B IT first; Qwen3-14B as an older 1xB alternative (YaRN required for 128K) | Qwen3.5-9B if the 1xB candidates exceed the measured runtime budget | Gemma 4 26B A4B IT or Qwen3.8-27B ONLY after full-weight loading AND longest-input runtime memory validation; otherwise use 12B/14B |
| L40S 48GB | Gemma 4 12B IT, conditional on optimized extraction and longest-input validation | Qwen3.5-9B, then Gemma 4 E4B IT | Not recommended as a BF16 baseline on this device; quantization requires a separate experiment |

추천 근거는 2026-11-25 확인한 공식 공개 자료입니다. [Gemma 4 모델 카드](https://huggingface.co/google/gemma-4-12B-it)는 12B/26B A4B/31B의 256K context와 local/global attention, global layer의 통합 Keys/Values를 설명합니다. [Google 메모리 안내](https://ai.google.dev/gemma/docs/core)는 로딩과 20% overhead를 포함한 BF16 추산을 12B 약 26.7GB, 26B A4B 약 57.7GB, 31B 약 69.9GB로 제시합니다. 이 값은 이번 장문 표현 추출의 peak memory가 아니며 KV cache·중간 tensor를 추가 검증해야 합니다. 26B A4B는 활성 parameter가 약 4B여도 전체 weight를 저장하므로 L40S에서 BF16 기본 후보로 쓰지 않습니다. 31B도 A100 80GB에서 장문 여유가 줄어 기본 순위 밖의 상위 실험 후보로 둡니다.

[Qwen 공식 배포 목록](https://huggingface.co/Qwen)에 Qwen3.8-27B가 공개돼 있으며 [Qwen3.8-27B 모델 카드](https://huggingface.co/Qwen/Qwen3.8-27B)와 revision을 고정해 검증해야 합니다. Qwen4 서비스 출시를 로컬 실행용 소형 공개 weight 출시와 혼동하지 않습니다. 공식 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)는 Qwen4 아키텍처 preview라고 명시하지만 공식 목록상 180B급이므로 단일 A100/L40S의 우선 후보가 아닙니다. 검증 가능한 공식 checkpoint 없이 Qwen4 소형 모델을 추천하지 않습니다.

[Qwen3.5 공식 문서](https://huggingface.co/docs/transformers/model_doc/qwen3_5)는 빠른 kernel 부재 시 느리고 메모리를 더 쓰는 구현으로 fallback할 수 있음을 설명합니다. 모델이 새롭다는 사실만으로 메모리 효율·제약 준수·probe 성능이 보장되지는 않습니다. 후보별 tokenizer가 다르므로 기존 Qwen3-4B 기준 76,848 token 예산도 다시 측정해야 합니다. Gemma 3 12B와 Qwen3-14B는 기존 비교·재현 후보로 남기되 최신 모델 추천 순서에서는 제외합니다.

우선 BF16으로 검증하며 양자화는 별도 조건으로 비교합니다. Hidden state를 직접 사용하는 연구이므로 양자화에 따른 표현·응답·라벨 변화도 실험 조건으로 기록합니다. L40S에서 메모리 검증에 실패하면 context를 줄여 후반 대화를 버린 채 같은 실험이라고 보고하지 않고, A100 80GB 이상으로 옮기는 것을 우선합니다.

GPU별 선택은 다음 원칙으로 확정합니다.

1. 실제 VRAM을 확인하고 모델 후보·revision·tokenizer·dtype·context·추론 backend를 고정합니다. 짧은 turn만 있는 pilot은 장문 메모리 검증을 대신하지 못합니다. Train에서 가장 긴 대화와 약 77K 예산의 긴 입력으로 표현 추출과 생성을 모두 확인하고 peak memory·초당 생성량·오류를 기록합니다. 이 검증 결과는 본 학습·평가 데이터에 섞지 않습니다.
2. OOM 또는 처리 시간 초과가 발생하면 해당 GPU에서는 부족한 것으로 판정하고 추론 최적화·모델 변경·장비 변경 중 하나를 선택합니다. 단순히 context를 줄여 대화 후반을 버리는 것은 “전체 대화 보존” 조건을 충족하지 않습니다. 131,072는 길이 상한이며 실제 메모리는 입력 길이와 backend에 따라 달라집니다.
3. 모델이 바뀌면 Target 응답·실패 라벨·표현도 달라지므로 새 run에서 수집→라벨→probe 학습→평가를 모두 수행합니다. 같은 run의 split별·trajectory별로 모델을 자동 교체하거나 서로 다른 모델의 cache·checkpoint를 혼용하지 않습니다.

**현재 코드에는 GPU 감지에 따른 자동 모델 교체 기능이 없습니다.** 등록된 기본 모델은 여전히 4B 하나입니다. 다른 모델을 채택하려면 YAML뿐 아니라 `src/contracts.py`의 모델 고정 검사, `configs/sources.json`의 고정 모델 metadata와 hash, revision/tokenizer/chat template·thinking 설정 및 필요한 장문 설정을 함께 등록하고 데이터셋 token 길이도 다시 측정해야 합니다. Gemma/Qwen3.5는 loader와 텍스트 backbone·최종 층 표현 추출 경로도 모델에 맞춰 구현해야 합니다. 결과에는 GPU·가용 VRAM·모델·설정과 변경 이유를 기록합니다. 본 표는 실측으로 성공이 확인된 호환 목록이 아니라 실행 전 선택·검증 기준입니다.

### 현재 적용된 효율화와 미적용 항목

현재는 4-bit/8-bit/FP8 양자화를 사용하지 않습니다. Target은 BF16 지원 GPU에서 BF16, 그렇지 않으면 FP16으로 로드하며 이는 저비트 양자화가 아닙니다. Target 전체를 fine-tuning하지 않고 동결한 표현으로 작은 probe만 학습합니다. LoRA/QLoRA도 사용하지 않습니다.

| 항목 | 현재 구현 |
|---|---|
| Target 동결·gradient 비활성화 | 적용: `requires_grad_(False)`와 `torch.inference_mode()` |
| BF16/FP16 Target 추론 | 적용. Probe는 FP32이며 현재 CPU에서 학습 |
| 응답·표현 cache 및 학습 재개 | 적용. 이미 수집한 표현을 여러 probe에 재사용 |
| 생성 중 KV cache | 적용. 한 응답의 생성 내부에서 사용하며 turn 간 prefix KV 재사용은 구현하지 않음 |
| Early stopping | 적용. validation AP 최소 개선량 0.001, patience 10 |
| 4-bit/8-bit/FP8 weight 또는 KV cache 양자화 | 미적용 |
| LoRA/QLoRA·Target gradient checkpointing | 미적용. 현재 Target을 학습하지 않으므로 우선 해결할 병목이 아님 |
| FlashAttention/SDPA backend 명시 선택·검증 | 미적용. Transformers가 자동 선택할 수 있으나 실제 backend·성능을 확인하지 않음 |
| 마지막 층만 추출·불필요한 전체 logits 제거 | 미적용. 현재 모든 층 hidden state와 전체 logits를 반환 |
| 표현 추출과 생성의 prefill 공유 | 미적용. 현재 표현 추출 forward와 generation prefill을 별도로 수행 |

다음 효율화의 우선순위는 최종 층 표현만 추출하고 불필요한 logits를 제거하는 것, attention/backend 확인, 긴 입력의 메모리·속도 측정입니다. 그 후 prefix/prefill 재사용이나 양자화를 검토합니다. 기능 추가 시 표현의 정의·dtype·provenance와 재개 검증을 함께 맞춥니다. 본 항목은 현재 구현 상태 설명이며 미적용 기능을 활성화한 것이 아닙니다.

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

단계 진행 상태와 차단 사유는 실행 시 `outputs/runs/context56k/reports/progress.json`에 기록합니다. 원본이 이미 있으면 audit의 `--fetch`를 생략할 수 있습니다.

## 입력과 버전 고정

| 경로 | 용도 |
|---|---|
| `configs/experiment.yaml` | Target revision·생성 설정·학습 설정·평가 설정·입출력 경로 |
| `configs/sources.json` | 공식 EvolIF 파일·checker 코드·Target 설정의 URL·revision·SHA-256 |
| `data/runs/context56k/raw/official_data/dialog_v0.1/` | 공식 원본 trajectory JSONL |
| `data/runs/context56k/raw/official_code/` | 고정된 공식 checker와 대화 경계 근거 |
| `data/runs/context56k/raw/target_metadata/` | 고정 모델의 config·tokenizer 설정 |
| `data/runs/context56k/normalized/turns.jsonl` | 원본 `user_query_verified`를 그대로 유지한 정규화 레코드 |
| `data/runs/context56k/manifests/` | 실행 시 생성되는 source·split·collection lock |

Dataset revision은 `47115ae2af4830948f3f15697221a1acca6078a7`, checker repository commit은 `00be379c882c15353d43dbabe0bf91ee4152b6c8`, model/tokenizer revision은 `cdbee75f17c01a7cc42f958dc650907174af0554`입니다. 버전 고정은 데이터·checker 변경 및 서로 다른 cache의 혼용을 방지합니다. GitHub commit이나 hash는 성능 수치가 아닙니다.

각 파일은 실제 공식 원본에서 받습니다. Source manifest와 split이 고정된 이후에는 임의로 삭제하거나 바꾸지 않습니다. Dataset attribution은 Qi Jia et al., EvolIF, CC BY-NC 4.0이며 원문을 재작성하지 않습니다. 공식 출처: [데이터](https://huggingface.co/datasets/KikiNLP/EvolIF), [코드](https://github.com/JiaQiSJTU/EvolIF), [논문](https://aclanthology.org/2026.acl-long.433/).

Git으로 전달되는 `data/`에는 폴더 유지용 `.gitkeep`만 포함됩니다. 원본 JSONL과 checker는 첫 실행의 `--fetch`에서 고정 버전으로 자동 다운로드합니다. 로컬에 이미 있는 원본은 hash가 일치하면 재사용합니다. 모델 weight도 지정한 revision으로 내려받으므로 최초 실행에는 인터넷 연결이 필요합니다.

## 응답 수집과 라벨

Context 길이는 고정 Qwen tokenizer로 공식 원본 150개 trajectory / 7,620 turn을 측정하여 결정했습니다. 144개는 50 turn, 6개는 70 turn입니다. 가장 긴 `dialog_103`은 사용자 발화 합계 4,470 token, 빈 assistant frame을 포함한 마지막 prompt 5,168 token입니다. 모든 응답에 각각 최대 1,024 token을 잡은 전체 예산 최대는 76,848 token입니다. 현재는 비용과 일부 제외 허용 조건을 반영해 57,344를 채택합니다. 이 예산에서는 70-turn trajectory 6개의 후반부 일부가 제외됩니다. 원본에는 실제 Target 응답이 없어 계산상 포함률은 실측 결과가 아니며, 실제 decode 후 재토큰화된 history와 GPU 메모리는 실행 시 확인합니다. 생성 출력 상한은 여전히 1,024이므로 OUTPUT_TRUNCATED 제외는 별개입니다.

각 trajectory에서 원본 사용자 발화와 동일 Target의 실제 이전 응답으로 history를 구성합니다. 현재 응답 생성 전에 마지막 non-padding prompt token의 최종 반환 hidden state를 추출합니다. Prompt 및 token hash, 모델·tokenizer·template·dtype·코드·패키지 버전이 cache와 일치해야 재사용합니다.

기술 오류는 같은 설정으로 한 번 재시도합니다. Prompt 길이+최대 생성 길이 1024가 57344를 초과하면 자르지 않고 해당 turn과 후속 의존 turn을 제외합니다. 생성된 응답의 truncation은 그 turn의 라벨을 제외하고 history는 유지합니다.

공식 deterministic checker 9종을 사용합니다. 모든 제약 PASS이면 0, UNKNOWN 없이 FAIL이 하나 이상이면 1입니다. Judge가 필요한 emotion/reader_age/style, checker 오류, 활성 제약 없음, truncation은 주 cohort에서 제외합니다. 첫 turn과 현재 또는 직전 표현이 누락된 turn도 제외합니다. 원본의 `instruction_success`·`topic_success`는 Target failure label로 사용하지 않습니다.

공식 EvolIF의 interactive termination과 달리 고정 원본 trajectory를 재생하는 실험 프로토콜입니다. 공식 leaderboard 수치와 직접 비교하지 않습니다.

## Split과 최소 coverage

동일·파생 trajectory를 group으로 묶고, OOD 조합을 먼저 holdout한 뒤 seed=42로 train/validation/ID test를 70/15/15 비율로 정수 배정합니다. OOD 후보는 서로 다른 instruction ID 2개 이상, 최소 20개 group에서 관측되는 조합입니다. Holdout 이후 각 구성 ID가 train에 남는 후보 중 group 수가 가장 많은 조합 하나를 선택하고 동률이면 사전순으로 정합니다. OOD 조합이 나타나는 group 전체를 격리하며, 평가에는 해당 조합을 실제로 가진 적격 turn만 포함합니다.

`outputs/runs/context56k/reports/label_coverage.json`에서 다음 조건을 검사합니다.

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
| Batch size / 최대 epoch | 64 / 200 |
| Patience | 10 epoch |
| Loss | BCEWithLogitsLoss + head weight L2, 후보 0.0001 / 0.001 / 0.01 / 0.1 |
| Checkpoint 선택 | Validation AP 최대 → 동률이면 Brier 최소 → 동률이면 이른 epoch |
| F1 threshold | Validation에서 선택하고 ID/OOD에 고정 |

P1은 현재 normalized 표현, T1은 현재 표현과 직전 turn 대비 차분, ConShift는 cosine distance로 차분 기여를 조절하는 학습 가능한 gate를 사용합니다. Gate a/c는 0에서 시작합니다. 새로운 auxiliary task·state head·operation head는 없습니다. G1은 train-only TF-IDF+LR, L1은 train-only 길이 StandardScaler+LR입니다.

P1/T1/ConShift 각각 동일한 L2 후보 4개를 seed 42, Adam lr 0.001, batch 64, 최대 200 epoch, **patience 10 고정**으로 탐색합니다. 모델별 최적 L2는 validation AP 최대 → Brier 최소 → 이른 epoch → 미리 정한 후보 순서로 고릅니다. Test는 선택에 사용하지 않습니다. Gate 구조·BCE·head weight만의 L2 정의·split·checker·라벨은 유지합니다. 기존 최소 AP 개선량 0.001도 유지합니다. G1/L1의 학습 설정은 바꾸지 않습니다.

각 후보의 checkpoint는 `checkpoints/search/{모델}/l2-00`~`l2-03` 아래 `checkpoints/*.last.pt`, `*.best.pt`, `epochs/`에, epoch 로그는 후보 폴더의 `logs/`에 저장합니다. 최적 후보의 최종 `p1.pt`·`t1.pt`·`conshift.pt`는 출력 루트의 `checkpoints/`에 배치하며 선택 후보 로그도 루트 `logs/`로 복사합니다. `reports/{모델}_l2_search.json`에는 모든 후보의 validation 결과·최적 L2·epoch를 기록합니다. Metadata의 `training_config`는 선택한 계수이고 `search_config`는 전체 탐색 설정입니다. 매 epoch `last`에는 optimizer·epoch·patience·best state·난수·shuffle 상태까지 저장합니다. G1/L1은 joblib와 metadata를 저장합니다.

Epoch 로그에는 train/validation의 AP·BCE·Brier, train–validation AP 차이, validation–train BCE·Brier 차이와 patience 초기화 여부·stale epoch 수를 기록합니다. BCE는 epoch 종료 시 동일한 모델로 계산한 정규화 항 제외 loss입니다. 이 기록으로 train 성능만 개선되고 validation 성능은 악화되는지 확인합니다. L2·early stopping·group 분리는 과적합 위험을 줄이는 장치이며 과적합이 없음을 보장하지 않습니다. `early_stopping_min_delta=0.001`이며 마지막 patience 초기화 시점 대비 validation AP가 0.001 이상 증가해야 patience를 초기화합니다. 작은 AP 개선은 누적해서 판단하며 Brier 개선만으로는 초기화하지 않습니다. Best checkpoint는 중단 기준과 별개로 실제 validation AP 최대 → Brier 최소 → 이른 epoch 순서로 계속 저장합니다.

각 probe의 매 epoch 가중치도 `outputs/runs/context56k/checkpoints/search/conshift/l2-00/checkpoints/epochs/conshift.epoch-012.pt` 같은 경로에 보관합니다. 각 파일에는 해당 epoch의 validation 기반 threshold와 provenance가 포함됩니다. Train/validation 곡선을 보고 자동 best와 다른 epoch를 수동 선택할 수 있습니다. Epoch와 threshold 선택은 test를 보기 전에 확정합니다. 기본 평가 명령은 계속 자동 선택된 최종 checkpoint를 사용하므로, 수동 선택 시에는 선택한 epoch와 이유를 기록하고 해당 checkpoint로 평가 경로를 구성해야 합니다. 이미 평가한 run의 checkpoint를 덮어쓰거나 기존 평가 cache와 혼용하지 않습니다. BCE는 이진 분류의 cross entropy이고 AP는 average precision입니다.

`05_train.py --resume`은 완료 모델을 재학습하지 않고 중단된 동일 run만 이어갑니다. Liblinear는 내부 epoch checkpoint API가 없으므로 fit 도중 중단된 baseline만 같은 고정 설정으로 다시 fit합니다. 설정·데이터·feature가 바뀌면 재개를 차단합니다.

## 평가와 결과 저장

### 수집 완료 cache에서 CPU 학습만 다시 실행

Target 수집(03)과 라벨링(04)이 완료됐다면 아래 **05/06만** 실행합니다. 이 경로는 Target loader나 CUDA 추론을 호출하지 않습니다. L2·epoch 등 probe 설정 변경 때문에 01~04 또는 전체 실행 스크립트를 다시 돌리지 않습니다. 모델·context·생성 설정이 같은 응답/표현 cache를 재사용하며, 수집 code hash는 당시 provenance로 보존합니다. 학습 코드 hash는 별도 training lock에 기록합니다.

```bash
python scripts/05_train.py --config configs/experiment.yaml --output-dir outputs/runs/context56k-l2-v1 --resume
python scripts/06_evaluate.py --config configs/experiment.yaml --output-dir outputs/runs/context56k-l2-v1 --resume
```

`--output-dir`은 학습·평가 산출물만 분리합니다. 기존 config의 data 경로와 output의 `reports/label_coverage.json`을 그대로 읽으므로 응답·표현·라벨을 다시 만들지 않습니다. 기존 학습 결과도 덮어쓰지 않습니다. 설정이 달라진 학습은 새 output-dir을 사용하고, 같은 탐색의 중단만 `--resume`으로 잇습니다. Test 평가(06)는 후보 선택이 끝난 뒤 실행합니다. GPU 서버에서 결과를 가져올 때는 `data/representations/*.safetensors`, `data/responses/*.json`, `data/labeled/turns.jsonl`, manifests와 label coverage도 필요합니다. 앞서 공유한 결과 압축파일처럼 예측·checkpoint만 있는 파일은 전체 재학습 cache가 아닙니다. 경로 표기는 config의 data/output 루트 기준입니다.

CPU에서도 PyTorch·safetensors·scikit-learn 등 requirements는 필요하지만 Target weight 다운로드나 GPU는 필요하지 않습니다. 메모리는 전체 feature tensor를 담을 시스템 RAM이 필요합니다. 후보 탐색은 기존 고정 lambda 한 번의 fit과 다른 학습 프로토콜이며, 작은 validation에 대한 선택 편향을 없애는 방법은 아닙니다. 탐색 범위를 먼저 고정하고 test 결과로 후보를 추가하지 않습니다.

### P1 T1 ConShift 학습 진단

각 probe의 epoch 로그 `outputs/runs/context56k/logs/*_training.jsonl`의 `diagnostics`에 train/validation 각각의 gate 평균·표준편차·분위수·포화 비율, gate a/c, distance 분포, 현재 표현과 raw/gated 차분의 logit 기여, 기여의 평균 절댓값 비율과 부호 충돌 비율을 저장합니다. 클래스별 gate·distance·차분 기여와 train에서 정한 turn/distance 구간별 AP·BCE·Brier·표본 및 group 수도 기록합니다. Epoch 진단의 threshold는 고정 0.5이며 최종 선택 모델 보고서는 validation 선택 threshold를 사용합니다. 핵심 지표와 gate 분포는 epoch마다 콘솔에도 출력합니다.

`gate_0`, `gate_05`, `gate_1`, `gate_train_mean`은 동일하게 학습된 head에 gate만 바꾸는 추론 ablation입니다. 마지막 값은 해당 epoch의 train gate 평균을 validation에도 그대로 적용합니다. 이 비교는 각각 새로 학습한 P1/T1/고정 gate baseline이 아니며 인과적인 성능 보증으로 해석하지 않습니다. Gate가 적응적으로 동작하는지, 단순한 차분 크기 조절 효과인지 진단하는 자료입니다.

자동 best로 선택된 모델의 train/validation 상세 보고서는 `reports/{p1,t1,conshift}_diagnostics.json`, 표본별 예측·라벨·group·turn·gate·distance·logit 기여는 `diagnostics/{모델}_{train,validation}.jsonl`에 저장합니다. 이를 모델 간 sample_id로 비교할 수 있습니다. Test 데이터는 이 학습 진단에 사용하지 않으며 진단값은 loss·best 선택·early stopping 기준을 바꾸지 않습니다. 추가 epoch 진단이 없는 이전 학습 checkpoint는 새 signature와 맞지 않아 재개가 차단됩니다. 수집의 코드 hash도 변경되므로 기존 수집 run을 임의로 덮어쓰거나 hash를 수정해 재개하지 않습니다.

`06_evaluate.py --resume`은 `outputs/runs/context56k/evaluation/`의 완료 모델 예측을 재사용합니다. Paired group bootstrap은 50 replicate마다 RNG와 진행 상태를 저장하고 같은 난수 순서로 재개합니다.

실제로 평가한 후 다음 결과를 생성합니다.

- `outputs/runs/context56k/predictions/`: 동일 cohort의 ID/OOD raw prediction JSONL
- `outputs/runs/context56k/reports/`: AP·AUROC·F1·Brier·ECE·prevalence, paired group bootstrap 2,000회·95% CI, latency, parameter count, gate·turn/shift 분석
- `outputs/runs/context56k/figures/`: PR·ROC·reliability·gate-distance 그림
- `outputs/runs/context56k/logs/`: 실제 epoch별 학습 이력

표현 추출 forward 비용과 생성 비용을 모두 기록합니다. Subgroup cutpoint는 train에서만 구합니다. Single-class AP/AUROC는 null로 기록하고 OOD 제외 사유를 명시합니다. 실행하지 않은 checkpoint·prediction·metric·결과표를 미리 생성하지 않습니다. 생성물은 Git에서 제외하고 폴더 구조만 `.gitkeep`으로 유지합니다.

## 개인정보 응용 데모

```bash
python scripts/07_demo.py --config configs/experiment.yaml --input /path/to/synthetic_privacy_case.json
```

실제 학습된 ConShift checkpoint가 필요합니다. 입력은 사용자가 제공한 합성 identifier·발화로 구성하며 `synthetic_identifiers_confirmed: true`, `turns`, turn별 `query`, `forbidden_identifiers`(value, 선택적 kind=phone), `policy_description`을 포함합니다. 첫 turn risk는 N/A이고, 이후 실제 생성 전 risk를 먼저 출력한 뒤 응답과 별도 identifier checker 결과를 표시합니다. 이는 일반 제약 predictor의 응용 시연이며 개인정보 유출 확률이나 방지 성능이 아닙니다.
