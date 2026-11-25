# Qwen3-4B: vLLM Tensor Parallel 2

`originaldata-2` 브랜치에서는 **vLLM `tensor_parallel_size: 2`로 GPU 0과 GPU 1이 함께 응답을 생성**합니다. Qwen3-4B의 생성용 weight와 attention 연산을 두 GPU에 분산합니다. 최종 층 표현을 기존 방식으로 추출하기 위한 Transformers backbone 사본은 GPU 1에 별도로 상주시킵니다. 표현 추출을 마친 뒤 두 GPU에서 병렬 생성하므로 GPU가 총 세 장 필요한 구성은 아닙니다.

이것은 **하나의 응답을 두 GPU가 함께 계산하는 모델 병렬화**입니다. 서로 다른 대화를 GPU마다 하나씩 처리하는 데이터 병렬화는 적용하지 않습니다. 각 대화의 turn 순서도 유지합니다. A100 80GB 두 장 또는 L40S 두 장을 대상으로 하며, 실제 긴 입력의 메모리와 처리 시간은 서버에서 검증해야 합니다.

모델은 기존 `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`를 유지합니다. BF16, greedy 생성, 출력 1,024 token, context 57,344, seed 42, 원본 대화 순서와 probe 학습 설정도 유지합니다. Context를 늘리거나 양자화를 적용하지 않습니다.

## 설치

Linux x86_64, Python 3.11 또는 3.12, BF16 지원 NVIDIA GPU 두 장을 사용합니다. Windows native는 이 vLLM 수집 경로의 실행 대상이 아닙니다. 새 환경에서 vLLM과 호환되는 PyTorch를 함께 설치합니다. 설치 기준은 vLLM 0.11.2, Transformers 4.x입니다. [vLLM 설치 문서](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)

CUDA 12.8용 기본 wheel과 호환되는 NVIDIA 드라이버가 필요합니다. `nvidia-smi`의 드라이버와 CUDA 지원 버전을 확인합니다. GPU 두 장이 같은 Linux 호스트/컨테이너에서 보여야 하고 NCCL 통신이 가능해야 합니다. A100의 NVLink 연결은 성능에 유리하지만 필수 조건은 아닙니다. L40S는 PCIe로 통신하므로 TP=2가 한 장보다 빠른지는 실측해야 합니다. [vLLM 병렬 실행 문서](https://docs.vllm.ai/en/v0.11.2/serving/parallelism_scaling/)

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
python -m nltk.downloader -d data/runs/context56k-vllm-2gpu/raw/nltk_data -e punkt punkt_tab
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c 'import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.device_count())'
```

L40S 공식 VRAM은 48GB입니다. 실제 할당량이 40GB라면 그 용량을 기준으로 검증합니다. 두 GPU의 메모리는 하나로 합쳐지지 않습니다. [NVIDIA 사양](https://www.nvidia.com/en-us/data-center/l40s/)

## 실행

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_experiment.sh configs/experiment.yaml
```

Collector가 vLLM worker와 두 개의 TP rank를 자동 시작·종료하므로 **별도 `vllm serve` 명령이나 API key/포트 설정은 필요 없습니다.** 같은 GPU에서 별도 vLLM 서버를 띄우면 메모리 경쟁이 생깁니다. `CUDA_VISIBLE_DEVICES=2,3`으로 노출할 때도 YAML GPU 번호는 노출 후의 논리 번호입니다. 기본 생성 GPU 목록은 `[0, 1]`, 표현 추출 GPU는 `1`입니다. 두 GPU가 보이지 않거나 BF16 미지원이면 weight 로딩 전에 중단합니다.

## vLLM 시작 확인

첫 실행은 모델 weight 다운로드, NCCL 초기화, kernel compile과 CUDA graph 준비로 시간이 걸릴 수 있습니다. 기본 시작 timeout은 1,800초입니다. 모델 다운로드용 인터넷과 디스크 공간, GPU 두 장의 여유 VRAM이 필요합니다. 설치 및 시작만 먼저 확인하려면 다음을 실행합니다. 원본 데이터 다운로드 없이 두 추론 backend를 로드하고 정상 종료하며, 실험 응답·표현 파일은 생성하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/check_vllm.py --config configs/experiment.yaml
```

시작 과정의 `[ConShift] Starting vLLM tensor_parallel_size=2` 메시지와 `TP=2 ready`, 마지막 JSON의 `status: LOADED`, `tensor_parallel_size: 2`, 두 GPU 이름을 확인합니다. 시작 검사는 긴 입력의 생성·표현 추출 검사를 대신하지 않습니다. 실제 pilot을 실행하는 동안 다른 터미널에서 두 GPU 사용량을 확인할 수 있습니다.

```bash
watch -n 1 nvidia-smi
```

컨테이너에서는 두 GPU를 모두 전달하고 NCCL multiprocessing용 shared memory를 충분히 제공합니다. NCCL 초기화에서 멈추거나 실패하면 같은 호스트의 GPU 노출·PCIe P2P·shared memory부터 확인하고, 다음처럼 디버그 로그를 확보합니다.

```bash
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python scripts/check_vllm.py --config configs/experiment.yaml
```

## 단계별 실행

Pilot과 전체 수집을 나누어 실행하려면:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/01_audit.py --config configs/experiment.yaml --fetch
python scripts/02_split.py --config configs/experiment.yaml
python scripts/03_collect.py --config configs/experiment.yaml --pilot --resume
# Pilot 결과와 긴 train 대화의 메모리 사용량을 확인한 다음 실행
python scripts/03_collect.py --config configs/experiment.yaml --resume
python scripts/04_label.py --config configs/experiment.yaml
python scripts/05_train.py --config configs/experiment.yaml --resume
python scripts/06_evaluate.py --config configs/experiment.yaml --resume
```

데이터는 `data/runs/context56k-vllm-2gpu/`, 결과는 `outputs/runs/context56k-vllm-2gpu/`에 저장합니다. 기존 56K run은 덮어쓰지 않습니다. Backend/kernel 및 TP 설정 변경으로 BF16 계산과 응답이 이전 run과 달라질 수 있으므로 이전 Transformers 또는 GPU 역할 분리 버전의 응답·표현·라벨·checkpoint를 복사해서 재개하지 않습니다. 역할 분리 버전을 이미 실행했다면 새 `paths.data`와 `paths.output`으로 시작합니다.

## 추론과 표현 추출

각 turn의 마지막 non-padding prompt token에서 final decoder norm 이후 hidden vector를 추출한 다음 응답을 생성합니다. `AutoModel` backbone과 SDPA를 사용하므로 LM logits와 모든 층의 hidden-state tuple을 만들지 않습니다. 이는 기존 `hidden_states[-1]`과 같은 위치의 표현입니다.

vLLM에는 기존 tokenizer로 만든 prompt token ID를 그대로 전달합니다. Temperature 0, top-p 1, top-k 비활성화, repetition penalty 1, 고정 revision의 EOS ID를 사용하고, 반환된 token ID는 같은 tokenizer로 decode합니다. [vLLM SamplingParams](https://docs.vllm.ai/en/v0.11.2/api/vllm/sampling_params/)

생성은 별도 spawn 프로세스에서 TP=2, multiprocessing executor, prefix caching과 chunked prefill을 사용합니다. Prefix cache는 worker가 살아 있는 동안만 유지되며 디스크 재개 시 다시 계산합니다. 표현 추출과 생성은 순서대로 실행하며, 생성 단계 내부에서는 두 GPU가 병렬 연산합니다. 여러 trajectory의 동시 batch 처리는 구현하지 않습니다. 속도 향상 배수는 아직 측정하지 않았습니다.

모델·설정·코드·vLLM 및 관련 패키지 버전·GPU 이름을 collection fingerprint에 기록합니다. 같은 run 도중 코드·설정·패키지를 변경하지 않습니다. 응답·표현 cache와 학습 재개 검사는 유지합니다.

## 메모리와 오류 처리

기본 vLLM 예산은 **GPU마다 전체 VRAM의 50%**, 표현 추출 프로세스의 PyTorch allocator 상한은 **공유 GPU의 40%**입니다. 예를 들어 장당 40GB이면 vLLM은 각 장에서 약 20GB를 예산으로 잡고, 표현 추출은 GPU 1에서 약 16GB의 allocator 상한을 사용합니다. 두 비율은 전체 VRAM 기준이며 vLLM이 두 장의 합계에서 50%를 쓰는 뜻이 아닙니다. 이 값은 측정된 필요량이나 실제 할당량이 아닙니다.

```yaml
target:
  gpu_memory_fraction: 0.40
  representation_gpu: 1
  vllm:
    generation_gpus: [0, 1]
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.50
```

두 예산의 합계가 90%를 넘는 설정은 차단합니다. 표현 추출이 끝나면 미사용 PyTorch allocator cache를 반환합니다. CUDA context·NCCL·비-PyTorch 메모리와 다른 프로세스의 사용량은 별도이므로 이 설정도 OOM 방지를 보장하지 않습니다. 실제 가용량에 맞춘 조정은 **새 run 시작 전에** 합니다. vLLM 비율을 올리면 공유 GPU에서 표현 추출에 쓸 수 있는 예산이 줄어듭니다. 두 비율을 함께 검토하며, 각각의 비율을 단독으로 0.9 이상으로 올리지 않습니다. CUDA graph 메모리 문제를 진단하려면 새 run에서 `enforce_eager: true`를 비교할 수 있습니다.

vLLM이 KV cache 부족으로 시작에 실패하거나 longest-input 검증에서 OOM이 나면 해당 장비·설정의 실행 성공이 확인된 것이 아닙니다. 생성과 표현 추출 양쪽의 메모리를 확인하고 예산을 다시 배분합니다. Context·모델·dtype을 몰래 바꾸어 이어서 실행하지 않습니다.

Pilot에서 생성과 표현 추출을 모두 확인하고, 긴 train 대화에서 GPU별 peak memory와 실행 시간을 측정한 뒤 전체 수집을 진행합니다. 짧은 pilot 성공만으로 56K 입력 성공을 보장하지 않습니다. 상한을 넘는 turn과 후속 의존 turn은 기존 기준으로 제외합니다.

정상 종료·예외 발생 시 worker를 정리합니다. Worker가 죽거나 timeout이면 collection을 중단하여 나머지 데이터를 오류 cache로 채우지 않습니다. 원인 해결 후 같은 코드·설정으로 `--resume`합니다. 일반 생성 오류는 기존처럼 한 번 재시도하며, 두 번 실패한 trajectory는 기술 오류로 기록합니다.

수집 중 `Ctrl+C`로 중단하면 worker와 TP rank 정리를 기다립니다. 중단한 단계는 동일한 `03_collect.py ... --pilot --resume` 또는 `03_collect.py ... --resume` 명령으로 다시 시작합니다. vLLM worker는 새로 기동되며 이미 commit된 응답·표현 cache는 재사용합니다. 임의로 GPU 프로세스를 전체 종료하지 말고, 비정상 종료 후 남은 프로세스가 있다면 `nvidia-smi`로 이 실행에 속한 PID인지 확인합니다.

Probe만 재학습할 때는 `requirements.txt`만 필요하고 vLLM이나 CUDA는 필요하지 않습니다. 다만 해당 run에서 만든 응답·표현·라벨과 provenance가 있어야 합니다. 데이터·split·checker·학습·평가 프로토콜은 [기존 실행 문서](RUNBOOK.md)를 참고합니다.
