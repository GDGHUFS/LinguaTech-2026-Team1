# ConShift GPU 담당자 실행 안내 — L40S 1장

작성 기준: 2026-11-18, 현재 작업 트리의 코드. **VSCode에서 GPU 서버에 연결한 통합 터미널의 Bash**와 Python 3.11을 기준으로 한다.

이 문서의 명령은 모두 GPU 서버의 Bash에 입력한다. VSCode에서 `Terminal → New Terminal`로 Bash를 열고 저장소 폴더로 이동한다. Windows에서 VSCode를 사용하더라도 실행 위치는 Remote-SSH로 연결한 Linux GPU 서버다. 로컬 Git Bash는 원격 GPU를 자동으로 사용하지 않는다. PowerShell 명령이나 Docker 설치·컨테이너 실행은 이 절차에 필요 없다.

코드 블록 안의 명령만 순서대로 복사한다.

## 1. 사용할 모델과 실행 원칙

| 역할 | 권장 모델 | 판단 |
|---|---|---|
| 대화 생성 + 2차 의미 검증 | `Qwen/Qwen3-8B` | 유지. BF16 가중치는 약 16GB이며 40GB 가용 GPU에서 단독 추론을 시작하기에 적절하다. Thinking을 끄고 JSON을 받는다. |
| 실패 관찰용 Target | `Qwen/Qwen3-4B-Instruct-2507` | 유지. 현재 `src/data/labeling.py`에 고정돼 있다. 모델 변경은 실험 대상과 기존 label/provenance 계약을 바꾸므로 단순 설정 교체가 아니다. |
| 실패 예측기 학습 | `answerdotai/ModernBERT-base` | 유지. 현재 A/B_pred/C/C+ 구현과 맞고 영어 데이터에 적합하다. Qwen을 학습하는 작업이 아니다. |

**지금은 위 조합을 그대로 추천한다.** 더 큰 생성기나 ModernBERT-large가 이 과제에서 더 낫다는 실측 근거는 없다. 먼저 이 조합으로 생성 품질·실패 양성 비율·VRAM·입력 길이를 확인한다. 생성기와 Target이 같은 계열이라 오류가 상관될 수 있으므로 Qwen 의미 검사와 symbolic oracle을 함께 적용한다. 자동 의미 검사는 독립적인 사람 평가 결과로 보고하지 않는다.

표준 L40S는 48GB 제품이지만 이 절차는 사용자에게 주어진 **실제 가용 40GB**를 예산으로 삼는다. `nvidia-smi`로 실제 모델/메모리를 확인한다. Qwen 8B 서버 → 서버 종료 → Target 4B 라벨링/activation → 프로세스 종료 → ModernBERT 순서로 실행한다. 세 모델을 동시에 띄우지 않는다. 양자화·LoRA는 기본 경로에 쓰지 않는다. 가중치 크기는 추정치이며 KV cache/activation/runtime 메모리는 별도다.

근거: [Qwen3-8B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-8B), [Target 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [ModernBERT 논문](https://arxiv.org/abs/2412.13663), [vLLM serve 옵션](https://docs.vllm.ai/en/stable/cli/serve/), [NVIDIA L40S 제품 정보](https://www.nvidia.com/en-us/data-center/l40s/).

## 2. 자동 생성 원칙

기본 `generation_mode: camber_automatic`이다. CAMBER 원문 상황·등장인물·정보·공개 맥락을 Qwen에 전달해 인용 근거와 함께 자동 정규화한다. 독립적인 새 상황으로 대체하지 않는다. 원문에서 공개 목적이나 필수 scope를 확인할 수 없으면 이유를 기록하고 제외한다. 원본 동의는 추정하지 않으며 초기 ALLOW는 별도의 가상 정책 개입이다. 기존 저장소의 합성 예시 7개를 자연화 prompt에 첨부한다. 사람 검수·수동 승인 파일 없이 의미 검사와 symbolic oracle 재계산으로 진행한다. holdout은 Target 실행 전에 seed 42로 배정한다.

seed 42, max_epochs 10, patience 3, AdamW, batch 8 × accumulation 2가 기본이다. 이 문서는 브랜치의 코드·config·prompt와 실행 시 생성하는 산출물만 사용한다.

## 3. 서버 환경 준비

관리자가 NVIDIA driver, Python 3.11 + venv, curl을 준비해야 한다. RAM 64GB 이상, 여유 디스크 100GB 이상은 운영 권장치다. 인터넷은 최초 패키지/모델/원본 다운로드에 필요하며 유료 외부 추론 API는 필요 없다.

전달받은 저장소 루트에서 시작한다. 아래 `/path/to/conshift`만 **GPU 서버의 실제 저장소 경로**로 바꾼다. 이후 모든 명령은 같은 Bash 터미널과 저장소 루트에서 실행한다. 터미널을 바꾸면 아래 재접속 명령으로 변수를 복원한다.

SSH 연결을 끊어도 장시간 학습을 유지하려면 선택적으로 `tmux new -s conshift`를 먼저 실행하고, 그 안에서 아래 명령을 실행한다. tmux는 필수가 아니다. tmux 없이 실행할 때는 VSCode 연결과 터미널을 유지한다. `set -euo pipefail`은 실패 시 이후 단계를 중단하며, 오류로 Bash가 종료되면 새 터미널에서 원인을 해결하고 재접속 절차로 복원한다.

```bash
cd /path/to/conshift
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/.hf-cache"
export RUN="l40s_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$RUN" "runs/$RUN/approvals" "runs/$RUN/logs"
printf '%s\n' "$RUN" > runs/ACTIVE_RUN
export R="runs/$RUN"
nvidia-smi | tee "$R/logs/nvidia-smi-before.txt"
git status --short > "$R/logs/git-status.txt"
git rev-parse HEAD > "$R/logs/git-head.txt"

python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install --upgrade pip
.venv-train/bin/python -m pip install -r requirements-training.txt 'transformers>=4.55,<5'
.venv-train/bin/python -m pip check
.venv-train/bin/python -m pip freeze > "$R/logs/train-requirements.lock.txt"

# vLLM은 자체 torch 의존성이 있으므로 학습 환경과 분리한다.
python3.11 -m venv .venv-serve
.venv-serve/bin/python -m pip install --upgrade pip
.venv-serve/bin/python -m pip install vllm
.venv-serve/bin/python -m pip check
.venv-serve/bin/python -m pip freeze > "$R/logs/serve-requirements.lock.txt"
.venv-serve/bin/vllm serve --help > "$R/logs/vllm-help.txt"
grep -- '--default-chat-template-kwargs' "$R/logs/vllm-help.txt"

source .venv-train/bin/activate
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA가 보이지 않음: driver/torch 설치를 먼저 해결'
print(torch.cuda.get_device_name(0))
print('total GB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('BF16:', torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported()
PY
python -m pytest tests -q | tee "$R/logs/tests.txt"
```

설치는 서버 driver와 호환되는 torch/vLLM wheel이 필요하다. 위 설치가 실패하면 해당 오류를 먼저 해결한다. 성공 환경의 lock 파일을 이후 재현에 사용한다. 이 설치 조합은 이 문서 작성 PC에서 검증한 고정 lock이 아니다. Transformers는 저장소의 `<6` 허용 범위 중 4.x를 선택해 API 변화 범위를 줄였다.

VSCode에서 새 Bash 터미널을 열거나 GPU 서버에 재접속했을 때:

```bash
cd /path/to/conshift
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/.hf-cache"
export RUN="$(cat runs/ACTIVE_RUN)" R="runs/$(cat runs/ACTIVE_RUN)"
source .venv-train/bin/activate
```

## 4. 모델 revision 고정 및 다운로드

아래는 최초 시점의 각 모델 commit을 조회하고 가중치를 캐시에 받는다. 이후에는 기록한 SHA를 유지한다. 서버에만 configs를 수정한다. freeze 후에는 configs를 수정하지 않는다.

```bash
python - <<'PY'
import os, json, hashlib
from pathlib import Path
import yaml
from huggingface_hub import HfApi, snapshot_download
r = Path(os.environ['R'])
models = ['Qwen/Qwen3-8B', 'Qwen/Qwen3-4B-Instruct-2507', 'answerdotai/ModernBERT-base']
pins = {m: HfApi().model_info(m).sha for m in models}
for m, rev in pins.items():
    snapshot_download(m, revision=rev)
(r/'model-pins.json').write_text(json.dumps(pins, indent=2)+'\n')
p = Path('configs/model.yaml'); c = yaml.safe_load(p.read_text())
c['encoder']['revision'] = pins[models[2]]
c['encoder']['tokenizer_revision'] = pins[models[2]]
c['target']['revision'] = pins[models[1]]
digest = hashlib.sha256()
for f in ['prompts/generation.txt','prompts/validation.txt','prompts/naturalization_shots.json']:
    digest.update(f.encode()); digest.update(Path(f).read_bytes())
c['generation']['model_id'] = models[0]
c['generation']['revision'] = pins[models[0]]
c['generation']['prompt_version'] = 'sha256:'+digest.hexdigest()
p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
export GEN_REV="$(python -c 'import json,os; print(json.load(open(os.environ["R"]+"/model-pins.json"))["Qwen/Qwen3-8B"])')"
export TARGET_REV="$(python -c 'import json,os; print(json.load(open(os.environ["R"]+"/model-pins.json"))["Qwen/Qwen3-4B-Instruct-2507"])')"
```

재접속 후에는 마지막 `GEN_REV` / `TARGET_REV` 두 줄도 실행한다. `SPLIT_VERSION`은 9절에서 다시 설정한다.

## 5. canonical 준비

```bash
python scripts/01_data_setup.py --canonical-out "$R/canonical.jsonl" --out "$R/seed_inventory.csv"
python scripts/05_preflight.py --out "$R/preflight-before.json"
```

canonical 100개가 이미 준비됐다면 재사용한다. 원본이 준비돼 있으면 setup에 `--offline`을 붙일 수 있다. canonical의 pending 상태는 사람 승인으로 해소할 필요가 없다. purpose는 원문 근거로 자동 추출하며, 확인할 수 없으면 제외한다. 승인 수량을 맞추기 위한 대체 생성은 없다. preflight는 config의 기본 경로를 읽으므로 이 run의 별도 경로는 뒤의 직접 검증으로 확인한다.

## 6. Qwen3-8B 서버 및 생성

여기서 말하는 서버는 **GPU 서버 안에서 Python/vLLM으로 실행하는 Qwen 추론 프로세스**다. Docker를 쓰는 단계가 아니다. 저장소의 `src/data/generation.py`에 있는 `QwenClient.complete()`가 HTTP 요청을 보내도록 구현돼 있어 모델을 받는 프로세스를 먼저 띄운다. vLLM은 이 인터페이스를 제공하기 위해 이 문서에서 선택한 실행 도구이며 저장소 자체에 내장된 서버가 아니다.

`127.0.0.1:8000`은 같은 GPU 서버 안의 Qwen 프로세스 주소다. 아래 `curl`은 Bash에서 해당 프로세스의 준비 상태와 JSON 응답을 확인한다. 외부 API 호출이나 컨테이너 명령이 아니다. `/v1/models` 확인은 기동 대기용이고 `/v1/chat/completions` 확인은 실제 생성 전 간단한 응답 테스트다.

현재 QwenClient는 `/v1/chat/completions`에 temperature 0, max_tokens 4096, JSON object를 요청한다. 클라이언트가 thinking 옵션을 보내지 않으므로 **서버 기본값에서 끈다**. 대화 자연화는 저장소의 합성 예시를 사용하는 7-shot이다. 입력과 응답이 함께 들어가도록 context 16384로 시작한다.

```bash
export QWEN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
VLLM_SERVER_DEV_MODE=1 nohup .venv-serve/bin/vllm serve Qwen/Qwen3-8B \
  --revision "$GEN_REV" --tokenizer-revision "$GEN_REV" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --api-key "$QWEN_API_KEY" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 --max-model-len 16384 --max-num-seqs 1 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  > "$R/logs/vllm.log" 2>&1 &
export SERVE_PID=$!
printf '%s\n' "$SERVE_PID" > "$R/vllm.pid"
# 모델을 메모리에 올리는 동안 준비될 때까지 기다린다.
READY=0
for i in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer $QWEN_API_KEY" http://127.0.0.1:8000/v1/models > "$R/served-models.json"; then READY=1; break; fi
  kill -0 "$SERVE_PID" || { tail -n 80 "$R/logs/vllm.log"; exit 1; }
  sleep 2
done
test "$READY" = 1
# 같은 GPU 서버의 Qwen에 JSON 응답을 요청하는 연결 테스트다.
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $QWEN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Return JSON only: {\"ok\":true}"}],"temperature":0,"max_tokens":64,"response_format":{"type":"json_object"}}' \
  > "$R/server-smoke.json"
python - <<'PY'
import json,os
x=json.load(open(os.environ['R']+'/server-smoke.json'))
assert json.loads(x['choices'][0]['message']['content'])=={'ok':True}
PY

# 먼저 2개 source로 API/생성/검증 연결만 확인한다.
python scripts/02_prepare.py --input "$R/canonical.jsonl" \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B --revision "$GEN_REV" \
  --limit-seeds 2 --execution-mode pilot \
  --output "$R/smoke-generated.jsonl" --audit "$R/smoke-generation-audit.csv" \
  | tee "$R/logs/generation-smoke.txt"
```

smoke audit의 오류와 `semantic_complete_oracle_verified` family 비율을 확인한 뒤 전체 canonical group을 생성한다. smoke는 처리량 측정이며 최종 데이터에 합치지 않는다. 아래는 limit을 canonical 수로 주므로 묵시적 20개 제한을 피한다.

```bash
N_SOURCES=$(wc -l < "$R/canonical.jsonl")
python scripts/02_prepare.py --input "$R/canonical.jsonl" \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B --revision "$GEN_REV" \
  --limit-seeds "$N_SOURCES" --execution-mode final \
  --output "$R/generated-all.jsonl" --audit "$R/generation-audit.csv" \
  | tee "$R/logs/generation.txt"
kill "$SERVE_PID"
wait "$SERVE_PID" || true
unset QWEN_API_KEY
nvidia-smi
```

클라이언트는 같은 서버의 `/server_info?config_format=json`을 읽고 모델 ID, revision, tokenizer revision과 로드된 HF config commit을 대조한다. 이 endpoint를 쓰는 위 연구용 vLLM은 loopback에만 바인딩한다. dev endpoint를 외부에 공개하지 않는다. 서버가 로드된 commit을 제공하지 않거나 값이 다르면 `NOT VERIFIED`로 실패하며, 지정한 SHA만으로 검증을 우회하지 않는다. 해당 서버 버전의 실제 JSON 노출 여부는 GPU pilot에서 확인해야 한다.

정규화·제외 기록은 audit CSV와 같은 이름의 `.source.jsonl`에 자동 저장된다. 추출 실패 때문에 최종 source 수가 canonical 수보다 적거나 0일 수 있다. 이유와 raw 정규화 응답을 확인하고, 원문 사실을 채워 넣거나 독립 상황으로 대체하지 않는다.

서버 종료와 GPU 메모리 회수를 확인한다. OOM이면 다른 GPU 프로세스를 먼저 확인하고 서버 메모리/context를 조정한다. context를 무조건 줄이면 입력이 잘릴 수 있다. generation은 마지막에 파일을 쓰므로 중간 장애 시 자동 resume가 아니다. 재시도는 새 output/audit 경로를 사용한다. 실제 1회 요청 timeout은 180초다.

## 7. 자동 검증 통과 family 선별

```bash
python - <<'PY'
import os,json
from pathlib import Path
from collections import Counter
from src.data.approvals import enforce_execution
r=Path(os.environ['R'])
rows=[json.loads(x) for x in (r/'generated-all.jsonl').read_text().splitlines() if x.strip()]
print(Counter(x['validation_status'] for x in rows))
ok=[x for x in rows if x.get('controlled_family_status')=='semantic_complete_oracle_verified']
assert ok, '완전 검증 family 없음: generation_raw/validation_raw/errors 확인'
enforce_execution(ok,mode='pilot')
(r/'generated.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in ok))
print('자동 검증 통과:',len(ok))
PY
```

Qwen 의미 검사, CAP별 인용 근거, oracle gold/support 재계산을 통과한 family만 사용한다. 실패 family는 제외한다. 사람 검수 파일을 작성하는 단계는 없다.

## 8. Target pilot → 전체 라벨링과 split

```bash
python scripts/03_label_and_split.py --input "$R/generated.jsonl" \
  --revision "$TARGET_REV" --execute --limit 200 --execution-mode pilot \
  --labeled-output "$R/pilot-target.jsonl" --splits-dir "$R/pilot-splits" \
  --report "$R/pilot-target.csv" | tee "$R/logs/target-pilot.txt"
```

pilot에서 attempted 최소 100, JSON valid rate, 첫 응답 valid rate, OOM/latency, max_new_tokens 64 충분 여부를 확인한다. pilot subset은 eligible sample_id 정렬 앞 200개라 모든 partition을 대표하지 않을 수 있다. 본 실험은 아래 전체 라벨로 별도 생성한다. freeze 전 전체 데이터 준비 단계이므로 execution-mode는 pilot을 유지한다.

```bash
python scripts/03_label_and_split.py --input "$R/generated.jsonl" \
  --revision "$TARGET_REV" --execute --full-run --execution-mode pilot \
  --labeled-output "$R/target-all.jsonl" --splits-dir "$R/splits" \
  --report "$R/target-all.csv" | tee "$R/logs/target-all.txt"
python - <<'PY'
import os,json
from pathlib import Path
from collections import Counter
from evaluate import read_rows
from src.evaluation.protocol import _manifest,_valid
r=Path(os.environ['R']); rows=read_rows(r/'splits')
m=json.loads((r/'splits/split_manifest.json').read_text()); _manifest(m,rows)
for p in ['train','validation','id_test','ood_composition','ood_template']:
    counts=Counter(x['failure_label_unsafe'] for x in rows if x['split']==p and _valid(x))
    print(p,dict(counts))
    assert set(counts)=={0,1}, f'{p}: 두 class 미확보. 최종 실험 중단'
PY
```

unsafe label은 **gold DENY → Target ALLOW**다. 단순 불일치와 다르다. 클래스가 한쪽뿐이면 그 상태로 AUPRC 모델 선택을 강행하지 않는다. Target을 약하게 바꾸거나 test 실패를 보고 split을 재배치하지 말고, 사전에 정한 source/CAP 확보 계획을 검토한다. INVALID_OUTPUT/EXECUTION_ERROR는 음성 label로 치환하지 않는다.

## 9. ModernBERT 실제 pilot과 최종 config 확정

```bash
export SPLIT_VERSION="$(python -c 'import json,os; print(json.load(open(os.environ["R"]+"/splits/split_manifest.json"))["manifest_hash"])')"
python train.py --variant C+ --train "$R/splits/train.jsonl" \
  --validation "$R/splits/validation.jsonl" --split-version "$SPLIT_VERSION" \
  --seed 42 --device cuda --pilot --execution-mode pilot \
  --output-dir "$R/pilot-checkpoints" \
  | tee "$R/logs/train-pilot.txt"
```

`--pilot`는 dry-run/1-step이 아니다. 실제 학습과 pilot 지표 기록이며 기본 max10/patience3가 적용된다. ModernBERT trainer는 이 문서에서 BF16/gradient checkpointing이 구현됐다고 가정하지 않는다. 기본 batch8/length2048이 실제 40GB에서 되는지 여기서 측정한다. OOM이면 freeze **이전**에 `configs/train.yaml`의 batch를 4, accumulation을 4로 바꿔 effective16을 유지하고 새 pilot 경로로 다시 실행한다. 2×8도 같은 원칙이다. 입력 truncation >5%, evidence loss 또는 request 손실이면 공통 max_length 정책을 검토하고 모든 모델에 똑같이 적용한다. length를 늘리면 메모리도 다시 측정한다. test 성능으로 설정을 고르지 않는다.

최종 baseline 경로와 실행 승인을 서버 config에 연결한다. 변경은 최종 freeze 전에 완료한다.

```bash
python - <<'PY'
import os,yaml
from pathlib import Path
r=Path(os.environ['R'])
p=Path('configs/baselines.yaml'); c=yaml.safe_load(p.read_text())
c.update(records=str(r/'splits'),manifest=str(r/'splits/split_manifest.json'),
    target_revision=os.environ['TARGET_REV'],execution_mode='final',
    qa_manifest=None,freeze_manifest=str(r/'approvals/freeze.json'),
    g1_output=str(r/'baselines/tfidf_lr'),p1_output=str(r/'baselines/target_probe'),
    activations=str(r/'baselines/target_probe/activations.jsonl'))
p.write_text(yaml.safe_dump(c,sort_keys=False))
p=Path('configs/model_registry.yaml'); c=yaml.safe_load(p.read_text())
for name in ['G1','majority']: c['models'][name]['artifact']=str(r/'baselines/tfidf_lr')
c['models']['P1']['artifact']=str(r/'baselines/target_probe')
for name,folder in [('A','A'),('B_pred','B_pred'),('C','C'),('C+','C_plus')]:
    c['models'][name]['artifact']=str(r/'checkpoints'/folder)
p.write_text(yaml.safe_dump(c,sort_keys=False))
PY
```

## 10. 자동 freeze

최종 config 조정 후 실행한다. split manifest와 group/controlled-pair 누수, 실제 Qwen 7-shot·의미 검사·oracle evidence trace, Target provenance, 생성 Qwen의 서버 보고 revision 및 record/config/holdout hash를 검증하고 자동 고정한다. split 전체가 있는 디렉터리를 입력해야 한다.

```bash
python scripts/08_freeze.py --records "$R/splits" --output "$R/approvals/freeze.json"
```

데이터나 config가 바뀌면 새 경로에 자동 freeze를 다시 만든다. 이후 명령은 이 freeze 경로를 참조한다.

## 11. 최종 A/B_pred/C/C+ 학습

```bash
for MODE in A B_pred C C+; do
  python train.py --variant "$MODE" --train "$R/splits/train.jsonl" \
    --validation "$R/splits/validation.jsonl" --split-version "$SPLIT_VERSION" \
    --seed 42 --device cuda --execution-mode final \
    --freeze-manifest "$R/approvals/freeze.json" \
    --output-dir "$R/checkpoints" | tee "$R/logs/train-${MODE}.txt"
done
```

순차 실행한다. `C+` checkpoint 폴더명은 `C_plus`다. 기본값은 `seed_42/trial_0/best.pt`와 `last.pt`다. 선택은 validation AUPRC, 동률은 낮은 Brier이며 test는 선택에 사용하지 않는다. 중단 재개는 동일 명령에 `--resume "$R/checkpoints/C/seed_42/trial_0/last.pt"`처럼 해당 모델의 last.pt를 추가한다. config와 데이터는 그대로 유지한다. `best.pt`는 평가용이다. 여러 모델을 한꺼번에 background로 실행하지 않는다.

## 12. 필수 비교 baseline G1/P1

```bash
# Qwen 8B 서버가 종료된 상태. 라벨 수집과 동일한 Target revision을 사용한다.
python scripts/04_export_target_pregen.py --config configs/baselines.yaml \
  | tee "$R/logs/p1-activations.txt"
python -m src.evaluation.baselines_pre --config configs/baselines.yaml \
  | tee "$R/logs/g1.txt"
python -m src.evaluation.baselines_target_probe --config configs/baselines.yaml \
  | tee "$R/logs/p1.txt"
```

P1은 Target 응답 후 feature가 아니라 응답 생성 전 중간층 마지막 non-padding token feature다. G1은 TF-IDF + logistic regression이다. 둘 다 train fit/validation 선택을 지킨다. P1 extraction은 긴 입력에 대해 모든 hidden state를 반환하므로 별도 peak VRAM을 확인한다. OOM 시 test 데이터를 줄여 결과를 만들지 말고 입력 길이/구현 문제를 보고한다. B_oracle/P2는 이 기본 비교에 넣지 않는다.

## 13. 예측 export → validation threshold freeze → 최종 평가

```bash
mkdir -p "$R/predictions"
for MODE in A B_pred C C_plus; do
  python export_predictions.py \
    --checkpoint "$R/checkpoints/$MODE/seed_42/trial_0/best.pt" \
    --records "$R/splits" --out "$R/predictions/$MODE.jsonl" --device cuda \
    --execution-mode final \
    --freeze-manifest "$R/approvals/freeze.json"
done
cp "$R/baselines/tfidf_lr/seed_42/predictions.jsonl" "$R/predictions/G1.jsonl"
cp "$R/baselines/target_probe/seed_42/predictions.jsonl" "$R/predictions/P1.jsonl"
python evaluate.py --records "$R/splits" --predictions "$R/predictions" \
  --split-manifest "$R/splits/split_manifest.json" --protocol configs/eval.yaml \
  --freeze --out "$R/eval_protocol.json" --execution-mode final \
  --freeze-manifest "$R/approvals/freeze.json"
python evaluate.py --records "$R/splits" --predictions "$R/predictions" \
  --split-manifest "$R/splits/split_manifest.json" --protocol "$R/eval_protocol.json" \
  --out "$R/evaluation.json" --execution-mode final \
  --freeze-manifest "$R/approvals/freeze.json"
```

predictions 폴더에는 위 여섯 모델의 예측만 넣는다. majority sanity/embedding/activation 파일을 섞으면 중복 또는 미등록 모델 오류가 난다. eval freeze는 validation에서만 threshold를 정한다. ID/OOD test 결과는 고정 protocol로 평가한다. 단일 seed 결과이므로 seed 간 안정성을 검증했다고 주장하지 않는다.

## 13.1. A/C/C+ frozen representation probing

최종 평가 protocol을 고정한 뒤 같은 checkpoint와 공통 sample ID에서 CLS를 export한다. P1 Target activation probe와 다른 state decodability 분석이다.

```bash
mkdir -p "$R/embeddings"
for MODE in A C C_plus; do
  python export_predictions.py \
    --checkpoint "$R/checkpoints/$MODE/seed_42/trial_0/best.pt" \
    --records "$R/splits" --out "$R/embeddings/${MODE}-scores.jsonl" \
    --embeddings-out "$R/embeddings/${MODE}.jsonl" --device cuda \
    --execution-mode final --freeze-manifest "$R/approvals/freeze.json"
done
cat "$R/embeddings/A.jsonl" "$R/embeddings/C.jsonl" "$R/embeddings/C_plus.jsonl" > "$R/frozen-cls.jsonl"
python -m src.evaluation.representation_probe \
  --embeddings "$R/frozen-cls.jsonl" --records "$R/splits" \
  --manifest "$R/splits/split_manifest.json" --protocol "$R/eval_protocol.json" \
  --out "$R/state-probing.json"
```

## 14. 결과 전달 및 장애 대응

```bash
nvidia-smi > "$R/logs/nvidia-smi-after.txt"
cp configs/{data,model,train,baselines,eval,model_registry}.yaml "$R/"
python - <<'PY'
import os,hashlib
from pathlib import Path
r=Path(os.environ['R'])
with (r/'SHA256SUMS').open('w') as out:
    for p in sorted(r.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':
            h=hashlib.sha256()
            with p.open('rb') as src:
                for block in iter(lambda:src.read(1024*1024),b''): h.update(block)
            out.write(f'{h.hexdigest()}  {p.relative_to(r)}\n')
PY
tar -czf "${RUN}-results.tar.gz" "$R"
```

전달물: run 전체, model-pins/환경 lock, source/generation/Target audit, 자동 검증 기록·freeze, split manifest, 학습 로그·checkpoint, G1/P1 fit/metadata, 예측, eval protocol/최종 지표, SHA256SUMS. 로그/데이터는 민감할 수 있으므로 팀의 승인된 전달 경로를 사용한다. HF 모델 캐시는 결과 압축에 포함하지 않는다.

| 증상 | 확인/조치 |
|---|---|
| `No generation-ready seeds` | endpoint/model 및 Qwen 시나리오 JSON 응답 확인. scenario_errors에 실패 seed와 오류 종류가 기록된다. |
| JSON parse / `<think>` 출력 | vLLM thinking 기본값, smoke content, generation_raw/validation_raw 확인. |
| 자동 의미 검사 실패 | generation_raw/validation_raw/errors 확인 후 새 output에서 재생성. |
| `Existing ... preserved` | 기존 결과 덮어쓰기 방지. 새 run/output 경로 사용. |
| Target pilot가 incomplete | valid 생성물이 100개 미만인지, parser/OOM/실행 오류인지 audit 확인. |
| validation에 한 class만 있음 | 데이터 확보 계획 재검토. test를 보고 분할/Target 변경 금지. |
| freeze hash mismatch | freeze 이후 변경한 config/record가 있는지 확인. 고정한 파일 복원 또는 새 자동 freeze. |
| GPU OOM | 단계별 모델 종료 확인, ModernBERT batch/accumulation pilot 재측정. 최종 freeze 전 공통 설정 확정. |

완료 기준은 명령이 종료된 것만이 아니다. Qwen/oracle/freeze 검증 PASS, 실제 Target 관찰 label과 두 class 확보, 여섯 모델의 실제 artifact/예측, 고정 protocol 평가 성공, 재현 자료 전달까지 확인한다.
