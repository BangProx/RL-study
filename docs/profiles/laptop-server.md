# 실제 모델 laptop에서 verl server까지

> C8 기준일: 2026-08-19. 실제 검증과 실행하지 않은 recipe를 같은 표에서
> 구분한다. 모델 weight, dataset, checkpoint는 이 저장소에 재배포하지 않는다.

## 1. 먼저 preflight만 실행하기

```bash
rl-study preflight \
  --profile laptop \
  --model laptop-smoke \
  --device cpu \
  --json
```

preflight는 network를 호출하지 않는다. exact Hub ID/revision, license, 예상 bytes,
cache 상태, dependency version, 실제 tensor forward/backward device probe와 메모리
산식을 출력한다. SmolLM2가 cache에 없으면 `requires_accept_download=true`다.

임의 model ID도 가능하지만 Hub metadata를 즉석에서 믿지 않는다. 사용자가 먼저
감사한 세 값을 함께 줘야 한다.

```bash
rl-study preflight --profile laptop --model ORG/MODEL \
  --revision FULL_COMMIT --license-id SPDX_ID \
  --expected-weight-bytes INTEGER --device cpu --json
```

revision/license/size 중 하나라도 없으면 network 전에 실패한다.

## 2. download guard

100,000,000 bytes를 넘는 uncached asset은 `--accept-download` 없이는 exit code 4로
종료한다. guard는 Transformers, Datasets, Hub client를 import하기 전에 실행되므로
실패 경로에 DNS/HTTP 요청이 없다.

```bash
# 정보만 확인: download 없음
rl-study train --config configs/laptop/smollm2_lora_sft.yaml --json

# ID/revision/license/bytes를 검토한 뒤에만 실행
rl-study train --config configs/laptop/smollm2_lora_sft.yaml \
  --accept-download --stop-after 1 --json
```

현재 preset은 다음 세 개다.

| preset | exact model | weight bytes | 용도/상태 |
|---|---|---:|---|
| laptop-smoke | SmolLM2-135M-Instruct `12fd25f…` | 269,060,552 | 이 Mac CPU에서 실제 LoRA 검증 |
| laptop-quality | Qwen3-0.6B `c1899de…` | 1,503,300,328 | preflight만; 미다운로드·미실행 |
| server | Qwen3-4B `1cfa9a7…` | 8,044,982,000 | verl external-manual |

## 3. LoRA one-step에서 실제로 학습되는 것

Transformers로 exact base revision을 읽고 PEFT `LoraConfig`를 `q_proj,v_proj`에
부착한다. base parameter는 모두 frozen이어야 하고 `lora_*`가 아닌 parameter에
gradient가 하나라도 생기면 실패한다. prompt token은 label `-100`으로 가리고
completion token만 SFT loss에 넣는다.

SmolLM2 실제 결과는 전체 134,975,808 parameter 중 460,800개, 즉 `0.3414%`만
trainable이었다. adapter-only safetensors, tokenizer, optimizer, data cursor,
resolved config와 file별 SHA-256을 checkpoint에 저장한다. base weight는 복제하지
않는다.

```text
base model (frozen, Hub cache)
  └─ q_proj/v_proj LoRA A·B (trainable)
       ├─ adapter_model.safetensors
       ├─ optimizer.pt
       ├─ state.json: step, split/prompt IDs, token/forward budget
       └─ experiment-card.json
```

continuous 2 step과 1 step + resume + 1 step의 adapter 파일은 byte 단위로 같았고
SHA-256은 `0c56d05d…c0890f9`였다. `training.steps` 연장만 허용하며 model revision,
dataset revision, LoRA rank/target, dtype와 mask 의미가 바뀌면 load 전에 거부한다.

## 4. GSM8K contamination 경계

`openai/gsm8k@740312a…`의 `main/train`만 학습 loader가 읽는다. question SHA-256
ID를 seed와 함께 안정 정렬해 256개 validation을 train에서 분리한다. 실제 C8
split hash는 `sha256:7c2cb77e…49978cb3`다. 공식 `test` 요청은
`purpose=final_evaluation`인 evaluator-only 호출이 아니면 optional dependency를
import하기 전 차단한다.

C8 metric의 validation loss는 이 파생 validation 중 고정된 첫 8개 예시 평균이다.
이는 GSM8K exact-match benchmark가 아니며, 짧은 adapter update가 유한하고
재현되는지 확인하는 smoke metric이다.

## 5. 메모리 추정은 항별로 보기

preflight는 다음을 따로 출력한다.

\[
M = M_{base}+M_{adapter}+M_{grad}+M_{Adam}+M_{activation}+M_{runtime}
\]

- base: dtype당 parameter bytes, QLoRA일 때만 0.5 byte heuristic
- adapter: preset의 q/v LoRA parameter 예상치
- gradient: trainable value당 4 bytes
- Adam state: trainable value당 8 bytes
- activation: `batch × sequence × hidden × layers × dtype × 8`
- runtime: Python/PyTorch/Transformers를 위한 768MiB
- 권장치: 합계에 20% margin

초기 runtime 항이 없던 추정 741MB보다 실제 peak RSS 1.258GB가 컸다. 이 실패를
근거로 runtime 항을 분리 추가했다. 현재 산식은 OOM 방지 참고치이지 보장이 아니며,
실제 card의 peak와 계속 보정해야 한다.

## 6. QLoRA를 지원한다고 말할 수 있는 조건

QLoRA는 config 이름만으로 켜지지 않는다. CUDA forward/backward probe와
bitsandbytes 설치가 모두 통과해야 한다. 이 Mac의 CPU probe에서는 명시적으로
`supported=false`이며 LoRA나 CPU full fine-tuning으로 조용히 fallback하지 않는다.

## 7. TRL laptop adapter의 역할

`requirements/laptop.lock`은 실행 환경의 exact version을 보존한다. C8 실행은
TRL 1.10.0 / Transformers 4.57.6 / PEFT 0.20.0 / Datasets 4.8.5 /
Accelerate 1.14.0을 사용했다. 작은 SFT smoke runner는 mask·gradient·checkpoint를
눈으로 추적하기 위해 직접 Transformers+PEFT를 사용한다. 이후 실제 DPO/GRPO/RLOO
확장에는 `TRLAdapterSpec`이 trainer class와 TRL revision `a7be897…`을 명시한다.
DAPO/Dr. GRPO/GSPO는 TRL 이름 아래 억지로 합치지 않고 verl 경로로 보낸다.

## 8. verl server recipe: 검증됨과 실행됨은 다르다

```bash
rl-study render-server \
  --config configs/server/qwen3_4b_dapo_verl.yaml --json
```

renderer는 verl 0.9.0 / commit `483b8a0…`, model revision, algorithm variant와
Hydra override를 검증한다. DAPO는 `token-mean`, group filter, asymmetric clip,
overlong buffer를, GSPO는 `loss_mode=gspo`, `seq-mean-token-mean`, 좁은 sequence
clip을 별도 recipe로 만든다.

서버에는 실제 parquet 경로를 사용자가 채워야 한다. 렌더된 argv의 핵심 형태는:

```text
python3 -m verl.trainer.main_ppo --config-name ppo_trainer
data.train_files=REPLACE_WITH_TRAIN_PARQUET
data.val_files=REPLACE_WITH_VALIDATION_PARQUET
actor_rollout_ref.model.path=Qwen/Qwen3-4B
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.loss_agg_mode=token-mean
algorithm.filter_groups.enable=True
reward_model.overlong_buffer.enable=True
trainer.n_gpus_per_node=8 trainer.nnodes=1
```

현재 환경에는 Linux CUDA 8-GPU와 verl이 없으므로 결과는 반드시
`run_status=external-manual`, `result_origin=not_executed`, `local_executed=null`이다.
성공 판정은 verl version/GPU probe, 유한한 optimizer step, checkpoint, token·forward·
peak-memory card가 모두 생기는 것이다. 실행하지 않은 성능 수치는 없다.

## 9. 이 Mac에서 발견한 import 지연

PyTorch 2.13의 external device backend 자동 탐색이 설치된 plugin 환경에서 수 분
걸렸다. CPU toy/laptop 검증은 다음처럼 사용하지 않는 backend autoload를 끄면
즉시 시작됐다.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 rl-study preflight --profile laptop --device cpu
```

CUDA/XPU 같은 외부 backend plugin이 실제로 필요하면 이 값을 무조건 복사하지
말고 해당 backend의 설치 문서를 먼저 확인한다.

## Sources

- `model-smollm2-135m-instruct`, `model-qwen3-0.6b`, `model-qwen3-4b`
- `dataset-gsm8k`, `dataset-ultrafeedback-binarized`
- `framework-trl`, `framework-verl`, `framework-pytorch`
- PEFT 공식 LoRA/quantization 문서: `LoraConfig`, `get_peft_model`, QLoRA 조건
