# C2 기술 아키텍처

> 설계 버전: 1.0
> 구현 기준: Python 3.10~3.12, PyTorch 2.13.x
> 원칙: offline core, strict config, explicit device/download, reproducible artifacts

## 1. 책임 경계

```text
notebook / CLI / demo
          │
          ▼
 application layer: experiment runner, comparison, report, checkpoint
          │
     ┌────┴───────────────┐
     ▼                    ▼
reference layer       adapter layer
clean-room PyTorch    TRL / verl / ALFWorld
     │                    │
     └──── typed contracts┘
          │
          ▼
math · data · env · model · artifact primitives
```

core import는 Transformers, TRL, verl, PEFT, ALFWorld, CUDA 또는 network를
요구하지 않는다. optional dependency를 top-level에서 import하지 않고 adapter
factory 내부에서만 import한다. 없는 extra는 설치할 extra와 현재 profile을
포함한 설명적인 오류를 낸다.

## 2. Package 경계

| module | 책임 | 의존 가능한 내부 module |
|---|---|---|
| `rl_study.math` | probability, KL, mask reduction, return, GAE | torch만 |
| `rl_study.types` | immutable batch/transition/result dataclass | torch, stdlib |
| `rl_study.config` | strict config parse/validate/hash | stdlib, PyYAML |
| `rl_study.data` | TinyReasoning 생성, split/hash, preference, guarded external loader | types/config |
| `rl_study.envs` | bandit와 GridWorld typed env | types/config |
| `rl_study.models` | tiny tokenizer/causal LM, policy/value/reward head | math/types |
| `rl_study.algorithms` | 순수 loss/update와 작은 trainer | math/types/models |
| `rl_study.agentic` | tool env, trace, parser, mask, credit, policy | models/math |
| `rl_study.training` | runner, seed, device, checkpoint/resume | config/algorithms |
| `rl_study.evaluation` | metrics, fair-comparison audit | config/types |
| `rl_study.diagnostics` | KL/entropy/length/reward-hacking checks | evaluation/math |
| `rl_study.adapters` | TRL/verl/ALFWorld optional boundary | public core contracts |
| `rl_study.reporting` | experiment card, JSON/PNG/HTML | artifact schema |
| `rl_study.ui` | 기존 artifact의 read-only explorer | reporting schema |
| `rl_study.cli` | `train/eval/preflight/inspect-run` routing | application layer |

algorithm module은 report HTML이나 CLI argument를 알지 못한다. adapter는 core
private function을 import하지 않고 parity가 필요한 public loss function만 쓴다.

## 3. Typed data contract

실제 구현은 `dataclass(frozen=True, slots=True)`를 기본으로 한다. tensor type은
runtime validation에서 dtype, rank와 동일 batch dimension을 확인한다.

```python
@dataclass(frozen=True, slots=True)
class TransitionBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor       # bool [B]
    truncated: Tensor        # bool [B]
    behavior_logprobs: Tensor | None = None

@dataclass(frozen=True, slots=True)
class PreferenceBatch:
    prompt_ids: Tensor       # int64 [B, P]
    chosen_ids: Tensor       # int64 [B, Tc]
    rejected_ids: Tensor     # int64 [B, Tr]
    chosen_mask: Tensor      # bool [B, Tc]
    rejected_mask: Tensor    # bool [B, Tr]
    prompt_uids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class TokenTrajectoryBatch:
    prompt_ids: Tensor
    response_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor
    old_logprobs: Tensor | None
    reference_logprobs: Tensor | None
    values: Tensor | None
    rewards: Tensor
    advantages: Tensor | None
    returns: Tensor | None
    episode_ids: Tensor
    step_ids: Tensor | None
    terminated: Tensor
    truncated: Tensor
```

불변 조건:

- ID tensor는 `int64`, mask와 종료 신호는 `bool`, loss 입력은 floating dtype이다.
- `old_logprobs`와 `reference_logprobs`는 rollout 저장 시 detach되어 있다.
- `action_mask=True`인 위치는 response를 생성한 model token뿐이다.
- prompt, padding, tool output, environment observation token은 기본 policy loss에서
  제외한다.
- `terminated=True`와 `truncated=True`를 한 signal로 합치지 않는다.
- sequence reward `[B]`를 token reward `[B,T]`로 broadcast할 때 credit strategy를
  명시적으로 기록한다.

Agentic 전이는 원 token을 보존한다.

```python
@dataclass(frozen=True, slots=True)
class AgentStep:
    observation: AgentObservation
    context_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    candidate_action_token_ids: tuple[tuple[int, ...], ...]
    chosen_candidate_index: int
    behavior_logprob: float
    action: AgentAction
    tool_output: ToolOutput
    process_reward: float
    outcome_reward: float
    next_observation: AgentObservation | None
    terminated: bool
    truncated: bool
    policy_version: int
```

`action_token_ids → decode → parse`는 허용하지만 parse 결과를 다시 encode한 token을
학습 action으로 대체하지 않는다.

## 4. Environment와 algorithm protocol

```python
class Environment(Protocol[ObservationT, ActionT]):
    def reset(self, *, seed: int) -> tuple[ObservationT, dict[str, object]]: ...
    def step(self, action: ActionT) -> StepResult[ObservationT]: ...

class Algorithm(Protocol[BatchT]):
    name: str
    def loss(self, batch: BatchT) -> LossOutput: ...
    def update(self, batch: BatchT) -> UpdateMetrics: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

`StepResult`는 `observation`, `reward`, `terminated`, `truncated`, `info`를
분리한다. `info`에 성공 여부, invalid action, task ID 같은 관찰용 값을 두되
trainer가 숨은 정답을 policy 입력에 넣지 못하도록 observation과 타입을 나눈다.

loss 함수는 가능한 한 optimizer와 분리한 pure function이다. 예:

```python
def ppo_policy_loss(
    current_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_low: float,
    clip_high: float,
    reduction: MaskedReduction,
) -> PPOLossOutput: ...
```

`PPOLossOutput`은 scalar loss뿐 아니라 unclipped/clipped objective, approximate KL,
clip fraction과 valid token count를 반환한다. notebook이 private intermediate를
재계산할 필요가 없게 한다.

## 5. Strict config

config는 YAML을 사람이 쓰되 load 직후 immutable dataclass로 변환한다. unknown
key, 잘못된 enum, 범위 밖 값, profile과 맞지 않는 option은 학습 전에 실패한다.
환경변수는 config를 조용히 덮어쓰지 않는다.

```yaml
schema_version: 1
profile: toy
algorithm:
  name: grpo
  variant: paper
  gamma: 0.99
  clip_low: 0.2
  clip_high: 0.2
  kl_coefficient: 0.0
  advantage_normalization: group_std
  loss_reduction: token_mean
data:
  id: tiny_reasoning
  revision: generated-v1
  split: train
  seed: 42
model:
  policy: tiny-v1
  reference: tiny-v1-frozen
  reward: deterministic-verifier
training:
  seed: 42
  steps: 100
  batch_size: 16
  group_size: 4
  response_token_budget: 32768
  device: cpu
  allow_device_fallback: false
evaluation:
  every_steps: 25
  split: validation
output:
  root: artifacts
```

### Immutable resume fields

- schema/profile/algorithm name와 수학 의미가 달라지는 variant
- model architecture와 initial policy hash
- dataset revision, split IDs/hash와 generator version
- optimizer 종류, token/environment budget 정의
- tokenizer/vocabulary와 action mask policy

`steps`, evaluation cadence, logging verbosity처럼 공정성을 깨지 않는 연장은
명시적 resume override로 허용하고 변경 전후 값을 experiment card에 남긴다.

config canonical hash는 key 정렬 JSON, UTF-8, compact separator의 SHA-256이다.
float는 Python JSON round-trip 표현을 사용하고 NaN/Infinity는 금지한다.

## 6. Device 계약

`device`는 `cpu | mps | cuda | cuda:N | auto`다.

- 명시 device가 unavailable이면 실패한다.
- `auto`는 `cuda → mps → cpu` 후보를 **실제 one-step probe**로 확인한다.
- probe 실패 이유와 선택 결과를 출력한다.
- fallback은 `allow_device_fallback=true`일 때만 가능하다.
- device/dtype별 deterministic 지원 여부와 알려진 operator 차이를 experiment
  card에 남긴다.

현재 C2 측정 runtime에서 PyTorch 2.13.0은 MPS build를 포함했지만 sandbox에서
`is_available=False`였다. 이 사실은 Mac 전체의 미지원 주장이 아니며 C8에서
다시 probe한다.

## 7. Model compatibility와 download guard

external model은 network 접속 전에 local manifest의 다음 필드를 통과해야 한다.

1. Hub ID와 exact revision이 `docs/sources.yml`에 있음
2. causal LM task와 tokenizer 존재
3. license/access/gating 정보 존재
4. 예상 weight/file bytes와 cache 위치 계산
5. 100,000,000 bytes 초과 시 `accept_download=True`
6. `trust_remote_code=False` 기본
7. chat template/padding/EOS가 없을 때의 명시적 정책
8. requested device/dtype의 예상 parameter·optimizer·activation memory

preflight는 machine-readable JSON과 사람이 읽는 요약을 내고 다운로드를 하지
않는다. arbitrary model override는 revision과 license를 사용자가 제공하거나 Hub
metadata를 새로 감사한 뒤에만 실행한다.

LoRA는 adapter parameter만 `requires_grad=True`인지 검사한다. QLoRA는
bitsandbytes와 device support probe가 모두 통과할 때만 활성화하고 macOS에
가짜 지원을 표시하지 않는다.

## 8. Algorithm variant registry

같은 이름 아래 다른 수학을 숨기지 않는다.

| canonical name | variant key | 핵심 의미 |
|---|---|---|
| `ppo` | `classic-paper` | timestep action, clipped surrogate |
| `rlhf_ppo` | `token-kl` | response token action + reference KL + value |
| `dpo` | `paper` | chosen/rejected sequence log-ratio |
| `grpo` | `deepseekmath-paper` | group-relative normalized reward |
| `rloo` | `paper` | leave-one-out group baseline |
| `dr_grpo` | `paper` | unbiased fixed denominator/reduction |
| `dapo` | `paper-four-components` | Clip-Higher + dynamic + token loss + overlong |
| `gspo` | `paper-sequence` | sequence-level ratio and clip |

TRL/verl adapter가 다른 default를 쓰면 별도 adapter config key로 기록한다. core
loss parity test는 같은 variant와 작은 tensor에서만 수행한다.

## 9. 공정 비교 계약

비교 run group은 `comparison_id`와 다음 immutable fingerprint를 공유한다.

```text
initial_policy_sha256
prompt_uids_sha256 / split_sha256
tokenizer_revision
optimizer family + learning-rate schedule
max optimizer steps
max environment steps
response_token_budget
decoding config
evaluation function revision
seed set
stopping rule
```

모든 budget을 동시에 같게 만들 수 없으면 “fair” boolean을 임의로 주지 않고
차이를 `budget_differences`에 기록한다. report는 optimizer step, environment
step, generated/processed token, model forward count, wall time, peak memory를
함께 보인다.

`evaluation/test`는 training이나 early stopping에서 접근할 수 없다. evaluator는
split role을 확인하고 optimizer가 존재하는 context에서 test 요청을 거부한다.

## 10. Gradient ownership

| model | RLHF-PPO | DPO | GRPO/DAPO | mode during policy update |
|---|---|---|---|---|
| current policy | grad | grad | grad | train |
| old policy/snapshot | frozen | n/a | frozen during epoch | eval |
| reference policy | frozen | frozen | frozen if KL used | eval |
| reward model | frozen | n/a | verifier/RM frozen | eval |
| value model | grad, policy와 optimizer 분리 가능 | n/a | n/a | train |

각 update 전후 parameter hash와 `.grad`를 test한다. `no_grad()`와
`requires_grad_(False)`를 역할에 맞게 함께 사용하되 detached log-prob만으로
current policy까지 끊지 않는다.

## 11. Checkpoint·resume

checkpoint는 임시 sibling directory에 완전히 쓴 뒤 같은 filesystem에서 atomic
rename한다. 최소 내용:

```text
checkpoint-N/
  manifest.json
  config.resolved.json
  model.pt
  optimizer.pt
  scheduler.pt
  rng.pt
  data_cursor.json
  metrics.jsonl
```

`manifest.json`은 schema/version, step, file별 SHA-256, immutable config hash,
initial hash와 created-at을 가진다. load는 hash를 먼저 검사하고 그 다음 tensor를
읽는다. PyTorch serialization은 신뢰하지 않는 checkpoint를 실행할 수 있으므로
local-created checkpoint만 기본 허용하고 가능한 경우 `weights_only=True`를
사용한다.

resume parity는 연속 N step과 K step+resume+(N-K) step의 parameter/metric을
허용 오차 내에서 비교한다. RNG는 Python과 PyTorch CPU/CUDA를 모두 보존한다.

## 12. Artifact와 experiment card schema

run directory 이름은 사람이 바꿔도 되지만 내부 `run_id`는 UUID다.

```text
artifacts/<run-id>/
  experiment-card.json
  config.resolved.json
  metrics.jsonl
  samples.jsonl
  comparison.json
  report.png
  report.html
  checkpoints/
```

`experiment-card.json` 필수 필드:

```json
{
  "schema_version": 1,
  "run_id": "uuid",
  "run_status": "completed|failed|interrupted|external-manual",
  "result_origin": "local_executed",
  "git": {"commit": "...", "dirty": true, "diff_sha256": "..."},
  "config_hash": "sha256:...",
  "dependency_lock_hash": "sha256:...",
  "environment": {
    "os": "...", "python": "...", "torch": "...",
    "device": "cpu", "dtype": "float32",
    "ram_bytes": null, "peak_memory_bytes": 0
  },
  "model": {"id": "tiny-v1", "revision": "repository", "license_id": "..."},
  "data": {"id": "tiny_reasoning", "revision": "generated-v1", "split_hash": "..."},
  "seed": 42,
  "budgets": {
    "optimizer_steps": 100, "environment_steps": 0,
    "generated_tokens": 0, "processed_tokens": 0,
    "model_forwards": 0
  },
  "timing": {"started_at": "...", "wall_seconds": 0.0},
  "metrics": {},
  "paper_reported": null,
  "upstream_reported": null,
  "local_executed": {},
  "known_deviations": [],
  "failures": []
}
```

NaN/Inf는 JSON에 쓰지 않고 metric failure로 기록한다. `paper_reported`와
`upstream_reported`는 source ID와 원문의 metric definition이 있을 때만 채운다.
local toy 결과와 한 chart 축에서 비교하더라도 model/data 규모가 다름을 직접
label한다.

## 13. Metric schema

각 JSONL metric record는 `step_type`, `step`, `name`, `value`, `unit`,
`aggregation`, `sample_count`, `wall_time`을 가진다. 최소 공통 metric:

- reward mean/std, task success/exact match
- policy loss, value/reward model loss
- KL direction과 estimator, entropy, clip fraction
- response length, EOS rate, format validity
- generated/processed tokens, optimizer/environment steps
- wall time와 peak memory

Agentic metric은 tool-validity, tool calls, unnecessary calls, timeout, final success,
process/outcome reward와 episode length를 추가한다.

## 14. CLI와 exit code

```text
python -m rl_study.demo --profile toy --non-interactive
rl-study train --config configs/toy/ppo.yaml
rl-study eval --checkpoint PATH --split validation
rl-study inspect-run ARTIFACT_DIR
rl-study preflight --profile laptop --model MODEL_ID
```

공통 option은 `--help`, `--dry-run`, `--json`이다. exit code:

- `0`: 성공
- `2`: CLI/config 사용 오류
- `3`: preflight/device/dependency 불일치
- `4`: download 승인이 필요함(아직 network 요청하지 않음)
- `5`: checkpoint/provenance 불일치
- `6`: training/evaluation numeric failure

secret, token, 전체 local environment와 home path를 오류 메시지에 출력하지 않는다.

## 15. Adapter 계약

- `TRLAdapter`: laptop one-step SFT/RM/DPO/PPO/GRPO/RLOO. TRL v1.10.0과
  Transformers/PEFT 호환 범위를 lock에 기록한다.
- `VerlRecipe`: server config render/validate와 core variant mapping. 실제 multi-GPU
  실행은 hardware가 없으면 `external-manual`이다.
- `ALFWorldAdapter`: 설치/asset preflight, text observation/action mapping,
  trajectory export. core에서 import하지 않는다.

adapter output은 core `ExperimentCard`와 metric schema로 정규화하되 upstream의
고유 metric 정의를 버리지 않고 `adapter_metadata`에 원 이름·버전을 남긴다.

## 16. 오류가 성공보다 먼저 설계되는 지점

- all-zero mask: scalar 0으로 조용히 반환하지 않고 명시적 policy에 따라 실패
- zero-variance reward group: finite zero advantage + diagnostic counter
- missing EOS/overlong: shaping과 truncation을 별도 기록
- invalid tool call: parser error를 observation으로 반환, shell/network 실행 없음
- stale rollout: policy version mismatch로 update 거부
- changed resume config: immutable diff를 표로 출력하고 load 전 실패
- test split in trainer: data access 전 실패
- 100MB+ asset: network call 전 exit 4
- missing optional dependency/device: 설치 extra/manual command와 exit 3

이 동작들은 C3~C9에서 각각 unit/integration test ID로 traceability에 연결한다.
