# DPO: 선호 데이터에서 직접 policy를 학습하기

> 구현 상태: C6 toy clean-room 구현. 출처 코드를 복사하지 않았으며
> `dpo-2023`, `repo-dpo`, `framework-trl`을 수식·동작 교차검산에 사용했다.

## 1. 먼저 잡을 큰 그림

DPO(Direct Preference Optimization)는 같은 prompt에 대해 `chosen`이
`rejected`보다 낫다는 offline pair를 사용한다. 새 응답을 rollout하거나 value
model을 학습하지 않는다. 그렇다고 RL과 무관한 단순 분류는 아니다. DPO 논문은
KL 제약 reward maximization의 최적 policy와 implicit reward의 관계를 이용해
reward model과 online RL 단계를 하나의 preference objective로 바꾼다.

한 pair에 대한 이 저장소의 기본 loss는 다음과 같다.

\[
z = \beta\left[
  (\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))
  -(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))
\right],
\qquad
\mathcal L_{DPO}=-\log\sigma(z).
\]

- `beta`가 크면 같은 policy/reference log-ratio 차이가 loss에 더 강하게 반영된다.
- reference는 SFT policy의 frozen 복사본이다.
- 네 log-prob은 모두 **response token의 합**이다. prompt와 padding은 제외하고
  EOS는 실제 response 종료 action이므로 포함한다.

## 2. 수식에서 코드까지

| 수식/계약 | 구현 위치 | 구현 이유 |
|---|---|---|
| chosen/rejected log-ratio | `dpo_loss` | 손계산 가능한 순수 tensor 함수로 분리 |
| response-only sequence 합 | `dpo_sequence_loss` → `response_sequence_log_probs` | prompt 길이가 objective에 섞이는 오류 방지 |
| frozen reference | `train_dpo` → `freeze_module`, `assert_frozen` | gradient와 mode까지 종료 시 재검증 |
| pair 순서 | `PreferenceExample(chosen, rejected)` | 뒤집힌 pair가 반대 loss를 내는 unit test 보유 |
| label smoothing | `label_smoothing` | 기본값 0; noisy preference ablation용 |
| 재개 가능한 sampling | `seed + global_step * 1_000_003` | 중단/재개와 연속 실행의 exact parity |

`src/rl_study/algorithms/dpo.py`는 논문 objective를 독립적으로 다시 작성한
교육용 구현이다. 공식 연구 repo는 batching과 부호를 확인하는 reference일 뿐
코드 provenance는 `clean-room-reimplemented`다.

## 3. 왜 sequence log-prob을 합하는가

autoregressive sequence 확률은 token 조건부 확률의 곱이고 log 공간에서는 합이다.
평균을 쓰면 길이에 따른 scale은 줄지만 논문의 sequence 확률과 다른 objective가
된다. 이 저장소는 paper-faithful 기본값으로 합을 사용한다. token 평균, 길이 보정,
reference-free loss는 중요한 대안이지만 같은 이름으로 조용히 바꾸지 않는다.

reference-free DPO는 reference 항을 0으로 놓을 수 있지만 KL-constrained 유도와
초기 policy anchoring의 의미가 달라진다. 초심자가 두 설정을 혼동하지 않도록 C6
기본 trainer에는 노출하지 않았다.

## 4. label noise를 다루는 선택지

`label_smoothing=ε`이면 positive와 reversed label loss를 `(1-ε):ε`로 섞는다.
이는 원 DPO의 필수 요소가 아니라 noisy label에 대한 보수적 변형이다. 따라서
기본 config는 `0.0`이며 experiment card에 값이 남는다. `ε >= 0.5`는 선호
방향을 잃으므로 config가 거부한다.

## 5. 실행과 관찰

```bash
rl-study train --config configs/toy/dpo.yaml --stop-after 10 --json
rl-study eval --checkpoint artifacts/dpo-verifier-seed42/checkpoint-000010 --json
```

C6 공정 비교(8 policy step)에서는 validation preference accuracy `0.805`, greedy
exact match `0.0`, format rate `0.875`였다. preference margin이 좋아졌다고 산술
생성 정답률까지 좋아졌다고 말할 수 없다는 좋은 반례다. 결과 원문과 hash/budget은
`docs/research/C6_ALIGNMENT_BENCHMARK.json`에 있다.

## 6. 체크

```python
import math, torch
from rl_study.algorithms.dpo import dpo_loss

out = dpo_loss(torch.tensor([-1.]), torch.tensor([-2.]),
               torch.tensor([-1.5]), torch.tensor([-1.5]), beta=1.0)
assert out.logits.item() == 1.0
assert math.isclose(out.loss.item(), -math.log(1 / (1 + math.exp(-1))), rel_tol=1e-6)
```

## Sources

- `dpo-2023`: 원 논문 objective와 KL-constrained 유도
- `repo-dpo`: 공식 연구 구현의 부호·batching 교차검산
- `framework-trl`: 현대 trainer의 label smoothing/reduction 차이 감사
