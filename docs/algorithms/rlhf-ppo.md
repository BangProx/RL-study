# RLHF-PPO: SFT부터 token update까지

> 구현 상태: C6 toy clean-room 구현. `learning-to-summarize-2020`,
> `instructgpt-2022`, `ppo-2017`의 계보를 따르되 대규모 결과를 재현한다고
> 주장하지 않는다.

## 1. 전체 파이프라인

```text
정답 응답 ──SFT──> policy ──복사/동결──> reference
선호 pair ──pairwise loss──> reward model(동결)
prompt ──old policy rollout──> response token
response ──task/RM score + reference KL──> token reward/return
token trajectory ──PPO──> current policy + value model
```

SFT, reward model, rollout, PPO를 분리하는 이유는 각 단계가 학습하는 대상과
오류 양상이 다르기 때문이다. verifier reward는 정답을 직접 검사할 수 있는 RLVR
ablation이고, learned reward model은 인간 선호를 근사하는 RLHF 경로다.

## 2. reward와 KL을 분해하기

response action token `t`마다 sampled KL estimator를

\[
k_t=\log\pi_{old}(a_t|s_t)-\log\pi_{ref}(a_t|s_t)
\]

로 두고 `-β k_t`를 non-score reward로 준다. sequence task/RM score는 마지막
유효 action token에 한 번 더한다. 한 sample의 `k_t`는 음수일 수 있지만 policy
분포에 대한 기대값이 forward KL이다. 따라서 음수 관측을 버리거나 clamp하지
않는다.

고정 KL coefficient를 선택한 이유는 짧은 toy run에서 controller 상태까지
추가하지 않고 reward decomposition을 눈으로 확인하기 위해서다. 대규모 학습에서는
target KL을 둔 adaptive controller가 유용할 수 있지만 controller state를 반드시
checkpoint해야 한다.

## 3. PPO update

rollout 직후 policy를 `old`로 동결하고, current policy의 token log-prob과 비교해
importance ratio를 만든다. clipped surrogate, value MSE, entropy bonus를 합치되
모든 reduction은 `action_mask`의 token만 대상으로 한다. advantage는 유효 token
전체에서 whitening한다. 이 선택은 작은 batch 안정화를 위한 구현 옵션이며 PPO
원 논문의 유일한 정답은 아니다.

| 역할 | update에서 gradient? | mode/수명 | 코드 |
|---|---:|---|---|
| current policy | 예 | train, checkpoint | `train_rlhf_ppo` |
| old rollout policy | 아니오 | update마다 snapshot/eval | `old_policy` |
| SFT reference | 아니오 | run 전체 frozen/eval | `reference` |
| reward model | 아니오 | run 전체 frozen/eval | `reward_model` |
| value model | 예 | train, 별도 optimizer/checkpoint | `value_model` |

종료 시 reference와 reward model은 trainable parameter, gradient, train mode 및
parameter hash를 모두 검사한다. checkpoint에는 policy/reference/reward/value와
두 optimizer 상태가 함께 들어간다.

## 4. mask와 reward truth table

| 위치 | policy loss | value loss | KL reward | task/RM score |
|---|---:|---:|---:|---:|
| prompt token | 제외 | 제외 | 제외 | 없음 |
| response token | 포함 | 포함 | 포함 | 마지막 token만 |
| EOS | 생성됐다면 포함 | 포함 | 포함 | 마지막 token이면 포함 |
| padding | 제외 | 제외 | 제외 | 없음 |
| tool/environment token | C6에는 없음 | C6에는 없음 | C6에는 없음 | C9에서 별도 mask |

prompt는 state/context이고 response token이 action이라는 계약이다. padding을 평균에
넣으면 짧은 응답의 gradient가 batch padding 길이에 따라 바뀌므로 금지한다.

## 5. classic PPO와 무엇이 다른가

| 항목 | GridWorld PPO | LLM RLHF-PPO |
|---|---|---|
| state | 환경 관측 | prompt + 지금까지의 token |
| action | 4개 이동 | vocabulary token |
| reward | 환경 step reward | sequence score + token KL |
| horizon | 환경 transition | response token |
| reduction | timestep/episode | response action-mask token |
| reference | 없음 | frozen SFT policy |
| 부가 모델 | critic | critic + optional reward model |

수학적 ratio/clip은 같지만 tensor 의미와 mask가 다르다. classic PPO 함수를 그대로
재사용하지 않은 이유다.

## 6. 실제 실패를 숨기지 않기

C6 8-step 비교에서 reward model은 held-out preference accuracy `0.797`이었지만
score-length correlation이 `+0.832`였다. 그 reward로 PPO한 policy의 greedy
format rate는 `1.0 → 0.0`으로 붕괴했다. 반면 verifier ablation은 format `1.0`,
exact match `0.031`이었다. 이는 작은 표본의 우승 결과가 아니라 **reward shortcut이
policy 최적화에서 증폭될 수 있음**을 보여주는 failure case다.

대안은 length-balanced preference, reward calibration, held-out adversarial set,
KL/entropy monitoring, format constraint, verifier와 learned score의 분리다. 단순히
학습 step을 늘리는 것은 shortcut의 해결책이 아니다.

## 7. 실행과 체크

```bash
rl-study train --config configs/toy/rlhf_ppo.yaml --stop-after 2 --json
python scripts/run_alignment_comparison.py --steps 8 --batch-size 8
```

```python
import torch
from rl_study.algorithms.rlhf_ppo import compose_rlhf_rewards

mask = torch.tensor([[True, True]])
out = compose_rlhf_rewards(torch.tensor([1.0]), torch.tensor([[0.2, 0.1]]),
                           torch.tensor([[0.0, 0.0]]), mask, kl_coefficient=0.5)
assert torch.allclose(out.token_rewards, torch.tensor([[-0.1, 0.95]]))
```

## Sources

- `learning-to-summarize-2020`: preference RM과 KL-regularized RLHF 계보
- `instructgpt-2022`: SFT → RM → PPO 파이프라인
- `ppo-2017`: clipped surrogate와 여러 update epoch
- `repo-summarize-from-feedback`, `framework-trl`: 역사/현대 구현 차이 교차검산

---

[← LLM 기반](llm-foundations.md) · [강좌 지도](../course-map.md) · [L10 notebook](../../notebooks/ko/L10_rlhf_ppo.ipynb) · [다음: DPO →](dpo.md)
