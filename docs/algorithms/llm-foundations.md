# LLM policy·preference·reward 기반

> source: `learning-to-summarize-2020`, `instructgpt-2022`
> lessons: L02, L09, L10의 기반
> provenance: clean-room PyTorch implementation

## 1. LLM에서 action은 어느 token인가

causal LM logits는 `[B,L,V]`이고 `logits[:,t]`는 `input_ids[:,t+1]`의 분포다.
따라서 mask도 input token 위치가 아니라 **next-token target 위치 `[B,L-1]`**에
맞춰야 한다.

[`build_sequence_batch`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/models/sequence.py)의 작은 예:

```text
input       : [BOS, p, r, EOS]
target      : [p,   r, EOS]
prompt mask : [1,   0, 0  ]
action mask : [0,   1, 1  ]
```

EOS는 model이 선택해야 하는 response action이므로 포함한다. prompt와 padding은
policy action loss에서 제외한다. prompt embedding은 response 예측의 context이므로
response loss의 gradient가 prompt 위치 hidden state를 통과할 수 있다. “prompt
target을 loss에서 제외”와 “prompt가 계산 graph에서 완전히 분리”는 다른 말이다.

`response_token_log_probs`는 `[B,L-1]`, `response_sequence_log_probs`는 action
mask 위치를 합한 `[B]`다. 길이가 다른 sequence의 **합**은 DPO의 sequence
log-ratio에 직접 쓰지만, token 평균은 PPO/GRPO의 reduction 옵션이 될 수 있다.
어떤 reduction인지 함수 이름과 config에서 숨기지 않는다.

관련 test: `test_prompt_and_action_mask_truth_table`, variable length/EOS/padding,
silent truncation rejection, token sum parity.

## 2. 64-token context와 데이터 설계

TinyReasoning prompt는 다음 고정 형식이다.

```text
31 * 31 = ? Answer: <answer>N</answer>
```

response는 `<answer>961</answer>`이다. BOS/EOS를 포함한 combined 길이는 전체
448 prompt에서 56~60 token이므로 canonical `tiny-v1`의 64 context 안에
truncation 없이 들어간다. builder는 넘칠 때 잘라내지 않고 오류를 낸다.

현재 split hash는
`sha256:f238657bbf6c0a112debf7ef3ffafb452c14308dfb5ce57d9abe4f77ac1deedd`다.
train 256, validation 64, test 128의 prompt UID는 서로 겹치지 않는다. test는
최종 평가 전용이다.

## 3. Response-only SFT

$$
L_{SFT}=-\frac{\sum_{b,t}m^{action}_{b,t}
\log\pi_\theta(y_{b,t}\mid x_b,y_{b,<t})}
{\sum_{b,t}m^{action}_{b,t}}.
$$

[`sft_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/sft.py)은 prompt·pad target을 제외하고
response와 EOS token만 평균한다. sequence별 평균 후 batch 평균을 내는 대안은
짧은 response와 긴 response에 같은 sequence weight를 준다. 기본 token mean은
모든 response token에 같은 weight를 주며, L12~L13에서 length bias와 함께 다시
비교한다.

local CPU 100 step에서 첫 10-step loss 평균 16.121 → 마지막 10-step 0.311,
validation response-token accuracy 0.902였다. teacher forcing token accuracy이지
free-running exact-match가 아니다.

## 4. Deterministic verifier

[`verifier_reward`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/data/tiny_reasoning.py)는 세 신호를 분리한다.

| response | reward | 의미 |
|---|---:|---|
| 정확한 `<answer>N</answer>` | 1.0 | answer와 format 모두 정확 |
| format은 맞고 숫자는 틀림 | 0.1 | format partial reward |
| format도 틀림 | 0.0 | invalid |

regex는 전체 문자열을 match하므로 뒤에 불필요한 설명을 붙이면 valid format이
아니다. partial format reward는 학습 curriculum을 줄 수 있지만, 최종 answer보다
format shortcut을 쉽게 만드는 trade-off가 있다. exact-match reward만 쓰는
toggle도 C6/C7 비교에서 제공한다.

## 5. Preference pair와 Bradley–Terry reward loss

train prompt마다 두 pair를 만든다.

- numeric: chosen correct vs 같은 format의 틀린 숫자
- format: chosen correct vs `The answer is N.`

따라서 train 512, held-out validation 128 pair다. pairwise loss:

$$
L_{RM}=-\log\sigma(r_\phi(x,y_c)-r_\phi(x,y_r)).
$$

[`pairwise_reward_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/reward_model.py)은 chosen-
rejected margin, loss와 accuracy를 모두 반환한다. margin=0을 correct로 세지 않는다.

[`TinyRewardModel`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/models/reward.py)은 causal backbone의 마지막
실제 token(EOS) hidden state를 linear scalar head에 넣는다. mean pooling은 모든
token을 직접 반영하고 last-response-token pooling도 가능하다. EOS pooling은
shape가 단순하고 full prefix를 본다는 장점이 있지만 EOS representation이 길이와
format feature를 쉽게 담을 수 있다.

## 6. Shortcut 진단은 부가 metric이 아니다

local CPU 120-step reward model:

| held-out metric | 값 |
|---|---:|
| 전체 preference accuracy | 0.797 |
| numeric correctness | 0.594 |
| format | 1.000 |
| score↔response length Pearson correlation | +0.832 |

loss는 0.446(첫 10-step 평균)에서 0.268(마지막 10-step)로 줄었지만, model은
숫자 정답보다 format을 훨씬 잘 구분하고 길이와 score가 강하게 엮였다. 이것은
성공을 과장할 수 없는 **shortcut 경고**다.

[`diagnose_reward_model`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/diagnostics/reward.py)은 전체 accuracy,
pair reason별 accuracy와 length correlation을 함께 낸다. 개선 대안:

- numeric/format pair의 sampling weight를 균형화한다.
- 같은 길이의 hard negative를 늘린다.
- deterministic verifier와 reward model을 분리 보고한다.
- held-out counterfactual format/length slice를 둔다.

이 범위에서는 관찰을 숨기지 않고 C6에서 policy가 이 shortcut을 exploit하는지
진단한다.

## 7. Gradient ownership

| 역할 | policy update 중 mode | `requires_grad` |
|---|---|---:|
| current policy | train | true |
| old snapshot | eval | false |
| reference policy | eval | false |
| reward model/verifier | eval | false |
| value model | train | true, 별도 value objective |

[`freeze_module`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/models/roles.py)은 `requires_grad=False`, 기존
`.grad=None`, `eval()`을 함께 적용한다. [`parameter_sha256`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/models/roles.py)
로 update 전후 reference/reward hash가 같은지 확인한다. current policy만 실제
SGD step으로 바뀌는 test와 `assert_frozen` contract가 있다.

[`TinyValueModel`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/models/roles.py)은 각 input token에 `[B,L]`
value를 낸다. prompt context를 볼 수 있지만 C6 RLHF-PPO의 value loss는 어떤
response/action 위치를 학습하는지 별도 mask로 제한한다.

## 8. 실행 결과의 경계

exact 결과는 [`C5_LLM_FOUNDATION_BENCHMARK.json`](../research/C5_LLM_FOUNDATION_BENCHMARK.json)에
있다. 이 단계는 다음을 아직 주장하지 않는다.

- 생성 exact-match가 90%라는 뜻이 아니다.
- reward score가 사람 preference를 잘 대변한다는 뜻이 아니다.
- RLHF-PPO나 DPO policy update가 끝났다는 뜻이 아니다.

C5가 보장하는 것은 mask/log-prob shape, SFT/RM finite training, deterministic
verifier, held-out pair 평가와 frozen-role gradient contract다.

---

[← 고전 RL](classic.md) · [강좌 지도](../course-map.md) · [L02 Causal LM](../../notebooks/ko/L02_causal_lm.ipynb) · [L09 LLM policy](../../notebooks/ko/L09_llm_as_policy.ipynb) · [다음: RLHF-PPO →](rlhf-ppo.md)
