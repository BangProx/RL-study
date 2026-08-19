# 알고리즘 카드

알고리즘 이름보다 **데이터가 어디서 오고, 무엇을 baseline으로 빼며, 어떤
parameter가 update되는지** 먼저 비교하세요.

## 한눈에 비교

| 알고리즘 | 데이터 | 핵심 target/advantage | update 대상 | 대표 함정 |
|---|---|---|---|---|
| Q-learning | off-policy transition | $r+\gamma\max Q(s',a')$ | Q table/network | terminal bootstrap |
| DQN | replay transition | frozen target network TD | online Q network | target detach 누락 |
| REINFORCE | on-policy episode | return − optional baseline | policy | loss 부호·큰 variance |
| Actor-Critic | on-policy transition | TD advantage | actor + critic | advantage detach 누락 |
| PPO | old-policy rollout | GAE + clipped ratio | policy + value | old/current 혼동 |
| RLHF-PPO | token rollout | reward − reference KL + value | policy + value | action mask·세 model 혼동 |
| DPO | offline preference pair | chosen/rejected log-ratio | policy | response 합·pair 순서 |
| GRPO | online response group | group-relative advantage | policy | zero-variance group |
| RLOO | online response group | leave-one-out sequence reward | policy | token action으로 오해 |
| Dr. GRPO | online response group | mean-centered reward | policy | fixed denominator 누락 |
| DAPO | filtered online groups | GRPO + 네 안정화 요소 | policy | 변형 이름을 숨김 |
| GSPO | online response group | sequence likelihood ratio | policy | token ratio와 혼합 |
| Agentic REINFORCE | multi-turn tool trajectory | outcome/process credit | action policy | tool output까지 loss 처리 |

## 카드 읽는 법

각 카드의 `언제`는 추천 순위가 아니라 문제 설정의 조건입니다. toy 결과로 특정
알고리즘이 우월하다고 말하지 않습니다.

### Q-learning / DQN

- **질문:** 다음 상태의 greedy value로 현재 action value를 고칠 수 있는가?
- **언제:** 작은 discrete MDP는 Q-learning, observation이 커지면 DQN.
- **구현:** `algorithms/tabular.py`, `algorithms/dqn.py`.
- **대안:** SARSA는 다음 behavior action을 써서 on-policy target을 만듭니다.
- **검사:** terminal mask, target network detach, replay sample shape.
- **출처:** `sutton-barto-rl2`, `dqn-2013`.

### REINFORCE / Actor-Critic / GAE

- **질문:** sampled action의 log-prob을 return 방향으로 얼마나 움직일 것인가?
- **언제:** differentiable stochastic policy와 on-policy trajectory가 있을 때.
- **구현:** `algorithms/policy_gradient.py`, `math/returns.py`.
- **대안:** baseline 없이 unbiased하게 둘 수 있지만 variance가 큽니다. critic은
  bias 가능성을 받아들이고 variance를 낮춥니다.
- **검사:** loss sign, baseline invariance, critic gradient, $\lambda$ 경계.
- **출처:** `sutton-barto-rl2`, `gae-2015`.

### PPO

- **질문:** old policy rollout을 여러 번 써도 update가 너무 멀리 가지 않게 할 수
  있는가?
- **언제:** on-policy actor-critic에서 간단한 trust-region 근사가 필요할 때.
- **구현:** `algorithms/ppo.py`; [상세 노트](classic.md).
- **대안:** KL penalty/early stop, TRPO. clipping은 보장된 hard constraint가
  아닙니다.
- **검사:** ratio=1, advantage 부호별 clip, entropy/value coefficient, GAE mask.
- **출처:** `ppo-2017`.

### RLHF-PPO

- **질문:** LLM response reward를 높이되 SFT reference에서 지나치게 멀어지지 않게
  할 수 있는가?
- **언제:** online response 생성과 reward/verifier를 실제로 호출할 수 있을 때.
- **구현:** `algorithms/rlhf_ppo.py`; [상세 노트](rlhf-ppo.md).
- **대안:** adaptive KL, reward whitening, DPO 같은 offline preference objective.
- **검사:** prompt/action mask, old/reference 분리, reward+KL 합, value ownership.
- **출처:** `instructgpt-2022`, `ppo-2017`.

### DPO

- **질문:** 별도 reward model과 online rollout 없이 preference pair로 policy를
  update할 수 있는가?
- **언제:** chosen/rejected가 고정된 offline alignment.
- **구현:** `algorithms/dpo.py`; [상세 노트](dpo.md).
- **대안:** IPO, label smoothing/cDPO, explicit reward-model + RLHF.
- **검사:** chosen/rejected 순서, response-only sequence log-prob, frozen reference.
- **출처:** `dpo-2023`.

### GRPO / RLOO / Dr. GRPO / GSPO

- **질문:** value model 없이 같은 prompt의 여러 response를 비교해 advantage를 만들
  수 있는가?
- **언제:** deterministic verifier가 있고 group rollout 비용을 감당할 때.
- **구현:** `algorithms/grpo.py`, `algorithms/group_policy.py`;
  [변형 비교](grpo-family.md).
- **대안:** learned critic, sequence-level RLOO, token-level GRPO, sequence-ratio GSPO.
- **검사:** constant group, group size, ratio granularity, reduction denominator.
- **출처:** `deepseekmath-grpo-2024`, `rloo-2024`, `dr-grpo-2025`, `gspo-2025`.

### DAPO

- **질문:** reasoning RL의 entropy collapse, uninformative group, 길이 편향을 어떤
  독립 구성요소로 다룰 것인가?
- **언제:** GRPO형 online RL에서 네 구성요소와 추가 rollout budget을 명시할 때.
- **구현:** `algorithms/dapo.py`; [상세 노트](dapo.md).
- **대안:** 각 구성요소를 별도 ablation, standard GRPO, Dr. GRPO.
- **검사:** asymmetric clip, bounded dynamic sampling exhaustion, token reduction,
  overlong 경계.
- **출처:** `dapo-2025`. 감사 시 공식 repo에 license가 없어 paper-only
  clean-room으로 작성했습니다.

### Agentic REINFORCE

- **질문:** 여러 tool step 뒤의 outcome을 어느 action에 배분할 것인가?
- **언제:** action parser, tool allowlist, termination과 trajectory를 통제할 수 있을 때.
- **구현:** `agentic/`, `training/agentic_runner.py`;
  [상세 노트](agentic-rl.md).
- **대안:** outcome broadcast, discounted step return, learned process reward,
  hierarchical credit.
- **검사:** tool output mask, stale policy version, invalid action, timeout, reward hacking.
- **출처:** `agent-lightning-2025`, `agent-r1-2025`.
