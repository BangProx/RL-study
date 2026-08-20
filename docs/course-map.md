# 강좌 지도

큰 질문에서 시작해 필요한 수학과 구현 디테일로 내려갑니다. 각 lesson은 이전
notebook의 kernel 상태에 의존하지 않습니다.

English learners can use the [English course map](course-map.en.md).

```text
왜 RL인가?
  ├─ 확률·gradient ── RL 언어 ── 가치 기반 ── policy gradient ── PPO
  └─ LLM policy ── preference/reward ── RLHF·DPO ── GRPO·DAPO
                                                └─ Agentic RL
```

## 두 가지 경로

| 경로 | 추천 대상 | 순서 | 예상 학습 시간 |
|---|---|---|---:|
| 빠른 | 전체 지도를 먼저 잡고 싶은 사람 | 아래 ★ lesson의 `[필수/CORE]` | 369분, 약 6시간 |
| 전체 | 수식·구현·오답까지 직접 확인할 사람 | L00~L16 전체 | 845분, 약 14시간 |

시간은 읽기, 예측, 손계산, exercise를 포함한 학습자 예상치입니다. notebook의
실제 코드 실행 시간과는 다른 값입니다.

## L00~L16

| Lesson | 핵심 질문 | 빠른 경로 |
|---|---|:---:|
| [L00 · LLM RL 전체 지도](../notebooks/ko/L00_rl_map.ipynb) | 전체 지형에서 지금 어디인가? | ★ |
| [L01 · 확률·미분·PyTorch](../notebooks/ko/L01_probability_autograd.ipynb) | 확률·log-prob·entropy·KL과 autodiff는 어떻게 연결되나? | ★ |
| [L02 · Causal LM](../notebooks/ko/L02_causal_lm.ipynb) | 언어모델은 다음 token의 확률을 어떻게 학습하나? |  |
| [L03 · Bandit](../notebooks/ko/L03_bandit.ipynb) | exploration과 exploitation을 어떻게 균형 잡나? | ★ |
| [L04 · MDP와 Bellman](../notebooks/ko/L04_mdp_bellman.ipynb) | Bellman 식이 value iteration 코드가 되는 과정은? | ★ |
| [L05 · MC·TD·Q-learning](../notebooks/ko/L05_mc_td_q.ipynb) | MC·TD·Q-learning의 target 차이는? |  |
| [L06 · DQN](../notebooks/ko/L06_dqn.ipynb) | DQN의 replay와 target network가 왜 필요한가? |  |
| [L07 · REINFORCE](../notebooks/ko/L07_reinforce.ipynb) | REINFORCE와 baseline은 왜 작동하나? | ★ |
| [L08 · Actor-Critic·GAE·PPO](../notebooks/ko/L08_actor_critic_gae_ppo.ipynb) | actor-critic·GAE·PPO는 variance와 update를 어떻게 제어하나? | ★ |
| [L09 · LLM을 policy로 보기](../notebooks/ko/L09_llm_as_policy.ipynb) | LLM에서 token action과 mask는 무엇인가? | ★ |
| [L10 · RLHF-PPO](../notebooks/ko/L10_rlhf_ppo.ipynb) | old/reference/value/reward model은 왜 역할을 나누나? | ★ |
| [L11 · DPO](../notebooks/ko/L11_dpo.ipynb) | preference를 어떻게 직접 분류 loss로 바꾸나? | ★ |
| [L12 · GRPO와 RLVR](../notebooks/ko/L12_grpo_rlvr.ipynb) | verifier reward와 group-relative advantage는 어떻게 작동하나? | ★ |
| [L13 · DAPO와 최신 변형](../notebooks/ko/L13_dapo_variants.ipynb) | DAPO·RLOO·Dr. GRPO·GSPO는 무엇을 다르게 고치나? | ★ |
| [L14 · 실제 모델과 서버](../notebooks/ko/L14_real_model_profiles.ipynb) | toy에서 실제 공개 LM·server로 어떻게 확장하나? |  |
| [L15 · Agentic RL](../notebooks/ko/L15_agentic_rl.ipynb) | multi-turn tool trajectory의 mask와 credit은 무엇이 다른가? | ★ |
| [L16 · 평가와 Capstone](../notebooks/ko/L16_evaluation_capstone.ipynb) | 실패 모드·평가·재현성을 어떻게 감사하나? | ★ |

## 막혔을 때 돌아올 곳

- 기호가 낯설다 → [수학 최소 도구함](math.md)
- 단어가 섞인다 → [용어집](glossary.md)
- 알고리즘 선택이 헷갈린다 → [알고리즘 카드](algorithms/cards.md)
- 코드가 실행되지 않는다 → [문제 해결](troubleshooting.md)
