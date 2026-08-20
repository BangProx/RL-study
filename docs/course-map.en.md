# Course Map

Start with the big picture, then descend into the math and implementation details
only when you need them. Every lesson runs independently in a fresh kernel.

> 한국어로 공부하려면 [한국어 강좌 지도](course-map.md)를 사용하세요.

```text
Why RL?
  ├─ probability · gradients ── RL language ── value methods ── PPO
  └─ LLM policy ── preference/reward ── RLHF · DPO ── GRPO · DAPO
                                                   └─ Agentic RL
```

## Pick One Path

| Path | Best for | Order | Estimated study time |
|---|---|---|---:|
| Fast | Building the map before the details | `[CORE]` sections in the ★ lessons below | 369 min, about 6 hours |
| Full | Checking equations, code, and mistakes yourself | Every section in L00–L16 | 845 min, about 14 hours |

These estimates include reading, prediction, hand calculations, and exercises. They
are not code runtime estimates.

## L00–L16

| Lesson | Guiding question | Fast path |
|---|---|:---:|
| [L00 · The LLM RL Map](../notebooks/en/L00_rl_map.ipynb) | Where am I in the full landscape? | ★ |
| [L01 · Probability, Gradients, and PyTorch](../notebooks/en/L01_probability_autograd.ipynb) | How do log-probability, entropy, KL, and autodiff connect? | ★ |
| [L02 · Causal Language Models](../notebooks/en/L02_causal_lm.ipynb) | How does an LM learn the next-token distribution? |  |
| [L03 · Bandits](../notebooks/en/L03_bandit.ipynb) | How do we balance exploration and exploitation? | ★ |
| [L04 · MDPs and Bellman Equations](../notebooks/en/L04_mdp_bellman.ipynb) | How does a Bellman equation become value-iteration code? | ★ |
| [L05 · MC, TD, and Q-learning](../notebooks/en/L05_mc_td_q.ipynb) | How do their learning targets differ? |  |
| [L06 · DQN](../notebooks/en/L06_dqn.ipynb) | Why do replay and target networks matter? |  |
| [L07 · REINFORCE](../notebooks/en/L07_reinforce.ipynb) | Why do policy gradients and baselines work? | ★ |
| [L08 · Actor-Critic, GAE, and PPO](../notebooks/en/L08_actor_critic_gae_ppo.ipynb) | How do they control variance and update size? | ★ |
| [L09 · Viewing an LLM as a Policy](../notebooks/en/L09_llm_as_policy.ipynb) | What are token actions and action masks? | ★ |
| [L10 · RLHF-PPO](../notebooks/en/L10_rlhf_ppo.ipynb) | Why are old, reference, value, and reward models separate? | ★ |
| [L11 · DPO](../notebooks/en/L11_dpo.ipynb) | How does a preference pair become a classification loss? | ★ |
| [L12 · GRPO and RLVR](../notebooks/en/L12_grpo_rlvr.ipynb) | How do verifier rewards and group-relative advantages work? | ★ |
| [L13 · DAPO and Recent Variants](../notebooks/en/L13_dapo_variants.ipynb) | What do DAPO, RLOO, Dr. GRPO, and GSPO fix differently? | ★ |
| [L14 · Real Models and Servers](../notebooks/en/L14_real_model_profiles.ipynb) | How do we scale from toy code to a public LM and a server? |  |
| [L15 · Agentic RL](../notebooks/en/L15_agentic_rl.ipynb) | How do masks and credit differ in multi-turn tool trajectories? | ★ |
| [L16 · Evaluation and Capstone](../notebooks/en/L16_evaluation_capstone.ipynb) | How do we audit failures, evaluation, and reproducibility? | ★ |

## When You Get Stuck

- Symbols feel unfamiliar → [minimum math toolkit](math.md)
- Algorithms blur together → [algorithm cards](algorithms/cards.md)
- Code does not run → [troubleshooting](troubleshooting.md)
- Need the Korean explanation → [Korean course map](course-map.md)
