#!/usr/bin/env python3
"""Generate the bilingual L00-L16 notebooks from one audited lesson catalog."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "lessons" / "catalog.yml"

TEST_IDS = {
    "L00": ["test_reinforce_sign"],
    "L01": ["test_probability_matches_torch", "test_reinforce_sign"],
    "L02": ["test_prompt_and_action_mask_truth_table"],
    "L03": ["test_bandit_seed_is_deterministic"],
    "L04": ["test_value_iteration_terminal_value_is_zero"],
    "L05": ["test_q_terminal_target", "test_q_off_policy_target"],
    "L06": ["test_dqn_target_detached"],
    "L07": ["test_reinforce_sign", "test_actor_advantage_detached"],
    "L08": ["test_gae_analytic", "test_ppo_clip_sign_cases"],
    "L09": ["test_rlhf_reward_plus_token_kl_decomposition"],
    "L10": ["test_rlhf_ppo_ratio_one_and_gradient_ownership"],
    "L11": ["test_dpo_loss_matches_hand_calculation"],
    "L12": ["test_group_relative_advantages_and_zero_variance_group"],
    "L13": ["test_overlong_reward_shaping_boundaries"],
    "L14": ["test_large_download_guard_runs_before_optional_import"],
    "L15": ["test_rollout_preserves_original_action_tokens_and_masks_tool_output"],
    "L16": ["test_alignment_comparison_audits_shared_start_and_prompts"],
}

DEMO_CODE = {
    "L00": """logits = torch.zeros(2, requires_grad=True)
optimizer = torch.optim.SGD([logits], lr=0.4)
reward_by_action = torch.tensor([0.0, 1.0])
probability_history = []
for _ in range(20):
    probabilities = torch.softmax(logits, dim=-1)
    loss = -(probabilities * reward_by_action).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    probability_history.append(float(probabilities[1].detach()))
print({"p_good_start": round(probability_history[0], 3),
       "p_good_end": round(probability_history[-1], 3)})""",
    "L01": """from rl_study.math import categorical_entropy, categorical_kl
logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
other = torch.tensor([[0.4, 0.2, -0.3]])
entropy = categorical_entropy(logits)
forward_kl = categorical_kl(logits, other)
objective = entropy.mean() - forward_kl.mean()
objective.backward()
print({"entropy": round(float(entropy.detach()), 4),
       "kl_p_q": round(float(forward_kl.detach()), 4),
       "gradient": [round(x, 4) for x in logits.grad[0].tolist()]})""",
    "L02": """from rl_study.models import TinyCausalLM, TinyTokenizer, build_sequence_batch
from rl_study.models.sequence import response_sequence_log_probs
tokenizer = TinyTokenizer()
model = TinyCausalLM()
sequence_batch = build_sequence_batch(
    ["Q:1+1="], ["2"], tokenizer=tokenizer, max_length=64
)
sequence_logp = response_sequence_log_probs(model, sequence_batch)
print({"input_shape": list(sequence_batch.input_ids.shape),
       "target_shape": list(sequence_batch.action_mask.shape),
       "action_tokens": int(sequence_batch.action_mask.sum()),
       "sequence_logp": round(float(sequence_logp[0].detach()), 3)})""",
    "L03": """from rl_study.envs import BernoulliBandit
def run_bandit(epsilon):
    env = BernoulliBandit(horizon=120)
    env.reset(seed=42)
    counts, estimates, regret = [0] * 5, [0.0] * 5, 0.0
    rng = random.Random(42)
    for step in range(120):
        explore = rng.random() < epsilon
        action = rng.randrange(5) if explore else max(range(5), key=estimates.__getitem__)
        result = env.step(action)
        counts[action] += 1
        estimates[action] += (result.reward - estimates[action]) / counts[action]
        regret += float(result.info["expected_regret"])
    return regret, counts, estimates
greedy_run = run_bandit(0.0)
exploring_run = run_bandit(0.1)
print({"greedy_regret": round(greedy_run[0], 2),
       "epsilon_regret": round(exploring_run[0], 2),
       "epsilon_counts": exploring_run[1]})""",
    "L04": """from rl_study.algorithms.tabular import gridworld_model, value_iteration
from rl_study.envs import TinyGridWorld
grid = TinyGridWorld()
transitions, rewards, terminal = gridworld_model(grid)
dp_result = value_iteration(transitions, rewards, terminal, gamma=0.99)
print("values")
print(dp_result.values.reshape(4, 4).round(decimals=3))
print({"iterations": dp_result.iterations,
       "converged": dp_result.converged,
       "terminal_value": float(dp_result.values[-1])})""",
    "L05": """from rl_study.algorithms.tabular import (
    monte_carlo_returns, q_learning_target, td_target
)
rewards = torch.tensor([-0.01, -0.01, 1.0])
mc_targets = monte_carlo_returns(rewards, gamma=0.9)
td_targets = td_target(
    rewards, torch.tensor([0.7, 0.8, 99.0]),
    torch.tensor([False, False, True]), gamma=0.9
)
q_target = q_learning_target(
    torch.tensor([1.0]), torch.tensor([[2.0, 4.0]]),
    torch.tensor([False]), gamma=0.9
)
print({"mc": mc_targets.tolist(), "td": td_targets.tolist(),
       "off_policy_q_target": float(q_target[0])})""",
    "L06": """from rl_study.algorithms.dqn import DQNBatch, DQNNetwork, dqn_loss, hard_update
policy_net = DQNNetwork(16, 4)
target_net = DQNNetwork(16, 4)
hard_update(target_net, policy_net)
dqn_batch = DQNBatch(
    states=torch.tensor([0, 1]), actions=torch.tensor([1, 2]),
    rewards=torch.tensor([-0.01, 1.0]), next_states=torch.tensor([1, 15]),
    terminated=torch.tensor([False, True]), truncated=torch.tensor([False, False])
)
dqn_output = dqn_loss(policy_net, target_net, dqn_batch, gamma=0.99)
dqn_output.loss.backward()
target_has_gradient = any(p.grad is not None for p in target_net.parameters())
print({"loss": round(float(dqn_output.loss.detach()), 4),
       "targets": dqn_output.targets.tolist(),
       "target_has_gradient": target_has_gradient})""",
    "L07": """from rl_study.algorithms.policy_gradient import reinforce_loss
from rl_study.algorithms.tabular import monte_carlo_returns
chosen_log_probs = torch.tensor([-0.7, -0.5, -0.2], requires_grad=True)
rewards = torch.tensor([0.0, 0.0, 1.0])
reward_to_go = monte_carlo_returns(rewards, gamma=0.9)
pg_loss = reinforce_loss(chosen_log_probs, reward_to_go, baseline=0.2)
pg_loss.backward()
print({"returns": reward_to_go.tolist(),
       "loss": round(float(pg_loss.detach()), 4),
       "logprob_gradient": chosen_log_probs.grad.tolist()})""",
    "L08": """from rl_study.algorithms.ppo import ppo_policy_loss
from rl_study.math import generalized_advantage_estimate
rewards = torch.tensor([0.0, 1.0])
values = torch.tensor([0.2, 0.4, 0.0])
terminated = torch.tensor([False, True])
truncated = torch.tensor([False, False])
gae, gae_returns = generalized_advantage_estimate(
    rewards, values, terminated, truncated, gamma=0.9, gae_lambda=0.95
)
old_logp = torch.zeros(2)
current_logp = torch.log(torch.tensor([1.25, 1.25]))
ppo_output = ppo_policy_loss(current_logp, old_logp, torch.tensor([1.0, -1.0]))
print({"gae": gae.tolist(), "returns": gae_returns.tolist(),
       "ratios": ppo_output.ratio.tolist(),
       "clipped": ppo_output.clipped_objective.tolist()})""",
    "L09": """from rl_study.algorithms.rlhf_ppo import compose_rlhf_rewards
policy_logp = torch.tensor([[-0.2, -0.3, 0.0]])
reference_logp = torch.tensor([[-0.3, -0.25, 0.0]])
token_action_mask = torch.tensor([[True, True, False]])
reward_parts = compose_rlhf_rewards(
    torch.tensor([1.0]), policy_logp, reference_logp,
    token_action_mask, kl_coefficient=0.1
)
print({"sampled_kl": reward_parts.sampled_kl.tolist(),
       "token_rewards": reward_parts.token_rewards.tolist(),
       "total_reward": reward_parts.total_rewards.tolist()})""",
    "L10": """from rl_study.algorithms.rlhf_ppo import train_rlhf_ppo
from rl_study.algorithms.sft import train_sft
from rl_study.models.roles import parameter_sha256
sft_stage = train_sft(steps=2, batch_size=4, seed=42)
sft_hash = parameter_sha256(sft_stage.model)
rlhf_stage = train_rlhf_ppo(
    updates=1, batch_size=2, seed=42, policy=sft_stage.model,
    reward_source="verifier", update_epochs=1
)
print({"sft_steps": 2, "rlhf_updates": 1,
       "initial_sft_hash": sft_hash[:20],
       "generated_tokens": rlhf_stage.generated_tokens,
       "reference_hash": rlhf_stage.reference_hash[:20],
       "policy_loss": round(rlhf_stage.policy_losses[-1], 4)})""",
    "L11": """from rl_study.algorithms.dpo import dpo_loss
policy_chosen = torch.tensor([-1.0, -0.5])
policy_rejected = torch.tensor([-2.0, -0.4])
reference_chosen = torch.tensor([-1.4, -0.7])
reference_rejected = torch.tensor([-1.8, -0.6])
dpo_output = dpo_loss(
    policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.2
)
swapped_output = dpo_loss(
    policy_rejected, policy_chosen, reference_rejected, reference_chosen, beta=0.2
)
print({"logits": dpo_output.logits.tolist(),
       "loss": round(float(dpo_output.loss), 4),
       "swapped_loss": round(float(swapped_output.loss), 4)})""",
    "L12": """from rl_study.algorithms.grpo import group_relative_advantages, rloo_advantages
group_rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
grpo_adv = group_relative_advantages(group_rewards)
rloo_adv = rloo_advantages(group_rewards)
print({"grpo_advantages": grpo_adv.advantages.tolist(),
       "informative": grpo_adv.informative_groups.tolist(),
       "rloo_first_group": rloo_adv[0].tolist(),
       "all_finite": bool(torch.isfinite(grpo_adv.advantages).all())})""",
    "L13": """from rl_study.algorithms.dapo import dynamic_sampling_filter, overlong_reward_shaping
from rl_study.algorithms.grpo import dr_grpo_advantages, gspo_sequence_loss
candidate_rewards = torch.tensor([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
dynamic = dynamic_sampling_filter(candidate_rewards, required_groups=2)
penalties = overlong_reward_shaping(
    torch.tensor([5, 6, 8, 10]), max_response_length=10, buffer_length=4
)
current = torch.log(torch.tensor([[2., 8.], [1., 1.]]))
mask = torch.ones_like(current, dtype=torch.bool)
gspo = gspo_sequence_loss(
    current, torch.zeros_like(current), torch.zeros_like(current),
    torch.tensor([1., -1.]), mask, clip_low=10., clip_high=10.
)
print({"dynamic_indices": dynamic.selected_group_indices.tolist(),
       "overlong_penalty": penalties.tolist(),
       "dr_adv": dr_grpo_advantages(candidate_rewards[1:2]).tolist(),
       "gspo_ratio": gspo.ratio.tolist()})""",
    "L14": """from rl_study.adapters import MODEL_PRESETS, enforce_download_guard, estimate_training_memory
from rl_study.errors import DownloadApprovalRequired
manifest = MODEL_PRESETS["laptop-smoke"]
memory = estimate_training_memory(
    manifest, adapter="lora", dtype="float32", batch_size=1, sequence_length=128
)
guard_blocked = False
try:
    enforce_download_guard(manifest, cached=False, accept_download=False)
except DownloadApprovalRequired:
    guard_blocked = True
print({"model": manifest.hub_id, "revision": manifest.revision[:12],
       "expected_download_mb": round(manifest.expected_bytes / 1e6, 1),
       "recommended_memory_gib": round(memory.recommended_bytes / 2**30, 2),
       "guard_blocked_before_download": guard_blocked})""",
    "L15": """from rl_study.agentic.envs import CalculatorToolEnv
from rl_study.agentic.trajectory import rollout_episode, update_policy
from rl_study.models import TinyCausalLM, TinyLMConfig, TinyTokenizer
agent_model = TinyCausalLM(TinyLMConfig(
    max_sequence_length=192, hidden_size=32, num_heads=4,
    num_layers=1, intermediate_size=64
))
agent_tokenizer = TinyTokenizer()
agent_trajectory, generated, rollout_forwards = rollout_episode(
    agent_model, agent_tokenizer, CalculatorToolEnv(seed=42, max_steps=3),
    generator=torch.Generator().manual_seed(42), policy_version=0, task_index=0
)
agent_optimizer = torch.optim.AdamW(agent_model.parameters(), lr=1e-3)
agent_update = update_policy(
    agent_model, agent_tokenizer, agent_optimizer, agent_trajectory,
    current_policy_version=0, credit_mode="discounted_returns", gamma=0.95
)
print({"episode_steps": len(agent_trajectory.steps), "generated_tokens": generated,
       "rollout_forwards": rollout_forwards, "credits": agent_update.step_credits,
       "tool_outputs_masked": True, "loss": round(agent_update.loss, 4)})""",
    "L16": """report_paths = [
    ROOT / "docs/research/C6_ALIGNMENT_BENCHMARK.json",
    ROOT / "docs/research/C7_GROUP_BENCHMARK.json",
    ROOT / "docs/research/C9_AGENTIC_BENCHMARK.json",
]
audit_rows = []
for report_path in report_paths:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    audit_rows.append({
        "report": report_path.name,
        "origin": payload.get("result_origin"),
        "has_sources": bool(payload.get("sources")),
        "has_guardrail": bool(payload.get("interpretation") or payload.get("interpretation_guardrails")),
    })
print(audit_rows)""",
}

CHECK_CODE = {
    "L00": "assert probability_history[-1] > probability_history[0] > 0.0",
    "L01": "assert float(forward_kl.detach()) >= 0.0 and torch.isfinite(logits.grad).all()",
    "L02": "assert not bool((sequence_batch.prompt_target_mask & sequence_batch.action_mask).any())\nassert int(sequence_batch.action_mask.sum()) == 2",
    "L03": "assert sum(exploring_run[1]) == 120 and exploring_run[0] >= 0.0",
    "L04": "assert dp_result.converged and dp_result.values[-1].item() == 0.0",
    "L05": "assert td_targets[-1].item() == 1.0\nassert torch.allclose(q_target, torch.tensor([4.6]))",
    "L06": "assert not target_has_gradient and torch.isfinite(dqn_output.loss)",
    "L07": "assert chosen_log_probs.grad[-1].item() < 0.0\nassert reward_to_go[-1].item() == 1.0",
    "L08": "assert torch.allclose(ppo_output.ratio, torch.tensor([1.25, 1.25]))\nassert torch.isfinite(gae).all()",
    "L09": "assert reward_parts.token_rewards[0, 2].item() == 0.0\nassert torch.allclose(reward_parts.total_rewards, reward_parts.token_rewards.sum(-1))",
    "L10": "assert rlhf_stage.generated_tokens > 0\nassert rlhf_stage.reference_hash == sft_hash",
    "L11": "assert dpo_output.loss.item() != swapped_output.loss.item()\nassert torch.isfinite(dpo_output.loss)",
    "L12": "assert grpo_adv.informative_groups.tolist() == [True, False]\nassert torch.equal(grpo_adv.advantages[1], torch.zeros(4))",
    "L13": "assert dynamic.selected_group_indices.tolist() == [1, 3]\nassert penalties.tolist() == [0.0, 0.0, -0.5, -1.0]",
    "L14": "assert guard_blocked and manifest.expected_bytes > 100_000_000",
    "L15": "assert len(agent_trajectory.steps) >= 1\nassert all(step.action_token_ids for step in agent_trajectory.steps)",
    "L16": "assert len(audit_rows) == 3\nassert all(row[\"origin\"] == \"local_executed\" for row in audit_rows)",
}

# Each lesson keeps one small executable question, but its surrounding explanation is
# lesson-specific.  The equation string is shared verbatim by both languages so the
# mirror checker can audit mathematical parity rather than just code parity.
LESSON_GUIDE: dict[str, dict[str, str | tuple[str, str]]] = {
    "L00": {
        "equation": r"$$J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[r(a)]$$",
        "position": (
            "현재 위치: **전체 지도** → 확률·최적화 → bandit/MDP → policy gradient/PPO → LLM 정렬 → Agentic RL → 평가",
            "Position: **whole map** → probability/optimization → bandit/MDP → policy gradient/PPO → LLM alignment → Agentic RL → evaluation",
        ),
        "explain": (
            "RL은 agent가 관측을 보고 action을 고르고, 환경의 feedback으로 미래 action 분포를 바꾸는 학습입니다. LLM에서는 action이 token 또는 tool call이고, DPO는 저장된 선호쌍으로 이 효과를 직접 최적화하며, PPO·GRPO는 새 응답을 rollout해 reward를 받습니다.\n\n```mermaid\nflowchart LR\n  P[확률·최적화] --> B[Bandit·MDP]\n  B --> PG[Policy gradient·PPO]\n  PG --> L[LLM policy·reward]\n  L --> R[RLHF·DPO·GRPO·DAPO]\n  L --> A[Agentic RL]\n  R --> E[평가·재현]\n  A --> E\n```\n\nMermaid가 보이지 않을 때의 동등한 지도:\n\n```text\n확률·최적화 → bandit → MDP/Bellman → MC/TD/Q-learning → DQN\n             └→ policy gradient → actor-critic/GAE → PPO\n                                      └→ LLM policy + preference/reward\n                                           ├→ RLHF-PPO\n                                           ├→ DPO\n                                           ├→ GRPO/RLVR → DAPO\n                                           └→ Agentic RL\n모든 경로 → 평가·reward hacking 진단·재현성\n```",
            "RL changes an agent's future action distribution using feedback after it acts on an observation. For an LLM, an action is a token or tool call. DPO optimizes the effect from stored preference pairs, while PPO and GRPO roll out new responses and receive rewards.\n\n```mermaid\nflowchart LR\n  P[Probability·optimization] --> B[Bandit·MDP]\n  B --> PG[Policy gradient·PPO]\n  PG --> L[LLM policy·reward]\n  L --> R[RLHF·DPO·GRPO·DAPO]\n  L --> A[Agentic RL]\n  R --> E[Evaluation·reproducibility]\n  A --> E\n```\n\nEquivalent fallback when Mermaid is unavailable:\n\n```text\nprobability·optimization → bandit → MDP/Bellman → MC/TD/Q-learning → DQN\n                         └→ policy gradient → actor-critic/GAE → PPO\n                                                  └→ LLM policy + preference/reward\n                                                       ├→ RLHF-PPO\n                                                       ├→ DPO\n                                                       ├→ GRPO/RLVR → DAPO\n                                                       └→ Agentic RL\nall paths → evaluation·reward-hacking diagnosis·reproducibility\n```",
        ),
        "predict": ("reward=1인 action의 확률은 20회 update 뒤 어느 방향으로 갈까요?", "Which way will the probability of the reward-1 action move after 20 updates?"),
        "answer": ("목적함수의 부호가 맞다면 증가합니다. 이것이 뒤의 모든 알고리즘에서 확인할 최소 단위입니다.", "It increases when the objective sign is correct. This is the smallest unit checked throughout the course."),
        "why": ("두 action으로 시작하면 상태·credit assignment를 잠시 치워 두고 `확률 → reward → gradient` 고리만 볼 수 있습니다. 대안인 큰 trainer는 현실적이지만 첫 오류의 원인을 분리하기 어렵습니다.", "Two actions isolate the `probability → reward → gradient` loop before state and credit assignment enter. A full trainer is more realistic but makes the first failure harder to isolate."),
        "trap": ("loss에 음수를 빠뜨리면 optimizer가 좋은 action을 줄입니다. `p_good_end > p_good_start` 회귀 검사가 부호 오류를 잡습니다.", "Dropping the minus sign makes the optimizer suppress the useful action. The `p_good_end > p_good_start` regression check catches it."),
        "conclusion": ("출력에서 좋은 action 확률이 0.5에서 0.916으로 증가했습니다. 이는 이 toy 목적함수의 방향만 검증하며 알고리즘 간 우열을 뜻하지 않습니다.", "The useful-action probability rose from 0.5 to 0.916. This validates only the toy objective's direction, not an algorithm ranking."),
        "recall": ("PPO·DPO·GRPO 중 새 응답을 rollout하지 않아도 되는 것은 무엇이며, 왜 그런가요?", "Which of PPO, DPO, and GRPO does not require new response rollouts, and why?"),
        "next": ("L01에서 이 확률 변화의 재료인 log-probability, entropy, KL과 gradient를 직접 계산합니다.", "L01 computes the log-probability, entropy, KL, and gradients behind this probability change."),
    },
    "L01": {
        "equation": r"$$H(p)=-\sum_i p_i\log p_i,\qquad D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}$$",
        "position": ("현재 위치: 전체 지도 → **확률·미분** → policy gradient", "Position: whole map → **probability and gradients** → policy gradient"),
        "explain": ("log-prob는 곱을 합으로 바꾸고 선택한 action의 민감도를 표현합니다. entropy는 분포의 퍼짐, KL은 방향이 있는 두 분포의 차이입니다. `detach`는 값을 유지하면서 그 경로의 gradient 소유권을 끊습니다.", "Log-probability turns products into sums and exposes sensitivity of a chosen action. Entropy measures spread; KL is a directional discrepancy between distributions. `detach` preserves a value while removing gradient ownership along that path."),
        "predict": ("entropy를 키우고 forward KL을 줄이면 가장 큰 logit의 gradient 부호는 어떻게 될까요?", "When maximizing entropy and reducing forward KL, what sign do you expect on the largest logit's gradient?"),
        "answer": ("두 항이 경쟁하므로 직감만으로 확정하지 말고 autograd 값을 읽어야 합니다. 이 입력에서는 두 번째 logit gradient가 음수입니다.", "The terms compete, so inspect autograd rather than guessing. For this input the second-logit gradient is negative."),
        "why": ("확률을 직접 clamp하기보다 `log_softmax` 기반 package 함수를 쓰면 정규화와 수치 안정성을 함께 얻습니다. reverse KL은 다른 mode-seeking 성질을 가지므로 같은 값으로 취급할 수 없습니다.", "Package functions based on `log_softmax` provide normalization and numerical stability without manual probability clamping. Reverse KL has different mode-seeking behavior and is not interchangeable."),
        "trap": ("`float(tensor)`로 gradient tensor를 출력하면 경고가 납니다. 관찰용 값은 `detach()`한 뒤 scalar로 바꾸고, 학습 loss에는 detach하지 않습니다.", "Calling `float(tensor)` on a gradient-tracking tensor warns. Detach observation-only values before scalar conversion, but never detach the training loss."),
        "conclusion": ("entropy와 KL은 유한했고 세 logit gradient의 합은 거의 0입니다. softmax가 공통 logit 이동에 불변이라는 사실과 맞습니다.", "Entropy and KL were finite, and the three logit gradients sum to approximately zero, matching softmax invariance to a common logit shift."),
        "recall": ("`D_KL(p||q)`와 `D_KL(q||p)`를 바꾸면 왜 같은 regularizer가 아닌가요?", "Why does swapping `D_KL(p||q)` and `D_KL(q||p)` change the regularizer?"),
        "next": ("L02에서 class 하나를 고르는 확률을 token sequence의 log-probability와 mask로 확장합니다.", "L02 extends one categorical choice to token-sequence log-probabilities and masks."),
    },
    "L02": {
        "equation": r"$$\log\pi_\theta(y\mid x)=\sum_t m_t\log p_\theta(y_t\mid x,y_{<t})$$",
        "position": ("현재 위치: 확률·미분 → **causal LM과 token mask** → LLM policy", "Position: probability/gradients → **causal LM and token masks** → LLM policy"),
        "explain": ("causal LM logits는 `[batch, time, vocabulary]`이고 target은 한 칸 왼쪽 logits와 맞춥니다. prompt는 조건이며 action이 아니므로 policy loss에는 response와 EOS 위치만 남깁니다. padding은 계산량을 맞출 뿐 reward를 받지 않습니다.", "Causal-LM logits have shape `[batch, time, vocabulary]`, with targets aligned to logits shifted by one position. The prompt is context, not an action, so policy loss keeps only response and EOS positions. Padding carries no reward."),
        "predict": ("문자열 `2`가 response일 때 action mask 합은 1일까요, EOS까지 포함한 2일까요?", "For response `2`, is the action-mask sum 1, or 2 including EOS?"),
        "answer": ("이 저장소의 계약은 생성 종료를 action으로 포함하므로 2입니다. 이 선택은 rollout과 update에서 동일해야 합니다.", "This repository includes generation termination as an action, so the answer is 2. Rollout and update must share that contract."),
        "why": ("평균 token log-prob와 합계 sequence log-prob는 길이에 대한 의미가 다릅니다. package API가 mask와 reduction을 명시해 DPO·PPO·GRPO가 서로 다른 암묵적 규칙을 갖지 않게 합니다.", "Mean token log-probability and summed sequence log-probability encode length differently. Explicit masks and reductions in the package API prevent DPO, PPO, and GRPO from silently using different rules."),
        "trap": ("prompt target까지 합치면 긴 prompt가 update를 지배합니다. prompt/action mask의 교집합이 비어 있다는 truth-table 검사가 이를 잡습니다.", "Including prompt targets lets long prompts dominate updates. A truth-table test requiring an empty prompt/action-mask intersection catches it."),
        "conclusion": ("출력은 input 길이 9에 대해 예측 위치 8개, 실제 action 위치 2개를 보여 줍니다. sequence log-prob는 그 두 위치만 합친 값입니다.", "The output shows 8 prediction positions for input length 9 and only 2 action positions. Sequence log-probability sums only those two positions."),
        "recall": ("EOS를 mask에서 빼면 어떤 행동의 학습 신호가 사라지나요?", "What behavior loses its learning signal if EOS is removed from the mask?"),
        "next": ("L03에서 sequence를 잠시 내려놓고 exploration과 sampled reward의 가장 작은 실험을 만듭니다.", "L03 temporarily sets sequences aside to study exploration and sampled rewards in their smallest setting."),
    },
    "L03": {
        "equation": r"$$Q_{n+1}(a)=Q_n(a)+\frac{1}{N_n(a)}\left(R_n-Q_n(a)\right)$$",
        "position": ("현재 위치: 확률·미분 → **bandit** → MDP", "Position: probability/gradients → **bandits** → MDPs"),
        "explain": ("bandit에는 state transition이 없고 매 순간 arm 하나만 고릅니다. exploitation은 현재 최대 추정값을, exploration은 아직 모르는 arm을 시험합니다. regret는 실제 표본 손실이 아니라 최적 arm 기대보상과 선택 arm 기대보상의 누적 차이입니다.", "A bandit has no state transitions; each step chooses one arm. Exploitation follows the current best estimate, while exploration tests uncertain arms. Regret accumulates expected-reward gaps, not realized sample losses."),
        "predict": ("초기 추정값이 모두 0일 때 순수 greedy는 항상 좋은 arm을 찾을까요?", "With all estimates initialized to zero, will pure greedy always discover the best arm?"),
        "answer": ("아닙니다. 초반 우연과 tie-breaking에 갇힐 수 있습니다. 작은 epsilon이 이 실패 경로를 끊습니다.", "No. Early randomness and tie-breaking can trap it. A small epsilon breaks that failure path."),
        "why": ("incremental mean은 전체 reward 기록 없이 같은 표본평균을 만듭니다. UCB나 Thompson sampling도 대안이지만, epsilon-greedy가 exploration 자체의 역할을 가장 투명하게 드러냅니다.", "The incremental mean reproduces the sample mean without storing rewards. UCB and Thompson sampling are alternatives, but epsilon-greedy makes the role of exploration most transparent."),
        "trap": ("sample reward로 regret를 계산하면 운 좋은 나쁜 arm이 음의 regret를 만들 수 있습니다. 환경이 제공한 expected regret를 별도로 누적합니다.", "Using sampled rewards for regret can make a lucky bad arm appear to have negative regret. Accumulate the environment's expected regret separately."),
        "conclusion": ("고정 seed 출력에서 epsilon=0.1의 누적 regret가 순수 greedy보다 낮았습니다. 이는 이 한 환경의 실패 사례이며 보편적 우월성 주장이 아닙니다.", "For the fixed seed, epsilon=0.1 had lower cumulative regret than pure greedy. This is one environment's failure case, not a universal superiority claim."),
        "recall": ("exploration이 reward를 즉시 낮추면서도 장기 regret를 줄일 수 있는 이유는 무엇인가요?", "How can exploration reduce long-run regret while lowering immediate reward?"),
        "next": ("L04에서 state와 transition을 추가해 현재 action이 미래 reward까지 바꾸는 MDP로 갑니다.", "L04 adds states and transitions, moving to MDPs where actions change future rewards."),
    },
    "L04": {
        "equation": r"$$V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)\left[r+\gamma V_k(s')\right]$$",
        "position": ("현재 위치: bandit → **MDP·Bellman** → MC/TD/Q-learning", "Position: bandits → **MDPs and Bellman equations** → MC/TD/Q-learning"),
        "explain": ("MDP는 미래가 현재 state와 action으로 충분하다는 모델입니다. Bellman backup은 장기 return 문제를 한 step reward와 다음 state value로 재귀 분해합니다. terminal state에는 미래가 없으므로 bootstrap value가 0입니다.", "An MDP assumes the current state and action contain everything needed for the future. A Bellman backup decomposes long-run return into one-step reward plus next-state value. Terminal states have no future, so their bootstrap value is zero."),
        "predict": ("goal 바로 옆 state의 value는 terminal value 0보다 클까요?", "Will a state next to the goal have value above the terminal state's zero value?"),
        "answer": ("goal로 들어갈 때 reward를 받으므로 큽니다. terminal 자체에서 다시 reward를 받는 것은 아닙니다.", "Yes, because reward is received on entering the goal. The terminal state itself does not pay repeatedly."),
        "why": ("transition model이 알려진 작은 grid에서는 value iteration이 정확한 기준선을 제공합니다. model-free 학습과 비교할 때 환경·gamma 오류를 먼저 분리할 수 있습니다.", "With a known transition model, value iteration provides an exact small-grid baseline. It isolates environment and gamma errors before comparison with model-free learning."),
        "trap": ("terminal에서 bootstrap하면 goal에 머물며 reward를 무한 반복하는 잘못된 MDP가 됩니다. terminal value 0 검사가 경계를 고정합니다.", "Bootstrapping from a terminal state creates a different MDP with repeatedly collected goal reward. The terminal-zero check fixes the boundary."),
        "conclusion": ("value iteration은 7회에 수렴했고 terminal value는 정확히 0입니다. goal에 가까운 비종료 state일수록 할인된 value가 커집니다.", "Value iteration converged in 7 iterations and the terminal value is exactly zero. Nonterminal states nearer the goal have larger discounted values."),
        "recall": ("Bellman expectation equation과 optimality equation에서 `max`의 유무는 무엇을 뜻하나요?", "What does the presence or absence of `max` mean in Bellman expectation versus optimality equations?"),
        "next": ("L05에서는 transition model 없이 실제 trajectory로 이 Bellman target을 추정합니다.", "L05 estimates these Bellman targets from trajectories without a transition model."),
    },
    "L05": {
        "equation": r"$$G_t=R_{t+1}+\gamma G_{t+1},\qquad y_t^{TD}=R_{t+1}+\gamma(1-d_t)V(S_{t+1})$$",
        "position": ("현재 위치: MDP·Bellman → **MC·TD·Q-learning** → DQN", "Position: MDP/Bellman → **MC, TD, and Q-learning** → DQN"),
        "explain": ("MC는 episode 끝까지 관찰한 return을 쓰므로 편향은 작지만 분산이 큽니다. TD는 다음 value로 bootstrap해 더 일찍 배우지만 추정 오차를 물려받습니다. Q-learning target은 실제 다음 action이 아니라 최대 Q action을 써서 off-policy입니다.", "MC uses returns observed through episode end, giving low bias but high variance. TD bootstraps from the next value, learning earlier while inheriting estimation error. Q-learning is off-policy because its target uses the max-Q action rather than the action actually taken."),
        "predict": ("terminal transition의 next value가 99여도 TD target에 들어갈까요?", "Does a next-state value of 99 enter the TD target on a terminal transition?"),
        "answer": ("들어가지 않습니다. `(1-d_t)`가 bootstrap을 0으로 만들고 target은 마지막 reward 1입니다.", "No. `(1-d_t)` zeros the bootstrap, so the target is the final reward 1."),
        "why": ("세 target을 같은 trajectory에 놓으면 variance, bootstrap, off-policy라는 차이 하나씩만 바뀝니다. 별도 학습 curve보다 먼저 analytic 값으로 구현 경계를 검산하기 좋습니다.", "Putting all three targets on one trajectory changes one property at a time: variance, bootstrapping, and off-policy choice. Analytic values audit implementation boundaries before training curves do."),
        "trap": ("시간 제한 `truncated`를 환경 종결 `terminated`처럼 다루면 bootstrap 가능한 정보를 버립니다. 두 flag를 합치지 않는 API와 test가 필요합니다.", "Treating time-limit `truncated` as environment `terminated` discards valid bootstrap information. The API and tests keep the two flags separate."),
        "conclusion": ("출력에서 terminal TD target은 1.0으로 고정되고, Q-learning은 최대 next-Q 4를 사용해 4.6을 만들었습니다.", "The terminal TD target stays at 1.0, while Q-learning uses max next-Q of 4 to produce 4.6."),
        "recall": ("MC target과 TD target 중 어느 쪽이 현재 value estimate에 직접 의존하나요?", "Which target, MC or TD, directly depends on the current value estimate?"),
        "next": ("L06에서 Q table을 neural network로 바꾸고 target network와 replay가 왜 필요한지 봅니다.", "L06 replaces the Q table with a neural network and studies target networks and replay."),
    },
    "L06": {
        "equation": r"$$L(\theta)=\mathbb{E}\left[\left(Q_\theta(s,a)-\operatorname{stopgrad}(r+\gamma\max_{a'}Q_{\bar\theta}(s',a'))\right)^2\right]$$",
        "position": ("현재 위치: MC·TD·Q-learning → **DQN** → policy gradient", "Position: MC/TD/Q-learning → **DQN** → policy gradients"),
        "explain": ("DQN은 Q table을 network로 근사합니다. replay buffer는 연속 sample의 상관을 줄이고 재사용하며, target network는 target이 매 gradient step마다 함께 움직이지 않게 늦게 동기화합니다.", "DQN approximates a Q table with a network. Replay reduces correlation and reuses experience; a target network is synchronized slowly so the learning target does not move with every gradient step."),
        "predict": ("loss.backward 뒤 target network parameter에 gradient가 생겨야 할까요?", "Should target-network parameters have gradients after `loss.backward()`?"),
        "answer": ("아니요. target은 학습 신호의 숫자만 제공하며 optimizer 소유가 아닙니다.", "No. The target provides numbers for the learning signal and is not owned by the optimizer."),
        "why": ("여기서는 작은 batch로 target detach 계약을 먼저 검증합니다. Double DQN은 action 선택과 평가를 나눠 과대추정을 줄이는 대안이지만 기본 gradient ownership은 같습니다.", "A tiny batch first verifies the target-detach contract. Double DQN separates action selection from evaluation to reduce overestimation, but keeps the same gradient ownership."),
        "trap": ("target tensor만 detach해도 target network를 optimizer에 넣으면 이후 다른 loss에서 움직일 수 있습니다. frozen parameter와 optimizer membership도 함께 검사해야 합니다.", "Detaching the target tensor is insufficient if target parameters remain in an optimizer and can move under another loss. Check freezing and optimizer membership too."),
        "conclusion": ("loss와 target은 유한했고 `target_has_gradient=False`였습니다. 이 출력은 update 방향보다 frozen target 경계를 검증합니다.", "The loss and targets were finite, with `target_has_gradient=False`. This output validates the frozen-target boundary rather than training quality."),
        "recall": ("replay와 target network가 각각 줄이려는 불안정성은 어떻게 다른가요?", "How do the instabilities addressed by replay and target networks differ?"),
        "next": ("L07에서는 action value를 맞추는 대신 policy 확률을 reward 방향으로 직접 움직입니다.", "L07 moves policy probabilities directly toward reward instead of fitting action values."),
    },
    "L07": {
        "equation": r"$$\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))\right]$$",
        "position": ("현재 위치: 확률·미분 → **Policy Gradient·REINFORCE** → Actor-Critic·PPO", "Position: probability/gradients → **Policy Gradient and REINFORCE** → Actor-Critic/PPO"),
        "explain": ("log-derivative trick은 sample한 action의 log-prob에 return을 곱해 expectation의 gradient를 추정합니다. reward-to-go는 과거 action에 미래 reward를 배분합니다. action과 무관한 baseline은 기대 gradient를 바꾸지 않으면서 분산을 줄입니다.", "The log-derivative trick estimates an expectation gradient by multiplying sampled-action log-probability by return. Reward-to-go assigns future rewards to earlier actions. An action-independent baseline reduces variance without changing the expected gradient."),
        "predict": ("positive advantage가 곱해진 chosen log-prob의 loss gradient 부호는 무엇인가요?", "What is the loss-gradient sign for a chosen log-probability multiplied by positive advantage?"),
        "answer": ("음수입니다. gradient descent가 log-prob를 키우려면 loss 미분은 음수여야 합니다.", "Negative. Gradient descent increases log-probability only when the loss derivative is negative."),
        "why": ("return과 baseline은 policy parameter 관점에서 detach해야 credit 값이 actor를 통해 다시 만들어지지 않습니다. actor-critic은 learned baseline으로 이를 일반화합니다.", "Returns and baselines are detached with respect to policy parameters so credit values are not recreated through the actor. Actor-critic generalizes this with a learned baseline."),
        "trap": ("loss 부호를 직관으로만 외우면 maximize/minimize 변환에서 자주 뒤집힙니다. positive advantage의 log-prob gradient가 음수인지 직접 assert합니다.", "Memorizing the sign verbally often fails when switching between maximize and minimize forms. Assert directly that positive advantage gives a negative log-probability gradient."),
        "conclusion": ("세 reward-to-go가 모두 양수였고 chosen log-prob gradient도 모두 음수였습니다. 더 큰 advantage일수록 절댓값이 컸습니다.", "All three reward-to-go values were positive and all chosen-log-probability gradients were negative. Larger advantages produced larger magnitudes."),
        "recall": ("action에 의존하는 baseline을 빼면 왜 policy-gradient 추정이 편향될 수 있나요?", "Why can an action-dependent baseline bias a policy-gradient estimate?"),
        "next": ("L08에서 baseline을 critic으로 학습하고 GAE로 bias-variance를 조절한 뒤 PPO로 update 폭을 제한합니다.", "L08 learns the baseline as a critic, tunes bias/variance with GAE, and limits updates with PPO."),
    },
    "L08": {
        "equation": r"$$\hat A_t=\delta_t+(\gamma\lambda)\hat A_{t+1},\qquad L^{clip}=\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)$$",
        "position": ("현재 위치: REINFORCE → **Actor-Critic·GAE·PPO** → LLM policy", "Position: REINFORCE → **Actor-Critic, GAE, and PPO** → LLM policy"),
        "explain": ("critic은 state value를 예측하고 TD residual `δ`가 advantage의 재료가 됩니다. GAE의 lambda는 짧은 bootstrap과 긴 return 사이를 잇습니다. PPO ratio는 rollout 당시 old policy와 현재 policy의 선택확률 비율입니다.", "The critic predicts state value, and TD residual `δ` supplies advantage estimates. GAE lambda interpolates between short bootstraps and long returns. The PPO ratio compares current selected-action probability with the rollout-time old policy."),
        "predict": ("ratio=1.25, epsilon=0.2일 때 positive와 negative advantage 모두 같은 방식으로 잘릴까요?", "At ratio 1.25 and epsilon 0.2, are positive and negative advantages clipped in the same way?"),
        "answer": ("아닙니다. `min` 때문에 positive advantage는 1.2에서 제한되지만 negative advantage는 더 나쁜 -1.25가 선택됩니다. 양쪽 부호를 따로 검산해야 합니다.", "No. The `min` caps positive advantage at 1.2, while negative advantage keeps the worse -1.25. Audit both signs separately."),
        "why": ("old log-prob는 rollout snapshot이고 detach되어야 합니다. KL penalty나 early stopping도 대안이며, clip 하나가 실제 trust region을 보장한다고 해석해서는 안 됩니다.", "Old log-probability is a detached rollout snapshot. KL penalties and early stopping are alternatives; clipping alone should not be interpreted as a guaranteed trust region."),
        "trap": ("current policy로 old log-prob를 다시 계산하면 ratio가 늘 1이 되어 update 진단이 무력화됩니다. rollout artifact의 policy version과 log-prob를 보존합니다.", "Recomputing old log-probability with the current policy makes the ratio always 1 and destroys update diagnostics. Preserve rollout log-probabilities and policy version."),
        "conclusion": ("GAE는 유한한 두 advantage를 만들었고 ratio는 둘 다 1.25였습니다. 출력의 clipped 항은 advantage 부호에 따라 비대칭입니다.", "GAE produced two finite advantages and both ratios were 1.25. The printed clipped terms are asymmetric across advantage signs."),
        "recall": ("lambda=0과 lambda=1은 각각 어떤 target에 가까워지나요?", "What targets do lambda=0 and lambda=1 approach?"),
        "next": ("L09에서 PPO의 한 action을 LLM response token들로 바꾸고 KL·mask·reward 위치를 다시 정의합니다.", "L09 replaces one PPO action with LLM response tokens and redefines KL, masks, and reward placement."),
    },
    "L09": {
        "equation": r"$$r_t^{total}=r_t^{task}-\beta\left(\log\pi_\theta(a_t)-\log\pi_{ref}(a_t)\right)$$",
        "position": ("현재 위치: PPO + causal LM → **LLM policy·reward·reference KL** → RLHF/DPO/GRPO", "Position: PPO + causal LM → **LLM policy, reward, and reference KL** → RLHF/DPO/GRPO"),
        "explain": ("prompt는 초기 state, 생성 token은 action, prefix는 다음 state입니다. scalar reward는 보통 response 끝에 붙지만 KL shaping은 action token마다 계산할 수 있습니다. reference model은 SFT 근처에서 policy가 너무 빨리 벗어나는 것을 측정합니다.", "The prompt is the initial state, generated tokens are actions, and each prefix is the next state. Scalar reward often lands at response end, while KL shaping can be computed per action token. A reference model measures rapid drift from the SFT policy."),
        "predict": ("action mask가 false인 세 번째 위치의 KL과 token reward는 얼마여야 하나요?", "What should KL and token reward be at the third position where the action mask is false?"),
        "answer": ("둘 다 0이어야 합니다. prompt/pad/tool token은 policy가 선택한 action이 아닙니다.", "Both must be zero. Prompt, padding, and tool-output tokens are not actions selected by the policy."),
        "why": ("sampled KL은 선택된 token 위의 log-ratio라 빠르지만 분산이 있고 음수가 될 수도 있습니다. full-distribution KL은 더 비싸지만 다른 진단 의미를 가집니다.", "Sampled KL is a fast selected-token log-ratio, but it has variance and can be negative. Full-distribution KL is costlier and answers a different diagnostic question."),
        "trap": ("scalar reward를 모든 token에 반복해서 더하면 response 길이만큼 보상을 복제합니다. terminal action에 한 번 붙이고 KL 항과 분리해 합계를 검산합니다.", "Repeating scalar reward on every token duplicates it by response length. Attach it once at the terminal action and audit it separately from KL shaping."),
        "conclusion": ("세 번째 masked 위치의 값은 0이고 두 action token의 합계 reward는 0.995였습니다. task reward와 KL 비용을 따로 볼 수 있습니다.", "The third masked position is zero, and the two action-token rewards sum to 0.995. Task reward and KL cost remain separately inspectable."),
        "recall": ("sampled KL 한 항이 음수여도 전체 regularization이 잘못되었다고 단정할 수 없는 이유는 무엇인가요?", "Why does one negative sampled-KL term not prove the regularizer is wrong?"),
        "next": ("L10에서 SFT, reward source, rollout, PPO update를 하나의 lifecycle로 연결합니다.", "L10 connects SFT, reward source, rollout, and PPO update into one lifecycle."),
    },
    "L10": {
        "equation": r"$$\text{SFT}\rightarrow\text{preference/RM}\rightarrow\text{rollout}\rightarrow\text{reward+KL}\rightarrow\text{PPO update}$$",
        "position": ("현재 위치: LLM policy → **RLHF-PPO end-to-end** → 비교·평가", "Position: LLM policy → **end-to-end RLHF-PPO** → comparison/evaluation"),
        "explain": ("RLHF-PPO에는 trainable policy와 value, frozen reference와 reward model이라는 서로 다른 역할이 있습니다. 이 toy는 deterministic verifier를 reward source로 써 pipeline을 완전히 offline으로 실행하지만 ownership과 mask 계약은 실제 모델과 같습니다.", "RLHF-PPO separates trainable policy/value from frozen reference/reward models. This toy uses a deterministic verifier for a fully offline pipeline, while preserving the ownership and mask contracts used with real models."),
        "predict": ("PPO가 시작될 때 reference hash는 SFT policy의 어느 시점 hash와 같아야 하나요?", "At PPO start, which SFT-policy hash should the reference hash match?"),
        "answer": ("rollout 전에 복제한 초기 SFT policy hash와 같아야 하며 이후 고정됩니다.", "It must match the initial SFT policy copied before rollout and remain frozen afterward."),
        "why": ("단계별 public API를 호출해 전체 lifecycle을 보되 trainer 내부를 notebook에 복사하지 않습니다. learned reward model은 대안이며 C5 구현에서 verifier와 동일한 interface를 공유합니다.", "Public stage APIs expose the lifecycle without copying trainer internals into the notebook. A learned reward model is an alternative and shares the verifier interface in the C5 implementation."),
        "trap": ("reference가 policy와 함께 update되면 KL anchor가 움직여 penalty가 작아 보입니다. hash와 `requires_grad=False`, optimizer membership을 함께 검사합니다.", "If the reference updates with the policy, the KL anchor moves and the penalty appears artificially small. Audit its hash, `requires_grad=False`, and optimizer membership."),
        "conclusion": ("2-step SFT 뒤 44 token을 rollout했고 reference hash가 초기 SFT hash와 일치했습니다. 한 update의 policy loss 0.0은 ratio=1 시작점 결과이지 학습 성공 주장이 아닙니다.", "After two SFT steps, the run generated 44 tokens and the reference hash matched the initial SFT hash. A one-update policy loss of 0.0 reflects the ratio-1 starting point, not training success."),
        "recall": ("policy, value, reference, reward 네 역할 중 optimizer가 소유해야 하는 것은 무엇인가요?", "Which of policy, value, reference, and reward roles should be optimizer-owned?"),
        "next": ("L11에서는 online rollout 없이 chosen/rejected preference pair로 policy를 직접 최적화합니다.", "L11 optimizes the policy directly from chosen/rejected pairs without online rollouts."),
    },
    "L11": {
        "equation": r"$$L_{DPO}=-\log\sigma\left(\beta\left[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))\right]\right)$$",
        "position": ("현재 위치: preference data → **DPO** → offline alignment 평가", "Position: preference data → **DPO** → offline-alignment evaluation"),
        "explain": ("DPO는 chosen이 rejected보다 reference 대비 얼마나 더 좋아졌는지를 logistic loss로 학습합니다. 별도 reward model과 online rollout이 없지만 reference policy와 preference data가 암묵적인 RL 문제를 정의합니다.", "DPO applies a logistic loss to how much more the chosen response improves over the rejected one relative to a reference. It removes a separate reward model and online rollout, but reference policy and preference data still define an implicit RL problem."),
        "predict": ("chosen/rejected를 바꾸면 loss가 같을까요?", "Does the loss stay the same when chosen and rejected are swapped?"),
        "answer": ("일반적으로 다릅니다. preference margin의 부호가 뒤집혀 sigmoid의 반대쪽을 평가합니다.", "Generally no. The preference margin changes sign and evaluates the opposite side of the sigmoid."),
        "why": ("sequence log-prob 합계와 평균은 length bias가 다르므로 reduction을 명시해야 합니다. IPO, KTO 등은 선호 noise와 데이터 형태에 대한 다른 가정을 둔 대안입니다.", "Summed and mean sequence log-probabilities have different length bias, so reduction must be explicit. IPO and KTO are alternatives with different assumptions about preference noise and data shape."),
        "trap": ("reference 항을 빼거나 chosen과 rejected의 순서를 뒤집어도 loss는 유한합니다. 손계산 parity와 swapped-pair test가 의미 오류를 잡습니다.", "The loss remains finite if the reference term is dropped or pair order is reversed. Hand-calculation parity and a swapped-pair test catch semantic errors."),
        "conclusion": ("원래 pair loss 0.664와 swapped loss 0.724가 달랐습니다. 출력된 두 margin 중 하나는 거의 0이라 그 sample은 약한 학습 신호를 줍니다.", "The original-pair loss 0.664 differs from swapped loss 0.724. One printed margin is near zero, so that sample carries a weak preference signal."),
        "recall": ("DPO에서 reference model을 제거하면 어떤 기준점이 사라지나요?", "What anchor disappears if the reference model is removed from DPO?"),
        "next": ("L12에서 한 prompt의 여러 rollout reward를 group 안에서 상대화하는 GRPO로 돌아갑니다.", "L12 returns to online rollouts and normalizes several rewards within each prompt group."),
    },
    "L12": {
        "equation": r"$$\hat A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G})+\epsilon}$$",
        "position": ("현재 위치: LLM policy + verifier → **GRPO·RLVR** → DAPO", "Position: LLM policy + verifier → **GRPO and RLVR** → DAPO"),
        "explain": ("GRPO는 같은 prompt에서 G개 completion을 뽑고 group reward 평균을 baseline으로 사용해 별도 critic을 없앱니다. RLVR은 정답·형식처럼 검증 가능한 reward를 사용해 reward model의 모호함을 줄입니다.", "GRPO samples G completions for one prompt and uses group-mean reward as a baseline, removing a separate critic. RLVR uses verifiable rewards such as correctness and format, reducing reward-model ambiguity."),
        "predict": ("한 group의 reward가 모두 2라면 normalized advantage는 NaN일까요, 0일까요?", "If every reward in a group is 2, are normalized advantages NaN or zero?"),
        "answer": ("안전한 구현은 0으로 만들고 그 group을 informative하지 않다고 표시합니다.", "A safe implementation returns zeros and marks the group uninformative."),
        "why": ("epsilon만 분모에 더하면 유한값은 만들지만 거의 같은 reward를 과장할 수 있습니다. 명시적 zero-variance mask가 sample budget 낭비를 관찰하게 합니다. RLOO는 다른 baseline 대안입니다.", "Adding epsilon alone keeps values finite but can amplify nearly identical rewards. An explicit zero-variance mask exposes wasted sampling budget. RLOO is an alternative baseline."),
        "trap": ("서로 다른 prompt의 reward를 한 batch에서 함께 normalize하면 난이도 차이가 credit으로 섞입니다. group axis와 prompt ID를 보존해야 합니다.", "Normalizing rewards across different prompts mixes prompt difficulty into credit. Preserve the group axis and prompt IDs."),
        "conclusion": ("첫 group은 약 ±1 advantage를 만들었고, 상수 reward인 두 번째 group은 정확히 0이며 `informative=False`였습니다.", "The first group produced advantages near ±1; the constant-reward second group produced exact zeros with `informative=False`."),
        "recall": ("critic을 없애면 줄어드는 비용과 새로 커지는 sample 의존성은 각각 무엇인가요?", "What cost shrinks when removing the critic, and what sampling dependency grows?"),
        "next": ("L13에서 zero-variance group, 길이 편향, clipping 문제를 DAPO·Dr.GRPO·GSPO 변형으로 분해합니다.", "L13 decomposes zero-variance groups, length bias, and clipping through DAPO, Dr.GRPO, and GSPO variants."),
    },
    "L13": {
        "equation": r"$$L_{DAPO}=L_{clip\text{-}higher}+L_{dynamic\ sampling}+L_{token\text{-}level}+L_{overlong}$$",
        "position": ("현재 위치: GRPO → **DAPO·Dr.GRPO·GSPO** → 안정성 평가", "Position: GRPO → **DAPO, Dr.GRPO, and GSPO** → stability evaluation"),
        "explain": ("DAPO는 하나의 마법 공식이 아니라 asymmetric clipping, informative group만 남기는 dynamic sampling, token-level loss, overlong shaping의 묶음입니다. Dr.GRPO는 normalization을 단순화하고 GSPO는 sequence-level ratio로 긴 응답의 token 변동을 모읍니다.", "DAPO is not one magic equation; it combines asymmetric clipping, dynamic sampling of informative groups, token-level loss, and overlong shaping. Dr.GRPO simplifies normalization, while GSPO aggregates token changes into a sequence-level ratio."),
        "predict": ("reward가 `[0,0]` 또는 `[1,1]`인 group은 dynamic sampling에서 남을까요?", "Do groups with rewards `[0,0]` or `[1,1]` survive dynamic sampling?"),
        "answer": ("아니요. group 내부 비교 신호가 없으므로 `[0,1]`, `[1,0]`만 남습니다.", "No. They contain no within-group comparison signal, so only `[0,1]` and `[1,0]` remain."),
        "why": ("각 변형을 독립 함수로 두면 어떤 안정화 요소가 결과를 바꿨는지 ablation할 수 있습니다. 논문 recipe를 이름 하나로 뭉치면 reduction과 mask 차이가 사라집니다.", "Separate functions make each stabilization component ablatable. Treating a paper recipe as one name hides reduction and masking differences."),
        "trap": ("overlong penalty의 buffer 경계를 off-by-one으로 구현하면 최대 길이 직전부터 갑자기 -1이 됩니다. 시작·중간·끝 네 지점을 analytic test로 고정합니다.", "An off-by-one overlong buffer can jump to -1 just before max length. Analytic tests pin the start, midpoint, and end boundaries."),
        "conclusion": ("dynamic sampling은 group 1과 3만 골랐고 overlong penalty는 0→-0.5→-1로 변했습니다. GSPO ratio는 token ratio가 아니라 sequence 집계값입니다.", "Dynamic sampling retained groups 1 and 3, and overlong penalty moved 0→-0.5→-1. The GSPO ratio is a sequence aggregate, not a token ratio."),
        "recall": ("DAPO의 네 요소 중 sample 효율을 직접 겨냥하는 요소는 무엇인가요?", "Which of DAPO's four components directly targets sample efficiency?"),
        "next": ("L14에서 같은 API를 공개 소형 모델과 GPU 서버로 옮길 때 다운로드·메모리·framework 경계를 확인합니다.", "L14 moves the same APIs toward public small models and GPU servers while auditing download, memory, and framework boundaries."),
    },
    "L14": {
        "equation": r"$$M_{train}\approx M_{weights}+M_{gradients}+M_{optimizer}+M_{activations}+M_{headroom}$$",
        "position": ("현재 위치: toy 알고리즘 → **실제 모델 profile** → server recipe", "Position: toy algorithms → **real-model profiles** → server recipes"),
        "explain": ("toy와 실제 공개 모델은 같은 알고리즘 API를 쓰지만 다운로드, tokenizer revision, dtype, adapter, device가 새 실패 경계가 됩니다. laptop preset은 LoRA와 보수적 headroom을 사용하고, server preset은 분산 framework 책임을 adapter 밖으로 분리합니다.", "Toy and public-model paths share algorithm APIs, but downloads, tokenizer revisions, dtype, adapters, and devices add failure boundaries. The laptop preset uses LoRA with conservative headroom; server presets isolate distributed-framework responsibilities behind adapters."),
        "predict": ("모델이 cache에 없고 승인 flag도 없을 때 optional import와 download 중 무엇보다 먼저 멈춰야 하나요?", "If the model is uncached and no approval flag is set, what must stop first: optional import or download?"),
        "answer": ("download guard가 optional framework import보다 먼저 멈춰야 환경 변경과 대용량 전송이 일어나지 않습니다.", "The download guard must stop before optional framework imports so no environment mutation or large transfer begins."),
        "why": ("정확한 peak memory는 hardware와 kernel에 따라 달라지므로 estimate와 measured 값을 구분합니다. QLoRA는 더 작은 memory 대안이지만 quantization backend 호환성이라는 새 경계를 만듭니다.", "Exact peak memory depends on hardware and kernels, so estimates stay separate from measurements. QLoRA lowers memory but adds quantization-backend compatibility boundaries."),
        "trap": ("`model_id`만 pin하고 revision을 비우면 같은 config가 다른 weights를 받을 수 있습니다. model·tokenizer revision과 예상 byte를 manifest에 함께 둡니다.", "Pinning only `model_id` allows the same config to fetch different weights. Store model/tokenizer revisions and expected bytes together."),
        "conclusion": ("laptop smoke preset은 약 269.1MB download와 1.59GiB 권장 memory를 보고했고, 승인 없는 download를 실제 전송 전에 차단했습니다.", "The laptop smoke preset reports about 269.1MB download and 1.59GiB recommended memory, and blocks an unapproved transfer before download."),
        "recall": ("memory estimate가 통과해도 실제 train 전 preflight가 다시 확인해야 할 세 조건은 무엇인가요?", "Which three conditions must preflight recheck even when the memory estimate passes?"),
        "next": ("L15에서 single response를 넘어 tool call과 여러 observation을 가진 trajectory를 학습합니다.", "L15 moves beyond single responses to trajectories with tool calls and multiple observations."),
    },
    "L15": {
        "equation": r"$$G_t=r_t^{process}+\sum_{k=t}^{T}\gamma^{k-t}r_k^{outcome}$$",
        "position": ("현재 위치: LLM policy → **Agentic RL multi-turn MDP** → trajectory 평가", "Position: LLM policy → **Agentic RL multi-turn MDP** → trajectory evaluation"),
        "explain": ("agent는 observation에서 `CALL` 또는 `FINAL` action을 만들고 tool output을 다음 observation으로 받습니다. policy가 생성한 action token만 loss 대상이며 tool output은 context일 뿐입니다. outcome reward broadcast와 discounted process return은 서로 다른 credit 가정입니다.", "An agent emits a `CALL` or `FINAL` action from an observation and receives tool output as the next observation. Only policy-generated action tokens enter the loss; tool output is context. Outcome broadcast and discounted process return make different credit assumptions."),
        "predict": ("tool output text를 다음 context에 넣을 때 그 token도 현재 policy loss mask에 포함할까요?", "When tool-output text enters the next context, should those tokens enter the current policy-loss mask?"),
        "answer": ("아니요. 환경이 만든 token을 policy action처럼 학습하면 gradient ownership이 깨집니다.", "No. Training environment-generated tokens as policy actions breaks gradient ownership."),
        "why": ("rollout 당시 token ID와 candidate set, policy version을 trajectory에 보존해 retokenization과 stale-policy drift를 탐지합니다. text만 저장하는 단순 log는 사람이 읽기 쉽지만 정확한 update 재현에는 부족합니다.", "The trajectory preserves rollout-time token IDs, candidate set, and policy version to detect retokenization and stale-policy drift. Text-only logs are readable but insufficient for exact updates."),
        "trap": ("outcome reward를 모든 step에 넣고 discounted return에서도 다시 더하면 reward를 이중 계산합니다. outcome은 종료 step에 한 번 저장하고 credit 함수가 배분합니다.", "Storing outcome reward at every step and adding it again in discounted returns double-counts it. Store outcome once at termination and let the credit function distribute it."),
        "conclusion": ("이번 seeded rollout은 한 step에서 종료되어 credit -0.25를 받았고 update loss는 유한했습니다. 실패 rollout도 mask와 credit 계약을 검증하는 유효한 증거입니다.", "This seeded rollout terminated after one step with credit -0.25 and a finite update loss. A failed rollout still provides valid evidence for mask and credit contracts."),
        "recall": ("process reward와 outcome reward를 같은 scalar로 미리 합치면 어떤 진단 능력을 잃나요?", "What diagnostic ability is lost when process and outcome rewards are pre-merged into one scalar?"),
        "next": ("L16에서 success 하나가 아니라 reward hacking, budget, split, initial hash까지 묶어 실험을 감사합니다.", "L16 audits reward hacking, budgets, splits, and initial hashes rather than looking at success alone."),
    },
    "L16": {
        "equation": r"$$\text{fair comparison}=\text{same init}+\text{same data}+\text{same budget}+\text{same metric contract}$$",
        "position": ("현재 위치: 모든 학습 경로 → **평가·실패 진단·재현 감사**", "Position: every training path → **evaluation, failure diagnosis, and reproducibility audit**"),
        "explain": ("좋은 score 하나는 reward hacking, length bias, entropy collapse를 숨길 수 있습니다. 공정 비교는 동일 initial hash, data order, token/forward/env budget, metric 정의를 요구합니다. local toy 결과와 논문 benchmark는 규모와 조건이 달라 같은 표에 순위처럼 놓지 않습니다.", "One good score can hide reward hacking, length bias, or entropy collapse. Fair comparisons require matching initial hashes, data order, token/forward/environment budgets, and metric definitions. Local toy results and paper benchmarks must not be ranked in one table as if conditions matched."),
        "predict": ("세 report가 모두 성공 파일이어도 `result_origin`이 빠지면 local 실행 증거로 받아들일 수 있을까요?", "Even if three report files look successful, can they count as local evidence without `result_origin`?"),
        "answer": ("아니요. origin·환경·config·budget이 없으면 실행 결과와 예시 데이터를 구분할 수 없습니다.", "No. Without origin, environment, config, and budget, executed results cannot be distinguished from example data."),
        "why": ("capstone은 새 대규모 train보다 기존 artifact의 계약을 기계적으로 읽습니다. 더 많은 seed는 통계 불확실성을 줄이지만 비교 조건 불일치를 고치지는 못합니다.", "The capstone machine-reads contracts from existing artifacts instead of starting another large run. More seeds reduce statistical uncertainty but do not repair mismatched comparison conditions."),
        "trap": ("서로 다른 token budget을 같은 `steps`로 비교하면 긴 response 알고리즘에 계산량 이점이 생깁니다. 여러 budget counter와 split hash를 같이 보고합니다.", "Comparing equal `steps` under different token budgets favors algorithms with longer responses. Report multiple budget counters and split hashes together."),
        "conclusion": ("C6·C7·C9 report 모두 `local_executed`와 해석 guardrail을 가졌습니다. C6은 report 내부 `sources` 배열이 없어 source traceability 보강이 필요하다는 gap도 드러났습니다.", "C6, C7, and C9 reports all declare `local_executed` and include interpretation guardrails. The audit also exposes a gap: C6 lacks an embedded `sources` array and needs stronger source traceability."),
        "recall": ("한 알고리즘의 exact-match가 높고 entropy가 0에 가까우면 어떤 두 해석을 추가로 구분해야 하나요?", "If exact match is high and entropy is near zero, which two interpretations must you distinguish next?"),
        "next": ("이제 README의 빠른/전체 경로로 돌아가 약한 영역을 복습하고, reproducibility checklist로 자신의 실험을 설계합니다.", "Return to the README fast/full routes, revisit weak areas, and design your own experiment with the reproducibility checklist."),
    },
}


def _hash_code(source: str) -> str:
    normalized = "\n".join(line.rstrip() for line in source.splitlines()).rstrip() + "\n"
    return "sha256:" + hashlib.sha256(normalized.encode()).hexdigest()


def _cell_metadata(
    stable_id: str,
    *,
    kind: str,
    concepts: list[str],
    sources: list[str],
    tests: list[str] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stable_id": stable_id,
        "kind": kind,
        "path": "core",
        "concept_ids": concepts,
        "source_ids": sources,
        "test_ids": tests or [],
    }
    if code is not None:
        payload["code_hash"] = _hash_code(code)
    return {"rl_study": payload, "tags": ["rl-study-core"]}


def _setup_code(lesson_id: str) -> str:
    return f'''import hashlib, json, os, platform, random, sys
from pathlib import Path
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file()), None)
if ROOT is None:
    raise RuntimeError("Run this notebook inside the RL-study repository")
sys.path.insert(0, str(ROOT / "src"))
import torch
from rl_study import __version__
from rl_study.data import build_tiny_reasoning
from rl_study.runtime import resolve_device, seed_everything
# These notebooks use only deterministic CPU toy kernels.  PyTorch 2.13's global
# guard imports the full Inductor stack, so keep the package's strict default for
# trainers while avoiding that unrelated startup cost in fresh teaching kernels.
seed_everything(42, deterministic=False)
random.seed(42)
language = os.environ.get("RL_STUDY_NOTEBOOK_LANGUAGE", "ko")
resolution = resolve_device("cpu")
dataset = build_tiny_reasoning(seed=42)
config_hash = "sha256:" + hashlib.sha256(b"{lesson_id}:toy:42").hexdigest()
print(f"lesson={lesson_id} language={{language}} profile=toy")
print("seed=42 network_required=False deterministic_scope=seeded_cpu_toy")
print(f"python={{platform.python_version()}} rl_study={{__version__}} torch={{torch.__version__}}")
print(f"requested_device=cpu resolved_device={{resolution.resolved}} fallback_used={{resolution.fallback_used}}")
print(f"config_hash={{config_hash}} data_split_hash={{dataset.split_hash}}")'''


def _markdown_text(spec: dict[str, Any], language: str) -> dict[str, str]:
    ko = language == "ko"
    objectives = spec["objectives_ko" if ko else "objectives_en"]
    title = spec["title_ko" if ko else "title_en"]
    objective_lines = "\n".join(f"- {item}" for item in objectives)
    tests = ", ".join(f"`{item}`" for item in TEST_IDS[spec["id"]])
    guide = LESSON_GUIDE[spec["id"]]

    def localized(key: str) -> str:
        value = guide[key]
        if not isinstance(value, tuple):
            raise TypeError(f"{spec['id']} guide field {key} must be bilingual")
        return value[0 if ko else 1]

    equation = guide["equation"]
    if not isinstance(equation, str):
        raise TypeError(f"{spec['id']} equation must be shared text")
    return {
        "title": f"# {spec['id']} · {title}",
        "goal": "## Goal\n\n" + objective_lines,
        "setup": "## Setup\n\n" + (
            "이 cell은 CPU·seed·offline 상태와 split hash를 먼저 고정합니다. "
            "toy 연산은 결정론적인 CPU 연산만 쓰며, package trainer의 전역 결정론 기본값은 유지합니다."
            if ko
            else "This cell fixes CPU, seed, offline status, and the split hash first. "
            "Toy code uses deterministic CPU operations; package trainers retain their strict global default."
        ),
        "steps": "## Steps",
        "orientation": (
            "### 1. 현재 위치와 핵심 식\n\n⏱ 5분 · 1/3 section · [필수/CORE]\n\n"
            f"{localized('position')}\n\n{equation}\n\n{localized('explain')}"
            if ko
            else "### 1. Position and core equation\n\n⏱ 5 min · 1/3 section · [CORE]\n\n"
            f"{localized('position')}\n\n{equation}\n\n{localized('explain')}"
        ),
        "concept": (
            "### 2. 작은 숫자로 실행\n\n⏱ 6분 · 2/3 section · [필수/CORE]\n\n"
            f"**먼저 예측:** {localized('predict')} 20초 동안 답을 적은 뒤 실행하세요.\n\n"
            f"<details><summary>정답 보기</summary>{localized('answer')}</details>"
            if ko
            else "### 2. Run with small numbers\n\n⏱ 6 min · 2/3 section · [CORE]\n\n"
            f"**Predict first:** {localized('predict')} Write an answer for 20 seconds, then run the cell.\n\n"
            f"<details><summary>Show answer</summary>{localized('answer')}</details>"
        ),
        "why": (
            "### 3. 구현 해부\n\n⏱ 6분 · 3/3 section · [심화/DEEP DIVE]\n\n"
            f"**왜 이렇게 구현했나:** {localized('why')}\n\n"
            f"**흔한 함정:** {localized('trap')} 회귀 test: {tests}.\n\n"
            "**쉬어가기:** 지금 출력한 한 값만 설명할 수 있으면 다음 cell로 가세요."
            if ko
            else "### 3. Implementation anatomy\n\n⏱ 6 min · 3/3 section · [DEEP DIVE]\n\n"
            f"**Why this implementation:** {localized('why')}\n\n"
            f"**Common trap:** {localized('trap')} Regression tests: {tests}.\n\n"
            "**Checkpoint:** Continue when you can explain just one printed value."
        ),
        "checks": "## Checks",
        "recall": (
            f"**회상 문제:** {localized('recall')} 1~2문장으로 답하세요."
            if ko
            else f"**Recall:** {localized('recall')} Answer in one or two sentences."
        ),
        "mistakes": (
            "## 내가 자주 틀리는 것\n\n- loss가 유한하면 구현도 맞다고 생각한다.\n- `terminated`와 `truncated`, prompt와 action을 합친다.\n- 한 seed의 작은 결과를 알고리즘 순위로 확대한다."
            if ko
            else "## Mistakes I Revisit\n\n- Assuming a finite loss proves the implementation is correct.\n- Merging `terminated` with `truncated`, or prompt with action.\n- Turning one tiny seed into an algorithm ranking."
        ),
        "recap": (
            f"## 60초 요약\n\n- **실행 결론:** {localized('conclusion')}\n- 실제 확인: {tests}.\n- 출력은 고정 seed의 toy 실행이며 논문 규모 결과가 아닙니다."
            if ko
            else f"## 60-Second Recap\n\n- **Run conclusion:** {localized('conclusion')}\n- Executable checks: {tests}.\n- The output is a fixed-seed toy run, not a paper-scale result."
        ),
        "next": (
            f"## Next Steps\n\n1. {localized('next')}\n2. `[필수/CORE]` assertion을 한 번 깨뜨리고 오류를 읽습니다.\n3. package test를 열어 notebook의 작은 식과 production guard를 연결합니다."
            if ko
            else f"## Next Steps\n\n1. {localized('next')}\n2. Break one `[CORE]` assertion and read the failure.\n3. Open the package test and connect the notebook equation to its production guard."
        ),
        "sources": "## Sources\n\n" + "\n".join(
            f"- `{source_id}` — `docs/sources.yml`" for source_id in spec["sources"]
        ),
    }


def _make_markdown(
    source: str, cell_id: str, stable_id: str, spec: dict[str, Any]
) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source=source, id=cell_id)
    cell.metadata = _cell_metadata(
        stable_id,
        kind="markdown",
        concepts=[spec["title_key"]],
        sources=list(spec["sources"]),
    )
    return cell


def _make_code(
    source: str, cell_id: str, stable_id: str, spec: dict[str, Any]
) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(source=source, id=cell_id)
    cell.metadata = _cell_metadata(
        stable_id,
        kind="code",
        concepts=[spec["title_key"]],
        sources=list(spec["sources"]),
        tests=TEST_IDS[spec["id"]],
        code=source,
    )
    return cell


def generate(spec: dict[str, Any], language: str) -> Path:
    lesson = spec["id"]
    slug = lesson.lower()
    text = _markdown_text(spec, language)
    setup = _setup_code(lesson)
    demo = DEMO_CODE[lesson]
    check = CHECK_CODE[lesson] + '\nprint("checks=passed")'
    cells = [
        _make_markdown(text["title"], f"{slug}-title", f"{lesson}.S00.C01", spec),
        _make_markdown(text["goal"], f"{slug}-goal", f"{lesson}.S01.C01", spec),
        _make_markdown(text["setup"], f"{slug}-setup", f"{lesson}.S02.C01", spec),
        _make_code(setup, f"{slug}-setup-code", f"{lesson}.S02.C02", spec),
        _make_markdown(text["steps"], f"{slug}-steps", f"{lesson}.S03.C01", spec),
        _make_markdown(text["orientation"], f"{slug}-orientation", f"{lesson}.S03.C02", spec),
        _make_markdown(text["concept"], f"{slug}-predict", f"{lesson}.S03.C03", spec),
        _make_code(demo, f"{slug}-demo", f"{lesson}.S03.C04", spec),
        _make_markdown(text["why"], f"{slug}-why", f"{lesson}.S03.C05", spec),
        _make_markdown(text["checks"], f"{slug}-checks", f"{lesson}.S04.C01", spec),
        _make_code(check, f"{slug}-check-code", f"{lesson}.S04.C02", spec),
        _make_markdown(text["recall"], f"{slug}-recall", f"{lesson}.S04.C03", spec),
        _make_markdown(text["mistakes"], f"{slug}-mistakes", f"{lesson}.S05.C01", spec),
        _make_markdown(text["recap"], f"{slug}-recap", f"{lesson}.S06.C01", spec),
        _make_markdown(text["next"], f"{slug}-next", f"{lesson}.S07.C01", spec),
        _make_markdown(text["sources"], f"{slug}-sources", f"{lesson}.S08.C01", spec),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (RL-study)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "rl_study": {
            "schema_version": 1,
            "lesson_id": lesson,
            "language": language,
            "mirror_language": "en" if language == "ko" else "ko",
            "title_key": spec["title_key"],
            "profile": "toy",
            "estimated_minutes_full": spec["full"],
            "estimated_minutes_fast": spec["fast"],
            "prerequisites": spec["prerequisites"],
            "source_ids": spec["sources"],
            "network_required": False,
            "seed": 42,
            "generated_from": "lessons/catalog.yml",
        },
    }
    target = ROOT / "notebooks" / language / f"{lesson}_{spec['title_key']}.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("ko", "en", "both"), default="ko")
    parser.add_argument("--lesson", choices=tuple(TEST_IDS))
    args = parser.parse_args()
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    languages = ("ko", "en") if args.language == "both" else (args.language,)
    generated = [
        generate(spec, language)
        for language in languages
        for spec in catalog["lessons"]
        if args.lesson is None or spec["id"] == args.lesson
    ]
    print(f"generated={len(generated)}")


if __name__ == "__main__":
    main()
