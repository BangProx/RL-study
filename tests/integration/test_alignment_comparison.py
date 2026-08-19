from __future__ import annotations

import math

from rl_study.evaluation import run_alignment_comparison


def test_alignment_comparison_audits_shared_start_and_prompts() -> None:
    report = run_alignment_comparison(steps=1, batch_size=2, seed=42)
    fairness = report["fairness_contract"]
    assert isinstance(fairness, dict)
    initial = fairness["initial_policy_sha256"]
    hashes = fairness["per_algorithm_initial_hashes"]
    assert isinstance(hashes, dict)
    assert set(hashes.values()) == {initial}
    assert fairness["prompt_occurrences"] == 2

    policies = report["policies"]
    assert isinstance(policies, dict)
    for policy in policies.values():
        assert isinstance(policy, dict)
        metrics = policy["metrics"]
        assert isinstance(metrics, dict)
        assert all(math.isfinite(value) for value in metrics.values())
