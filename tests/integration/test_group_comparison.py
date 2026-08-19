from __future__ import annotations

from rl_study.evaluation import run_group_comparison


def test_group_comparison_has_shared_start_and_four_dapo_ablation_axes() -> None:
    report = run_group_comparison(steps=1, prompt_batch_size=1, group_size=2, seed=42)
    fairness = report["fairness_contract"]
    assert isinstance(fairness, dict)
    initial_hash = fairness["initial_policy_sha256"]
    hashes = fairness["per_variant_initial_hashes"]
    assert isinstance(hashes, dict)
    assert set(hashes.values()) == {initial_hash}

    variants = report["variants"]
    assert isinstance(variants, dict)
    expected = {
        "dapo_clip_higher_only",
        "dapo_dynamic_sampling_only",
        "dapo_token_loss_only",
        "dapo_overlong_only",
        "dapo_all_four",
    }
    assert expected <= set(variants)
    grpo = variants["grpo_paper"]
    assert isinstance(grpo, dict)
    if grpo["mean_informative_group_rate"] == 0.0:
        assert grpo["final_policy_sha256"] == initial_hash
