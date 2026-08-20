"""Offline comparison demo with honest, accessible, reproducible artifacts."""

from __future__ import annotations

import html
import json
import os
import platform
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rl_study.algorithms.rlhf_ppo import collect_rollouts
from rl_study.config import ExperimentConfig
from rl_study.data import build_tiny_reasoning, verifier_reward
from rl_study.models import TinyCausalLM, TinyTokenizer
from rl_study.models.sequence import response_token_log_probs
from rl_study.platform_metrics import peak_memory_bytes
from rl_study.training.agentic_runner import train_agentic
from rl_study.training.alignment_runner import (
    AlignmentRunResult,
    DPOState,
    RLHFState,
    train_alignment,
)
from rl_study.training.group_runner import GroupRunResult, train_group

DEMO_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DemoArtifacts:
    """Paths and machine-readable summary produced by one demo execution."""

    run_directory: Path
    summary_json: Path
    figure_png: Path
    report_html: Path
    interactive_html: Path
    checkpoints: tuple[Path, ...]
    experiment_cards: tuple[Path, ...]
    summary: dict[str, object]


def _atomic_text(path: Path, value: str) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _base_mapping(algorithm: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": "toy",
        "algorithm": {
            "name": algorithm,
            "variant": "demo-one-step",
            "reward_source": "verifier",
            "update_epochs": 1,
        },
        "data": {
            "id": "tiny_reasoning",
            "revision": "generated-v1",
            "split": "train",
            "seed": 42,
        },
        "model": {
            "policy": "tiny-v1",
            "reference": "tiny-v1-frozen",
            "reward": "deterministic-verifier",
        },
        "training": {
            "seed": 42,
            "steps": 1,
            "batch_size": 4,
            "group_size": 4,
            "response_token_budget": 8_192,
            "max_new_tokens": 22,
            "device": "cpu",
            "allow_device_fallback": False,
        },
        "evaluation": {"every_steps": 1, "split": "validation"},
        "output": {"root": "artifacts"},
    }


def _demo_config(algorithm: str) -> ExperimentConfig:
    mapping = _base_mapping(algorithm)
    algorithm_mapping = cast(dict[str, object], mapping["algorithm"])
    training_mapping = cast(dict[str, object], mapping["training"])
    if algorithm == "dpo":
        algorithm_mapping.update({"beta": 0.1, "label_smoothing": 0.0})
    elif algorithm == "rlhf_ppo":
        algorithm_mapping.update(
            {"kl_coefficient": 0.02, "clip_low": 0.2, "clip_high": 0.2}
        )
    elif algorithm == "grpo":
        algorithm_mapping.update(
            {
                "variant": "deepseekmath-paper-demo",
                "kl_coefficient": 0.04,
                "clip_low": 0.2,
                "clip_high": 0.2,
            }
        )
        training_mapping["batch_size"] = 1
    elif algorithm == "dapo":
        algorithm_mapping.update(
            {
                "variant": "paper-four-components-demo",
                "clip_low": 0.2,
                "clip_high": 0.28,
                "clip_higher": True,
                "dynamic_sampling": True,
                "dynamic_sampling_multiplier": 4,
                "token_level_loss": True,
                "overlong_reward_shaping": True,
                "overlong_buffer_length": 4,
                "overlong_penalty_scale": 1.0,
            }
        )
        training_mapping["batch_size"] = 1
    elif algorithm == "agentic_reinforce":
        algorithm_mapping.update(
            {
                "variant": "discounted-step-return-demo",
                "credit_assignment": "discounted_returns",
                "gamma": 0.95,
            }
        )
        data_mapping = cast(dict[str, object], mapping["data"])
        data_mapping.update(
            {
                "id": "agentic_offline",
                "revision": "generated-v1",
                "subset": "mixed",
            }
        )
        cast(dict[str, object], mapping["model"]).update(
            {"policy": "tiny-agent-v1", "reference": None, "reward": None}
        )
        training_mapping.update(
            {
                "batch_size": 1,
                "group_size": 2,
                "max_new_tokens": 48,
                "max_sequence_length": 192,
                "learning_rate": 0.001,
                "response_token_budget": 4_096,
            }
        )
    else:
        raise ValueError(f"unsupported demo algorithm: {algorithm}")
    return ExperimentConfig.from_mapping(mapping)


@torch.no_grad()
def _fixed_prompt_diagnostic(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    *,
    clip_low: float,
    clip_high: float,
) -> dict[str, object]:
    dataset = build_tiny_reasoning(seed=42)
    examples = dataset.validation[:4]
    tokenizer = TinyTokenizer()
    before = collect_rollouts(
        reference,
        examples,
        tokenizer=tokenizer,
        seed=91_001,
        do_sample=False,
    )
    after = collect_rollouts(
        policy,
        examples,
        tokenizer=tokenizer,
        seed=91_001,
        do_sample=False,
    )
    current_log_probs = response_token_log_probs(policy, after.batch)
    reference_log_probs = response_token_log_probs(reference, after.batch)
    mask = after.batch.action_mask
    valid_count = mask.sum()
    sampled_kl = ((current_log_probs - reference_log_probs) * mask).sum() / valid_count
    logits = policy(after.batch.input_ids, after.batch.attention_mask).logits[:, :-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    entropy = (token_entropy * mask).sum() / valid_count
    ratio = (current_log_probs - reference_log_probs).exp()
    clipped = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    clip_fraction = (clipped.to(torch.float32) * mask).sum() / valid_count
    rewards = [
        verifier_reward(example, response)
        for example, response in zip(examples, after.responses, strict=True)
    ]
    response_pairs = [
        {
            "prompt_id": example.uid,
            "prompt": example.prompt,
            "expected": example.target_response,
            "before": before_response,
            "after": after_response,
        }
        for example, before_response, after_response in zip(
            examples, before.responses, after.responses, strict=True
        )
    ]
    return {
        "scope": (
            "fixed_prompt_greedy_diagnostic; ratios use the frozen initial SFT "
            "policy as reference; not a training-time aggregate"
        ),
        "prompt_ids": [example.uid for example in examples],
        "reward": sum(rewards) / len(rewards),
        "sampled_kl": float(sampled_kl),
        "entropy": float(entropy),
        "clip_fraction": float(clip_fraction),
        "responses": response_pairs,
    }


def _alignment_record(
    algorithm: str,
    result: AlignmentRunResult,
    config: ExperimentConfig,
) -> tuple[dict[str, object], Path, Path]:
    if algorithm == "dpo":
        dpo_state = cast(DPOState, result.model)
        policy = dpo_state.policy
        reference = dpo_state.reference
    else:
        rlhf_state = cast(RLHFState, result.model)
        policy = rlhf_state.policy
        reference = rlhf_state.reference
    diagnostic = _fixed_prompt_diagnostic(
        policy,
        reference,
        clip_low=config.algorithm.clip_low,
        clip_high=config.algorithm.clip_high,
    )
    return (
        {
            "algorithm": algorithm,
            "family": "preference" if algorithm == "dpo" else "online-rlhf",
            "variant": config.algorithm.variant,
            "step": result.step,
            "training_metrics": result.metrics,
            "diagnostic": diagnostic,
            "checkpoint": str(result.checkpoint),
            "experiment_card": str(result.experiment_card),
        },
        result.checkpoint,
        result.experiment_card,
    )


def _group_record(
    algorithm: str,
    result: GroupRunResult,
    config: ExperimentConfig,
) -> tuple[dict[str, object], Path, Path]:
    diagnostic = _fixed_prompt_diagnostic(
        result.model.policy,
        result.model.reference,
        clip_low=config.algorithm.clip_low,
        clip_high=config.algorithm.clip_high,
    )
    return (
        {
            "algorithm": algorithm,
            "family": "group-policy",
            "variant": config.algorithm.variant,
            "step": result.step,
            "training_metrics": result.metrics,
            "diagnostic": diagnostic,
            "checkpoint": str(result.checkpoint),
            "experiment_card": str(result.experiment_card),
        },
        result.checkpoint,
        result.experiment_card,
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a numeric report value")
    return float(value)


def _figure(path: Path, records: list[dict[str, object]]) -> None:
    comparable = [record for record in records if record["family"] != "agentic"]
    labels = [
        str(record["algorithm"]).upper().replace("_", "-")
        for record in comparable
    ]
    diagnostics = [
        cast(Mapping[str, object], record["diagnostic"])
        for record in comparable
    ]
    panels = (
        ("reward", "Verifier reward", "0 to 1"),
        ("sampled_kl", "Sampled KL", "policy - reference log-prob"),
        ("entropy", "Token entropy", "nats"),
        ("clip_fraction", "Clip fraction", "fraction of action tokens"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    figure.subplots_adjust(top=0.88, bottom=0.08, hspace=0.38, wspace=0.18)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    hatches = ["//", "xx", "..", "\\\\"]
    for axis, (metric, title, unit) in zip(axes.flat, panels, strict=True):
        values = [_number(diagnostic[metric]) for diagnostic in diagnostics]
        bars = axis.bar(
            labels,
            values,
            color=colors,
            edgecolor="#111827",
            hatch=hatches,
        )
        axis.set_title(title, pad=10)
        axis.set_ylabel(unit)
        axis.axhline(0.0, color="#374151", linewidth=0.8)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.margins(y=0.18)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    figure.suptitle(
        "One-update fixed-prompt diagnostics (same TinyReasoning prompts)",
        fontsize=15,
        y=0.965,
    )
    figure.savefig(path, dpi=150, metadata={"Software": "RL-study"})
    plt.close(figure)


def _metric(record: Mapping[str, object], name: str) -> str:
    value = record.get("diagnostic")
    if not isinstance(value, Mapping):
        return "—"
    metric_value = value.get(name)
    if metric_value is None:
        return "—"
    return f"{_number(metric_value):.4f}"


_CSS = """
:root{color-scheme:light dark;--bg:#f8fafc;--ink:#172033;--card:#fff;--accent:#275dad;--line:#cbd5e1}
@media(prefers-color-scheme:dark){:root{--bg:#101827;--ink:#e5edf8;--card:#182338;--accent:#8dbbff;--line:#52647e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,sans-serif}
main{max-width:1080px;margin:auto;padding:2.5rem 1.25rem 5rem}.eyebrow{font-weight:750;letter-spacing:.08em;color:var(--accent)}
h1{font-size:clamp(2rem,5vw,3.7rem);line-height:1.05}h2{margin-top:2.5rem}a{color:var(--accent)}
img{display:block;max-width:100%;height:auto;background:white;border:1px solid var(--line);border-radius:.8rem}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card)}caption{text-align:left;font-weight:700;padding:.75rem 0}
th,td{border:1px solid var(--line);padding:.7rem;text-align:left}label{font-weight:700}select{font:inherit;margin:.4rem 1rem .4rem .3rem;padding:.35rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.card{background:var(--card);border:1px solid var(--line);border-radius:.8rem;padding:1rem}
.metric{font-variant-numeric:tabular-nums}.response{white-space:pre-wrap;overflow-wrap:anywhere;border-left:.25rem solid var(--accent);padding-left:.7rem}
.sr-status{min-height:1.5rem}button{font:inherit;padding:.45rem .8rem}code{overflow-wrap:anywhere}
"""


def _static_report(summary: Mapping[str, object]) -> str:
    records = cast(list[dict[str, object]], summary["records"])
    rows = "\n".join(
        "<tr>"
        f"<th scope='row'>{html.escape(str(record['algorithm']))}</th>"
        f"<td>{html.escape(str(record['family']))}</td>"
        f"<td>{_metric(record, 'reward')}</td>"
        f"<td>{_metric(record, 'sampled_kl')}</td>"
        f"<td>{_metric(record, 'entropy')}</td>"
        f"<td>{_metric(record, 'clip_fraction')}</td>"
        "</tr>"
        for record in records
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL-study toy 비교 보고서</title><style>{_CSS}</style></head>
<body><main><p class="eyebrow">LOCAL EXECUTED · TOY · ONE UPDATE</p>
<h1>RL-study 비교 보고서</h1>
<p>이 보고서는 같은 TinyReasoning 고정 prompt에서 계산한 진단값을 보여줍니다. 한 번의 toy update는 알고리즘 순위를 입증하지 않습니다.</p>
<img src="comparison.png" alt="DPO, RLHF-PPO, GRPO, DAPO의 고정 prompt reward, sampled KL, token entropy, clip fraction을 나타낸 네 개의 막대그래프. 정확한 값은 아래 표에 있습니다.">
<div class="table-wrap"><table><caption>고정 prompt greedy diagnostic</caption><thead><tr><th>알고리즘</th><th>계열</th><th>reward</th><th>sampled KL</th><th>entropy</th><th>clip fraction</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><strong>해석:</strong> Agentic RL은 환경과 지표가 달라 이 그래프에서 제외했습니다. 값이 없는 항목은 0이 아니라 적용 불가입니다.</p>
<p><a href="compare.html">응답과 지표를 필터로 탐색하기</a> · <a href="summary.json">원본 JSON 보기</a></p>
</main></body></html>"""


def _interactive_report(summary: Mapping[str, object]) -> str:
    payload = json.dumps(summary, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL-study 대화형 비교</title><style>{_CSS}</style></head>
<body><main><p class="eyebrow">OFFLINE INTERACTIVE REPORT</p><h1>학습 전후를 직접 비교하세요</h1>
<p>모든 값은 이 실행에서 나온 값입니다. 필터는 결과를 다시 계산하지 않고 표시만 바꿉니다.</p>
<form id="filters"><label for="family">계열</label><select id="family"><option value="all">전체</option></select>
<label for="metric">강조 지표</label><select id="metric"><option value="reward">reward</option><option value="sampled_kl">sampled KL</option><option value="entropy">entropy</option><option value="clip_fraction">clip fraction</option></select>
<button type="reset">초기화</button></form><p id="status" class="sr-status" aria-live="polite"></p><section id="cards" class="cards" aria-label="알고리즘 비교 카드"></section>
<noscript><p>JavaScript가 꺼져 있어 필터를 쓸 수 없습니다. <a href="report.html">정적 표</a>에서 모든 값을 확인하세요.</p></noscript>
<p><a href="report.html">정적 보고서</a> · <a href="summary.json">원본 JSON</a></p>
<script id="data" type="application/json">{payload}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const family=document.getElementById('family');const metric=document.getElementById('metric');const cards=document.getElementById('cards');const status=document.getElementById('status');
[...new Set(data.records.map(r=>r.family))].sort().forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;family.append(o);}});
const val=(r,k)=>r.diagnostic&&r.diagnostic[k]!==null&&r.diagnostic[k]!==undefined?Number(r.diagnostic[k]).toFixed(4):'적용 불가';
function add(parent,tag,text,cls){{const node=document.createElement(tag);node.textContent=text;if(cls)node.className=cls;parent.append(node);return node;}}
function render(){{cards.replaceChildren();const rows=data.records.filter(r=>family.value==='all'||r.family===family.value);rows.forEach(r=>{{const card=document.createElement('article');card.className='card';add(card,'h2',r.algorithm);add(card,'p',`${{metric.options[metric.selectedIndex].text}}: ${{val(r,metric.value)}}`,'metric');add(card,'p',`계열: ${{r.family}} · 실제 update: ${{r.step}}`);if(r.diagnostic){{add(card,'p',r.diagnostic.scope);r.diagnostic.responses.slice(0,2).forEach((x,i)=>{{add(card,'h3',`Prompt ${{i+1}} · ${{x.prompt_id}}`);add(card,'p',x.prompt);add(card,'p',`학습 전: ${{x.before}}`,'response');add(card,'p',`학습 후: ${{x.after}}`,'response');}});}}else{{add(card,'p',r.comparison_note||'동일 prompt 응답 비교가 적용되지 않습니다.');}}cards.append(card);}});status.textContent=`${{rows.length}}개 알고리즘 표시 중`;}}
family.addEventListener('change',render);metric.addEventListener('change',render);document.getElementById('filters').addEventListener('reset',()=>setTimeout(render,0));render();
</script></main></body></html>"""


def run_demo(
    *, output_root: str | Path = "artifacts/demo", device: str = "cpu"
) -> DemoArtifacts:
    """Run five real one-step toy trainers and write a self-contained report set."""
    if device != "cpu":
        raise ValueError("the reproducible demo currently requires --device cpu")
    started = time.perf_counter()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_directory = Path(tempfile.mkdtemp(prefix="demo-", dir=root))
    training_root = run_directory / "checkpoints"
    records: list[dict[str, object]] = []
    checkpoints: list[Path] = []
    cards: list[Path] = []

    for algorithm in ("dpo", "rlhf_ppo"):
        config = _demo_config(algorithm)
        alignment_result = train_alignment(
            config, output_root=training_root, stop_after=1
        )
        record, checkpoint, card = _alignment_record(
            algorithm, alignment_result, config
        )
        records.append(record)
        checkpoints.append(checkpoint)
        cards.append(card)
    for algorithm in ("grpo", "dapo"):
        config = _demo_config(algorithm)
        group_result = train_group(config, output_root=training_root, stop_after=1)
        record, checkpoint, card = _group_record(algorithm, group_result, config)
        records.append(record)
        checkpoints.append(checkpoint)
        cards.append(card)

    agentic_config = _demo_config("agentic_reinforce")
    agentic = train_agentic(agentic_config, output_root=training_root, stop_after=1)
    records.append(
        {
            "algorithm": "agentic_reinforce",
            "family": "agentic",
            "variant": agentic_config.algorithm.variant,
            "step": agentic.step,
            "training_metrics": agentic.metrics,
            "diagnostic": None,
            "comparison_note": (
                "Agentic RL uses multi-turn tool trajectories, so token-response KL/"
                "clip diagnostics from TinyReasoning are not semantically comparable."
            ),
            "checkpoint": str(agentic.checkpoint),
            "experiment_card": str(agentic.experiment_card),
        }
    )
    checkpoints.append(agentic.checkpoint)
    cards.append(agentic.experiment_card)

    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "status": "completed",
        "result_origin": "local_executed",
        "profile": "toy",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "peak_memory_bytes": peak_memory_bytes(),
        },
        "timing": {
            "training_and_diagnostics_seconds": elapsed,
            "command_target_seconds": 600,
        },
        "experiment": {
            "seed": 42,
            "optimizer_updates_per_algorithm": 1,
            "diagnostic_prompt_count": 4,
            "stopping_rule": (
                "one optimizer update per algorithm; no metric early stop"
            ),
        },
        "records": records,
        "interpretation_guardrails": [
            "This toy run does not reproduce paper-scale results or rank algorithms.",
            (
                "Diagnostic reward/KL/entropy/clip values share prompts only across "
                "the four TinyReasoning policies."
            ),
            (
                "Agentic success and process reward use a different environment and "
                "are not plotted beside TinyReasoning reward."
            ),
            "Missing metrics mean not applicable, never zero.",
        ],
        "paper_reported": None,
        "upstream_reported": None,
    }
    summary_json = run_directory / "summary.json"
    figure_png = run_directory / "comparison.png"
    report_html = run_directory / "report.html"
    interactive_html = run_directory / "compare.html"
    _atomic_text(
        summary_json,
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )
    _figure(figure_png, records)
    _atomic_text(report_html, _static_report(summary))
    _atomic_text(interactive_html, _interactive_report(summary))
    return DemoArtifacts(
        run_directory=run_directory,
        summary_json=summary_json,
        figure_png=figure_png,
        report_html=report_html,
        interactive_html=interactive_html,
        checkpoints=tuple(checkpoints),
        experiment_cards=tuple(cards),
        summary=summary,
    )
