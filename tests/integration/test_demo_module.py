from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def test_python_module_demo_contract(tmp_path: Path) -> None:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_study.demo",
            "--profile",
            "toy",
            "--non-interactive",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    process_wall_seconds = time.perf_counter() - started
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["result_origin"] == "local_executed"
    assert payload["algorithms"] == [
        "dpo",
        "rlhf_ppo",
        "grpo",
        "dapo",
        "agentic_reinforce",
    ]
    assert payload["wall_seconds"] < 600
    assert process_wall_seconds < 600
    assert payload["training_and_diagnostics_seconds"] > 0
    for key in ("summary_json", "figure_png", "report_html", "interactive_html"):
        assert Path(payload[key]).is_file()
    assert len(payload["checkpoints"]) == 5
    assert len(payload["experiment_cards"]) == 5

    summary = json.loads(Path(payload["summary_json"]).read_text(encoding="utf-8"))
    assert summary["paper_reported"] is None
    assert summary["upstream_reported"] is None
    assert len(summary["records"]) == 5
    assert all(Path(path).is_dir() for path in payload["checkpoints"])
    assert all(Path(path).is_file() for path in payload["experiment_cards"])
    assert Path(payload["figure_png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = Path(payload["report_html"]).read_text(encoding="utf-8")
    interactive = Path(payload["interactive_html"]).read_text(encoding="utf-8")
    assert 'alt="DPO, RLHF-PPO, GRPO, DAPO' in report
    assert "<caption>" in report
    assert 'aria-live="polite"' in interactive
    assert "textContent" in interactive
