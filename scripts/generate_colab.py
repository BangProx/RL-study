#!/usr/bin/env python3
"""Generate the auditable free-Colab quickstart notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/colab/RL_study_quickstart.ipynb"


def _hash(source: str) -> str:
    normalized = "\n".join(line.rstrip() for line in source.splitlines()).rstrip() + "\n"
    return "sha256:" + hashlib.sha256(normalized.encode()).hexdigest()


def _markdown(source: str, cell_id: str, stable_id: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source=source, id=cell_id)
    cell.metadata = {
        "rl_study": {"stable_id": stable_id, "kind": "markdown"},
        "tags": ["rl-study-core"],
    }
    return cell


def _code(
    source: str,
    cell_id: str,
    stable_id: str,
    *,
    tags: list[str] | None = None,
) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(source=source, id=cell_id)
    cell.metadata = {
        "rl_study": {
            "stable_id": stable_id,
            "kind": "code",
            "code_hash": _hash(source),
        },
        "tags": tags or ["rl-study-core"],
    }
    return cell


def generate() -> Path:
    cells = [
        _markdown(
            """# RL-study · 무료 Colab quickstart

새 runtime에서 **clone → install → toy demo → 선택형 실제 모델 1-step smoke** 순서로 실행합니다.

- 기본 경로는 CPU에서도 동작하며 유료 accelerator를 요구하지 않습니다.
- 실제 모델 cell은 약 1.5GB weights와 추가 package를 내려받으므로 직접 switch를 켜야 합니다.
- 출력은 Colab runtime의 실행 증거이지 논문 규모 benchmark가 아닙니다.

> English: Run the cells top to bottom in a fresh runtime. The real-model cell is opt-in and may download about 1.5GB.""",
            "colab-title",
            "COLAB.S00.C01",
        ),
        _markdown(
            """## 0. Runtime 확인

먼저 날짜·Python·platform·accelerator를 기록합니다. 이 정보가 없으면 나중에 같은 결과를 재현할 수 없습니다.""",
            "colab-runtime-note",
            "COLAB.S01.C01",
        ),
        _code(
            """from datetime import datetime, timezone
import json, os, platform, subprocess, sys
from pathlib import Path

started_at = datetime.now(timezone.utc).isoformat()
has_cuda = Path("/usr/local/cuda").exists() or bool(os.environ.get("COLAB_GPU"))
runtime = {
    "started_at": started_at,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "has_cuda_hint": has_cuda,
    "network_required": True,
}
print(json.dumps(runtime, ensure_ascii=False, indent=2))""",
            "colab-runtime",
            "COLAB.S01.C02",
        ),
        _markdown(
            """## 1. Clone

고정 대상은 `BangProx/RL-study`입니다. `pyproject.toml`이 없는 빈 저장소나 잘못된 branch는 즉시 실패합니다.""",
            "colab-clone-note",
            "COLAB.S02.C01",
        ),
        _code(
            """REPO_URL = "https://github.com/BangProx/RL-study.git"
REPO_DIR = Path("/content/RL-study")
if not (REPO_DIR / ".git").is_dir():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if not (REPO_DIR / "pyproject.toml").is_file():
    raise RuntimeError("The cloned repository has no pyproject.toml; stop instead of installing an unknown tree")
os.chdir(REPO_DIR)
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip()
print({"repo": REPO_URL, "commit": commit, "cwd": str(Path.cwd())})""",
            "colab-clone",
            "COLAB.S02.C02",
            tags=["rl-study-core", "rl-study-network"],
        ),
        _markdown(
            """## 2. Install

기본 package만 editable install합니다. optional model framework는 마지막 opt-in cell 전까지 설치하지 않습니다.""",
            "colab-install-note",
            "COLAB.S03.C01",
        ),
        _code(
            """install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    check=True,
    capture_output=True,
    text=True,
)
print("install=passed")
print("\\n".join(install.stdout.splitlines()[-8:]))""",
            "colab-install",
            "COLAB.S03.C02",
            tags=["rl-study-core", "rl-study-network"],
        ),
        _markdown(
            """## 3. Offline toy demo

이 단계부터는 model download 없이 package/data/device smoke와 2-action policy-gradient update를 실행합니다. 전체 알고리즘 비교는 강좌 notebook과 `rl-study train` config에서 이어집니다. subprocess가 0이 아니면 성공으로 표시하지 않습니다.""",
            "colab-toy-note",
            "COLAB.S04.C01",
        ),
        _code(
            """toy = subprocess.run(
    [
        sys.executable, "-m", "rl_study.demo", "--profile", "toy",
        "--device", "cpu", "--non-interactive", "--json",
    ],
    check=True,
    capture_output=True,
    text=True,
)
print("toy_demo=passed")
print("\\n".join(toy.stdout.splitlines()[-20:]))
assert '"status": "foundation-smoke"' in toy.stdout

import torch

logits = torch.zeros(2, requires_grad=True)
optimizer = torch.optim.SGD([logits], lr=0.4)
p_good = []
for _ in range(20):
    probabilities = torch.softmax(logits, dim=-1)
    loss = -probabilities[1]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    p_good.append(float(probabilities[1].detach()))
print({"p_good_start": round(p_good[0], 3), "p_good_end": round(p_good[-1], 3)})
assert p_good[-1] > p_good[0]""",
            "colab-toy",
            "COLAB.S04.C02",
        ),
        _markdown(
            """## 4. 선택형 실제 모델 1-step smoke

아래 cell은 기본적으로 skip합니다. 무료 GPU runtime을 선택했고 약 1.5GB download를 수락할 때만 `RUN_REAL_MODEL_SMOKE=True`로 바꾸세요. GPU가 없으면 조용히 CPU로 fallback하지 않고 이유를 출력합니다.""",
            "colab-real-note",
            "COLAB.S05.C01",
        ),
        _code(
            """RUN_REAL_MODEL_SMOKE = False  # learner-controlled download approval
real_model_status = "skipped: opt-in switch is False"
if RUN_REAL_MODEL_SMOKE:
    import torch

    if not torch.cuda.is_available():
        real_model_status = "external-manual: CUDA accelerator is unavailable"
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[laptop]"],
            check=True,
        )
        subprocess.run(
            [
                str(Path(sys.executable).with_name("rl-study")), "train",
                "--config", "configs/laptop/qwen3_lora_sft.yaml",
                "--accept-download", "--json",
            ],
            check=True,
        )
        real_model_status = "passed: Qwen3-0.6B LoRA one-step"
print({"real_model_smoke": real_model_status})""",
            "colab-real-model",
            "COLAB.S05.C02",
            tags=["rl-study-network", "rl-study-server"],
        ),
        _markdown(
            """## 5. 실행 증거

마지막 JSON을 저장하면 실행 날짜, commit, runtime, toy 성공, 실제 모델 선택 여부를 함께 남길 수 있습니다.""",
            "colab-evidence-note",
            "COLAB.S06.C01",
        ),
        _code(
            """evidence = {
    **runtime,
    "repo": REPO_URL,
    "git_commit": commit,
    "toy_demo": "passed",
    "real_model_smoke": real_model_status,
}
print(json.dumps(evidence, ensure_ascii=False, indent=2))
assert evidence["toy_demo"] == 'passed'""",
            "colab-evidence",
            "COLAB.S06.C02",
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "colab": {"name": "RL-study quickstart", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "rl_study": {
            "schema_version": 1,
            "profile": "colab-free",
            "order": ["clone", "install", "toy", "real-model-optional"],
            "repo_url": "https://github.com/BangProx/RL-study.git",
            "network_required": True,
        },
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(generate().relative_to(ROOT))
