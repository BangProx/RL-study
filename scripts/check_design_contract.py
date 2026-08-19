"""C2 설계 문서의 기계 검증.

외부 package 없이 lesson 수, 완료 조건 matrix와 source reference를 확인한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    sources = json.loads(read("docs/sources.yml"))
    source_ids = {
        item["id"]
        for section in ("papers", "repositories", "models", "datasets")
        for item in sources[section]
    }

    curriculum = read("docs/design/curriculum.md")
    lessons = re.findall(r"^### L(\d{2}) —", curriculum, flags=re.MULTILINE)
    expected_lessons = [f"{index:02d}" for index in range(17)]
    assert lessons == expected_lessons, (lessons, expected_lessons)

    referenced_sources = set(re.findall(r"`([a-z0-9][a-z0-9.-]+)`", curriculum))
    source_like = {
        value
        for value in referenced_sources
        if value.startswith(
            (
                "sutton-",
                "dqn-",
                "gae-",
                "ppo-",
                "learning-",
                "instructgpt-",
                "dpo-",
                "deepseekmath-",
                "rloo-",
                "dapo-",
                "dr-grpo-",
                "gspo-",
                "agent-",
                "repo-",
                "framework-",
                "benchmark-",
            )
        )
    }
    unknown_sources = source_like - source_ids
    assert not unknown_sources, f"unknown source IDs: {sorted(unknown_sources)}"

    traceability = read("docs/design/traceability.md")
    requirement_rows = re.findall(r"^\| R(\d{2}) \|", traceability, flags=re.MULTILINE)
    expected_requirements = [f"{index:02d}" for index in range(1, 21)]
    assert requirement_rows == expected_requirements, (
        requirement_rows,
        expected_requirements,
    )

    notebook_style = read("docs/design/notebook-style.md")
    for section in (
        "## Goal",
        "## Setup",
        "## Steps",
        "## Checks",
        "## 내가 자주 틀리는 것",
        "## 60초 요약",
        "## Next Steps",
        "## Sources",
    ):
        assert section in notebook_style, section

    benchmark = json.loads(read("docs/research/C2_TOY_BENCHMARK.json"))
    selected_name = benchmark["selection"]["tiny-v1"]
    selected = next(
        result
        for result in benchmark["results"]
        if result["candidate"]["name"] == selected_name
    )
    assert selected["parameters"] == 242976
    assert selected["median_train_step_ms"] < 50

    print("C2 design contract: PASS")
    print(
        f"lessons={len(lessons)}, requirements={len(requirement_rows)}, "
        f"sources={len(source_ids)}, tiny_v1_params={selected['parameters']}"
    )


if __name__ == "__main__":
    main()
