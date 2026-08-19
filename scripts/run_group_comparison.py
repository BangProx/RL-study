#!/usr/bin/env python3
"""Print the C7 group-policy comparison and DAPO ablations as JSON."""

from __future__ import annotations

import argparse
import json

from rl_study.evaluation import run_group_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_group_comparison(
        steps=args.steps,
        prompt_batch_size=args.prompt_batch_size,
        group_size=args.group_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
