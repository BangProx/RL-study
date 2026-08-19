#!/usr/bin/env python3
"""Print the C6 fair SFT/RM/DPO/RLHF comparison as JSON."""

from __future__ import annotations

import argparse
import json

from rl_study.evaluation import run_alignment_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_alignment_comparison(
        steps=args.steps, batch_size=args.batch_size, seed=args.seed
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
