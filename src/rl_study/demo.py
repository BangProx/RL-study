"""Module entry point for ``python -m rl_study.demo``."""

from __future__ import annotations

import sys

from rl_study.cli import main as cli_main


def main() -> None:
    cli_main(["demo", *sys.argv[1:]])


if __name__ == "__main__":
    main()
