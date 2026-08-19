"""Command-line entry points with stable, user-actionable exit codes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rl_study import __version__
from rl_study.adapters import (
    build_profile_preflight,
    render_verl_recipe,
    validate_verl_recipe,
)
from rl_study.algorithms.group_policy import GROUP_ALGORITHMS
from rl_study.config import ExperimentConfig
from rl_study.errors import (
    CheckpointError,
    ConfigError,
    DownloadApprovalRequired,
    NumericError,
    PreflightError,
    RLStudyError,
)
from rl_study.runtime import resolve_device, seed_everything
from rl_study.training.agentic_runner import (
    AGENTIC_ALGORITHMS,
    evaluate_agentic_checkpoint,
    train_agentic,
)
from rl_study.training.alignment_runner import (
    ALIGNMENT_ALGORITHMS,
    evaluate_alignment_checkpoint,
    train_alignment,
)
from rl_study.training.classic_runner import (
    CLASSIC_ALGORITHMS,
    evaluate_classic_checkpoint,
    train_classic,
)
from rl_study.training.group_runner import evaluate_group_checkpoint, train_group
from rl_study.training.laptop_runner import (
    evaluate_laptop_checkpoint,
    train_laptop_sft,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-study")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run the current offline toy demonstration"
    )
    demo.add_argument("--profile", choices=("toy",), default="toy")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--non-interactive", action="store_true")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--json", action="store_true")

    train = subparsers.add_parser("train", help="train from a strict experiment config")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--output-root", type=Path)
    train.add_argument("--cache-dir", type=Path)
    train.add_argument("--accept-download", action="store_true")
    train.add_argument("--json", action="store_true")

    evaluate = subparsers.add_parser("eval", help="evaluate a repository checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--json", action="store_true")
    evaluate.add_argument("--cache-dir", type=Path)

    inspect = subparsers.add_parser(
        "inspect-run", help="inspect an experiment artifact"
    )
    inspect.add_argument("artifact_directory", type=Path)
    inspect.add_argument("--json", action="store_true")

    preflight = subparsers.add_parser("preflight", help="validate a profile and device")
    preflight.add_argument(
        "--profile", choices=("toy", "laptop", "server"), required=True
    )
    preflight.add_argument("--device", default="auto")
    preflight.add_argument("--model", default="laptop-smoke")
    preflight.add_argument("--revision")
    preflight.add_argument("--license-id")
    preflight.add_argument("--expected-weight-bytes", type=int)
    preflight.add_argument("--adapter", choices=("lora", "qlora"), default="lora")
    preflight.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    preflight.add_argument("--batch-size", type=int, default=1)
    preflight.add_argument("--sequence-length", type=int, default=128)
    preflight.add_argument("--cache-dir", type=Path)
    preflight.add_argument("--allow-device-fallback", action="store_true")
    preflight.add_argument("--json", action="store_true")

    render_server = subparsers.add_parser(
        "render-server", help="render a pinned unexecuted verl recipe"
    )
    render_server.add_argument("--config", type=Path, required=True)
    render_server.add_argument("--json", action="store_true")
    return parser


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}={value}")


def _demo(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    from rl_study.reporting.demo import run_demo

    del args.non_interactive
    seed_everything(42)
    resolution = resolve_device(args.device)
    if str(resolution.resolved) != "cpu":
        raise ConfigError("the reproducible demo currently requires --device cpu")
    artifacts = run_demo(output_root=args.output_dir, device="cpu")
    summary = artifacts.summary
    payload: dict[str, object] = {
        "status": "completed",
        "result_origin": "local_executed",
        "profile": "toy",
        "device": str(resolution.resolved),
        "fallback_used": resolution.fallback_used,
        "run_directory": str(artifacts.run_directory),
        "summary_json": str(artifacts.summary_json),
        "figure_png": str(artifacts.figure_png),
        "report_html": str(artifacts.report_html),
        "interactive_html": str(artifacts.interactive_html),
        "checkpoints": [str(path) for path in artifacts.checkpoints],
        "experiment_cards": [str(path) for path in artifacts.experiment_cards],
        "algorithms": [
            str(record["algorithm"])
            for record in cast(list[dict[str, object]], summary["records"])
        ],
        "wall_seconds": time.perf_counter() - started,
        "training_and_diagnostics_seconds": cast(
            dict[str, float], summary["timing"]
        )["training_and_diagnostics_seconds"],
        "interpretation": "toy diagnostics; not a paper-scale ranking",
    }
    _emit(payload, as_json=args.json)
    return 0


def _train(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    resolution = resolve_device(
        config.training.device,
        allow_fallback=config.training.allow_device_fallback,
    )
    payload: dict[str, object] = {
        "status": "dry-run" if args.dry_run else "not-started",
        "config_hash": config.sha256,
        "profile": config.profile,
        "algorithm": config.algorithm.name,
        "device": str(resolution.resolved),
        "fallback_used": resolution.fallback_used,
    }
    if args.dry_run:
        _emit(payload, as_json=args.json)
        return 0
    if config.profile == "laptop":
        laptop_result = train_laptop_sft(
            config,
            accept_download=args.accept_download,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            output_root=args.output_root,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        payload.update(
            {
                "status": "completed",
                "step": laptop_result.step,
                "checkpoint": str(laptop_result.checkpoint),
                "experiment_card": str(laptop_result.experiment_card),
                "metrics": laptop_result.metrics,
                "trainable_parameters": laptop_result.trainable_parameters,
                "total_parameters": laptop_result.total_parameters,
            }
        )
    elif config.profile == "server":
        raise PreflightError(
            "server training is external-manual; validate with render-server on "
            "a machine that has the pinned verl stack"
        )
    elif config.algorithm.name in CLASSIC_ALGORITHMS:
        classic_result = train_classic(
            config,
            output_root=args.output_root,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        payload.update(
            {
                "status": "completed",
                "step": classic_result.step,
                "success_rate": classic_result.success_rate,
                "checkpoint": str(classic_result.checkpoint),
                "updates": len(classic_result.losses),
            }
        )
    elif config.algorithm.name in ALIGNMENT_ALGORITHMS:
        alignment_result = train_alignment(
            config,
            output_root=args.output_root,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        payload.update(
            {
                "status": "completed",
                "step": alignment_result.step,
                "checkpoint": str(alignment_result.checkpoint),
                "experiment_card": str(alignment_result.experiment_card),
                "metrics": alignment_result.metrics,
            }
        )
    elif config.algorithm.name in GROUP_ALGORITHMS:
        group_result = train_group(
            config,
            output_root=args.output_root,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        payload.update(
            {
                "status": "completed",
                "step": group_result.step,
                "checkpoint": str(group_result.checkpoint),
                "experiment_card": str(group_result.experiment_card),
                "metrics": group_result.metrics,
            }
        )
    elif config.algorithm.name in AGENTIC_ALGORITHMS:
        agentic_result = train_agentic(
            config,
            output_root=args.output_root,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        payload.update(
            {
                "status": "completed",
                "step": agentic_result.step,
                "checkpoint": str(agentic_result.checkpoint),
                "experiment_card": str(agentic_result.experiment_card),
                "metrics": agentic_result.metrics,
            }
        )
    else:
        raise ConfigError(
            f"trainer for {config.algorithm.name!r} is introduced in a later checkpoint"
        )
    _emit(payload, as_json=args.json)
    return 0


def _eval(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        raise CheckpointError(f"checkpoint does not exist: {args.checkpoint}")
    if args.dry_run:
        _emit(
            {
                "status": "dry-run",
                "checkpoint": str(args.checkpoint),
                "split": args.split,
            },
            as_json=args.json,
        )
        return 0
    config_path = args.checkpoint / "config.resolved.json"
    try:
        config = ExperimentConfig.from_mapping(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read checkpoint config: {error}") from error
    if config.profile == "laptop":
        algorithm, step, metrics = evaluate_laptop_checkpoint(
            args.checkpoint,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
        )
    elif config.algorithm.name in CLASSIC_ALGORITHMS:
        algorithm, step, success_rate = evaluate_classic_checkpoint(args.checkpoint)
        metrics = {"success_rate": success_rate}
    elif config.algorithm.name in ALIGNMENT_ALGORITHMS:
        algorithm, step, metrics = evaluate_alignment_checkpoint(args.checkpoint)
    elif config.algorithm.name in GROUP_ALGORITHMS:
        algorithm, step, metrics = evaluate_group_checkpoint(args.checkpoint)
    elif config.algorithm.name in AGENTIC_ALGORITHMS:
        algorithm, step, metrics = evaluate_agentic_checkpoint(args.checkpoint)
    else:
        raise ConfigError(
            f"evaluator for {config.algorithm.name!r} is introduced later"
        )
    _emit(
        {
            "status": "completed",
            "algorithm": algorithm,
            "step": step,
            "split": args.split,
            "metrics": metrics,
        },
        as_json=args.json,
    )
    return 0


def _inspect_run(args: argparse.Namespace) -> int:
    card_path = args.artifact_directory / "experiment-card.json"
    if not card_path.is_file():
        raise ConfigError(f"experiment card does not exist: {card_path}")
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("experiment card must contain a JSON object")
    _emit(payload, as_json=args.json)
    return 0


def _preflight(args: argparse.Namespace) -> int:
    if args.profile != "toy":
        external_payload = build_profile_preflight(
            profile=args.profile,
            model=args.model,
            device=args.device,
            adapter=args.adapter,
            revision=args.revision,
            license_id=args.license_id,
            expected_weight_bytes=args.expected_weight_bytes,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            dtype=args.dtype,
        )
        _emit(external_payload, as_json=args.json)
        return 0
    resolution = resolve_device(args.device, allow_fallback=args.allow_device_fallback)
    payload: dict[str, object] = {
        "status": "passed",
        "profile": args.profile,
        "requested_device": args.device,
        "resolved_device": str(resolution.resolved),
        "fallback_used": resolution.fallback_used,
        "probe": resolution.probe_message,
    }
    _emit(payload, as_json=args.json)
    return 0


def _render_server(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    recipe = render_verl_recipe(config)
    validate_verl_recipe(recipe)
    _emit(recipe, as_json=args.json)
    return 0


def _exit_code(error: RLStudyError) -> int:
    if isinstance(error, DownloadApprovalRequired):
        return 4
    if isinstance(error, PreflightError):
        return 3
    if isinstance(error, CheckpointError):
        return 5
    if isinstance(error, NumericError):
        return 6
    return 2


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    handlers = {
        "demo": _demo,
        "train": _train,
        "eval": _eval,
        "inspect-run": _inspect_run,
        "preflight": _preflight,
        "render-server": _render_server,
    }
    try:
        code = handlers[args.command](args)
    except (RLStudyError, json.JSONDecodeError) as error:
        wrapped = error if isinstance(error, RLStudyError) else ConfigError(str(error))
        print(f"error: {wrapped}", file=sys.stderr)
        raise SystemExit(_exit_code(wrapped)) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
