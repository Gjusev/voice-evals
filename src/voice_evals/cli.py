"""Command-line interface for voice-evals.

Exit codes: 0 = gate passed, 1 = gate failed, 2 = usage/dataset error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import evaluate
from .types import DatasetError, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-eval",
        description="Replay voice-agent calls and gate on WER, latency and outcomes.",
    )
    parser.add_argument("--version", action="version", version=f"voice-evals {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a replay dataset and optionally gate on thresholds")
    run.add_argument("dataset", help="Path to a .json array or .jsonl replay dataset")
    run.add_argument("--max-wer", type=float, default=None, help="Gate: fail if mean WER is higher")
    run.add_argument(
        "--min-task-completion", type=float, default=None, help="Gate: fail if task completion is lower"
    )
    run.add_argument("--min-fact-coverage", type=float, default=None, help="Gate: fail if fact coverage is lower")
    run.add_argument(
        "--max-e2e-p95-ms", type=float, default=None, help="Gate: fail if p95 end-to-end latency is higher"
    )
    run.add_argument("--max-hallucination-rate", type=float, default=None, help="Gate: fail if rate is higher")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    run.add_argument("--output", type=Path, default=None, help="Also write the full result JSON here")
    return parser


def _run_command(args: argparse.Namespace) -> int:
    try:
        records = load_dataset(args.dataset)
        result = evaluate(records)
    except (OSError, ValueError, DatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    problems: list[str] = []
    if args.max_wer is not None and result.mean_wer > args.max_wer:
        problems.append(f"mean_wer {result.mean_wer:.4f} > allowed {args.max_wer:.4f}")
    if args.min_task_completion is not None and result.task_completion < args.min_task_completion:
        problems.append(
            f"task_completion {result.task_completion:.4f} < required {args.min_task_completion:.4f}"
        )
    if args.min_fact_coverage is not None and result.fact_coverage < args.min_fact_coverage:
        problems.append(
            f"fact_coverage {result.fact_coverage:.4f} < required {args.min_fact_coverage:.4f}"
        )
    if args.max_e2e_p95_ms is not None and result.e2e_p95_ms is not None and result.e2e_p95_ms > args.max_e2e_p95_ms:
        problems.append(f"e2e_p95_ms {result.e2e_p95_ms:.1f} > allowed {args.max_e2e_p95_ms:.1f}")
    if args.max_hallucination_rate is not None and result.hallucination_rate > args.max_hallucination_rate:
        problems.append(
            f"hallucination_rate {result.hallucination_rate:.4f} > allowed {args.max_hallucination_rate:.4f}"
        )

    if args.json:
        print(json.dumps({**result.to_dict(), "gate_passed": not problems}, indent=2))
    else:
        print(result.summary())
        print("gate: " + ("PASSED" if not problems else "FAILED"))
        for problem in problems:
            print(f"  - {problem}")
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
