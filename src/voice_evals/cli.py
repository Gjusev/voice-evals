"""Command-line interface for voice-evals.

Exit codes: 0 = gate passed, 1 = gate/behavioral failure or no usable
response, 2 = usage/config/transport/provider/recording error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import __version__
from .evaluate import evaluate
from .gates import metric_gate_problems
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

    probe = subparsers.add_parser(
        "probe",
        help="Call a live voice agent with a scripted synthetic caller and score the session",
    )
    _add_probe_args(probe)
    return parser


def _add_probe_args(probe: argparse.ArgumentParser) -> None:
    probe.add_argument("scenario", help="Path to a scenario-v2 JSON script")
    probe.add_argument("--transport", default=None, help="Agent WebSocket URL (ws:// or wss://)")
    probe.add_argument(
        "--transport-env",
        default="PROBE_TRANSPORT_URL",
        help="Environment variable holding the transport URL (avoids signed URLs in argv)",
    )
    probe.add_argument(
        "--api-key-env",
        default="ELEVENLABS_API_KEY",
        help="Env var holding the caller TTS API key",
    )
    probe.add_argument(
        "--agent-api-key-env",
        default=None,
        help="Env var holding the agent transport API key (never the caller key)",
    )
    probe.add_argument("--protocol-map", default=None, help="Protocol map JSON path (default: bundled default-v1)")
    probe.add_argument(
        "--caller",
        choices=["elevenlabs", "http", "fixture", "mock"],
        default="elevenlabs",
        help="Caller voice engine",
    )
    probe.add_argument("--voice-id", default=None, help="Caller voice ID")
    probe.add_argument("--voice-id-env", default="ELEVENLABS_VOICE_ID", help="Env var holding the voice ID")
    probe.add_argument("--caller-model", default=None, help="Caller TTS model ID")
    probe.add_argument("--caller-url-env", default="CALLER_TTS_URL", help="Env var holding the open-model TTS URL")
    probe.add_argument("--caller-auth-env", default=None, help="Optional env var holding caller TTS auth token")
    probe.add_argument("--language", default="en", help="Caller language hint")
    probe.add_argument("--fixture-manifest", default=None, help="Fixture caller manifest path")
    probe.add_argument("--mock", action="store_true", help="MockTransport + MockCallerVoice, no network")
    # gates
    probe.add_argument("--max-wer", type=float, default=None)
    probe.add_argument("--min-task-completion", type=float, default=None)
    probe.add_argument("--min-fact-coverage", type=float, default=None)
    probe.add_argument("--max-e2e-p95-ms", type=float, default=None)
    probe.add_argument("--max-hallucination-rate", type=float, default=None)
    probe.add_argument(
        "--max-barge-in-stop-ms",
        type=int,
        default=None,
        help="Gate: fail if observed barge-in stop latency is higher; requests the barge-in gate",
    )
    # timeouts / knobs
    probe.add_argument("--connect-timeout-ms", type=int, default=None)
    probe.add_argument("--response-timeout-ms", type=int, default=None)
    probe.add_argument("--call-timeout-ms", type=int, default=None)
    probe.add_argument("--agent-quiet-ms", type=int, default=None, help="Quiet window inferring agent audio cessation")
    probe.add_argument("--frame-ms", type=int, default=None, help="Caller audio frame duration")
    probe.add_argument("--seed", type=int, default=None, help="Override the scenario seed")
    probe.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout")
    probe.add_argument("--output", type=Path, default=None, help="Also write the scored JSON here")
    probe.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/probe"),
        help="Recording directory (manifest, journal, audio, calls, result)",
    )


def _run_command(args: argparse.Namespace) -> int:
    try:
        records = load_dataset(args.dataset)
        result = evaluate(records)
    except (OSError, ValueError, DatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    problems = metric_gate_problems(
        result.to_dict(),
        {
            "max_wer": args.max_wer,
            "min_task_completion": args.min_task_completion,
            "min_fact_coverage": args.min_fact_coverage,
            "max_e2e_p95_ms": args.max_e2e_p95_ms,
            "max_hallucination_rate": args.max_hallucination_rate,
        },
    )

    if args.json:
        print(json.dumps({**result.to_dict(), "gate_passed": not problems}, indent=2))
    else:
        print(result.summary())
        print("gate: " + ("PASSED" if not problems else "FAILED"))
        for problem in problems:
            print(f"  - {problem}")
    return 0 if not problems else 1


_OPERATIONAL_EXIT2 = {
    "transport_error",
    "caller_auth",
    "caller_voice_error",
    "caller_format",
    "recording_error",
    "config_invalid",
    "protocol_failure",
    "protocol_map_invalid",
    "capability_unsupported",
    "scenario_invalid",
    "cancelled",
}


def _probe_command(args: argparse.Namespace) -> int:
    from .probe import (
        CallerConfig,
        ProbeConfig,
        ScenarioScript,
        SessionRunner,
        TransportConfig,
        build_caller,
        build_transport,
    )
    from .probe.errors import ProbeError

    caller = CallerConfig(kind="mock" if args.mock else args.caller)
    caller.voice_id = args.voice_id
    caller.voice_id_env = args.voice_id_env
    caller.api_key_env = args.api_key_env
    if args.caller_model:
        caller.model_id = args.caller_model
    caller.auth_env = args.caller_auth_env
    caller.language = args.language
    caller.fixture_manifest = str(args.fixture_manifest) if args.fixture_manifest else None

    transport = TransportConfig(
        endpoint=args.transport,
        endpoint_env=args.transport_env,
        api_key_env=args.agent_api_key_env,
        protocol_map_path=args.protocol_map,
    )
    if args.frame_ms:
        transport.frame_ms = args.frame_ms

    environment = "mock" if args.mock else os.environ.get("VOICE_EVALS_ENV", "local")
    config = ProbeConfig(
        environment=environment,
        output_dir=args.output_dir,
        seed_override=args.seed,
        transport=transport,
        caller=caller,
        max_wer=args.max_wer,
        min_task_completion=args.min_task_completion,
        min_fact_coverage=args.min_fact_coverage,
        max_e2e_p95_ms=args.max_e2e_p95_ms,
        max_hallucination_rate=args.max_hallucination_rate,
        max_barge_in_stop_ms=args.max_barge_in_stop_ms,
    )
    if args.connect_timeout_ms:
        config.connect_timeout_ms = args.connect_timeout_ms
    if args.response_timeout_ms:
        config.response_timeout_ms = args.response_timeout_ms
    if args.call_timeout_ms:
        config.call_timeout_ms = args.call_timeout_ms
    if args.agent_quiet_ms:
        config.agent_quiet_ms = args.agent_quiet_ms

    try:
        script = ScenarioScript.load(args.scenario)
        if not script.steps or script.outcome_rule is None:
            print(
                f"error: scenario {args.scenario!r} has no probe script (legacy replay "
                "scenario); provide a schema_version=2 script with steps and outcome_rule",
                file=sys.stderr,
            )
            return 2
        config.request_barge_in_gate = args.max_barge_in_stop_ms is not None or any(
            s.cue.mode == "interrupt" for s in script.steps
        )
        config.validate()
    except (OSError, ValueError, ProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.frame_ms is None and config.caller.kind != "mock":
        from .probe import load_protocol_map

        declared = load_protocol_map(config).input_frame_ms
        if declared is not None:
            config.transport.frame_ms = declared
    caller_voice = build_caller(config)
    agent_transport = build_transport(config)
    runner = SessionRunner(caller=caller_voice, transport=agent_transport, config=config)
    try:
        session = asyncio.run(runner.run(script, output_dir=args.output_dir))
    except ProbeError as error:
        print(f"error: {error.message}", file=sys.stderr)
        if error.remediation:
            print(f"remediation: {error.remediation}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return 2

    report = runner.last_report
    if report is None:  # finalize itself failed before scoring
        print("error: session finalized without a report", file=sys.stderr)
        return 2
    if session.errors:
        first = session.errors[0]
        print(f"session error: {first.get('message')}", file=sys.stderr)
    if args.output is not None:
        args.output.write_text(report.to_json(), encoding="utf-8")
    if args.json:
        print(report.to_json())
    else:
        report.print_summary()
        print("gate: " + ("PASSED" if report.gate_passed else "FAILED"))
        for reason in report.probe.get("gate_reasons", []):
            print(f"  - {reason}")

    exit_code = _probe_exit_code(session, report.gate_passed)
    print(f"artifacts: {args.output_dir}", file=sys.stderr)
    return exit_code


def _probe_exit_code(session, gate_passed: bool) -> int:
    """0 observed-and-scored pass; 2 operational; 1 measured/behavioral."""
    for error in session.errors:
        if error.get("category") in _OPERATIONAL_EXIT2:
            return 2
    if session.status_reason == "cancelled":
        return 2
    if session.status.value == "completed" and not session.errors:
        return 0 if gate_passed else 1
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "probe":
        return _probe_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
