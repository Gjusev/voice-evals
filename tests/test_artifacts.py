"""Recording artifacts: manifest, journal recovery, audio bounds, secrets."""

import asyncio
import importlib.resources
import json
from pathlib import Path

from voice_evals.probe import (
    MockCallerVoice,
    MockTransport,
    ProbeConfig,
    ScenarioScript,
    SessionRunner,
)
from voice_evals.probe.clock import VirtualClock
from voice_evals.probe.recording import read_journal

RESOURCES = Path(str(importlib.resources.files("voice_evals") / "resources"))
APPOINTMENT = RESOURCES / "scenarios" / "appointment-v2.json"


def run_session(tmp_path: Path, **config_kwargs):
    clock = VirtualClock(slice_ns=1_000_000)
    config = ProbeConfig(environment="mock", output_dir=tmp_path / "out", **config_kwargs)
    runner = SessionRunner(
        caller=MockCallerVoice(clock=clock),
        transport=MockTransport(clock=clock),
        config=config,
        clock=clock,
    )
    session = asyncio.run(runner.run(ScenarioScript.load(APPOINTMENT), output_dir=config.output_dir))
    return runner, session


def test_output_directory_layout(tmp_path: Path) -> None:
    runner, _session = run_session(tmp_path)
    out = runner.config.output_dir
    for name in ("manifest.json", "events.jsonl", "calls.jsonl", "result.json"):
        assert (out / name).is_file(), f"missing {name}"
    assert list((out / "audio" / "caller").glob("*.wav"))
    assert list((out / "audio" / "agent").glob("*.wav"))


def test_manifest_contents_and_hashes(tmp_path: Path) -> None:
    runner, _session = run_session(tmp_path)
    manifest = json.loads((runner.config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["complete"] is True
    assert manifest["environment"] == "mock"
    assert manifest["versions"]["voice_evals"] and manifest["versions"]["python"]
    assert manifest["script"]["schema_version"] == 2
    assert manifest["script_sha256"]
    assert manifest["capabilities"]["duplex"] is True
    assert manifest["scoring"]["eligible"] is True
    assert manifest["scoring"]["replayable"] is True
    assert manifest["files"]["calls.jsonl"]
    # Chosen alternatives recorded with their indices
    turns = manifest["turns"]
    assert all("alternative_index" in t and "selected_text" in t for t in turns)
    # Interruption observation carried in the manifest
    assert any(t.get("interrupt", {}).get("status") == "observed" for t in turns)


def test_manifest_config_has_no_secrets(tmp_path: Path) -> None:
    import os

    os.environ["ELEVENLABS_API_KEY"] = "sk-super-secret"
    os.environ["PROBE_TRANSPORT_URL"] = "wss://agent.example.com/ws?sig=abcd1234"
    try:
        runner, _session = run_session(tmp_path)
        text = (runner.config.output_dir / "manifest.json").read_text(encoding="utf-8")
        assert "sk-super-secret" not in text
        assert "abcd1234" not in text  # signed query redacted
        assert "PROBE_TRANSPORT_URL" in text or "mock" in text  # env names are fine
    finally:
        os.environ.pop("ELEVENLABS_API_KEY", None)
        os.environ.pop("PROBE_TRANSPORT_URL", None)


def test_journal_is_append_only_jsonl_and_recoverable(tmp_path: Path) -> None:
    runner, _session = run_session(tmp_path)
    journal = runner.config.output_dir / "events.jsonl"
    raw = journal.read_text(encoding="utf-8")
    entries = read_journal(journal)
    assert entries and all(isinstance(e, dict) and "kind" in e for e in entries)
    # Simulate hard termination: truncate the final line mid-way.
    lines = raw.splitlines(keepends=True)
    corrupted = lines[-1][: len(lines[-1]) // 2]
    journal.write_text("".join(lines[:-1]) + corrupted, encoding="utf-8")
    recovered = read_journal(journal)
    assert len(recovered) == len(entries) - 1  # complete lines only


def test_audio_files_contain_only_sent_and_received_bytes(tmp_path: Path) -> None:
    runner, _session = run_session(tmp_path)
    out = runner.config.output_dir
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    caller_wav = next((out / "audio" / "caller").glob("utt-request.wav"))
    import wave

    with wave.open(str(caller_wav), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    turn_frames = next(t for t in manifest["turns"] if t["step_id"] == "request")
    expected_samples = sum(f["sample_count"] for f in turn_frames["frames"])
    assert frames == expected_samples
    assert rate == 16000
    assert manifest["files"]["audio/caller/utt-request.wav"]


def test_journal_bound_fails_visibly(tmp_path: Path) -> None:
    _runner, session = run_session(tmp_path, max_event_journal_entries=5)
    assert session.status.value == "failed"
    assert session.status_reason == "recording_error"
    assert any(e["category"] == "recording_error" for e in session.errors)


def test_result_json_preserves_v01_fields_additively(tmp_path: Path) -> None:
    runner, _session = run_session(tmp_path)
    result = json.loads((runner.config.output_dir / "result.json").read_text(encoding="utf-8"))
    for key in (
        "samples", "failures", "mean_wer", "max_wer", "task_completion",
        "fact_coverage", "hallucination_rate", "e2e_p50_ms", "e2e_p95_ms",
        "e2e_p99_ms", "stage_means_ms", "interruption_count",
        "median_barge_in_stop_ms", "details",
    ):
        assert key in result
    assert "probe" in result and "gate_passed" in result
    assert result["probe"]["scoring_status"] == "scored"
    assert result["probe"]["stage_aggregation"] == "mean_of_observed_nonoverlap_turns"
    assert result["probe"]["measurement_basis"].startswith("local monotonic clock")


def test_not_scored_session_keeps_placeholder_convention(tmp_path: Path) -> None:
    # Force ineligibility: agent never answers (huge processing delay).
    from voice_evals.probe import MockAgentPolicy

    clock = VirtualClock(slice_ns=1_000_000)
    config = ProbeConfig(environment="mock", output_dir=tmp_path / "out")
    runner = SessionRunner(
        caller=MockCallerVoice(clock=clock),
        transport=MockTransport(clock=clock, policy=MockAgentPolicy(processing_ms=60_000)),
        config=config,
        clock=clock,
    )
    asyncio.run(runner.run(ScenarioScript.load(APPOINTMENT), output_dir=config.output_dir))
    result = json.loads((config.output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["probe"]["scoring_status"] == "not_scored"
    assert result["samples"] == 0 and result["failures"] == 0
    assert result["mean_wer"] == 0.0  # placeholder convention, not a measurement
    assert result["e2e_p50_ms"] is None and result["stage_means_ms"] == {}
    assert result["details"] == []
    assert result["gate_passed"] is False
    calls = (config.output_dir / "calls.jsonl").read_text(encoding="utf-8").strip()
    assert calls == ""
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scoring"]["replayable"] is False
    assert manifest["scoring"]["exclusion_reasons"]
    # The summary prints NOT SCORED, never placeholder rates.
    import io

    buffer = io.StringIO()
    runner.last_report.print_summary(buffer)
    assert "NOT SCORED" in buffer.getvalue()
