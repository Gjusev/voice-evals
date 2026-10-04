"""Kaggle Kernel B: live diagnostic probe with a full mock fallback.

Installs the pinned release from the same attached offline bundle (even the
mock path never touches live pip), then:

- ELEVENLABS_API_KEY + PROBE_TRANSPORT_URL present: one live scripted probe
  against the agent over outbound WSS. ElevenLabs is the default caller.
- Open-model variant: CALLER_TTS_URL (+ optional CALLER_TTS_AUTH) with
  caller=http instead of ElevenLabs credentials.
- Secrets missing: a full mock session through the SAME runner, recorder,
  conversion and evaluator with mode=mock. No network attempt. The missing
  secret names are printed.
- Live attempt failing: keep the failed live artifacts AND additionally run a
  separately named mock diagnostic; live_status stays "failed". The mock score
  is never substituted for a live result.

Latency measured from the Kaggle datacenter includes network transit/RTT and
client scheduling; it is not directly comparable with local runs. Mock latency
is simulated and not a provider benchmark.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")

def locate_bundle() -> Path:
    """Find the attached offline bundle wherever Kaggle mounted it.

    Observed layouts: the classic /kaggle/input/<slug>/... and the newer
    /kaggle/input/datasets/<owner>/<slug>/..., with the bundle zip either
    already extracted (nested one level) or still zipped. A recursive search
    from /kaggle/input covers all of them; hashes are verified against
    bundle-manifest.json after extraction, so a wrong directory can never
    silently install.
    """
    import glob
    import zipfile

    input_root = Path("/kaggle/input")
    manifests = [
        Path(p)
        for p in glob.glob(str(input_root / "**" / "bundle-manifest.json"), recursive=True)
    ]
    if manifests:
        return manifests[0].parent
    zips = sorted(input_root.rglob("*.zip"))
    if not zips:
        listing = sorted(str(p) for p in input_root.rglob("*") if p.is_file())
        raise SystemExit(
            "no bundle or zip found under /kaggle/input; attached files: "
            + "; ".join(listing[:40])
        )
    target = Path("/kaggle/working/bundle")
    if not (target / "bundle-manifest.json").is_file():
        with zipfile.ZipFile(zips[0]) as archive:
            archive.extractall(target)
        print(f"extracted {zips[0].name} -> {target}", flush=True)
    return target


BUNDLE = locate_bundle()


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd)}")


def load_secret(name: str) -> str:
    """Kaggle Secret when available, else environment (for local runs)."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - Kaggle-only import; env fallback elsewhere
        return os.environ.get(name, "")


def install_from_bundle() -> None:
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text(encoding="utf-8"))
    py_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    lock = next((e for e in manifest["locks"] if e["python"] == py_minor), None)
    if lock is None:
        raise SystemExit(f"bundle has no lock for Python {py_minor}; refusing to download online")
    run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", str(BUNDLE / "wheelhouse"),
            "--require-hashes", "-r", str(BUNDLE / lock["lock"]),
        ]
    )
    run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", str(BUNDLE / "wheelhouse"),
            f"voice-evals=={manifest['voice_evals_version']}",
        ]
    )
    import voice_evals

    print(f"installed voice-evals {voice_evals.__version__}", flush=True)


SCENARIO = "scenario.json"


def prepare_scenario() -> Path:
    """Copy the bundled four-turn scenario into the working directory."""
    source = BUNDLE / "evals" / "scenarios" / "appointment-v2.json"
    target = WORKING / SCENARIO
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


async def run_probe(mode: str, out_dir: Path) -> dict:
    """One probe session through the real runner (live or mock)."""
    from voice_evals.probe import (
        CallerConfig,
        ProbeConfig,
        ScenarioScript,
        SessionRunner,
        TransportConfig,
        build_caller,
        build_transport,
    )
    from voice_evals.probe.clock import MonotonicClock

    scenario_path = prepare_scenario()
    script = ScenarioScript.load(scenario_path)
    caller = CallerConfig(kind="mock" if mode == "mock" else "elevenlabs")
    transport = TransportConfig()
    config = ProbeConfig(
        environment="kaggle" if mode != "mock" else "mock",
        output_dir=out_dir,
        caller=caller,
        transport=transport,
        request_barge_in_gate=True,
        max_barge_in_stop_ms=800,
    )
    if mode == "http":
        config.caller = CallerConfig(
            kind="http",
            model_id=os.environ.get("CALLER_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            url_env="CALLER_TTS_URL",
            auth_env="CALLER_TTS_AUTH",
        )
    clock = MonotonicClock()
    runner = SessionRunner(
        caller=build_caller(config, clock=clock),
        transport=build_transport(config, clock=clock),
        config=config,
        clock=clock,
    )
    session = await runner.run(script, output_dir=out_dir)
    report = runner.last_report
    return {
        "mode": mode,
        "session_status": session.status.value,
        "status_reason": session.status_reason,
        "scoring_status": report.probe["scoring_status"],
        "gate_passed": report.gate_passed,
        "result": report.result,
        "errors": session.errors,
        "output_dir": str(out_dir),
    }


def write_report(live: dict | None, mock: dict | None, missing: list[str]) -> None:
    live_status = None
    if live is not None:
        live_status = (
            "completed_and_scored"
            if live["session_status"] == "completed" and live["scoring_status"] == "scored"
            else "failed"
        )
    report = {
        "live": live,
        "live_status": live_status,
        "mock_diagnostic": mock,
        "missing_secrets": missing,
        "location": "kaggle",
        "latency_warning": (
            "live latency measured from the Kaggle datacenter includes network "
            "transit/RTT and client scheduling; not comparable with local runs. "
            "Mock latency is simulated, not a provider benchmark."
        ),
    }
    (WORKING / "probe-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mock_diagnostic"}, indent=2), flush=True)


def main() -> int:
    WORKING.mkdir(parents=True, exist_ok=True)
    print("=== voice-evals live probe (Kernel B) ===", flush=True)
    install_from_bundle()

    elevenlabs_key = load_secret("ELEVENLABS_API_KEY")
    agent_url = load_secret("PROBE_TRANSPORT_URL")
    caller_tts_url = load_secret("CALLER_TTS_URL")
    agent_key = load_secret("PROBE_AGENT_API_KEY")  # optional, never the caller key
    # Optional voice id: fall back to the bundled documented stock voice choice.
    voice_id = load_secret("ELEVENLABS_VOICE_ID")

    mode = None
    missing: list[str] = []
    if caller_tts_url and agent_url:
        mode = "http"
        os.environ["CALLER_TTS_URL"] = caller_tts_url
    elif elevenlabs_key and agent_url:
        mode = "elevenlabs"
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
        if voice_id:
            os.environ["ELEVENLABS_VOICE_ID"] = voice_id
    else:
        if not elevenlabs_key and not caller_tts_url:
            missing.append("ELEVENLABS_API_KEY")
        if not agent_url:
            missing.append("PROBE_TRANSPORT_URL")
        if not voice_id and not caller_tts_url:
            missing.append("ELEVENLABS_VOICE_ID (or set a stock voice via --voice-id)")

    if agent_url:
        os.environ["PROBE_TRANSPORT_URL"] = agent_url
    if agent_key:
        os.environ["PROBE_AGENT_API_KEY"] = agent_key

    live_result: dict | None = None
    mock_result: dict | None = None
    if mode is None:
        print(f"secrets missing ({', '.join(missing)}): full MOCK run through the real runner", flush=True)
        mock_result = asyncio.run(run_probe("mock", WORKING / "mock-session"))
        _print_summary(mock_result)
        write_report(None, mock_result, missing)
        return 0

    print(f"secrets present: running ONE LIVE probe (caller={mode})", flush=True)
    live_result = asyncio.run(run_probe(mode, WORKING / "live-session"))
    _print_summary(live_result)
    if live_result["session_status"] != "completed" or live_result["scoring_status"] != "scored":
        print("live attempt failed: additionally running the separately named mock diagnostic", flush=True)
        mock_result = asyncio.run(run_probe("mock", WORKING / "mock-session"))
        _print_summary(mock_result)
    write_report(live_result, mock_result, [])
    # The report's live gate decides severity; a failed live run exits 1.
    live_ok = live_result["gate_passed"] and live_result["scoring_status"] == "scored"
    return 0 if live_ok else 1


def _print_summary(result: dict) -> None:
    probe = result["result"].get("probe", {})
    print(
        f"[{result['mode']}] status={result['session_status']} scoring={result['scoring_status']} "
        f"gate={result['gate_passed']} mean_wer={result['result'].get('mean_wer')} "
        f"e2e_p50={result['result'].get('e2e_p50_ms')} interruptions={probe.get('interruptions')}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
