"""Kaggle Kernel A: offline CPU reproduction of voice-evals v0.2.

enable_internet=false. Installs the exact release wheel and pinned dependency
wheels from the attached, checksummed dataset bundle -- never from PyPI at
runtime -- then reproduces the demo metrics, the frozen regression corpus,
a full four-turn mock probe (conditional branch, overlap barge-in, artifacts,
canonical conversion), the FixtureCallerVoice path, and the offline test
suite. No test here touches a network API.

Attach dataset: gjusev/voice-evals-v020-offline-bundle
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
BUNDLE = Path("/kaggle/input/voice-evals-v020-offline-bundle")
ENV = {**os.environ, "PIP_NO_INDEX": "1"}


def run(cmd: list[str], *, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env or os.environ, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        if check:
            raise SystemExit(f"failed: {' '.join(cmd)}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_from_bundle() -> None:
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text(encoding="utf-8"))
    print(f"bundle: {manifest['bundle']} voice-evals=={manifest['voice_evals_version']} "
          f"(published_wheel={manifest['published_wheel']})", flush=True)

    # 1. Verify every bundled file against the manifest before installing.
    mismatches = [
        rel for rel, expected in manifest["files"].items()
        if sha256(BUNDLE / rel) != expected
    ]
    if mismatches:
        raise SystemExit(f"bundle checksum mismatch (dataset must be pinned): {mismatches[:5]}")

    # 2. Select the lock matching this runtime; unsupported upgrades fail loudly.
    py_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    lock = next(
        (entry for entry in manifest["locks"] if entry["python"] == py_minor),
        None,
    )
    if lock is None:
        available = ", ".join(e["python"] for e in manifest["locks"])
        raise SystemExit(
            f"no bundled lock for Python {py_minor} on {platform.machine()}; "
            f"bundle supports {available}. Refusing to fetch packages online."
        )

    # 3. Offline install: hashed dependencies, then the verified release wheel.
    run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", str(BUNDLE / "wheelhouse"),
            "--require-hashes", "-r", str(BUNDLE / lock["lock"]),
        ],
        env=ENV,
    )
    wheel = manifest["wheel"]["file"]
    if sha256(BUNDLE / wheel) != manifest["wheel"]["sha256"]:
        raise SystemExit("release wheel hash mismatch against bundle manifest")
    run(
        [
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", str(BUNDLE / "wheelhouse"),
            f"voice-evals=={manifest['voice_evals_version']}",
        ],
        env=ENV,
    )
    installed = run([sys.executable, "-m", "pip", "show", "voice-evals"]).stdout
    print(installed, flush=True)
    import voice_evals

    print(f"installed voice-evals {voice_evals.__version__} at {voice_evals.__file__}", flush=True)
    assert "site-packages" in str(voice_evals.__file__), "must import the installed wheel, not a source path"


def reproduce_demo() -> None:
    from voice_evals import evaluate, load_dataset
    from voice_evals.cli import main as cli_main

    data = BUNDLE / "evals" / "data"
    result = evaluate(load_dataset(str(data / "demo_calls.jsonl")))
    assert result.samples == 3, result.samples
    assert abs(result.mean_wer - (1 / 11) / 3) < 0.01
    assert result.task_completion == 2 / 3
    assert result.fact_coverage == 5 / 6
    assert result.hallucination_rate == 1 / 3
    assert result.e2e_p50_ms == 950.0
    assert result.median_barge_in_stop_ms == 210.0
    print("demo metrics: unchanged v0.1 expected values PASS", flush=True)

    # Permissive passing gate and an intentionally failing gate (expected).
    demo = str(data / "demo_calls.jsonl")
    assert cli_main(["run", demo, "--max-wer", "0.05"]) == 0
    assert cli_main(["run", demo, "--max-wer", "0.001"]) == 1
    print("demo CLI gates: pass-case exit 0, intentional-fail exit 1 PASS", flush=True)


def reproduce_corpus() -> None:
    from voice_evals import evaluate, load_dataset

    data = BUNDLE / "evals" / "data"
    corpus = data / "replay_v02.jsonl"
    expected = json.loads((data / "replay_v02.expected.json").read_text(encoding="utf-8"))
    result = evaluate(load_dataset(str(corpus))).to_dict()
    for key, value in expected["metrics"].items():
        got = result[key]
        ok = abs(got - value) < 1e-9 if isinstance(value, float) else got == value
        assert ok, f"regression: {key} {got} != {value}"
    by_id = {d["id"]: d["wer"] for d in result["details"]}
    for case in expected["hand_checked"]["cases"]:
        assert abs(by_id[case["id"]] - case["hand_computed_wer"]) < 1e-9
    print(f"regression corpus: {result['samples']} calls, frozen metrics + hand-checked WERs PASS", flush=True)
    (WORKING / "replay_v02.result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_mock_probe() -> None:
    import asyncio

    from voice_evals import evaluate, load_dataset
    from voice_evals.probe import (
        MockCallerVoice,
        MockTransport,
        ProbeConfig,
        ScenarioScript,
        SessionRunner,
    )

    scenario = BUNDLE / "evals" / "scenarios" / "appointment-v2.json"
    out = WORKING / "mock-probe"
    config = ProbeConfig(environment="mock", output_dir=out)
    runner = SessionRunner(
        caller=MockCallerVoice(),
        transport=MockTransport(),
        config=config,
    )
    session = asyncio.run(runner.run(ScenarioScript.load(scenario), output_dir=out))
    report = runner.last_report
    assert session.status.value == "completed", session.status_reason
    assert report.probe["scoring_status"] == "scored"
    assert report.result["task_completion"] == 1.0
    interruptions = report.probe["interruptions"]
    assert interruptions["attempted"] == 1 and interruptions["observed"] == 1
    assert interruptions["max_stop_ms"] <= 340, interruptions
    for name in ("manifest.json", "events.jsonl", "calls.jsonl", "result.json"):
        assert (out / name).is_file()
    # Live-to-replay canonical equality on the exported record.
    replay = evaluate(load_dataset(str(out / "calls.jsonl"))).to_dict()
    for key in ("mean_wer", "task_completion", "fact_coverage", "median_barge_in_stop_ms"):
        assert replay[key] == report.result[key], key
    print("mock probe: conditional branch + overlap barge-in + artifacts + replay equality PASS", flush=True)

    # FixtureCallerVoice against the synthetic checksummed fixtures.
    from voice_evals.probe.voices.fixture import FixtureCallerVoice

    fixtures = BUNDLE / "evals" / "fixtures" / "audio" / "synthetic" / "manifest.json"
    voice = FixtureCallerVoice(fixtures)
    clip = asyncio.run(voice.synthesize("My name is Alex Morgan.", audio_format=config.input_format))
    assert clip.pcm and clip.provenance["sha256"]
    print("fixture caller: checksummed synthetic clip PASS", flush=True)


def run_tests() -> None:
    tests_dir = BUNDLE / "tests"
    # Integration tests are excluded by marker AND guard (they can never run here).
    result = run(
        [
            sys.executable, "-m", "pytest", str(tests_dir),
            "-q", "-m", "not live",
            "-p", "no:cacheprovider",
        ],
        check=False,
        env={**os.environ, "CI": "1"},
    )
    (WORKING / "pytest-report.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-5:])
    print(tail, flush=True)
    assert result.returncode == 0, "offline test suite failed"


def write_provenance() -> None:
    import voice_evals

    provenance = {
        "voice_evals": voice_evals.__version__,
        "voice_evals_path": str(voice_evals.__file__),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": sys.implementation.name,
        "bundle": json.loads((BUNDLE / "bundle-manifest.json").read_text(encoding="utf-8"))["bundle"],
        "network": "disabled (enable_internet=false)",
    }
    (WORKING / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)
    # Artifact checksums for downstream verification.
    checksums = {
        p.relative_to(WORKING).as_posix(): sha256(p)
        for p in sorted(WORKING.rglob("*"))
        if p.is_file() and p.name != "artifact-checksums.json"
    }
    (WORKING / "artifact-checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    WORKING.mkdir(parents=True, exist_ok=True)
    print("=== voice-evals offline reproduction (Kernel A) ===", flush=True)
    install_from_bundle()
    reproduce_demo()
    reproduce_corpus()
    run_mock_probe()
    run_tests()
    write_provenance()
    print("KERNEL A: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
