"""Opt-in live integration tests. Never run in CI.

These require BOTH an explicit opt-in (VOICE_EVALS_LIVE_TESTS=1) and the
relevant secrets. A CI env var guard blocks live calls even when credentials
accidentally exist. Mock transports here are never presented as live results.
"""

import os
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("VOICE_EVALS_LIVE_TESTS") != "1",
        reason="live tests require VOICE_EVALS_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        os.environ.get("CI") == "1" or os.environ.get("GITHUB_ACTIONS") == "true",
        reason="live calls are never made from CI",
    ),
]

SCENARIO = (
    Path(__file__).resolve().parents[2] / "src" / "voice_evals" / "resources" / "scenarios" / "appointment-v2.json"
)


def test_live_probe_against_real_endpoint(tmp_path: Path) -> None:
    """One scripted probe against PROBE_TRANSPORT_URL with ElevenLabs caller."""
    if not os.environ.get("PROBE_TRANSPORT_URL"):
        pytest.skip("PROBE_TRANSPORT_URL not set")
    if not os.environ.get("ELEVENLABS_API_KEY"):
        pytest.skip("ELEVENLABS_API_KEY not set")
    from voice_evals.cli import main

    out = tmp_path / f"live-{uuid.uuid4().hex[:6]}"
    code = main(
        [
            "probe", str(SCENARIO),
            "--max-wer", "0.5",
            "--max-barge-in-stop-ms", "800",
            "--output-dir", str(out),
        ]
    )
    # Live nondeterminism: assert artifacts and honest scoring, not fixed metrics.
    assert (out / "manifest.json").is_file()
    assert (out / "result.json").is_file()
    assert code in (0, 1)  # 2 would mean config/transport setup failure
    assert os.environ.get("PROBE_TRANSPORT_URL", "").startswith(("ws://", "wss://"))
