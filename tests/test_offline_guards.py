"""Offline guards: base-only importability and bundled trace fixtures."""

import json
import subprocess
import sys
from pathlib import Path


def test_base_install_imports_without_network_deps() -> None:
    """Replay APIs, probe interfaces, runner, and mocks import with jiwer only.

    Runs a clean subprocess that blocks httpx/websockets imports.
    """
    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name in ('httpx', 'websockets'):\n"
        "            raise ImportError('blocked: base install')\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in ('httpx', 'websockets'):\n"
        "            raise ImportError('blocked: base install')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import voice_evals\n"
        "import voice_evals.probe\n"
        "from voice_evals.probe import SessionRunner, MockTransport, MockCallerVoice, ScenarioScript\n"
        "from voice_evals import evaluate, load_dataset\n"
        "assert voice_evals.__version__\n"
        "print('base-import-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "base-import-ok" in result.stdout


def test_no_network_module_imports_from_probe_package_init() -> None:
    import voice_evals.probe as probe_pkg

    # The package itself must not import network libraries transitively.
    source = (Path(probe_pkg.__file__).parent / "__init__.py").read_text(encoding="utf-8")
    for banned in ("import httpx", "import websockets", "from httpx", "from websockets"):
        assert banned not in source


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_bundled_traces_are_consistent() -> None:
    for path in sorted(FIXTURES.glob("trace-*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        assert trace["note"].startswith("normalized trace"), path
        assert trace["events"], path
        kinds = {e["kind"] for e in trace["events"]}
        assert "session_ready" in kinds, path
        if trace["name"] == "trace-success":
            assert trace["session_status"] == "completed"
            assert "caller_transcript_final" in kinds
            assert "agent_audio" in kinds
        if trace["name"] == "trace-timeout":
            assert trace["session_status"] == "failed"
            assert trace["status_reason"] == "response_timeout"
        if trace["name"] == "trace-barge-in-ignored":
            assert trace["session_status"] == "completed"
