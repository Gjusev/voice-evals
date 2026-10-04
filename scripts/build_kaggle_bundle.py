"""Build the internet-disabled Kaggle reproduction bundle (release tooling).

Runs on an internet-enabled release machine AFTER the PyPI publication:

    uv run python scripts/build_kaggle_bundle.py --version 0.2.0 --output kaggle-kernel/bundle/staging

It downloads the *actual published* wheel from PyPI, verifies its published
hash, resolves and downloads all offline test/probe/base dependency wheels for
each supported (python-minor, platform) pair with hashes, copies tests and
datasets from the release commit, and writes ``bundle-manifest.json``.

For pre-release validation only, ``--allow-local-wheel`` substitutes a locally
built wheel; the manifest then records ``"published_wheel": false`` and the
bundle MUST NOT be published as a reproduction of a PyPI release.

Requires on the release machine: uv (resolution), pip (wheel download), git
(clean-tree provenance). No source builds, compilers, or apt at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Kaggle CPU images are Ubuntu-based (glibc >= 2.28): wheels tagged up to
# manylinux_2_28 install; older tags stay accepted for pure/legacy wheels.
PIP_PLATFORMS = [
    "manylinux_2_28_x86_64",
    "manylinux_2_27_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
]
LOCK_TARGETS = [
    {"python": "3.10", "tag": "cp310"},
    {"python": "3.11", "tag": "cp311"},
]
# The lock covers DEPENDENCIES ONLY: voice-evals itself comes from the
# verified bundle wheel (not from an index), so resolution cannot see a
# version that is not yet published.
REQUIREMENTS_IN = """
jiwer>=3.0
httpx>=0.27,<0.29
websockets>=15,<16
pytest>=8.0
jsonschema>=4.20
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], **kwargs) -> str:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return result.stdout


def ensure_pip() -> None:
    """uv venvs ship without pip; bootstrap it for ``pip download``."""
    probe = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        run([sys.executable, "-m", "ensurepip", "--upgrade"])


def pypi_wheel_info(version: str) -> tuple[str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/voice-evals/{version}/json") as response:
        data = json.load(response)
    for entry in data["urls"]:
        name = entry["filename"]
        if name.endswith(".whl") and "py3-none-any" in name:
            return entry["url"], entry["digests"]["sha256"]
    raise SystemExit(f"no universal wheel found on PyPI for voice-evals=={version}")


def download(url: str, dest: Path) -> None:
    print(f"download {url}", flush=True)
    urllib.request.urlretrieve(url, dest)


def git_tree_state() -> dict:
    commit = run(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    dirty = bool(run(["git", "status", "--porcelain"], cwd=REPO).strip())
    return {"commit": commit, "clean_tree": not dirty}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=_current_version())
    parser.add_argument("--output", default="kaggle-kernel/bundle/staging")
    parser.add_argument(
        "--allow-local-wheel",
        action="store_true",
        help="pre-release validation: use the locally built wheel instead of PyPI",
    )
    parser.add_argument("--local-wheel", default=None, help="path to a locally built wheel")
    parser.add_argument("--skip-download", action="store_true", help="resolve locks only (no wheelhouse)")
    args = parser.parse_args()

    version = args.version
    out = Path(args.output)
    wheelhouse = out / "wheelhouse"
    locks = out / "locks"
    tests = out / "tests"
    for directory in (wheelhouse, locks, tests, out / "evals"):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. The exact published wheel (or an explicitly-flagged local substitute).
    published = True
    if args.allow_local_wheel or args.local_wheel:
        published = False
        local = Path(args.local_wheel or f"dist/voice_evals-{version}-py3-none-any.whl")
        if not local.is_file():
            raise SystemExit(f"local wheel not found: {local} (run `make build` first)")
        wheel_path = wheelhouse / local.name
        wheel_path.write_bytes(local.read_bytes())
        wheel_hash = sha256_file(wheel_path)
        source = {"kind": "local-wheel", "path": str(local)}
    else:
        url, expected_hash = pypi_wheel_info(version)
        name = url.rsplit("/", 1)[-1]
        wheel_path = wheelhouse / name
        download(url, wheel_path)
        wheel_hash = sha256_file(wheel_path)
        if wheel_hash != expected_hash:
            raise SystemExit("wheel hash mismatch against PyPI digests; refusing")
        source = {"kind": "pypi", "url": url}

    # 2. Per-target hashed locks + wheelhouse downloads.
    lock_records = []
    for target in LOCK_TARGETS:
        lock_name = f"requirements-{target['tag']}-linux-x86_64.lock"
        lock_path = locks / lock_name
        requirements_in = out / f"requirements-{target['tag']}.in"
        requirements_in.write_text(REQUIREMENTS_IN, encoding="utf-8")
        run(
            [
                "uv", "pip", "compile",
                str(requirements_in),
                "--generate-hashes",
                "--python-version", target["python"],
                "--python-platform", "x86_64-unknown-linux-gnu",
                "--no-annotate", "--quiet",
                "-o", str(lock_path),
            ]
        )
        lock_records.append({"lock": f"locks/{lock_name}", "python": target["python"], "platform": "linux-x86_64"})
        if not args.skip_download:
            ensure_pip()
            run(
                [
                    sys.executable, "-m", "pip", "download",
                    "-r", str(lock_path),
                    "-d", str(wheelhouse),
                    "--only-binary", ":all:",
                    *[arg for plat in PIP_PLATFORMS for arg in ("--platform", plat)],
                    "--python-version", target["python"].replace(".", ""),
                    "--implementation", "cp",
                    "--abi", target["tag"],
                    "--abi", "abi3",  # rapidfuzz & friends ship abi3 wheels
                    "--no-deps",
                ]
            )

    # 3. Tests and datasets from the release tree.
    import shutil

    # Mirror the repo layout so tests' Path(__file__)-relative data paths hold.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for source_dir, dest_dir in (
        (REPO / "tests", tests),
        (REPO / "evals" / "data", out / "evals" / "data"),
        (REPO / "evals" / "scenarios", out / "evals" / "scenarios"),
        (REPO / "evals" / "fixtures", out / "evals" / "fixtures"),
    ):
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir, ignore=ignore)

    # 4. Manifest.
    files = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name == "bundle-manifest.json" or path.name.endswith(".in") or path.suffix == ".pyc":
            continue
        files[path.relative_to(out).as_posix()] = sha256_file(path)
    manifest = {
        "bundle": f"voice-evals-v{version.replace('.', '')}-offline-bundle",
        "voice_evals_version": version,
        "published_wheel": published,
        "wheel": {
            "file": f"wheelhouse/{wheel_path.name}",
            "sha256": wheel_hash,
            **source,
        },
        "locks": lock_records,
        "git": git_tree_state(),
        "contents": {
            "tests": "copied from the release commit; run against the installed wheel (no source path injection)",
            "evals": "demo + frozen replay corpus + expected metrics + scenarios + fixtures (repo layout mirrored)",
        },
        "files": files,
        "build_host_python": sys.version.split()[0],
        "notes": (
            "Kernel installs with --no-index --find-links=wheelhouse --require-hashes. "
            "A local-wheel bundle is pre-release validation only."
        ),
    }
    (out / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"bundle staged at {out} (published_wheel={published})")
    return 0


def _current_version() -> str:
    for line in (REPO / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return "0.2.0"


if __name__ == "__main__":
    raise SystemExit(main())
