# Kaggle reproduction bundle

The internet-disabled Kernel A installs everything from an immutable Kaggle
dataset (proposed slug `gjusev/voice-evals-v020-offline-bundle`) — never from
PyPI at runtime. This directory holds the committed lock files and the build
instructions; the dataset itself is staged by `scripts/build_kaggle_bundle.py`
on an internet-enabled release machine **after** the PyPI publication.

## Contents of the staged dataset

```
bundle-manifest.json        bundle id, wheel hash + provenance, lock list,
                            release commit, per-file sha256 of everything below
wheelhouse/*.whl            the exact published voice-evals wheel + all pinned
                            dependency wheels (base, probe, pytest, jsonschema)
locks/requirements-*.lock   per (python-minor, linux x86_64) hashed lock files
tests/                      tests from the exact release commit (run against the
                            installed wheel; no repository source on the path)
data/data/                  demo_calls.jsonl + replay_v02 corpus + expected
data/scenarios/             human-facing scenario copies
data/fixtures/              synthetic audio fixtures + traces
```

## Reproducible build

From a clean checkout of the release tag:

```bash
make build                                        # wheel + sdist
uv run python scripts/build_kaggle_bundle.py \
    --version 0.2.0 --output out/bundle
# pre-release validation only (never publish this as a reproduction):
uv run python scripts/build_kaggle_bundle.py \
    --version 0.2.0 --allow-local-wheel --output out/bundle-local
```

The script verifies the downloaded wheel's sha256 against the published PyPI
digests, resolves per-target locks with `uv pip compile --generate-hashes`,
downloads binary-only wheels with pip, copies tests/data from the release
tree, and writes the manifest. A `--allow-local-wheel` bundle records
`"published_wheel": false` and is validation-only.

## Publishing the dataset and kernels

```bash
KAGGLE_API_TOKEN=... python -m kaggle datasets create -p out/bundle \
    # then pin the dataset version used by the kernels' metadata
KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel/offline
KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/voice-evals-offline-benchmark
KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/voice-evals-offline-benchmark -p out/kaggle-offline/

KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel/probe
KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/voice-evals-live-probe
KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/voice-evals-live-probe -p out/kaggle-probe/
```

PowerShell equivalent: set the token once with `$env:KAGGLE_API_TOKEN="..."`,
then run the same commands without the Bash `VAR=...` prefix. A fork must
change the metadata `id` (and the commands' owner) to your own account.
`KAGGLE_API_TOKEN` is a publisher credential, not a kernel runtime secret.

## Verification performed before announcing reproduction

1. Bundle validated in a clean CPU environment with networking disabled and an
   empty pip cache: `--no-index --find-links --require-hashes` install only.
2. The actual Kaggle CPU kernel with `enable_internet=false` passes end to end
   (demo asserts, frozen corpus, mock probe, replay equality, test suite,
   provenance + artifact checksums under `/kaggle/working`).

Kernel runtime Python upgrades yield a clear unsupported-bundle error from the
kernel (it refuses to fetch online), not a silent behavior change.
