.PHONY: install test test-integration lint build demo-probe kaggle-bundle

install:
	uv sync --extra probe

test:
	uv run pytest -q -m "not live"

test-integration:
	uv run pytest -m live

lint:
	uv run ruff check src tests scripts kaggle-kernel examples

build:
	uv build

# Offline four-turn mock probe: no credentials, no network, ~25s real time.
demo-probe:
	uv run voice-eval probe src/voice_evals/resources/scenarios/appointment-v2.json \
		--mock --output-dir out/probe-demo \
		--max-wer 0.05 --max-barge-in-stop-ms 500

# Stage the internet-disabled Kaggle bundle (release machine; see
# kaggle-kernel/bundle/README.md). Validation-only variant uses the local wheel:
kaggle-bundle:
	uv build
	uv run python scripts/build_kaggle_bundle.py --allow-local-wheel --output out/bundle-local
	@echo "NOTE: local-wheel bundles are pre-release validation only; the published"
	@echo "reproduction requires the PyPI wheel (drop --allow-local-wheel)."
