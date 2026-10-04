.PHONY: install test lint build

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

build:
	uv build
