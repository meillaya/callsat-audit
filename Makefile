SHELL := /bin/bash
CURDIR := $(shell pwd)

export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python
export UV_TOOL_DIR := $(CURDIR)/.uv-tools
export MPLCONFIGDIR := $(CURDIR)/.mpl-cache
export HF_HOME := $(CURDIR)/.hf-cache
export JUPYTER_DATA_DIR := $(CURDIR)/.jupyter-data
export JUPYTER_CONFIG_DIR := $(CURDIR)/.jupyter-config
export JUPYTER_RUNTIME_DIR := $(CURDIR)/.jupyter-runtime
export JUPYTER_PATH := $(CURDIR)/.venv/share/jupyter

.PHONY: setup data summary notebooks run pilot test lint format typecheck all clean

setup:
	uv sync
	uv pip install -e .

data:
	uv run callsat data --force

summary:
	uv run callsat summary

notebooks:
	uv run python scripts/build_notebooks.py

run:
	uv run python scripts/run_notebooks.py

pilot:
	uv run --extra llm python scripts/run_llm_pilot.py --n 40

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

typecheck:
	uvx ty check src

all: notebooks run test lint typecheck

clean:
	rm -rf results/cache results/data/*.jsonl .pytest_cache .ruff_cache
