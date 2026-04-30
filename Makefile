PY ?= python3.11

.PHONY: help install install-mac install-linux test test-gpu lint dev dev-mac \
        docker-cpu docker-gpu docker-build clean

help:
	@echo "Targets:"
	@echo "  install          alias for install-linux (uv sync from lockfile)"
	@echo "  install-linux    uv sync (uses uv.lock; Linux+CUDA target)"
	@echo "  install-mac      pip install -e . + dev/test deps (CPU torch)"
	@echo "  test             pytest with default markers (CPU-safe)"
	@echo "  test-gpu         pytest with gpu_required + model_required + slo"
	@echo "  lint             ruff check"
	@echo "  dev              uv run python -m asr (Linux/uv path)"
	@echo "  dev-mac          .venv/bin/python -m asr (Mac/pip path)"
	@echo "  docker-cpu       docker compose --profile cpu up --build"
	@echo "  docker-gpu       docker compose --profile gpu up --build"
	@echo "  docker-build     docker build -t asr:local ."

install: install-linux

install-linux:
	uv sync --group test

install-mac:
	test -d .venv || uv venv --python 3.11 .venv
	.venv/bin/python -m ensurepip --upgrade
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .
	.venv/bin/python -m pip install pytest 'pytest-asyncio>=0.24' 'httpx>=0.27' testcontainers ruff

test:
	@if [ -x .venv/bin/python ]; then .venv/bin/python -m pytest; else uv run pytest; fi

test-gpu:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m pytest -m "gpu_required or model_required or slo"; \
	else \
		uv run pytest -m "gpu_required or model_required or slo"; \
	fi

lint:
	uvx ruff check src/asr tests

dev:
	uv run python -m asr

dev-mac:
	ASR_ALLOW_CPU=1 ASR_ENABLED_MODELS= .venv/bin/python -m asr

docker-cpu:
	docker compose --profile cpu up --build

docker-gpu:
	docker compose --profile gpu up --build

docker-build:
	docker build -t asr:local .

clean:
	rm -rf .venv .ruff_cache .gradio __pycache__ */__pycache__ */*/__pycache__
