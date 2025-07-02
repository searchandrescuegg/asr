
# Stage 1: Base image with system dependencies
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    python3-dev \
    g++ \
    clang \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc
ENV CXX=g++

COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"

WORKDIR /app

COPY uv.lock pyproject.toml /app/

RUN uv python install

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 3: Final image
FROM base AS final

COPY --from=builder /app /app
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
# Run the application
CMD ["fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]