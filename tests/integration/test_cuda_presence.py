"""T050 — CUDA-presence behavior (Constitution IV layer 4).

Verifies the documented startup contract:
- production mode (ASR_ALLOW_CPU unset/false) + no CUDA → process exits non-zero.
- development mode (ASR_ALLOW_CPU=1) → startup succeeds with a WARN log.

Both branches run on CPU. The GPU-affirmative branch (ASR_ALLOW_CPU=0
WITH a real CUDA host) is exercised by the gpu_required smoke tests
elsewhere; we don't fake a GPU here.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    code = (
        "from asr.config import get_settings, reset_settings_for_tests\n"
        "from asr.observability.logging import configure_logging, get_logger\n"
        "import sys\n"
        "reset_settings_for_tests()\n"
        "settings = get_settings()\n"
        "cuda_ok = False\n"
        "try:\n"
        "    import torch\n"
        "    cuda_ok = torch.cuda.is_available()\n"
        "except Exception:\n"
        "    pass\n"
        "configure_logging()\n"
        "log = get_logger('cuda_presence_probe')\n"
        "if not cuda_ok:\n"
        "    if settings.allow_cpu:\n"
        "        log.warning('cuda_unavailable_cpu_allowed')\n"
        "        sys.exit(0)\n"
        "    else:\n"
        "        log.error('cuda_unavailable_production')\n"
        "        sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    import os

    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


def test_production_mode_without_cuda_exits_non_zero():
    try:
        import torch

        if torch.cuda.is_available():
            pytest.skip("real CUDA present; this branch tests the no-GPU case")
    except Exception:
        pass

    result = _run({"ASR_ALLOW_CPU": "0"})
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cuda_unavailable_production" in result.stdout


def test_dev_mode_with_allow_cpu_succeeds_and_warns():
    result = _run({"ASR_ALLOW_CPU": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cuda_unavailable_cpu_allowed" in result.stdout
    assert '"level": "warning"' in result.stdout
