from __future__ import annotations

import sys

import uvicorn

from asr.config import get_settings
from asr.observability.logging import configure_logging, get_logger


def main() -> None:
    configure_logging()
    log = get_logger("asr.__main__")
    settings = get_settings()

    cuda_ok = False
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception as ex:
        log.error("torch_import_failed", exc_type=type(ex).__name__)

    if not cuda_ok:
        if settings.allow_cpu:
            log.warning(
                "cuda_unavailable_cpu_allowed",
                reason="ASR_ALLOW_CPU=1 — running on CPU; performance SLOs do not apply",
            )
        else:
            log.error(
                "cuda_unavailable_production",
                reason="set ASR_ALLOW_CPU=1 to permit CPU operation in dev/test",
            )
            sys.exit(1)

    uvicorn.run(
        "asr.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
