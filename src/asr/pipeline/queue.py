from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from asr.api.errors import ErrorCode, TranscriptionError
from asr.observability import metrics


class ModelQueue:
    """Per-model semaphore + bounded wait queue.

    Invariants:
      - At most one inference is in flight on a given model.
      - At most `depth` requests are *waiting* (not counting the in-flight
        one). The (depth + 1)th total reservation is rejected immediately
        with MODEL_BUSY. FR-017.
    """

    def __init__(self, model_id: str, depth: int) -> None:
        self.model_id = model_id
        self.depth = depth
        self._semaphore = asyncio.Semaphore(1)
        self._reserved = 0  # in-flight + currently-waiting
        self._lock = asyncio.Lock()

    async def _try_reserve(self) -> bool:
        async with self._lock:
            # Total in-system cap = 1 in-flight + `depth` waiting.
            if self._reserved >= self.depth + 1:
                return False
            self._reserved += 1
            # Reported queue depth excludes the in-flight slot when one is
            # held; if no slot is held yet, reserved == 1 and depth = 0.
            waiting = max(self._reserved - 1, 0)
            metrics.queue_depth.labels(model=self.model_id).set(waiting)
            return True

    async def _release_reservation(self) -> None:
        async with self._lock:
            self._reserved -= 1
            waiting = max(self._reserved - 1, 0)
            metrics.queue_depth.labels(model=self.model_id).set(waiting)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if not await self._try_reserve():
            metrics.queue_rejections_total.labels(model=self.model_id).inc()
            raise TranscriptionError(
                ErrorCode.MODEL_BUSY,
                f"model '{self.model_id}' queue is full ({self.depth})",
                details={"model": self.model_id, "queue_depth": self.depth},
            )
        try:
            async with self._semaphore:
                yield
        finally:
            await self._release_reservation()


class QueueRouter:
    def __init__(self, depth: int) -> None:
        self._depth = depth
        self._queues: dict[str, ModelQueue] = {}
        self._creation_lock = asyncio.Lock()

    async def for_model(self, model_id: str) -> ModelQueue:
        if model_id not in self._queues:
            async with self._creation_lock:
                if model_id not in self._queues:
                    self._queues[model_id] = ModelQueue(model_id, self._depth)
        return self._queues[model_id]
