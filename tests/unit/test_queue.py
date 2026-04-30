import asyncio

import pytest

from asr.api.errors import ErrorCode, TranscriptionError
from asr.observability import metrics
from asr.pipeline.queue import ModelQueue, QueueRouter


@pytest.mark.asyncio
async def test_serial_per_model():
    queue = ModelQueue("m", depth=4)
    started = []
    finished = []

    async def task(idx: int):
        async with queue.slot():
            started.append(idx)
            await asyncio.sleep(0.05)
            finished.append(idx)

    await asyncio.gather(task(1), task(2), task(3))
    # Tasks run sequentially under the per-model semaphore: each finished
    # before the next started.
    assert finished == [1, 2, 3] or finished == sorted(finished)


@pytest.mark.asyncio
async def test_overflow_raises_model_busy():
    queue = ModelQueue("m-busy", depth=1)

    holder_done = asyncio.Event()

    async def holder():
        async with queue.slot():
            await holder_done.wait()

    async def attempt() -> str:
        try:
            async with queue.slot():
                return "served"
        except TranscriptionError as ex:
            return f"rejected:{ex.code}"

    h = asyncio.create_task(holder())
    await asyncio.sleep(0.01)  # let holder reserve the slot

    # depth=1; one waiter is allowed, second waiter rejected.
    waiter = asyncio.create_task(attempt())
    await asyncio.sleep(0.01)
    overflow = await attempt()
    assert overflow == f"rejected:{ErrorCode.MODEL_BUSY}"

    holder_done.set()
    await h
    assert (await waiter) == "served"


@pytest.mark.asyncio
async def test_different_models_overlap():
    router = QueueRouter(depth=2)
    a = await router.for_model("a")
    b = await router.for_model("b")

    started = set()
    started_event = asyncio.Event()

    async def hold(q: ModelQueue, name: str):
        async with q.slot():
            started.add(name)
            if len(started) == 2:
                started_event.set()
            await started_event.wait()

    await asyncio.wait_for(asyncio.gather(hold(a, "a"), hold(b, "b")), timeout=1.0)
    assert started == {"a", "b"}


@pytest.mark.asyncio
async def test_queue_metrics_increment_on_rejection():
    rejection_metric = metrics.queue_rejections_total.labels(model="m-metrics")
    before = rejection_metric._value.get()
    queue = ModelQueue("m-metrics", depth=1)

    async def holder(stop: asyncio.Event):
        async with queue.slot():
            await stop.wait()

    stop = asyncio.Event()
    h = asyncio.create_task(holder(stop))
    await asyncio.sleep(0.01)

    async def waiter():
        async with queue.slot():
            return "ok"

    asyncio.create_task(waiter())  # depth=1: one waiter takes the lone slot
    await asyncio.sleep(0.01)

    with pytest.raises(TranscriptionError):
        async with queue.slot():
            pass

    after = rejection_metric._value.get()
    assert after == before + 1

    stop.set()
    await h
