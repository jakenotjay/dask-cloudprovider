import pytest

import asyncio
import time
from unittest.mock import AsyncMock, patch

from dask_cloudprovider.generic.vmcluster import VMCluster, VMInterface


class DummyWorker(VMInterface):
    """A dummy worker for testing."""


class DummyScheduler(VMInterface):
    """A dummy scheduler for testing."""


class DummyCluster(VMCluster):
    """A dummy cluster for testing."""

    scheduler_class = DummyScheduler
    worker_class = DummyWorker


@pytest.mark.asyncio
async def test_init():
    with pytest.raises(RuntimeError):
        _ = VMCluster(asynchronous=True)


@pytest.mark.asyncio
async def test_call_async():
    cluster = DummyCluster(asynchronous=True)

    def blocking(string):
        time.sleep(0.1)
        return string

    start = time.time()

    a, b, c, d = await asyncio.gather(
        cluster.call_async(blocking, "hello"),
        cluster.call_async(blocking, "world"),
        cluster.call_async(blocking, "foo"),
        cluster.call_async(blocking, "bar"),
    )

    assert a == "hello"
    assert b == "world"
    assert c == "foo"
    assert d == "bar"

    # Each call to ``blocking`` takes 0.1 seconds, but they should've been run concurrently.
    assert time.time() - start < 0.2

    await cluster.close()


@pytest.mark.asyncio
async def test_start_waits_for_workers():
    """VMCluster._start() waits for workers after creating VMs."""
    cluster = DummyCluster(n_workers=3, asynchronous=True)
    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait, patch.object(
        VMCluster, "_start", wraps=cluster._start
    ):
        # _start calls super()._start() which needs a running scheduler;
        # mock the parent _start to avoid real SpecCluster setup, then
        # call the wait logic directly.
        original_start = VMCluster._start

        async def patched_start(self):
            # Skip SpecCluster._start but still run the wait logic
            if self._n_workers:
                await self._wait_for_workers(
                    self._n_workers, timeout=self._worker_timeout
                )

        with patch.object(VMCluster, "_start", patched_start):
            await cluster._start()

    mock_wait.assert_called_once_with(3, timeout="600s")
    await cluster.close()


@pytest.mark.asyncio
async def test_start_skips_wait_with_zero_workers():
    """VMCluster._start() skips waiting when n_workers=0."""
    cluster = DummyCluster(n_workers=0, asynchronous=True)

    async def patched_start(self):
        if self._n_workers:
            await self._wait_for_workers(
                self._n_workers, timeout=self._worker_timeout
            )

    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait, patch.object(VMCluster, "_start", patched_start):
        await cluster._start()

    mock_wait.assert_not_called()
    await cluster.close()


@pytest.mark.asyncio
async def test_custom_worker_timeout():
    """Custom worker_timeout is forwarded to _wait_for_workers."""
    cluster = DummyCluster(n_workers=1, worker_timeout=120, asynchronous=True)

    async def patched_start(self):
        if self._n_workers:
            await self._wait_for_workers(
                self._n_workers, timeout=self._worker_timeout
            )

    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait, patch.object(VMCluster, "_start", patched_start):
        await cluster._start()

    mock_wait.assert_called_once_with(1, timeout=120)
    await cluster.close()
