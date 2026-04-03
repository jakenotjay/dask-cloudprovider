import pytest

import asyncio
import time
from unittest.mock import AsyncMock, patch

from distributed.deploy.spec import SpecCluster

from distributed.core import Status

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


class _FakeScheduler:
    """Stand-in for a scheduler ProcessInterface: awaitable and closable."""
    def __await__(self):
        yield  # make it a generator so await works
    async def close(self):
        pass


def _patch_cluster_for_await(cluster):
    """Mock internals so __await__ runs without real cluster infrastructure."""
    cluster.status = Status.running  # skip _start()
    cluster.scheduler = _FakeScheduler()
    cluster._correct_state = AsyncMock()  # skip worker VM creation
    cluster.workers = {}  # no real workers to await


@pytest.mark.asyncio
async def test_await_waits_for_workers():
    """VMCluster.__await__ waits for workers after VMs are created."""
    cluster = DummyCluster(n_workers=3, asynchronous=True)
    _patch_cluster_for_await(cluster)
    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait:
        await cluster

    mock_wait.assert_called_once_with(3, timeout="600s")
    await cluster.close()


@pytest.mark.asyncio
async def test_await_skips_wait_with_zero_workers():
    """VMCluster.__await__ skips waiting when n_workers=0."""
    cluster = DummyCluster(n_workers=0, asynchronous=True)
    _patch_cluster_for_await(cluster)
    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait:
        await cluster

    mock_wait.assert_not_called()
    await cluster.close()


@pytest.mark.asyncio
async def test_custom_worker_timeout():
    """Custom worker_timeout is forwarded to _wait_for_workers."""
    cluster = DummyCluster(n_workers=1, worker_timeout=120, asynchronous=True)
    _patch_cluster_for_await(cluster)
    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ) as mock_wait:
        await cluster

    mock_wait.assert_called_once_with(1, timeout=120)
    await cluster.close()


class _FakeWorker:
    """Stand-in for a worker ProcessInterface: awaitable."""
    def __init__(self):
        self.awaited = False

    def __await__(self):
        async def _mark():
            self.awaited = True
        return _mark().__await__()


@pytest.mark.asyncio
async def test_await_with_existing_workers():
    """VMCluster.__await__ correctly awaits workers in self.workers dict."""
    cluster = DummyCluster(n_workers=2, asynchronous=True)
    cluster.status = Status.running
    cluster.scheduler = _FakeScheduler()
    cluster._correct_state = AsyncMock()

    w1, w2 = _FakeWorker(), _FakeWorker()
    cluster.workers = {"w1": w1, "w2": w2}

    with patch.object(
        VMCluster, "_wait_for_workers", new_callable=AsyncMock
    ):
        await cluster

    assert w1.awaited
    assert w2.awaited
    await cluster.close()
