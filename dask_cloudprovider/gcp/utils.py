import asyncio
import logging

import aiohttp
import httplib2
import googleapiclient.http
import google_auth_httplib2
from distributed.diagnostics.plugin import WorkerPlugin
from tornado.ioloop import IOLoop

logger = logging.getLogger(__name__)

GCP_PREEMPTED_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
)


def build_request(credentials=None):
    def inner(http, *args, **kwargs):
        new_http = httplib2.Http()
        if credentials is not None:
            new_http = google_auth_httplib2.AuthorizedHttp(credentials, http=new_http)

        return googleapiclient.http.HttpRequest(new_http, *args, **kwargs)

    return inner


def is_inside_gce() -> bool:
    """
    Returns True is the client is running in the GCE environment,
    False otherwise.

    Doc: https://cloud.google.com/compute/docs/storing-retrieving-metadata
    """
    h = httplib2.Http()
    try:
        resp_headers, _ = h.request(
            "http://metadata.google.internal/computeMetadata/v1/",
            headers={"metadata-flavor": "Google"},
            method="GET",
        )
    except (httplib2.HttpLib2Error, OSError):
        return False
    return True


class GCPPreemptibleWorkerPlugin(WorkerPlugin):
    """A worker plugin for GCP Spot VMs.

    Monitors the GCP metadata service for preemption notifications using
    the efficient ``wait_for_change=true`` long-poll endpoint.  When
    preemption is detected, the plugin calls ``worker.close_gracefully()``
    to stop accepting new tasks, allow in-flight tasks to complete, and
    migrate intermediate data to other workers before the VM is terminated.

    GCP provides a best-effort window of ~30 seconds between the preemption
    signal and forced termination.  ``close_gracefully()`` will attempt to
    complete within that window, but tasks with very large intermediate
    state may not fully migrate in time.

    This plugin can be used on any Dask worker running on a GCP Spot VM,
    not just those created by ``dask-cloudprovider``.

    For more details on GCP Spot VMs see:
    https://cloud.google.com/compute/docs/instances/spot

    Parameters
    ----------
    metadata_url : str, optional
        The URL of the GCP metadata preemption endpoint.

        Defaults to
        ``"http://metadata.google.internal/computeMetadata/v1/instance/preempted"``

    poll_timeout_s : int, optional
        Timeout in seconds for each long-poll request to the metadata
        service.  When the timeout expires without a preemption signal
        the request is retried automatically.  Should exceed the GCP
        metadata server's own hold time (~60s) to avoid spurious
        client-side timeouts.

        Defaults to ``90``

    Examples
    --------

    Register the plugin on any Dask cluster whose workers run on GCP
    Spot VMs:

    >>> from distributed import Client
    >>> client = Client("<Any Dask cluster running on GCP Spot VMs>")

    >>> from dask_cloudprovider.gcp import GCPPreemptibleWorkerPlugin
    >>> client.register_worker_plugin(GCPPreemptibleWorkerPlugin())
    """

    def __init__(self, metadata_url=None, poll_timeout_s=90):
        self.metadata_url = metadata_url or GCP_PREEMPTED_METADATA_URL
        self.poll_timeout_s = poll_timeout_s
        self.worker = None
        self.terminating = False
        self._task = None
        self._session = None

    async def _watch_for_preemption(self):
        """Long-poll the metadata service until preemption is signaled."""
        self._session = aiohttp.ClientSession(
            headers={"Metadata-Flavor": "Google"},
            timeout=aiohttp.ClientTimeout(total=self.poll_timeout_s),
        )
        try:
            while not self.terminating:
                try:
                    async with self._session.get(
                        self.metadata_url,
                        params={"wait_for_change": "true"},
                    ) as response:
                        text = await response.text()
                        if text.strip() == "TRUE":
                            logger.info(
                                "Worker %s: GCP preemption detected, "
                                "attempting graceful shutdown",
                                self.worker.name,
                            )
                            self.terminating = True
                            await self.worker.close_gracefully()
                            return
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.warning(
                        "Worker %s: error polling GCP preemption metadata, "
                        "retrying in 1s",
                        self.worker.name,
                        exc_info=True,
                    )
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            return
        finally:
            if self._session and not self._session.closed:
                await self._session.close()

    def setup(self, worker):
        self.worker = worker
        if not is_inside_gce():
            logger.warning(
                "Worker %s: GCPPreemptibleWorkerPlugin registered but "
                "this instance does not appear to be running on GCE. "
                "Preemption monitoring will likely fail.",
                worker.name,
            )
        loop = IOLoop.current()
        loop.add_callback(self._start_watching)
        logger.debug(
            "Worker %s: registered GCP preemption plugin", worker.name
        )

    def _start_watching(self):
        """Callback to create the watch task on the event loop."""
        self._task = asyncio.ensure_future(self._watch_for_preemption())

    def teardown(self, worker):
        logger.debug("Worker %s: tearing down GCP preemption plugin", worker.name)
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        session = self._session
        if session and not session.closed:
            loop = IOLoop.current()
            loop.add_callback(
                lambda: asyncio.ensure_future(session.close())
            )
