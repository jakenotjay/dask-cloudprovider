import pytest

import dask
from dask_cloudprovider.gcp.instances import (
    GCPCluster,
    GCPCompute,
    GCPCredentialsError,
    GCPInstance,
)
from dask.distributed import Client
from distributed.core import Status


def skip_without_credentials():
    try:
        _ = GCPCompute()
    except GCPCredentialsError:
        pytest.skip(
            """
        You must configure your GCP credentials to run this test.

            $ google auth login

            or

            $ export GOOGLE_APPLICATION_CREDENTIALS=<path-to-gcp-json-credentials>

        """
        )

    if not dask.config.get("cloudprovider.gcp.projectid"):
        pytest.skip(
            """
        You must configure your Google project ID to run this test.

            # ~/.config/dask/cloudprovider.yaml
            cloudprovider:
              gcp:
                projectid: "YOUR PROJECT ID"

            or

            $ export DASK_CLOUDPROVIDER__GCP__PROJECTID="YOUR PROJECT ID"

        """
        )


@pytest.mark.asyncio
async def test_init():
    skip_without_credentials()

    cluster = GCPCluster(asynchronous=True)
    assert cluster.status == Status.created


@pytest.mark.asyncio
async def test_init_gpu_config_defaults():
    """Regression test for https://github.com/dask/dask-cloudprovider/pull/479."""
    skip_without_credentials()

    cluster = GCPCluster(asynchronous=True)
    assert cluster.ngpus is None
    assert cluster.scheduler_ngpus == 0
    assert cluster.worker_ngpus == 0


@pytest.mark.asyncio
async def test_get_cloud_init():
    skip_without_credentials()
    cloud_init = GCPCluster.get_cloud_init(
        security=True,
        docker_args="--privileged",
        extra_bootstrap=["gcloud auth print-access-token"],
    )
    assert "dask-scheduler" in cloud_init
    assert "# Bootstrap" in cloud_init
    assert " --privileged " in cloud_init
    assert "- gcloud auth print-access-token" in cloud_init


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
@pytest.mark.external
async def test_create_cluster():
    skip_without_credentials()

    async with GCPCluster(
        asynchronous=True, env_vars={"FOO": "bar"}, security=True
    ) as cluster:
        assert cluster.status == Status.running

        cluster.scale(2)
        await cluster
        assert len(cluster.workers) == 2

        async with Client(cluster, asynchronous=True) as client:

            def inc(x):
                return x + 1

            def check_env():
                import os

                return os.environ["FOO"]

            assert await client.submit(inc, 10).result() == 11
            assert await client.submit(check_env).result() == "bar"


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
@pytest.mark.external
async def test_create_spot_cluster():
    skip_without_credentials()

    async with GCPCluster(
        asynchronous=True, spot=True, security=True
    ) as cluster:
        assert cluster.status == Status.running

        cluster.scale(1)
        await cluster
        assert len(cluster.workers) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
@pytest.mark.external
async def test_create_cluster_sync():
    skip_without_credentials()

    cluster = GCPCluster(n_workers=1)
    client = Client(cluster)

    def inc(x):
        return x + 1

    assert client.submit(inc, 10).result() == 11


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
@pytest.mark.external
async def test_create_rapids_cluster():
    skip_without_credentials()

    async with GCPCluster(
        source_image="projects/nv-ai-infra/global/images/ngc-docker-11-20200916",
        zone="us-east1-c",
        machine_type="n1-standard-1",
        filesystem_size=50,
        ngpus=2,
        gpu_type="nvidia-tesla-t4",
        docker_image="rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.9",
        worker_class="dask_cuda.CUDAWorker",
        worker_options={"rmm_pool_size": "15GB"},
        asynchronous=True,
        auto_shutdown=True,
        bootstrap=False,
    ) as cluster:
        assert cluster.status == Status.running

        cluster.scale(1)

        await cluster

        assert len(cluster.workers) == 1

        client = Client(cluster, asynchronous=True)  # noqa
        await client

        def gpu_mem():
            from pynvml.smi import nvidia_smi

            nvsmi = nvidia_smi.getInstance()
            return nvsmi.DeviceQuery("memory.free, memory.total")

        results = await client.run(gpu_mem)
        for w, res in results.items():
            assert "total" in res["gpu"][0]["fb_memory_usage"].keys()
            print(res)


@pytest.mark.timeout(1200)
@pytest.mark.external
def test_create_rapids_cluster_sync():
    skip_without_credentials()
    cluster = GCPCluster(
        source_image="projects/nv-ai-infra/global/images/packer-1607527229",
        network="dask-gcp-network-test",
        zone="us-east1-c",
        machine_type="n1-standard-1",
        filesystem_size=50,
        ngpus=2,
        gpu_type="nvidia-tesla-t4",
        docker_image="rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.9",
        worker_class="dask_cuda.CUDAWorker",
        worker_options={"rmm_pool_size": "15GB"},
        asynchronous=False,
        bootstrap=False,
    )

    cluster.scale(1)

    client = Client(cluster)  # noqa
    client.wait_for_workers(2)

    def gpu_mem():
        from pynvml.smi import nvidia_smi

        nvsmi = nvidia_smi.getInstance()
        return nvsmi.DeviceQuery("memory.free, memory.total")

    results = client.run(gpu_mem)
    for w, res in results.items():
        assert "total" in res["gpu"][0]["fb_memory_usage"].keys()
        print(res)
    cluster.close()


# --- Spot VM scheduling config unit tests (no GCP credentials needed) ---


def test_build_scheduling_config_spot():
    """SPOT scheduling config has correct provisioningModel and constraints."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.spot = True
    instance.instance_termination_action = "DELETE"
    instance.on_host_maintenance = "TERMINATE"

    config = instance._build_scheduling_config()
    assert config["provisioningModel"] == "SPOT"
    assert config["instanceTerminationAction"] == "DELETE"
    assert config["preemptible"] is True
    assert config["automaticRestart"] is False
    assert config["onHostMaintenance"] == "TERMINATE"


def test_build_scheduling_config_standard():
    """STANDARD scheduling config has no instanceTerminationAction."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.spot = False
    instance.on_host_maintenance = "TERMINATE"

    config = instance._build_scheduling_config()
    assert config["provisioningModel"] == "STANDARD"
    assert config["preemptible"] is False
    assert config["automaticRestart"] is True
    assert config["onHostMaintenance"] == "TERMINATE"
    assert "instanceTerminationAction" not in config


def test_build_scheduling_config_stop_action():
    """instanceTerminationAction=STOP is respected."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.spot = True
    instance.instance_termination_action = "STOP"
    instance.on_host_maintenance = "TERMINATE"

    config = instance._build_scheduling_config()
    assert config["instanceTerminationAction"] == "STOP"


@pytest.mark.asyncio
async def test_spot_default_is_false():
    """Spot defaults to False when no spot config is provided."""
    skip_without_credentials()

    cluster = GCPCluster(asynchronous=True)
    assert cluster.worker_options.get("spot") in (None, False)


@pytest.mark.asyncio
async def test_spot_true_passed_to_workers():
    """spot=True flows through to worker options."""
    skip_without_credentials()

    cluster = GCPCluster(asynchronous=True, spot=True)
    assert cluster.worker_options["spot"] is True


# --- COS image detection unit tests ---


def test_is_cos_image_cos_cloud():
    """COS images from cos-cloud project are detected."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.source_image = "projects/cos-cloud/global/images/family/cos-125-lts"
    assert instance._is_cos_image() is True


def test_is_cos_image_cos_stable():
    """COS stable image family is detected."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.source_image = "projects/cos-cloud/global/images/cos-stable-121-18867-381-56"
    assert instance._is_cos_image() is True


def test_is_cos_image_ubuntu():
    """Ubuntu images are not detected as COS."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.source_image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
    assert instance._is_cos_image() is False


# --- Startup script rendering unit tests ---


def test_render_startup_script_no_bootstrap():
    """COS images skip Docker installation in startup script."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = True
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "#!/bin/bash" in script
    assert "docker run" in script
    assert "daskdev/dask:latest" in script
    assert "apt-get" not in script
    assert "curl -fsSL https://get.docker.com" not in script


def test_render_startup_script_with_bootstrap():
    """Ubuntu images include Docker installation in startup script."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = True
    instance.auto_shutdown = True
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "#!/bin/bash" in script
    assert "docker run" in script
    assert "curl -fsSL https://get.docker.com" in script


def test_render_startup_script_with_env_vars():
    """Environment variables are passed to the docker run command."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = False
    instance.env_vars = {"FOO": "bar", "OMP_NUM_THREADS": "4"}

    script = instance.render_startup_script()
    assert "FOO=bar" in script
    assert "OMP_NUM_THREADS=4" in script
    assert "shutdown" not in script


def test_render_startup_script_env_vars_shell_escaped():
    """Environment variable values with shell metacharacters are escaped."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = False
    instance.env_vars = {"EVIL": '"; rm -rf / #'}

    script = instance.render_startup_script()
    # shlex.quote wraps the value in single quotes, preventing shell interpretation
    assert """EVIL='"; rm -rf / #'""" in script
    # The unquoted form (which would allow injection) must not appear
    assert 'EVIL="; rm -rf / #' not in script


def test_render_startup_script_with_extra_bootstrap():
    """Extra bootstrap commands are included in the startup script."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = ["echo hello", "whoami"]
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = False
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "echo hello" in script
    assert "whoami" in script
