import pytest

import dask
from dask_cloudprovider.gcp.instances import (
    GCPCluster,
    GCPCompute,
    GCPCredentialsError,
    GCPInstance,
    GCPWorker,
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
@pytest.mark.timeout(600)
@pytest.mark.external
async def test_single_worker_joins_cluster():
    """Smoke test: one scheduler + one worker, verify the worker registers."""
    skip_without_credentials()

    async with GCPCluster(
        n_workers=1,
        asynchronous=True,
        security=True,
    ) as cluster:
        assert cluster.status == Status.running

        async with Client(cluster, asynchronous=True) as client:
            await client.wait_for_workers(1, timeout=300)
            info = client.scheduler_info()
            assert len(info["workers"]) == 1


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
    assert "_self_delete" in script
    assert "metadata.google.internal" in script


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


def test_render_startup_script_invalid_env_key():
    """Invalid env var key names raise ValueError."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "daskdev/dask:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = False
    instance.env_vars = {"INVALID KEY": "value"}

    with pytest.raises(ValueError, match="Invalid environment variable name"):
        instance.render_startup_script()


def test_render_startup_script_ar_image_has_auth():
    """AR images configure docker-credential-gcr for the registry hostname."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "europe-west2-docker.pkg.dev/my-project/my-repo/my-image:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = True
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "docker-credential-gcr configure-docker" in script
    assert "europe-west2-docker.pkg.dev" in script
    assert "HOME=/home/chronos" in script


def test_render_startup_script_dockerhub_image_no_auth():
    """Docker Hub images skip private registry auth configuration."""
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
    assert "docker-credential-gcr" not in script
    assert "configure-docker" not in script


def test_render_startup_script_non_gcp_registry_no_auth():
    """Non-GCP private registries (quay.io, etc.) skip GCP credential config."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "quay.io/org/my-image:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = False
    instance.auto_shutdown = True
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "docker-credential-gcr" not in script
    assert "configure-docker" not in script


def test_render_startup_script_bootstrap_ar_image_installs_credential_helper():
    """Bootstrap (non-COS) + AR image installs docker-credential-gcr."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = "europe-west2-docker.pkg.dev/my-project/my-repo/my-image:latest"
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = True
    instance.auto_shutdown = True
    instance.env_vars = {}

    script = instance.render_startup_script()
    assert "docker-credential-gcr configure-docker" in script
    assert "curl" in script and "docker-credential-gcr" in script
    assert "europe-west2-docker.pkg.dev" in script


def _make_instance(**overrides):
    """Helper to create a GCPInstance through __init__ with a mock cluster."""
    from unittest.mock import MagicMock

    mock_cluster = MagicMock()
    mock_cluster.uuid = "test-uuid"
    config = dask.config.get("cloudprovider.gcp", {})

    return GCPInstance(
        cluster=mock_cluster,
        config=config,
        **overrides,
    )


def test_network_tags_default():
    """Default network tags include http-server and https-server."""
    instance = _make_instance()
    assert instance.network_tags == ["http-server", "https-server"]


def test_network_tags_custom():
    """Custom network tags are used when provided."""
    instance = _make_instance(network_tags=["dask-scheduler"])
    assert instance.network_tags == ["dask-scheduler"]


def test_network_tags_empty_list():
    """Empty list explicitly disables all network tags."""
    instance = _make_instance(network_tags=[])
    assert instance.network_tags == []


def test_public_ingress_false():
    """Passing public_ingress=False is not silently ignored."""
    instance = _make_instance(public_ingress=False)
    assert instance.public_ingress is False


def test_preemptible_deprecated_maps_to_spot():
    """preemptible=True emits FutureWarning and sets spot=True."""
    with pytest.warns(FutureWarning):
        instance = _make_instance(preemptible=True)
    assert instance.spot is True


def _make_render_instance(bootstrap=False, **overrides):
    """Helper to create a GCPInstance for startup script rendering tests."""
    instance = GCPInstance.__new__(GCPInstance)
    instance.docker_image = overrides.get("docker_image", "daskdev/dask:latest")
    instance.command = "python -m distributed.cli.dask_scheduler"
    instance.docker_args = ""
    instance.extra_bootstrap = None
    instance.gpu_instance = False
    instance.bootstrap = bootstrap
    instance.auto_shutdown = True
    instance.env_vars = {}
    instance.port = overrides.get("port", 8786)
    instance._scheduler_options = overrides.get("_scheduler_options", {})
    return instance


def test_render_startup_script_cos_has_iptables():
    """COS (non-bootstrap) images open firewall for VPC and scheduler ports."""
    script = _make_render_instance().render_startup_script()
    # Guard clause checks for iptables and DROP policy
    assert "command -v iptables" in script
    assert '"-P INPUT DROP"' in script
    # VPC CIDR rule for internal Dask traffic
    assert "iptables -A INPUT -s 10.128.0.0/9 -j ACCEPT" in script
    # Scheduler/dashboard ports for external access
    assert "iptables -A INPUT -p tcp --dport 8786" in script
    assert "iptables -A INPUT -p tcp --dport 8787" in script


def test_render_startup_script_cos_custom_port_iptables():
    """Custom scheduler port propagates to iptables rules on COS."""
    instance = _make_render_instance(
        port=9786, _scheduler_options={"dashboard_address": ":9999"}
    )
    script = instance.render_startup_script()
    assert "iptables -A INPUT -p tcp --dport 9786" in script
    assert "iptables -A INPUT -p tcp --dport 9999" in script


def test_render_startup_script_bootstrap_no_iptables():
    """Bootstrap (Ubuntu) images do not modify iptables."""
    script = _make_render_instance(bootstrap=True).render_startup_script()
    assert "iptables" not in script


def test_build_scheduling_config_invalid_termination_action():
    """Invalid instance_termination_action raises ValueError during init."""
    from unittest.mock import MagicMock

    mock_cluster = MagicMock()
    mock_cluster.uuid = "test-uuid"
    config = dask.config.get("cloudprovider.gcp", {})

    with pytest.raises(ValueError, match="instance_termination_action must be"):
        GCPInstance(
            cluster=mock_cluster,
            config=config,
            instance_termination_action="RESTART",
        )


def _make_worker(**overrides):
    """Helper to create a GCPWorker with a mock cluster."""
    from unittest.mock import MagicMock

    mock_cluster = MagicMock()
    mock_cluster.uuid = "test-uuid"
    mock_cluster.protocol = "tcp"
    mock_cluster.scheduler_internal_ip = "10.128.0.5"
    mock_cluster.scheduler_port = 8786
    config = dask.config.get("cloudprovider.gcp", {})

    return GCPWorker(
        scheduler="tcp://35.200.0.1:8786",
        cluster=mock_cluster,
        config=config,
        worker_class="distributed.cli.dask_worker",
        **overrides,
    )


def test_worker_uses_internal_scheduler_address():
    """GCPWorker connects to internal scheduler IP, not the external address."""
    worker = _make_worker()
    assert worker.scheduler == "tcp://10.128.0.5:8786"
    assert "10.128.0.5" in worker.command
    assert "35.200.0.1" not in worker.command


def test_worker_command_built_by_mixin():
    """WorkerMixin sets self.command with dask_spec and worker name."""
    worker = _make_worker()
    assert worker.command is not None
    assert "distributed.cli.dask_spec" in worker.command
    assert worker.name in worker.command
    assert worker.name.startswith("dask-test-uuid-worker-")


def test_worker_options_forwarded():
    """worker_options are embedded in the dask_spec --spec JSON."""
    worker = _make_worker(worker_options={"nthreads": 4})
    assert '"nthreads": 4' in worker.command


def test_worker_startup_script_has_shell_quoted_spec():
    """render_startup_script converts YAML ''...'' quoting to shell '...' quoting."""
    worker = _make_worker()
    script = worker.render_startup_script()
    # The spec JSON should be shell-quoted with single quotes, not YAML ''...''
    assert "''\"" not in script, "YAML double-single-quote escaping leaked into bash"
    # Should have proper shell quoting: '{"cls": ...}'
    assert """--spec '{"cls":""" in script


def test_worker_requires_cluster_kwarg():
    """GCPWorker raises ValueError when cluster kwarg is missing."""
    with pytest.raises(ValueError, match="requires a 'cluster'"):
        GCPWorker(scheduler="tcp://10.0.0.1:8786")
