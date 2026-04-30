"""T049 — Containerized end-to-end (Constitution IV layer 3).

Builds the image from the repo's Dockerfile via testcontainers, runs it
with ASR_ALLOW_CPU=1 (no GPU on most CI runners), and exercises the API
end-to-end. This is gated behind `gpu_required` AND `model_required`
in practice — the CPU run is too slow to load real Parakeet/Seamless
weights, but a container that boots with `ASR_ENABLED_MODELS=` (no
real models) provides a structural smoke test that the build, package
layout, and entrypoint are correct.
"""

from __future__ import annotations

import shutil

import httpx
import pytest

testcontainers = pytest.importorskip("testcontainers.core.container")
DockerContainer = testcontainers.DockerContainer


@pytest.mark.gpu_required
def test_containerized_e2e_smoke():
    if shutil.which("docker") is None:
        pytest.skip("docker not available on this host")

    image = "asr-e2e:test"
    # Build the production image in-place (testcontainers doesn't build for us).
    import subprocess

    subprocess.run(
        ["docker", "build", "-t", image, "."], check=True
    )

    container = (
        DockerContainer(image)
        .with_env("ASR_ALLOW_CPU", "1")
        .with_env("ASR_ENABLED_MODELS", "")  # boot without loading real weights
        .with_exposed_ports(8000)
    )
    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(8000)
        base = f"http://{host}:{port}"

        # Wait for /healthz to come up (model load not required).
        for _ in range(60):
            try:
                r = httpx.get(f"{base}/api/v1/healthz", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            import time

            time.sleep(1.0)
        else:
            pytest.fail("server did not become healthy within 60s")

        # /api/v1/models must respond even with zero models registered.
        r = httpx.get(f"{base}/api/v1/models", timeout=5.0)
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
