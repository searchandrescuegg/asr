"""T052 — Performance regression suite (Constitution VI).

Asserts both:
  - the absolute SLOs from the constitution (p95 RTF ≤ 0.5, API overhead ≤
    100 ms, cold-start ≤ 60 s, ≤ 8 GB VRAM), AND
  - regression ≤ 10% versus the captured baseline in `tests/perf/baseline.json`.

Marked `slo` so it runs on the nightly GPU job, not the default CPU
test pass.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest

BASELINE = pathlib.Path(__file__).resolve().parent / "baseline.json"


def _load_baseline() -> dict:
    return json.loads(BASELINE.read_text())


@pytest.mark.slo
@pytest.mark.gpu_required
@pytest.mark.model_required
def test_slo_thresholds_and_regression():
    baseline = _load_baseline()
    captured = baseline["captured"]
    if captured.get("_status", "").startswith("PENDING"):
        pytest.skip(
            "tests/perf/baseline.json has no captured numbers yet — populate with a "
            "real GPU run, then remove the PENDING status string."
        )
    abs_thresholds = baseline["absolute_thresholds"]

    measured = _measure_slos()

    # Absolute checks.
    for model, rtf in measured["p95_rtf_per_model"].items():
        assert rtf <= abs_thresholds["p95_rtf"], (
            f"{model} p95 RTF {rtf:.3f} > absolute threshold {abs_thresholds['p95_rtf']}"
        )
    assert (
        measured["p95_api_overhead_seconds"] <= abs_thresholds["p95_api_overhead_seconds"]
    )
    for model, secs in measured["cold_start_seconds_per_model"].items():
        assert secs <= abs_thresholds["cold_start_seconds"], (
            f"{model} cold start {secs:.1f}s > {abs_thresholds['cold_start_seconds']}s"
        )
    assert measured["steady_state_vram_bytes"] <= abs_thresholds["steady_state_vram_bytes"]

    # Regression checks (≤ 10% worse than baseline).
    for model, rtf in measured["p95_rtf_per_model"].items():
        baseline_rtf = captured["p95_rtf_per_model"][model]
        assert rtf <= baseline_rtf * 1.10, (
            f"{model} p95 RTF regressed: {rtf:.3f} > {baseline_rtf:.3f} * 1.10"
        )
    assert (
        measured["p95_api_overhead_seconds"]
        <= captured["p95_api_overhead_seconds"] * 1.10
    )
    for model, secs in measured["cold_start_seconds_per_model"].items():
        assert secs <= captured["cold_start_seconds_per_model"][model] * 1.10
    assert (
        measured["steady_state_vram_bytes"]
        <= captured["steady_state_vram_bytes"] * 1.10
    )


def _measure_slos() -> dict:
    """Drive the standard reference set against each registered model.

    Implemented as a stub — the real harness loads each model, runs the
    fixture set from audio/REFERENCE.md, and computes percentiles.
    Filling this in is part of the operator runbook (see
    quickstart.md §7) once a GPU runner exists.
    """
    raise NotImplementedError(
        "_measure_slos must be implemented as part of the GPU-runner setup; "
        "see audio/REFERENCE.md and tests/perf/baseline.json"
    )


# Convenience for operators capturing a fresh baseline.
def capture_baseline() -> None:
    measured = _measure_slos()
    baseline = _load_baseline()
    baseline["captured"] = {
        **measured,
        "_status": "OK",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
