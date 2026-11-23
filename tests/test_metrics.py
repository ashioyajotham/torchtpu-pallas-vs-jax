import numpy as np

from torch_tpu_vs_jax_pallas.config import KernelConfig
from torch_tpu_vs_jax_pallas.metrics import (
    V5E_BF16_PEAK_TFLOPS,
    attention_flops,
    error_metrics,
    latency_summary,
    passes_tolerance,
    v5e_mfu,
)


def test_error_metrics():
    metrics = error_metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, 1.5]))
    assert metrics.max_abs == 0.5
    assert metrics.mean_abs == 0.25
    assert metrics.finite


def test_strict_gate_enforces_max_abs_even_when_relative_tolerance_passes():
    expected = np.asarray([100.0], dtype=np.float32)
    actual = np.asarray([100.06], dtype=np.float32)
    assert np.allclose(actual, expected, atol=0.05, rtol=0.05)
    assert not passes_tolerance(actual, expected, atol=0.05, rtol=0.05)


def test_nonfinite_values_fail_tolerance():
    assert not passes_tolerance(
        np.asarray([np.inf]), np.asarray([1.0]), atol=0.05, rtol=0.05
    )


def test_causal_reports_executed_and_useful_flops_separately():
    config = KernelConfig(causal=True)
    executed = attention_flops(config, useful_causal=False)
    useful = attention_flops(config, useful_causal=True)
    assert executed == 2 * useful


def test_v5e_mfu_uses_single_chip_peak():
    assert V5E_BF16_PEAK_TFLOPS == 197.0
    assert v5e_mfu(98.5) == 0.5


def test_latency_summary_has_percentiles():
    summary = latency_summary([1.0, 2.0, 3.0])
    assert summary["trials"] == 3
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] >= summary["p50_ms"]
