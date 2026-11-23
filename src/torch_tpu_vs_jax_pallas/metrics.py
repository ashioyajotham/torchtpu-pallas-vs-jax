from __future__ import annotations

from dataclasses import dataclass

import numpy as np


V5E_BF16_PEAK_TFLOPS = 197.0
V5E_SPEC_URL = "https://docs.cloud.google.com/tpu/docs/v5e"


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    max_rel: float
    finite: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "max_rel": self.max_rel,
            "finite": self.finite,
        }


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> ErrorMetrics:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    delta = np.abs(actual - expected)
    rel = delta / np.maximum(np.abs(expected), np.float32(1e-6))
    return ErrorMetrics(
        max_abs=float(delta.max(initial=0)),
        mean_abs=float(delta.mean()),
        max_rel=float(rel.max(initial=0)),
        finite=bool(np.isfinite(actual).all()),
    )


def passes_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> bool:
    metrics = error_metrics(actual, expected)
    return bool(
        metrics.finite
        and metrics.max_abs <= atol
        and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    )


def latency_summary(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        raise ValueError("at least one latency sample is required")
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "trials": len(samples_ms),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
    }


def attention_flops(config, *, useful_causal: bool) -> int:
    b, h, s, d = config.shape
    flops = 4 * b * h * s * s * d
    return flops // 2 if config.causal and useful_causal else flops


def achieved_tflops(flops: int, latency_ms: float) -> float:
    return flops / (latency_ms * 1e-3) / 1e12


def v5e_mfu(throughput_tflops: float) -> float:
    return throughput_tflops / V5E_BF16_PEAK_TFLOPS
