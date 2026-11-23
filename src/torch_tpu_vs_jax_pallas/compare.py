from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import load_fixture
from .io import read_json, write_json
from .metrics import (
    V5E_BF16_PEAK_TFLOPS,
    V5E_SPEC_URL,
    achieved_tflops,
    attention_flops,
    error_metrics,
    passes_tolerance,
    v5e_mfu,
)
from .reference import attention_reference


GATED_FIELDS = ("o", "dq", "dk", "dv")
AUX_FIELDS = ("m", "l")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "arrays.npz") as data:
        return {name: data[name].copy() for name in data.files}


def _comparison(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    fields: tuple[str, ...],
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], bool]:
    reports = {
        name: error_metrics(actual[name], expected[name]).to_dict() for name in fields
    }
    passed = all(
        passes_tolerance(actual[name], expected[name], atol=atol, rtol=rtol)
        for name in fields
    )
    return reports, passed


def compare_runs(
    fixture_path: str | Path,
    native_dir: str | Path,
    torch_dirs: list[str | Path],
    output: str | Path,
    *,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> dict[str, Any]:
    fixture = load_fixture(fixture_path)
    q, k, v, _ = fixture.arrays()
    reference = (
        attention_reference(q, k, v, causal=fixture.config.causal)
        if fixture.config.seq_len <= 1024
        else None
    )
    native_path = Path(native_dir)
    native_result = read_json(native_path / "result.json")
    native = _load_arrays(native_path)

    reference_report: dict[str, Any]
    reference_passed = True
    if reference is None:
        reference_report = {"skipped": "host reference is limited to sequence lengths <= 1024"}
    else:
        reference_report = error_metrics(native["o"], reference).to_dict()
        reference_passed = passes_tolerance(
            native["o"], reference, atol=atol, rtol=rtol
        )

    oracle_report: dict[str, Any]
    oracle_passed = True
    if all(f"oracle_{name}" in native for name in GATED_FIELDS):
        oracle_expected = {name: native[f"oracle_{name}"] for name in GATED_FIELDS}
        oracle_report, oracle_passed = _comparison(
            native, oracle_expected, GATED_FIELDS, atol=atol, rtol=rtol
        )
    else:
        oracle_report = {"skipped": "naive JAX oracle is limited to sequence lengths <= 1024"}

    executed_flops = attention_flops(fixture.config, useful_causal=False)
    useful_flops = attention_flops(fixture.config, useful_causal=True)
    report: dict[str, Any] = {
        "config": fixture.config.to_dict(),
        "tolerances": {
            "max_abs": atol,
            "atol": atol,
            "rtol": rtol,
            "finite_required": True,
        },
        "native_vs_numpy_forward": reference_report,
        "native_vs_naive_jax": oracle_report,
        "executed_forward_flops": executed_flops,
        "useful_forward_flops": useful_flops,
        "mfu_denominator": {
            "device": "TPU v5e",
            "chips_used": 1,
            "bf16_peak_tflops": V5E_BF16_PEAK_TFLOPS,
            "source": V5E_SPEC_URL,
        },
        "comparisons": [],
    }
    for torch_dir in torch_dirs:
        path = Path(torch_dir)
        result = read_json(path / "result.json")
        candidate = _load_arrays(path)
        fields, passed = _comparison(
            candidate, native, GATED_FIELDS, atol=atol, rtol=rtol
        )
        aux_names = tuple(name for name in AUX_FIELDS if name in candidate and name in native)
        aux = (
            _comparison(candidate, native, aux_names, atol=atol, rtol=rtol)[0]
            if aux_names
            else {}
        )
        native_ms = native_result["forward"]["p50_ms"]
        torch_ms = result["forward"]["p50_ms"]
        executed_tflops = achieved_tflops(executed_flops, torch_ms)
        useful_tflops = achieved_tflops(useful_flops, torch_ms)
        report["comparisons"].append(
            {
                "runner": result["runner"],
                "passed": passed,
                "errors": fields,
                "auxiliary_errors": aux,
                "forward_p50_overhead_ratio": torch_ms / native_ms,
                "forward_p50_overhead_ms": torch_ms - native_ms,
                "executed_forward_tflops": executed_tflops,
                "useful_forward_tflops": useful_tflops,
                "executed_forward_mfu": v5e_mfu(executed_tflops),
                "useful_forward_mfu": v5e_mfu(useful_tflops),
            }
        )
    report["passed"] = bool(
        reference_passed
        and oracle_passed
        and all(item["passed"] for item in report["comparisons"])
    )
    write_json(report, output)
    return report
