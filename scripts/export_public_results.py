"""Build a path-free public summary from a completed private TPU run.

This exporter allowlists fields rather than copying raw reports, whose
manifests and logs may contain local paths or environment details.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median


SEQUENCES = (256, 512, 1024, 2048, 4096)
MASKS = ("causal", "noncausal")
RUNNERS = ("torchtpu_direct", "torchtpu_torch_compile")
RUNNER_DIRS = {"torchtpu_direct": "direct", "torchtpu_torch_compile": "compiled"}
UPSTREAM_COMMIT = "c5bb931da44489f0697d97ff94bddaaffeee94d6"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: list[float]) -> dict[str, float]:
    _require(len(values) == 3, "expected three fresh-process measurements")
    _require(all(math.isfinite(value) and value >= 0 for value in values), "invalid timing")
    return {"median": median(values), "min": min(values), "max": max(values)}


def _check_manifest(manifest: dict, *, commit: str, wheel_hash: str) -> None:
    source = manifest["source_control"]
    packages = manifest["packages"]
    _require(source["comparison_commit"] == commit, "comparison commit mismatch")
    _require(source["comparison_dirty"] is False, "dirty comparison source")
    _require(source["upstream_commit"] == UPSTREAM_COMMIT, "upstream commit mismatch")
    _require(source["upstream_dirty"] is False, "dirty upstream source")
    _require(packages["torch_tpu"] == "0.3.0", "TorchTPU version mismatch")
    _require(packages["torch"] == "2.15.0+cpu", "PyTorch version mismatch")
    _require(packages["libtpu"] == "0.0.51", "libtpu version mismatch")
    _require(packages["jax"] == packages["jaxlib"] == "0.9.2", "JAX version mismatch")
    _require(manifest["artifacts"]["torch_tpu_wheel_sha256"] == wheel_hash, "wheel mismatch")
    _require(manifest["environment"].get("TORCH_TPU_SAFE_PROCESS_EXIT") == "0", "normal teardown was disabled")


def build(root: Path, *, commit: str) -> dict:
    doctor_lines = (root / "doctor10" / "returncodes.txt").read_text(encoding="utf-8").splitlines()
    _require(
        doctor_lines == [f"doctor {i:02d}: returncode=0 passed=true" for i in range(1, 11)],
        "ten-run normal-exit gate did not pass",
    )
    doctor = _read(root / "doctor.json")
    summary = _read(root / "publication" / "summary.json")
    adversarial = _read(root / "publication" / "adversarial" / "comparison.json")
    smoke = _read(root / "smoke" / "comparison.json")
    _require(doctor["passed"] and doctor["bridge"] == "torch.tpu.pallas.jax_op", "doctor gate failed")
    _require(smoke["passed"], "smoke gate failed")
    _require(adversarial["passed"], "adversarial gate failed")
    _require(summary["passed"] and summary["expected_repetitions"] == 3, "matrix gate failed")
    _require(summary["missing_cases"] == [] and summary["wrong_repetition_counts"] == {}, "incomplete matrix")
    _require(len(summary["sources"]) == 30 and len(summary["summaries"]) == 20, "unexpected matrix size")
    wheel_hash = doctor["manifest"]["artifacts"]["torch_tpu_wheel_sha256"]
    _require(isinstance(wheel_hash, str) and re.fullmatch(r"[0-9a-f]{64}", wheel_hash) is not None, "missing wheel hash")
    _check_manifest(doctor["manifest"], commit=commit, wheel_hash=wheel_hash)
    for i in range(1, 11):
        doctor_repeat = _read(root / "doctor10" / f"doctor-{i:02d}.json")
        _require(doctor_repeat["passed"], f"doctor repetition {i} failed")
        _check_manifest(doctor_repeat["manifest"], commit=commit, wheel_hash=wheel_hash)

    rows = []
    for seq in SEQUENCES:
        for mask in MASKS:
            case = f"s{seq}-{mask}"
            for runner in RUNNERS:
                native_forward: list[float] = []
                runner_forward: list[float] = []
                native_forward_backward: list[float] = []
                runner_forward_backward: list[float] = []
                forward_ratios: list[float] = []
                forward_backward_ratios: list[float] = []
                max_errors: list[float] = []
                for rep in range(1, 4):
                    case_root = root / "publication" / f"rep{rep}" / case
                    comparison = _read(case_root / "comparison.json")
                    _require(comparison["passed"], f"failed comparison: {case} rep{rep}")
                    matches = [item for item in comparison["comparisons"] if item["runner"] == runner]
                    _require(len(matches) == 1 and matches[0]["passed"], f"failed runner: {case} rep{rep} {runner}")
                    errors = matches[0]["errors"]
                    _require(all(errors[name]["finite"] for name in ("o", "dq", "dk", "dv")), "nonfinite output or gradient")
                    max_errors.append(max(float(errors[name]["max_abs"]) for name in ("o", "dq", "dk", "dv")))
                    native = _read(case_root / "native" / "result.json")
                    candidate = _read(case_root / RUNNER_DIRS[runner] / "result.json")
                    _check_manifest(native["manifest"], commit=commit, wheel_hash=wheel_hash)
                    _check_manifest(candidate["manifest"], commit=commit, wheel_hash=wheel_hash)
                    nf = float(native["forward"]["p50_ms"])
                    rf = float(candidate["forward"]["p50_ms"])
                    nb = float(native["forward_backward"]["p50_ms"])
                    rb = float(candidate["forward_backward"]["p50_ms"])
                    _require(nf > 0 and nb > 0, "invalid native latency")
                    native_forward.append(nf)
                    runner_forward.append(rf)
                    native_forward_backward.append(nb)
                    runner_forward_backward.append(rb)
                    forward_ratios.append(rf / nf)
                    forward_backward_ratios.append(rb / nb)
                _require(max(max_errors) <= 0.05, "maximum absolute error exceeded")
                rows.append({
                    "case": case,
                    "runner": runner,
                    "native_forward_p50_ms": _stats(native_forward),
                    "runner_forward_p50_ms": _stats(runner_forward),
                    "forward_ratio": _stats(forward_ratios),
                    "native_forward_backward_p50_ms": _stats(native_forward_backward),
                    "runner_forward_backward_p50_ms": _stats(runner_forward_backward),
                    "forward_backward_ratio": _stats(forward_backward_ratios),
                    "max_abs_error": max(max_errors),
                })

    packages = doctor["manifest"]["packages"]
    return {
        "schema_version": 1,
        "passed": True,
        "normal_exit_doctor_runs": 10,
        "matrix_case_runs": 30,
        "adversarial_passed": True,
        "scope": {"device": "TPU v5e", "devices_used": 1, "batch": 1, "heads": 1, "head_dim": 128,
                  "sequence_lengths": list(SEQUENCES), "masks": list(MASKS), "repetitions": 3,
                  "warmups": 5, "trials": 30, "timing": "synchronized host-observed latency"},
        "software": {name: packages[name] for name in ("torch_tpu", "torch", "libtpu", "jax", "jaxlib", "numpy")},
        "source_control": {"comparison_commit": commit, "upstream_commit": UPSTREAM_COMMIT},
        "torch_tpu_wheel_sha256": wheel_hash,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(re.fullmatch(r"[0-9a-f]{40}", args.comparison_commit) is not None, "invalid source commit")
    report = build(args.run_root, commit=args.comparison_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} public-safe rows to {args.output}")


if __name__ == "__main__":
    main()
