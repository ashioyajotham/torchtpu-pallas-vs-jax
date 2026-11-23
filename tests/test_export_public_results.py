import json

import pytest

from scripts import export_public_results as exporter


COMMIT = "a" * 40
WHEEL = "b" * 64


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    root = tmp_path / "run"
    doctor10 = root / "doctor10"
    doctor10.mkdir(parents=True)
    (doctor10 / "returncodes.txt").write_text(
        "".join(f"doctor {i:02d}: returncode=0 passed=true\n" for i in range(1, 11)),
        encoding="utf-8",
    )
    manifest = {
        "source_control": {"comparison_commit": COMMIT, "comparison_dirty": False,
                           "upstream_commit": exporter.UPSTREAM_COMMIT, "upstream_dirty": False},
        "packages": {"torch_tpu": "0.3.0", "torch": "2.15.0+cpu", "libtpu": "0.0.51",
                     "jax": "0.9.2", "jaxlib": "0.9.2", "numpy": "2.5.2"},
        "artifacts": {"torch_tpu_wheel_sha256": WHEEL, "torch_tpu_wheel": "secret-marker"},
        "environment": {"TORCH_TPU_SAFE_PROCESS_EXIT": "0"},
        "pip_freeze": ["secret-marker"],
    }
    _write(root / "doctor.json", {"passed": True, "bridge": "torch.tpu.pallas.jax_op", "manifest": manifest})
    for i in range(1, 11):
        _write(doctor10 / f"doctor-{i:02d}.json", {"passed": True, "manifest": manifest})
    _write(root / "smoke" / "comparison.json", {"passed": True})
    _write(root / "publication" / "adversarial" / "comparison.json", {"passed": True})
    _write(root / "publication" / "summary.json", {
        "passed": True, "expected_repetitions": 3, "missing_cases": [],
        "wrong_repetition_counts": {}, "sources": ["redacted"] * 30,
        "summaries": [{}] * 20,
    })
    errors = {name: {"finite": True, "max_abs": 0.001} for name in ("o", "dq", "dk", "dv")}
    for seq in exporter.SEQUENCES:
        for mask in exporter.MASKS:
            for rep in range(1, 4):
                case = root / "publication" / f"rep{rep}" / f"s{seq}-{mask}"
                _write(case / "comparison.json", {"passed": True, "comparisons": [
                    {"runner": runner, "passed": True, "errors": errors}
                    for runner in exporter.RUNNERS
                ]})
                _write(case / "native" / "result.json", {
                    "manifest": manifest, "forward": {"p50_ms": 1.0},
                    "forward_backward": {"p50_ms": 2.0},
                })
                for runner in exporter.RUNNERS:
                    _write(case / exporter.RUNNER_DIRS[runner] / "result.json", {
                        "manifest": manifest, "forward": {"p50_ms": 1.25},
                        "forward_backward": {"p50_ms": 2.5},
                    })
    return root


def test_export_is_complete_and_allowlisted(tmp_path):
    root = _fixture(tmp_path)
    report = exporter.build(root, commit=COMMIT)
    assert report["passed"]
    assert report["normal_exit_doctor_runs"] == 10
    assert len(report["rows"]) == 20
    assert report["rows"][0]["forward_ratio"]["median"] == 1.25
    assert "secret-marker" not in json.dumps(report)


def test_export_rejects_disabled_normal_teardown(tmp_path):
    root = _fixture(tmp_path)
    doctor = json.loads((root / "doctor.json").read_text(encoding="utf-8"))
    doctor["manifest"]["environment"]["TORCH_TPU_SAFE_PROCESS_EXIT"] = "1"
    _write(root / "doctor.json", doctor)
    with pytest.raises(ValueError, match="normal teardown was disabled"):
        exporter.build(root, commit=COMMIT)
