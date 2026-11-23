import json

from torch_tpu_vs_jax_pallas.aggregate import aggregate_publication


def test_aggregate_requires_complete_three_repetition_matrix(tmp_path):
    root = tmp_path / "publication"
    metric_values = {
        "forward_p50_overhead_ratio": 1.1,
        "forward_p50_overhead_ms": 0.1,
        "executed_forward_tflops": 100.0,
        "useful_forward_tflops": 50.0,
        "executed_forward_mfu": 0.5,
        "useful_forward_mfu": 0.25,
    }
    for repetition in range(1, 4):
        for sequence in (256, 512, 1024, 2048, 4096):
            for suffix in ("causal", "noncausal"):
                path = root / f"rep{repetition}" / f"s{sequence}-{suffix}"
                path.mkdir(parents=True)
                comparison = {
                    "passed": True,
                    "comparisons": [
                        {"runner": "torchtpu_direct", "passed": True, **metric_values},
                        {"runner": "torchtpu_torch_compile", "passed": True, **metric_values},
                    ],
                }
                (path / "comparison.json").write_text(json.dumps(comparison))

    report = aggregate_publication(root, tmp_path / "summary.json")
    assert report["passed"]
    assert len(report["summaries"]) == 20
    assert all(item["repetitions"] == 3 for item in report["summaries"])
