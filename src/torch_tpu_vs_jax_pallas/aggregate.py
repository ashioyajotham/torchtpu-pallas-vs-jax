from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .io import read_json, write_json


METRICS = (
    "forward_p50_overhead_ratio",
    "forward_p50_overhead_ms",
    "executed_forward_tflops",
    "useful_forward_tflops",
    "executed_forward_mfu",
    "useful_forward_mfu",
)


def aggregate_publication(
    root: str | Path,
    output: str | Path,
    *,
    expected_repetitions: int = 3,
) -> dict[str, Any]:
    root_path = Path(root)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    case_passes: dict[str, list[bool]] = defaultdict(list)
    sources: list[str] = []
    for path in sorted(root_path.glob("rep*/s*/comparison.json")):
        report = read_json(path)
        case = path.parent.name
        case_passes[case].append(bool(report["passed"]))
        sources.append(str(path))
        for comparison in report["comparisons"]:
            groups[(case, comparison["runner"])].append(comparison)

    summaries = []
    for (case, runner), values in sorted(groups.items()):
        summary = {
            "case": case,
            "runner": runner,
            "repetitions": len(values),
            "passed": all(value["passed"] for value in values),
            "metrics": {},
        }
        for metric in METRICS:
            samples = [float(value[metric]) for value in values]
            summary["metrics"][metric] = {
                "median": median(samples),
                "min": min(samples),
                "max": max(samples),
            }
        summaries.append(summary)

    expected_cases = {
        f"s{seq_len}-{suffix}"
        for seq_len in (256, 512, 1024, 2048, 4096)
        for suffix in ("causal", "noncausal")
    }
    missing_cases = sorted(expected_cases - set(case_passes))
    wrong_counts = {
        case: len(values)
        for case, values in case_passes.items()
        if len(values) != expected_repetitions
    }
    passed = bool(
        summaries
        and not missing_cases
        and not wrong_counts
        and all(all(values) for values in case_passes.values())
        and all(item["passed"] for item in summaries)
    )
    report = {
        "passed": passed,
        "expected_repetitions": expected_repetitions,
        "missing_cases": missing_cases,
        "wrong_repetition_counts": wrong_counts,
        "sources": sources,
        "summaries": summaries,
    }
    write_json(report, output)
    return report
