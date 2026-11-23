from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .compare import compare_runs
from .config import KernelConfig
from .fixtures import generate_fixture, save_fixture


def _run(command: list[str], *, env: dict[str, str], timeout: int) -> None:
    try:
        subprocess.run(command, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"runner exceeded {timeout}s and was terminated: {' '.join(command)}"
        ) from exc


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--process-timeout", type=int, default=1800)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=10)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpu-compare")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--out", default="artifacts/doctor.json")
    doctor.add_argument(
        "--expected-torch-tpu",
        help=(
            "override the pinned TorchTPU version for an isolated compatibility "
            "experiment"
        ),
    )
    doctor.add_argument("--expected-torch")
    doctor.add_argument("--expected-libtpu")
    doctor.add_argument("--expected-jax")

    fixture = sub.add_parser("fixture")
    fixture.add_argument("--out", required=True)
    fixture.add_argument("--batch", type=int, default=1)
    fixture.add_argument("--heads", type=int, default=8)
    fixture.add_argument("--seq-len", type=int, default=256)
    fixture.add_argument("--head-dim", type=int, default=128)
    fixture.add_argument("--block-q", type=int, default=128)
    fixture.add_argument("--block-kv", type=int, default=128)
    fixture.add_argument("--seed", type=int, default=0)
    fixture.add_argument("--causal", action="store_true")
    fixture.add_argument("--adversarial", action="store_true")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--results", default="artifacts/smoke")
    _add_run_options(smoke)
    smoke.set_defaults(warmup=1, trials=3)

    suite = sub.add_parser("suite")
    suite.add_argument("--fixture", required=True)
    suite.add_argument("--results", required=True)
    _add_run_options(suite)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--expected-repetitions", type=int, default=3)

    compare = sub.add_parser("compare")
    compare.add_argument("--fixture", required=True)
    compare.add_argument("--native-dir", required=True)
    compare.add_argument("--torch-dir", action="append", required=True)
    compare.add_argument("--out", required=True)
    return parser


def _run_suite(
    *,
    fixture: str | Path,
    results: str | Path,
    warmup: int,
    trials: int,
    timeout: int,
    profile: bool,
    profile_steps: int,
) -> dict:
    results_path = Path(results)
    results_path.mkdir(parents=True, exist_ok=True)
    base = [
        "--fixture",
        str(Path(fixture).resolve()),
        "--warmup",
        str(warmup),
        "--trials",
        str(trials),
    ]
    env = os.environ.copy()
    python = sys.executable
    _run(
        [
            python,
            "-m",
            "torch_tpu_vs_jax_pallas.native_runner",
            *base,
            "--output-dir",
            str(results_path / "native"),
        ],
        env=env,
        timeout=timeout,
    )
    _run(
        [
            python,
            "-m",
            "torch_tpu_vs_jax_pallas.torch_runner",
            *base,
            "--output-dir",
            str(results_path / "direct"),
        ],
        env=env,
        timeout=timeout,
    )
    compiled_command = [
        python,
        "-m",
        "torch_tpu_vs_jax_pallas.torch_runner",
        *base,
        "--compiled",
        "--output-dir",
        str(results_path / "compiled"),
    ]
    if profile:
        compiled_command.extend(["--profile", "--profile-steps", str(profile_steps)])
    _run(compiled_command, env=env, timeout=timeout)
    return compare_runs(
        fixture,
        results_path / "native",
        [results_path / "direct", results_path / "compiled"],
        results_path / "comparison.json",
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "doctor":
        from .doctor import run_doctor

        from .process_exit import finalize_tpu_process

        print(
            json.dumps(
                run_doctor(
                    args.out,
                    expected_torch_tpu=args.expected_torch_tpu,
                    expected_torch=args.expected_torch,
                    expected_libtpu=args.expected_libtpu,
                    expected_jax=args.expected_jax,
                ),
                indent=2,
            ),
            flush=True,
        )
        finalize_tpu_process()
        return
    if args.command == "fixture":
        config = KernelConfig(
            batch=args.batch,
            heads=args.heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            causal=args.causal,
            block_q=args.block_q,
            block_kv=args.block_kv,
            seed=args.seed,
        )
        print(save_fixture(generate_fixture(config, adversarial=args.adversarial), args.out))
        return
    if args.command == "aggregate":
        from .aggregate import aggregate_publication

        report = aggregate_publication(
            args.root,
            args.out,
            expected_repetitions=args.expected_repetitions,
        )
    elif args.command == "compare":
        report = compare_runs(args.fixture, args.native_dir, args.torch_dir, args.out)
    elif args.command == "smoke":
        results = Path(args.results)
        fixture = results / "fixture.npz"
        save_fixture(
            generate_fixture(
                KernelConfig(batch=1, heads=1, seq_len=256, head_dim=128)
            ),
            fixture,
        )
        report = _run_suite(
            fixture=fixture,
            results=results,
            warmup=args.warmup,
            trials=args.trials,
            timeout=args.process_timeout,
            profile=args.profile,
            profile_steps=args.profile_steps,
        )
    else:
        report = _run_suite(
            fixture=args.fixture,
            results=args.results,
            warmup=args.warmup,
            trials=args.trials,
            timeout=args.process_timeout,
            profile=args.profile,
            profile_steps=args.profile_steps,
        )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
