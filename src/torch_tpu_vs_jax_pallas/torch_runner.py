from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bridge import feasibility_report
from .fixtures import load_fixture
from .io import write_json
from .manifest import environment_manifest
from .metrics import latency_summary


def _synchronize(torch) -> None:
    accelerator_sync = getattr(getattr(torch, "accelerator", None), "synchronize", None)
    if callable(accelerator_sync):
        accelerator_sync()
        return
    tpu_sync = getattr(getattr(torch, "tpu", None), "synchronize", None)
    if callable(tpu_sync):
        tpu_sync()
        return
    torch.empty((), device="tpu").cpu()


def _device_count(torch) -> int | None:
    for owner in (getattr(torch, "accelerator", None), getattr(torch, "tpu", None)):
        function = getattr(owner, "device_count", None)
        if callable(function):
            try:
                return int(function())
            except Exception:
                continue
    return None


def _reset_peak_memory(torch, device) -> None:
    function = getattr(getattr(torch, "accelerator", None), "reset_peak_memory_stats", None)
    if callable(function):
        try:
            function(device)
        except RuntimeError as exc:
            if "ClearMemoryStats is not supported" not in str(exc):
                raise


def _memory_snapshot(torch, device) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in ("memory_allocated", "max_memory_allocated"):
        function = getattr(getattr(torch, "accelerator", None), name, None)
        if callable(function):
            try:
                result[name] = int(function(device))
            except Exception:
                pass
    return result


def _capture_profile(torch, call, q, k, v, output_dir: Path, steps: int) -> str | None:
    profiler = getattr(torch, "profiler", None)
    activity = getattr(getattr(profiler, "ProfilerActivity", None), "TPU", None)
    handler = getattr(profiler, "xprof_trace_handler", None)
    if profiler is None or activity is None or not callable(handler):
        return None
    profile_dir = output_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [profiler.ProfilerActivity.CPU, activity]
    with profiler.profile(
        activities=activities,
        on_trace_ready=handler(dir_name=str(profile_dir)),
    ):
        for _ in range(steps):
            call(q, k, v)
        _synchronize(torch)
    return str(profile_dir)


def run(
    fixture_path: str,
    output_dir: str,
    warmup: int,
    trials: int,
    compiled: bool,
    *,
    profile: bool = False,
    profile_steps: int = 10,
) -> dict:
    import torch
    import torch_tpu  # noqa: F401

    from .pallas_ops import flash_forward_op

    fixture = load_fixture(fixture_path)
    config = fixture.config
    feasibility = feasibility_report(config)
    if not feasibility["available"]:
        raise RuntimeError(json.dumps(feasibility, indent=2))
    q_np, k_np, v_np, do_np = fixture.arrays()
    device = torch.device("tpu")

    def tensor(value, *, requires_grad=False):
        return torch.tensor(
            value, dtype=torch.bfloat16, device=device, requires_grad=requires_grad
        )

    q_infer, k_infer, v_infer = (tensor(x) for x in (q_np, k_np, v_np))
    q, k, v = (tensor(x, requires_grad=True) for x in (q_np, k_np, v_np))
    do = tensor(do_np)

    def raw_forward(q_arg, k_arg, v_arg):
        return flash_forward_op(
            q_arg,
            k_arg,
            v_arg,
            config.causal,
            config.block_q,
            config.block_kv,
        )

    call = (
        torch.compile(
            raw_forward,
            backend="tpu",
            dynamic=False,
            fullgraph=True,
        )
        if compiled
        else raw_forward
    )

    def step():
        for tensor_ in (q, k, v):
            tensor_.grad = None
        o_value, _, _ = call(q, k, v)
        o_value.backward(do)
        return o_value

    _reset_peak_memory(torch, device)
    memory_before = _memory_snapshot(torch, device)
    t0 = time.perf_counter()
    first_forward = call(q_infer, k_infer, v_infer)
    _synchronize(torch)
    compile_forward_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        call(q_infer, k_infer, v_infer)
    _synchronize(torch)
    _reset_peak_memory(torch, device)
    forward_samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        last_forward = call(q_infer, k_infer, v_infer)
        _synchronize(torch)
        forward_samples.append((time.perf_counter() - t0) * 1e3)
    memory_after_forward = _memory_snapshot(torch, device)

    t0 = time.perf_counter()
    o = step()
    _synchronize(torch)
    compile_train_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        step()
    _synchronize(torch)
    _reset_peak_memory(torch, device)
    train_samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        o = step()
        _synchronize(torch)
        train_samples.append((time.perf_counter() - t0) * 1e3)
    memory_after_train = _memory_snapshot(torch, device)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    profile_dir = (
        _capture_profile(torch, call, q_infer, k_infer, v_infer, target, profile_steps)
        if profile
        else None
    )
    o_infer, m_infer, l_infer = last_forward if trials else first_forward
    np.savez_compressed(
        target / "arrays.npz",
        o=o.detach().float().cpu().numpy(),
        dq=q.grad.detach().float().cpu().numpy(),
        dk=k.grad.detach().float().cpu().numpy(),
        dv=v.grad.detach().float().cpu().numpy(),
        inference_o=o_infer.detach().float().cpu().numpy(),
        m=m_infer.detach().float().cpu().numpy(),
        l=l_infer.detach().float().cpu().numpy(),
    )
    mode = "torch_compile" if compiled else "direct"
    result = {
        "runner": f"torchtpu_{mode}",
        "config": config.to_dict(),
        "device": str(device),
        "devices_available": _device_count(torch),
        "devices_used": 1,
        "bridge": feasibility,
        "compile_forward_ms": compile_forward_ms,
        "compile_forward_backward_ms": compile_train_ms,
        "forward": latency_summary(forward_samples),
        "forward_backward": latency_summary(train_samples),
        "memory": {
            "before": memory_before,
            "warm_forward": memory_after_forward,
            "warm_forward_backward": memory_after_train,
        },
        "profile_dir": profile_dir,
        "manifest": environment_manifest(runner=f"torchtpu_{mode}"),
    }
    write_json(result, target / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=10)
    args = parser.parse_args()
    from .process_exit import finalize_tpu_process

    print(
        json.dumps(
            run(
                args.fixture,
                args.output_dir,
                args.warmup,
                args.trials,
                args.compiled,
                profile=args.profile,
                profile_steps=args.profile_steps,
            ),
            indent=2,
        ),
        flush=True,
    )
    finalize_tpu_process()


if __name__ == "__main__":
    main()
