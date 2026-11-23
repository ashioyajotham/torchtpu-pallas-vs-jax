from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .fixtures import load_fixture
from .io import write_json
from .manifest import environment_manifest
from .metrics import latency_summary


def _ready(jax, value):
    return jax.tree.map(lambda x: x.block_until_ready(), value)


def _memory_stats(jax) -> dict:
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return {}
    return {
        key: int(stats[key])
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if key in stats
    }


def _naive_oracle(jax, jnp, q, k, v, do, *, causal: bool):
    scale = jnp.asarray(1.0 / np.sqrt(q.shape[-1]), dtype=jnp.float32)

    def attention(q_arg, k_arg, v_arg):
        logits = jnp.einsum(
            "bhsd,bhtd->bhst",
            q_arg.astype(jnp.float32),
            k_arg.astype(jnp.float32),
        )
        logits = logits * scale
        if causal:
            rows = jnp.arange(q_arg.shape[2])[:, None]
            cols = jnp.arange(k_arg.shape[2])[None, :]
            logits = jnp.where(rows >= cols, logits, jnp.asarray(-1e30, logits.dtype))
        probabilities = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum(
            "bhst,bhtd->bhsd", probabilities, v_arg.astype(jnp.float32)
        )

    output, pullback = jax.vjp(attention, q, k, v)
    grads = pullback(do.astype(jnp.float32))
    return output, grads


def run(fixture_path: str, output_dir: str, warmup: int, trials: int) -> dict:
    import jax
    import jax.numpy as jnp
    import recompute_dont_restore as kernels

    fixture = load_fixture(fixture_path)
    config = fixture.config
    q_np, k_np, v_np, do_np = fixture.arrays()
    q, k, v, do = (jnp.asarray(x, dtype=jnp.bfloat16) for x in (q_np, k_np, v_np, do_np))
    blocks = kernels.BlockSizes(config.block_q, config.block_kv)

    def forward(q_arg, k_arg, v_arg):
        return kernels.flash_attention_forward(
            q_arg, k_arg, v_arg, causal=config.causal, block_sizes=blocks
        )

    def forward_backward(q_arg, k_arg, v_arg, do_arg):
        o_value, m_value, l_value = forward(q_arg, k_arg, v_arg)
        grads_value = kernels.flash_attention_backward(
            q_arg,
            k_arg,
            v_arg,
            o_value,
            m_value,
            l_value,
            do_arg,
            causal=config.causal,
            block_sizes=blocks,
        )
        return (o_value, m_value, l_value), grads_value

    forward_jit = jax.jit(forward)
    train_jit = jax.jit(forward_backward)

    memory_before = _memory_stats(jax)
    t0 = time.perf_counter()
    first_forward = _ready(jax, forward_jit(q, k, v))
    compile_forward_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        _ready(jax, forward_jit(q, k, v))
    forward_samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ready(jax, forward_jit(q, k, v))
        forward_samples.append((time.perf_counter() - t0) * 1e3)
    memory_after_forward = _memory_stats(jax)

    t0 = time.perf_counter()
    (o, m, l), grads = _ready(jax, train_jit(q, k, v, do))
    compile_train_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        _ready(jax, train_jit(q, k, v, do))
    train_samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ready(jax, train_jit(q, k, v, do))
        train_samples.append((time.perf_counter() - t0) * 1e3)

    oracle = None
    if config.seq_len <= 1024:
        oracle = _ready(
            jax,
            _naive_oracle(jax, jnp, q, k, v, do, causal=config.causal),
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    arrays = {
        "o": np.asarray(o, dtype=np.float32),
        "dq": np.asarray(grads[0], dtype=np.float32),
        "dk": np.asarray(grads[1], dtype=np.float32),
        "dv": np.asarray(grads[2], dtype=np.float32),
        "m": np.asarray(m, dtype=np.float32),
        "l": np.asarray(l, dtype=np.float32),
    }
    if oracle is not None:
        arrays.update(
            oracle_o=np.asarray(oracle[0], dtype=np.float32),
            oracle_dq=np.asarray(oracle[1][0], dtype=np.float32),
            oracle_dk=np.asarray(oracle[1][1], dtype=np.float32),
            oracle_dv=np.asarray(oracle[1][2], dtype=np.float32),
        )
    np.savez_compressed(target / "arrays.npz", **arrays)
    result = {
        "runner": "native_jax",
        "config": config.to_dict(),
        "devices": [str(device) for device in jax.devices()],
        "devices_available": len(jax.devices()),
        "devices_used": 1,
        "process_count": jax.process_count(),
        "oracle_available": oracle is not None,
        "compile_forward_ms": compile_forward_ms,
        "compile_forward_backward_ms": compile_train_ms,
        "forward": latency_summary(forward_samples),
        "forward_backward": latency_summary(train_samples),
        "memory": {
            "fresh_process_before": memory_before,
            "after_forward": memory_after_forward,
        },
        "manifest": environment_manifest(runner="native_jax"),
    }
    write_json(result, target / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.output_dir, args.warmup, args.trials), indent=2))


if __name__ == "__main__":
    main()
