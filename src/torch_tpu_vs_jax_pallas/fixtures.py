from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import KernelConfig


def float32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (round-to-nearest-even) and return raw uint16."""
    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


@dataclass(frozen=True, slots=True)
class Fixture:
    config: KernelConfig
    q_bits: np.ndarray
    k_bits: np.ndarray
    v_bits: np.ndarray
    do_bits: np.ndarray

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            bf16_bits_to_float32(x) for x in (self.q_bits, self.k_bits, self.v_bits, self.do_bits)
        )  # type: ignore[return-value]


def generate_fixture(config: KernelConfig, *, adversarial: bool = False) -> Fixture:
    rng = np.random.default_rng(config.seed)
    q = rng.standard_normal(config.shape, dtype=np.float32)
    k = rng.standard_normal(config.shape, dtype=np.float32)
    v = rng.standard_normal(config.shape, dtype=np.float32)
    do = rng.standard_normal(config.shape, dtype=np.float32)
    if adversarial:
        if config.seq_len < 2 * config.block_kv:
            raise ValueError("adversarial fixture requires at least two KV tiles")
        q.fill(1.0)
        k[..., : config.block_kv, :].fill(1.0)
        k[..., config.block_kv :, :].fill(0.0)
        v[..., : config.block_kv, :].fill(1.0)
        v[..., config.block_kv :, :].fill(100.0)
    return Fixture(config, *(float32_to_bf16_bits(x) for x in (q, k, v, do)))


def save_fixture(fixture: Fixture, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        metadata=np.asarray(json.dumps(fixture.config.to_dict())),
        q=fixture.q_bits,
        k=fixture.k_bits,
        v=fixture.v_bits,
        do=fixture.do_bits,
    )
    return target


def load_fixture(path: str | Path) -> Fixture:
    with np.load(path, allow_pickle=False) as data:
        config = KernelConfig.from_dict(json.loads(str(data["metadata"])))
        return Fixture(config, *(data[name].copy() for name in ("q", "k", "v", "do")))
