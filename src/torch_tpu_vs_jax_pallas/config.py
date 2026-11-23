from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class KernelConfig:
    batch: int = 1
    heads: int = 8
    seq_len: int = 256
    head_dim: int = 128
    causal: bool = False
    block_q: int = 128
    block_kv: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        positive = ("batch", "heads", "seq_len", "head_dim", "block_q", "block_kv")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 128:
            raise ValueError("head_dim must be a multiple of 128")
        if self.block_q % 128 or self.block_kv % 128:
            raise ValueError("block sizes must be multiples of 128")
        if self.seq_len % self.block_q or self.seq_len % self.block_kv:
            raise ValueError("seq_len must be divisible by both block sizes")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.heads, self.seq_len, self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelConfig":
        return cls(**data)
