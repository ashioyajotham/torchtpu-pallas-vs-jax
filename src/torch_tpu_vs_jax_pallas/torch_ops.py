from __future__ import annotations

from typing import Any

import torch

from .config import KernelConfig
from .pallas_ops import flash_forward_op


def _config_for(q, *, causal: bool, block_sizes: tuple[int, int] | None) -> KernelConfig:
    if q.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 B,H,S,D tensors")
    block_q, block_kv = block_sizes or (128, 128)
    return KernelConfig(
        batch=q.shape[0],
        heads=q.shape[1],
        seq_len=q.shape[2],
        head_dim=q.shape[3],
        causal=causal,
        block_q=block_q,
        block_kv=block_kv,
    )


def _validate_tensor(name: str, tensor: Any, config: KernelConfig) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != config.shape:
        raise ValueError(f"{name}.shape must be {config.shape}, got {tuple(tensor.shape)}")
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16")
    if tensor.device.type != "tpu":
        raise ValueError(f"{name} must be on a TorchTPU device, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def flash_attention_with_aux(
    q,
    k,
    v,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    block_sizes: tuple[int, int] | None = None,
):
    """Return O/M/L from the registered forward custom op."""
    if sm_scale is not None:
        raise NotImplementedError("explicit sm_scale is outside the pinned v1 contract")
    config = _config_for(q, causal=causal, block_sizes=block_sizes)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _validate_tensor(name, tensor, config)
    return flash_forward_op(q, k, v, config.causal, config.block_q, config.block_kv)


def flash_attention(
    q,
    k,
    v,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    block_sizes: tuple[int, int] | None = None,
):
    """Model-facing Flash Attention call backed by TorchTPU/Pallas."""
    o, _, _ = flash_attention_with_aux(
        q, k, v, causal=causal, sm_scale=sm_scale, block_sizes=block_sizes
    )
    return o
