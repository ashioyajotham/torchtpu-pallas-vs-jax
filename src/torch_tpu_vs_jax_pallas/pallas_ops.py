"""TorchTPU custom-op registrations for the upstream Pallas kernels."""

from __future__ import annotations

import jax
import torch
import torch_tpu  # noqa: F401 - registers torch.tpu before tracing
import recompute_dont_restore as kernels


FORWARD_OP_NAME = "recompute_pallas::flash_attention_forward"
BACKWARD_OP_NAME = "recompute_pallas::flash_attention_backward"


def _jax_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool,
    block_q: int,
    block_kv: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    blocks = kernels.BlockSizes(block_q, block_kv)
    return kernels.flash_attention_forward(q, k, v, causal=causal, block_sizes=blocks)


def _jax_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    m: jax.Array,
    l: jax.Array,
    do: jax.Array,
    causal: bool,
    block_q: int,
    block_kv: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    blocks = kernels.BlockSizes(block_q, block_kv)
    return kernels.flash_attention_backward(
        q, k, v, o, m, l, do, causal=causal, block_sizes=blocks
    )


flash_forward_op = torch.tpu.pallas.jax_op(FORWARD_OP_NAME, _jax_forward)
flash_backward_op = torch.tpu.pallas.jax_op(BACKWARD_OP_NAME, _jax_backward)


def _backward_fake(q, k, v, o, m, l, do, causal, block_q, block_kv):
    del o, m, l, do, causal, block_q, block_kv
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


flash_backward_op.register_fake(_backward_fake)


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, causal, block_q, block_kv = inputs
    o, m, l = output
    ctx.save_for_backward(q, k, v, o, m, l)
    ctx.causal = causal
    ctx.block_q = block_q
    ctx.block_kv = block_kv
    ctx.mark_non_differentiable(m, l)


def _backward_impl(ctx, grad_o, grad_m, grad_l):
    del grad_m, grad_l
    if grad_o is None:
        return None, None, None, None, None, None
    q, k, v, o, m, l = ctx.saved_tensors
    dq, dk, dv = flash_backward_op(
        q, k, v, o, m, l, grad_o.contiguous(), ctx.causal, ctx.block_q, ctx.block_kv
    )
    return dq, dk, dv, None, None, None


flash_forward_op.register_autograd(_backward_impl, setup_context=_setup_context)
