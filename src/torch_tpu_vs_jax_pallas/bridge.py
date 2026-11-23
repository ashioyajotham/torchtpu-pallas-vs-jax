from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config import KernelConfig


class BridgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WrappedKernels:
    forward: Any
    backward: Any
    api_name: str


@lru_cache(maxsize=1)
def load_wrapped_kernels() -> WrappedKernels:
    """Load the two process-global TorchTPU custom operators."""
    try:
        module = importlib.import_module("torch_tpu_vs_jax_pallas.pallas_ops")
    except Exception as exc:
        raise BridgeUnavailable(
            "TorchTPU Pallas operators could not be registered through "
            "torch.tpu.pallas.jax_op"
        ) from exc
    return WrappedKernels(
        forward=module.flash_forward_op,
        backward=module.flash_backward_op,
        api_name="torch.tpu.pallas.jax_op",
    )


def feasibility_report(config: KernelConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "config": config.to_dict(),
        "available": False,
        "required_api": "torch.tpu.pallas.jax_op",
        "operator_names": {
            "forward": "recompute_pallas::flash_attention_forward",
            "backward": "recompute_pallas::flash_attention_backward",
        },
    }
    try:
        wrapped = load_wrapped_kernels()
    except Exception as exc:
        cause = exc.__cause__ or exc
        report["error"] = f"{type(cause).__name__}: {cause}"
        return report
    report.update(
        available=True,
        api=wrapped.api_name,
        forward_callable=callable(wrapped.forward),
        backward_callable=callable(wrapped.backward),
        forward_has_register_autograd=hasattr(wrapped.forward, "register_autograd"),
    )
    return report
