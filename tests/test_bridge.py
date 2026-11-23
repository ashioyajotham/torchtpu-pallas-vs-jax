import ast
import sys
import types
from pathlib import Path

from torch_tpu_vs_jax_pallas.bridge import (
    feasibility_report,
    load_wrapped_kernels,
)
from torch_tpu_vs_jax_pallas.config import KernelConfig


def test_loads_public_jax_op_registration_module(monkeypatch):
    forward = types.SimpleNamespace(register_autograd=lambda *args, **kwargs: None)
    fake_module = types.ModuleType("torch_tpu_vs_jax_pallas.pallas_ops")
    fake_module.flash_forward_op = forward
    fake_module.flash_backward_op = object()
    monkeypatch.setitem(
        sys.modules, "torch_tpu_vs_jax_pallas.pallas_ops", fake_module
    )
    load_wrapped_kernels.cache_clear()
    report = feasibility_report(KernelConfig())
    assert report["available"]
    assert report["api"] == "torch.tpu.pallas.jax_op"
    assert report["operator_names"]["forward"].startswith("recompute_pallas::")
    load_wrapped_kernels.cache_clear()


def test_pallas_source_matches_torchtpu_contract():
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "torch_tpu_vs_jax_pallas"
        / "pallas_ops.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("_jax_forward", "_jax_backward"):
        function = functions[name]
        assert all(argument.annotation is not None for argument in function.args.args)
        assert function.returns is not None
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "jax_op"
    ]
    assert len(calls) == 2
    assert all(len(call.args) == 2 for call in calls)
    assert "torch.tpu.pallas.jax_op" in source
    assert "torch_tpu._internal" not in source
    assert "register_autograd" in source
    assert "torch.autograd.Function" not in source


def test_compiled_runner_uses_tpu_static_fullgraph():
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "torch_tpu_vs_jax_pallas"
        / "torch_runner.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert 'backend="tpu"' in source
    assert "dynamic=False" in source
    assert "fullgraph=True" in source
    assert "from .pallas_ops import flash_forward_op" in source


def test_model_facing_op_is_bound_before_dynamo_tracing():
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "torch_tpu_vs_jax_pallas"
        / "torch_ops.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "from .pallas_ops import flash_forward_op" in source
    assert "load_wrapped_kernels" not in source
