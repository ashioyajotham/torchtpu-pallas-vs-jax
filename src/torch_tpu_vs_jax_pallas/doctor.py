import json
import platform
from pathlib import Path

from .io import write_json
from .manifest import environment_manifest


EXPECTED_TORCH_TPU = "0.3.0"
EXPECTED_TORCH = "2.15.0+cpu"
EXPECTED_LIBTPU = "0.0.51"
EXPECTED_JAX = "0.9.2"


def run_doctor(
    output: str | Path | None = None,
    *,
    expected_torch_tpu: str | None = None,
    expected_torch: str | None = None,
    expected_libtpu: str | None = None,
    expected_jax: str | None = None,
) -> dict:
    import importlib.metadata

    import torch
    import torch_tpu  # noqa: F401
    import jax
    from jax.experimental import pallas as pl

    failures: list[str] = []
    if platform.python_version_tuple()[:2] != ("3", "12"):
        failures.append(f"Python 3.12 required, got {platform.python_version()}")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch_tpu", "jax", "jaxlib", "libtpu", "torch")
    }
    required_torch_tpu = expected_torch_tpu or EXPECTED_TORCH_TPU
    required_torch = expected_torch or EXPECTED_TORCH
    required_libtpu = expected_libtpu or EXPECTED_LIBTPU
    required_jax = expected_jax or EXPECTED_JAX
    if versions["torch_tpu"] != required_torch_tpu:
        failures.append(
            f"torch_tpu=={required_torch_tpu} required, got {versions['torch_tpu']}"
        )
    if versions["torch"] != required_torch:
        failures.append(f"torch=={required_torch} required, got {versions['torch']}")
    if versions["libtpu"] != required_libtpu:
        failures.append(
            f"libtpu=={required_libtpu} required, got {versions['libtpu']}"
        )
    if versions["jax"] != required_jax or versions["jaxlib"] != required_jax:
        failures.append(
            f"jax/jaxlib=={required_jax} required, got "
            f"{versions['jax']}/{versions['jaxlib']}"
        )

    def add_kernel(x_ref, y_ref, out_ref):
        out_ref[...] = x_ref[...] + y_ref[...]

    def add_jax(x: jax.Array, y: jax.Array) -> jax.Array:
        return pl.pallas_call(
            add_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y)

    # jax_op evaluates annotations in the function's global namespace. The
    # imports above are deliberately local so CPU-only tooling can import us.
    add_jax.__annotations__ = {"x": jax.Array, "y": jax.Array, "return": jax.Array}
    add_op = torch.tpu.pallas.jax_op("recompute_doctor::add_vectors", add_jax)
    device = torch.device("tpu")
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16, device=device)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.bfloat16, device=device)
    direct = add_op(x, y)

    @torch.compile(backend="tpu", dynamic=False, fullgraph=True)
    def compiled_add(a, b):
        return add_op(a, b)

    compiled = compiled_add(x, y)
    torch.accelerator.synchronize()
    expected = torch.tensor([5.0, 7.0, 9.0])
    if not torch.equal(direct.float().cpu(), expected):
        failures.append("direct pallas.jax_op vector add returned incorrect values")
    if not torch.equal(compiled.float().cpu(), expected):
        failures.append("compiled pallas.jax_op vector add returned incorrect values")

    manifest = environment_manifest(runner="doctor")
    source = manifest["source_control"]
    required_prefix = source["required_upstream_prefix"]
    actual_upstream = source["upstream_commit"]
    if required_prefix and (
        actual_upstream is None or not actual_upstream.startswith(required_prefix)
    ):
        failures.append(
            f"upstream HEAD {actual_upstream!r} does not match {required_prefix!r}"
        )
    report = {
        "passed": not failures,
        "failures": failures,
        "versions": versions,
        "device": str(device),
        "devices_available": torch.accelerator.device_count(),
        "bridge": "torch.tpu.pallas.jax_op",
        "manifest": manifest,
    }
    if output is not None:
        write_json(report, output)
    if failures:
        raise RuntimeError(json.dumps(report, indent=2))
    return report
