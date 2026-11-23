from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(path: str | Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_revision(path: str | Path) -> str | None:
    return _git(path, "rev-parse", "HEAD")


def git_dirty(path: str | Path) -> bool | None:
    status = _git(path, "status", "--porcelain")
    return None if status is None else bool(status)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pip_freeze() -> list[str]:
    try:
        output = subprocess.check_output(
            [os.sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in output.splitlines() if line)


def environment_manifest(*, runner: str) -> dict[str, Any]:
    packages = {
        name: _version(name)
        for name in ("torch", "torch_tpu", "jax", "jaxlib", "libtpu", "numpy")
    }
    wheel_path = os.environ.get("TORCH_TPU_WHEEL")
    comparison_repo = Path(os.environ.get("COMPARISON_REPO", Path.cwd()))
    upstream_repo_value = os.environ.get("UPSTREAM_REPO")
    upstream_repo = Path(upstream_repo_value) if upstream_repo_value else None
    return {
        "runner": runner,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "pip_freeze": _pip_freeze(),
        "source_control": {
            "comparison_commit": git_revision(comparison_repo),
            "comparison_dirty": git_dirty(comparison_repo),
            "upstream_commit": (
                git_revision(upstream_repo) if upstream_repo is not None else None
            ),
            "upstream_dirty": (
                git_dirty(upstream_repo) if upstream_repo is not None else None
            ),
            "required_upstream_prefix": os.environ.get("UPSTREAM_SHA"),
        },
        "artifacts": {
            "torch_tpu_version": os.environ.get("TORCH_TPU_VERSION"),
            "torch_tpu_wheel": wheel_path,
            "torch_tpu_wheel_sha256": (
                file_sha256(wheel_path)
                if wheel_path and Path(wheel_path).is_file()
                else None
            ),
        },
        "execution": {
            "compile_backend": "tpu",
            "dynamic_shapes": False,
            "fullgraph": True,
            "operator_names": [
                "recompute_pallas::flash_attention_forward",
                "recompute_pallas::flash_attention_backward",
            ],
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TPU_NAME",
                "TPU_WORKER_ID",
                "TPU_LAUNCH_BLOCKING",
                "TPU_DEFER_AND_FUSE",
                "JAX_INTERPRET_PALLAS",
                "JAX_COMPILATION_CACHE_DIR",
                "TORCH_TPU_SAFE_PROCESS_EXIT",
                "TORCH_TPU_TIER2_COMPILATION_CACHE",
                "UPSTREAM_REPO",
                "UPSTREAM_SHA",
            )
            if os.environ.get(key) is not None
        },
    }
