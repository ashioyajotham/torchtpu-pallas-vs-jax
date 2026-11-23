from torch_tpu_vs_jax_pallas.cli import _parser
from torch_tpu_vs_jax_pallas import doctor


def test_doctor_accepts_explicit_torch_tpu_version():
    args = _parser().parse_args(
        [
            "doctor",
            "--out",
            "candidate.json",
            "--expected-torch-tpu",
            "0.2.0",
            "--expected-torch",
            "2.15.0+cpu",
            "--expected-libtpu",
            "0.0.51",
            "--expected-jax",
            "0.9.2",
        ]
    )
    assert args.out == "candidate.json"
    assert args.expected_torch_tpu == "0.2.0"
    assert args.expected_torch == "2.15.0+cpu"
    assert args.expected_libtpu == "0.0.51"
    assert args.expected_jax == "0.9.2"


def test_doctor_uses_pinned_version_by_default():
    args = _parser().parse_args(["doctor"])
    assert args.expected_torch_tpu is None
    assert args.expected_torch is None
    assert args.expected_libtpu is None
    assert args.expected_jax is None
    assert doctor.EXPECTED_TORCH_TPU == "0.3.0"
    assert doctor.EXPECTED_TORCH == "2.15.0+cpu"
    assert doctor.EXPECTED_LIBTPU == "0.0.51"


def test_vm_runner_checks_selected_torchtpu_version():
    from pathlib import Path

    run = Path("scripts/run_v5e8.sh").read_text(encoding="utf-8")
    assert 'tpu-compare doctor --expected-torch-tpu "${TORCH_TPU_VERSION}"' in run


def test_doctor_uses_public_operator():
    from pathlib import Path

    source = Path(doctor.__file__).read_text(encoding="utf-8")
    assert 'torch.tpu.pallas.jax_op("recompute_doctor::add_vectors"' in source
    assert "from torch_tpu._internal import pallas" not in source


def test_active_vm_scripts_do_not_use_preview_registry():
    from pathlib import Path

    root = Path(__file__).parents[1]
    bootstrap = (root / "scripts/bootstrap_tpu_vm.sh").read_text(encoding="utf-8")
    run = (root / "scripts/run_v5e8.sh").read_text(encoding="utf-8")
    for source in (bootstrap, run):
        assert "0.1.1.dev" not in source
        assert "torch-tpu-virtual-registry" not in source
        assert "c5bb931da44489f0697d97ff94bddaaffeee94d6" in source
    assert "TORCH_TPU_SAFE_PROCESS_EXIT=${TORCH_TPU_SAFE_PROCESS_EXIT:-0}" in run
