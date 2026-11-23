from torch_tpu_vs_jax_pallas import process_exit


def test_normal_teardown_is_default(monkeypatch):
    monkeypatch.delenv("TORCH_TPU_SAFE_PROCESS_EXIT", raising=False)
    monkeypatch.setattr(process_exit.os, "_exit", lambda code: (_ for _ in ()).throw(AssertionError(code)))
    process_exit.finalize_tpu_process()


def test_safe_process_exit_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TORCH_TPU_SAFE_PROCESS_EXIT", "0")
    monkeypatch.setattr(process_exit.os, "_exit", lambda code: (_ for _ in ()).throw(AssertionError(code)))
    process_exit.finalize_tpu_process()


def test_safe_process_exit_uses_zero_after_success(monkeypatch):
    exits = []
    monkeypatch.setenv("TORCH_TPU_SAFE_PROCESS_EXIT", "1")
    monkeypatch.setattr(process_exit.os, "_exit", exits.append)
    process_exit.finalize_tpu_process()
    assert exits == [0]
