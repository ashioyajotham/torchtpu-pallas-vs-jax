import pytest

from torch_tpu_vs_jax_pallas.config import KernelConfig


def test_default_config_is_valid():
    config = KernelConfig()
    assert config.shape == (1, 8, 256, 128)
    assert KernelConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seq_len": 0},
        {"head_dim": 64},
        {"block_q": 64},
        {"seq_len": 384, "block_q": 256},
    ],
)
def test_invalid_configs_fail_early(kwargs):
    with pytest.raises(ValueError):
        KernelConfig(**kwargs)
