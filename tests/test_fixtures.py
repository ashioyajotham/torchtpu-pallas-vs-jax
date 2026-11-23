import numpy as np

from torch_tpu_vs_jax_pallas.config import KernelConfig
from torch_tpu_vs_jax_pallas.fixtures import (
    bf16_bits_to_float32,
    float32_to_bf16_bits,
    generate_fixture,
)


def test_bf16_roundtrip_uses_nearest_even():
    values = np.asarray([0.0, 1.0, -2.5, 3.1415927, 1e-8], dtype=np.float32)
    rounded = bf16_bits_to_float32(float32_to_bf16_bits(values))
    assert rounded.dtype == np.float32
    np.testing.assert_allclose(rounded, values, rtol=8e-3, atol=1e-8)


def test_fixture_is_bitwise_reproducible():
    config = KernelConfig(batch=1, heads=1, seq_len=256, seed=17)
    first = generate_fixture(config)
    second = generate_fixture(config)
    assert np.array_equal(first.q_bits, second.q_bits)
    assert first.config == config
    assert np.array_equal(first.do_bits, second.do_bits)


def test_adversarial_fixture_orders_tile_maxima():
    config = KernelConfig(batch=1, heads=1, seq_len=256)
    fixture = generate_fixture(config, adversarial=True)
    _, k, v, _ = fixture.arrays()
    assert k[..., :128, :].mean() > k[..., 128:, :].mean()
    assert v[..., :128, :].mean() < v[..., 128:, :].mean()
