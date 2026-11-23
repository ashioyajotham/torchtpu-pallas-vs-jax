import numpy as np

from torch_tpu_vs_jax_pallas.reference import attention_reference


def test_causal_first_token_returns_first_value():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 1, 4, 8), dtype=np.float32)
    k = rng.standard_normal((1, 1, 4, 8), dtype=np.float32)
    v = rng.standard_normal((1, 1, 4, 8), dtype=np.float32)
    out = attention_reference(q, k, v, causal=True)
    np.testing.assert_allclose(out[:, :, 0], v[:, :, 0], atol=1e-6)


def test_reference_is_numerically_stable():
    q = np.full((1, 1, 4, 8), 100.0, dtype=np.float32)
    k = np.full_like(q, 100.0)
    v = np.arange(q.size, dtype=np.float32).reshape(q.shape)
    assert np.isfinite(attention_reference(q, k, v, causal=False)).all()
