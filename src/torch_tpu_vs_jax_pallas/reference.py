from __future__ import annotations

import numpy as np


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool) -> np.ndarray:
    """Float32 host reference for correctness-sized fixtures."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(q.shape[-1]))
    logits = np.einsum("bhsd,bhtd->bhst", q, k, dtype=np.float32) * scale
    if causal:
        mask = np.arange(q.shape[2])[:, None] >= np.arange(k.shape[2])[None, :]
        logits = np.where(mask[None, None], logits, np.float32(-1e30))
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits, dtype=np.float32)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", weights, v, dtype=np.float32)
