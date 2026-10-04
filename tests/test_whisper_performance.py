from types import SimpleNamespace

import mlx.core as mx

from stt_mlx.whisper_performance import fast_qkv_attention


def reference_attention(q: mx.array, k: mx.array, v: mx.array, n_head: int) -> mx.array:
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // n_head) ** -0.25
    q = q.reshape(n_batch, n_ctx, n_head, -1).transpose(0, 2, 1, 3) * scale
    k = k.reshape(n_batch, k.shape[1], n_head, -1).transpose(0, 2, 3, 1) * scale
    v = v.reshape(n_batch, v.shape[1], n_head, -1).transpose(0, 2, 1, 3)
    weights = mx.softmax(q @ k, axis=-1, precise=True)
    return (weights @ v).transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)


def test_fast_attention_matches_reference() -> None:
    q = mx.random.normal((2, 12, 32)).astype(mx.float16)
    k = mx.random.normal((2, 12, 32)).astype(mx.float16)
    v = mx.random.normal((2, 12, 32)).astype(mx.float16)

    actual, qk = fast_qkv_attention(SimpleNamespace(n_head=4), q, k, v)
    expected = reference_attention(q, k, v, n_head=4)
    mx.eval(actual, expected)

    # Fused and unfused float16 kernels can differ by a few quantization steps.
    assert mx.allclose(actual, expected, rtol=1e-3, atol=5e-3)
    assert qk is None


def test_incremental_attention_can_see_all_cached_keys() -> None:
    q = mx.random.normal((1, 1, 32)).astype(mx.float16)
    k = mx.random.normal((1, 10, 32)).astype(mx.float16)
    v = mx.random.normal((1, 10, 32)).astype(mx.float16)
    causal_mask = mx.full((20, 20), -mx.inf)

    actual, _ = fast_qkv_attention(SimpleNamespace(n_head=4), q, k, v, causal_mask)
    expected = reference_attention(q, k, v, n_head=4)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-3)
