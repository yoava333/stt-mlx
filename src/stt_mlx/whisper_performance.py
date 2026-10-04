"""Performance overrides for the mlx_whisper inference path used by stt."""

from __future__ import annotations

import mlx.core as mx


def fast_qkv_attention(
    attention: object,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, None]:
    """Compute attention with MLX's fused SDPA kernel.

    stt uses segment timestamps, not the optional word-alignment path, so the
    otherwise returned query-key matrix is unused.
    """
    n_batch, n_ctx, n_state = q.shape
    n_head = attention.n_head
    head_dim = n_state // n_head

    q = q.reshape(n_batch, n_ctx, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(n_batch, k.shape[1], n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(n_batch, v.shape[1], n_head, head_dim).transpose(0, 2, 1, 3)

    if mask is not None:
        # During incremental decoding the single query is the newest token, so
        # every cached key is visible. Multi-token decoding uses the causal mask.
        mask = mask[:n_ctx, :n_ctx] if n_ctx > 1 else None

    output = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=head_dim**-0.5,
        mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
    return output, None


def enable_fast_attention() -> None:
    from mlx_whisper.whisper import MultiHeadAttention

    MultiHeadAttention.qkv_attention = fast_qkv_attention
