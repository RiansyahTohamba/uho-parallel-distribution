"""
attention_numpy.py — Transformer internals from scratch, NumPy only.
Teaching reference: every tensor shape is printed so students can trace them.
Run: python3 attention_numpy.py
"""
import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. Softmax (numerically stable) — the only nonlinearity inside attention
# ----------------------------------------------------------------------
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)                      # subtract max -> no overflow
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# 2. Scaled dot-product attention
#    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
# ----------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., n_q, d_k)
    K: (..., n_k, d_k)
    V: (..., n_k, d_v)
    mask: broadcastable to (..., n_q, n_k); True = keep, False = block
    returns: out (..., n_q, d_v), weights (..., n_q, n_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # (..., n_q, n_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


# ----------------------------------------------------------------------
# 3. Causal mask — what makes a decoder autoregressive
# ----------------------------------------------------------------------
def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


# ----------------------------------------------------------------------
# 4. Multi-head attention as ONE batched matmul (the parallelism trick)
# ----------------------------------------------------------------------
def split_heads(X, h):
    """(B, T, d_model) -> (B, h, T, d_head). Pure reshape+transpose: no math."""
    B, T, d_model = X.shape
    assert d_model % h == 0, "d_model must be divisible by n_heads"
    d_head = d_model // h
    return X.reshape(B, T, h, d_head).transpose(0, 2, 1, 3)


def merge_heads(X):
    """(B, h, T, d_head) -> (B, T, d_model)"""
    B, h, T, d_head = X.shape
    return X.transpose(0, 2, 1, 3).reshape(B, T, h * d_head)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng):
        self.h = n_heads
        s = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))

    def __call__(self, X, mask=None):
        # 1) three projections — independent, could run concurrently
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        # 2) split into heads -> heads become a BATCH dimension
        Qh, Kh, Vh = (split_heads(t, self.h) for t in (Q, K, V))
        # 3) one call computes all heads at once
        Oh, W = scaled_dot_product_attention(Qh, Kh, Vh, mask)
        # 4) concat heads + output projection
        return merge_heads(Oh) @ self.Wo, W


# ----------------------------------------------------------------------
# 5. Layer norm, FFN, and a full encoder block (pre-norm variant)
# ----------------------------------------------------------------------
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class FeedForward:
    def __init__(self, d_model, d_ff, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.W2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))

    def __call__(self, x):
        return gelu(x @ self.W1) @ self.W2      # position-wise: no token talks to another


class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ffn = FeedForward(d_model, d_ff, rng)

    def __call__(self, x, mask=None):
        a, _ = self.attn(layer_norm(x), mask)
        x = x + a                               # residual 1
        x = x + self.ffn(layer_norm(x))         # residual 2
        return x


# ----------------------------------------------------------------------
# 6. Sinusoidal positional encoding
# ----------------------------------------------------------------------
def positional_encoding(T, d_model):
    pos = np.arange(T)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angle = pos / np.power(10000.0, i / d_model)
    pe = np.zeros((T, d_model))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ----------------------------------------------------------------------
# 7. The point of the whole lecture: RNN is sequential, attention is not
# ----------------------------------------------------------------------
def rnn_forward(X, Wx, Wh):
    """O(T) dependent steps. Step t cannot start before step t-1 finishes."""
    B, T, d = X.shape
    h = np.zeros((B, d))
    outs = []
    for t in range(T):                          # <-- this loop cannot be parallelised
        h = np.tanh(X[:, t] @ Wx + h @ Wh)
        outs.append(h)
    return np.stack(outs, axis=1)


if __name__ == "__main__":
    B, T, d_model, h, d_ff = 2, 6, 32, 4, 128
    X = rng.normal(0, 1, (B, T, d_model)) + positional_encoding(T, d_model)

    print(f"input                 {X.shape}")

    block = TransformerBlock(d_model, h, d_ff, rng)
    mask = causal_mask(T)                       # (T, T) broadcasts over (B, h, T, T)
    Y = block(X, mask)
    print(f"encoder block output  {Y.shape}")

    # inspect one head's attention pattern
    _, W = block.attn(layer_norm(X), mask)
    print(f"attention weights     {W.shape}   (batch, heads, query, key)")
    print("head 0, sample 0 (rows sum to 1, upper triangle is zero):")
    np.set_printoptions(precision=3, suppress=True)
    print(W[0, 0])
    print("row sums:", W[0, 0].sum(-1))

    # sanity check 1: uniform K,V -> attention returns the mean of V
    Q = rng.normal(size=(1, 3, 8))
    K = np.zeros((1, 5, 8))
    V = rng.normal(size=(1, 5, 8))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(out, V.mean(1, keepdims=True)), "uniform-attention check failed"
    print("\ncheck: identical keys -> uniform weights -> mean(V)  OK")

    # sanity check 2: causal masking really blocks the future
    _, w = scaled_dot_product_attention(Q[:, :3], K[:, :3], V[:, :3], causal_mask(3))
    assert np.allclose(np.triu(w[0], 1), 0), "causal mask leaked"
    print("check: causal mask has zero weight above the diagonal      OK")

    # sanity check 3: multi-head via batched matmul == looping over heads
    mha = MultiHeadAttention(d_model, h, rng)
    fast, _ = mha(X)
    Q, K, V = X @ mha.Wq, X @ mha.Wk, X @ mha.Wv
    dh = d_model // h
    slow = np.concatenate(
        [scaled_dot_product_attention(Q[..., i*dh:(i+1)*dh],
                                      K[..., i*dh:(i+1)*dh],
                                      V[..., i*dh:(i+1)*dh])[0]
         for i in range(h)], axis=-1) @ mha.Wo
    assert np.allclose(fast, slow), "batched heads != looped heads"
    print("check: batched all-heads matmul == per-head loop           OK")

    # sequential vs parallel work
    Wx = rng.normal(0, 0.1, (d_model, d_model))
    Wh = rng.normal(0, 0.1, (d_model, d_model))
    r = rnn_forward(X, Wx, Wh)
    print(f"\nRNN needed {T} dependent steps for T={T}")
    print("Attention needed 1 matmul  ->  depth O(1) vs O(T)")
