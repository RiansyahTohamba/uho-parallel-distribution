# Transformer Internals & GPU Parallelism
### Teaching module — informatics undergraduates
**Duration:** 2 × 100-minute sessions + 1 lab
**Prerequisites:** linear algebra (matrix product), Python + NumPy, basic complexity notation
**Not required:** deep learning background, calculus of backprop

---

## Learning outcomes

By the end, a student can:

1. Write scaled dot-product attention from the formula, with correct tensor shapes.
2. Explain *why* the RNN was replaced — in terms of **sequential depth**, not accuracy.
3. Show that multi-head attention is one batched matrix multiply, not a loop over heads.
4. Map the attention computation onto the CUDA execution hierarchy (grid → block → warp → thread).
5. Explain why attention is **memory-bandwidth bound**, and what tiling does about it.

Companion code (both tested):
- `attention_numpy.py` — full block from scratch, NumPy only, with self-checking assertions
- `attention_cuda.py` — the same math as CUDA kernels via Numba, verified against the NumPy version

---

## Session 1 — The architecture

### 1.1 The framing question (10 min)

Do **not** open with "attention is all you need." Open with a complexity table on the board.

Sequence length `T`, hidden size `d`:

| Layer type | FLOPs per layer | **Sequential steps** | Max path between two tokens |
|---|---|---|---|
| Recurrent | `O(T · d²)` | **`O(T)`** | `O(T)` |
| Convolutional (kernel `k`) | `O(k · T · d²)` | `O(1)` | `O(log_k T)` |
| Self-attention | `O(T² · d)` | **`O(1)`** | `O(1)` |

The middle column is the whole lecture. Self-attention does *more* arithmetic than an RNN for short sequences, and it won anyway — because a GPU has tens of thousands of cores sitting idle while an RNN walks a `for` loop over timesteps.

> **Board line to keep up all session:** *Transformers traded work for depth. GPUs make that trade profitable.*

Run this to make it concrete — the `rnn_forward` function in `attention_numpy.py` has a Python loop that **cannot** be vectorised, because step `t` needs `h` from step `t-1`. Attention has no such loop.

### 1.2 Scaled dot-product attention (25 min)

Build it up as three questions.

**Q1: how should token `i` decide which other tokens matter?** Dot product of a *query* with each *key*:

```
s_ij = q_i · k_j
```

**Q2: why divide by √d_k?** If `q` and `k` have i.i.d. entries with unit variance, then `q·k` has variance `d_k`. For `d_k = 64` the scores land in roughly ±8 before scaling; softmax of that is nearly one-hot, and its gradient vanishes. Dividing by `√d_k` restores unit variance.

*Ask students to verify this numerically — it takes four lines and it is the single most-forgotten detail in the paper:*

```python
import numpy as np
rng = np.random.default_rng(0)
for d_k in (8, 64, 512):
    q = rng.normal(size=(10000, d_k)); k = rng.normal(size=(10000, d_k))
    s = (q * k).sum(-1)
    print(d_k, s.var().round(1), (s / np.sqrt(d_k)).var().round(3))
```

**Q3: how do we turn scores into a mixture?** Row-wise softmax, then a weighted sum of *values*:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

Shapes, written on the board and never erased:

```
Q  (B, T, d_k)      K  (B, T, d_k)      V  (B, T, d_v)
QKᵀ                 (B, T, T)      <-- the quadratic object
softmax(·)V         (B, T, d_v)
```

Two sanity checks students should run themselves (both are assertions in `attention_numpy.py`):

- Make all keys identical → weights become uniform → output equals `mean(V)`. Attention degenerates to average pooling.
- Apply the causal mask → every weight above the diagonal is exactly 0.

Emphasise the mask is `-inf` **before** softmax, never `0` after. Post-hoc zeroing breaks the normalisation.

### 1.3 Multi-head attention (20 min)

The intuition: one softmax produces one mixture. Syntax, coreference, and positional locality are different mixtures. So project into `h` subspaces of size `d_head = d_model / h` and attend independently.

The implementation point matters more than the intuition:

```
(B, T, d_model)
  --reshape-->    (B, T, h, d_head)
  --transpose-->  (B, h, T, d_head)      # h is now a BATCH dimension
```

No arithmetic happens in those two steps — only a view change. Then **one** batched matmul computes all heads simultaneously. `attention_numpy.py` asserts that the batched form is numerically identical to a Python loop over heads. Have students delete that assertion, run the loop version, and time both.

Total parameter count is unchanged by `h`: `4 · d_model²` for the Q, K, V, O projections regardless of head count. Students routinely guess it scales with `h`.

### 1.4 The rest of the block (25 min)

A transformer block is attention plus three things that look like plumbing and are not:

| Component | Does it mix tokens? | Why it exists |
|---|---|---|
| Multi-head attention | **Yes** — the only place | information routing |
| Position-wise FFN (`d_model → d_ff → d_model`, `d_ff ≈ 4·d_model`) | No | per-token nonlinear capacity; holds most of the parameters |
| Residual connection | No | gradient highway; makes 100-layer stacks trainable |
| Layer norm | No | keeps activation scale stable across depth |

**Pre-norm vs post-norm.** The 2017 paper used post-norm (`x + Sublayer(LayerNorm(x))` was *not* what it did — it did `LayerNorm(x + Sublayer(x))`). Every modern LLM uses **pre-norm**, `x + Sublayer(LayerNorm(x))`, because post-norm needs learning-rate warmup to avoid divergence at depth. The companion code uses pre-norm, and this is worth one slide — it is a case where the canonical paper is no longer the canonical practice.

**Positional information.** Attention is permutation-equivariant: shuffle the input tokens and the output shuffles identically. Prove it in one line with a permutation matrix. Therefore position must be injected. Sinusoidal encoding is in the code; mention that current models use RoPE (rotary) instead, applied to Q and K rather than added to the input.

**Homework for session 1:** three exercises at the end of this document.

---

## Session 2 — Where the parallelism is

### 2.1 The CUDA execution model, mapped onto attention (25 min)

Teach the hierarchy against the actual computation rather than in the abstract:

| CUDA level | Shares | Attention work that maps here |
|---|---|---|
| **Thread** | registers | one output element of `QKᵀ` |
| **Warp** (32 threads, lockstep) | registers, via shuffle | a partial softmax reduction |
| **Block** (≤1024 threads, one SM) | **shared memory**, `__syncthreads()` | one query row's full softmax; one output tile |
| **Grid** | global memory only | all rows × all heads × all batch items |

The rule students must internalise: **threads can only communicate cheaply inside a block.** That single constraint determines the shape of every kernel in `attention_cuda.py`.

So the parallelism decomposes as:

- **Embarrassingly parallel** (independent, any granularity): batch × heads × query rows. This is `B · h · T` independent problems — thousands of them, which is exactly why GPUs suit this.
- **Needs cooperation**: softmax over each row, because max and sum are reductions across the row.
- **Strictly sequential**: layer `L+1` needs layer `L`. Depth is not parallelisable. Nor is autoregressive *generation* — one token at a time, which is why inference is latency-bound while training is throughput-bound. Good exam question.

### 2.2 Kernel 1 & 2 — naive vs tiled matmul (30 min)

This is the core lab. Both kernels are in `attention_cuda.py`.

**Naive** (`matmul_naive`): one thread per output element, reading a full row of `A` and column of `B` from global memory. For `N×N`, each element of `A` is fetched from DRAM `N` times. Arithmetic intensity: **2 FLOPs per 8 bytes loaded ≈ 0.25 FLOP/byte.**

**Tiled** (`matmul_tiled`): each block stages a `16×16` sub-tile of `A` and `B` into shared memory, syncs, then every thread reuses those staged values 16 times.

```python
sA[ty, tx] = A[row, k + tx]      # cooperative load, coalesced
sB[ty, tx] = B[k + ty, col]
cuda.syncthreads()               # nobody computes until all have loaded
for i in range(TILE):
    acc += sA[ty, i] * sB[i, tx] # ~100x lower latency than global
cuda.syncthreads()               # nobody overwrites until all have computed
```

Same FLOPs. DRAM traffic cut by a factor of `TILE`. Intensity rises to ~4 FLOP/byte.

Then do the **roofline** calculation on the board with real numbers, e.g. an A100: ~1.5 TB/s HBM, ~19.5 TFLOP/s FP32. The break-even intensity is `19500 / 1500 ≈ 13 FLOP/byte`. Both kernels are well below it, so both are **bandwidth-bound** — which is the punchline. Students arrive believing GPUs are about FLOPs.

The two `syncthreads()` calls are also the place to teach race conditions concretely: comment out the second one, rerun, watch the answer become wrong and *non-deterministically* wrong.

### 2.3 Kernel 3 — softmax needs a reduction (20 min)

`softmax_rows_causal` launches **one block per query row**. Inside, threads stride over the row, then perform two shared-memory tree reductions — max, then sum:

```python
s = nthreads // 2
while s > 0:
    if tid < s:
        buf[tid] += buf[tid + s]
    cuda.syncthreads()
    s //= 2
```

Points to draw out:

- `log₂(nthreads)` steps instead of `nthreads`. Classic parallel reduction.
- The max pass exists purely for numerical stability, and it costs an extra full read of the row. Remember this — it is the setup for FlashAttention.
- Strided access (`i += nthreads`) is **coalesced**; a contiguous-chunk-per-thread split is not. Have students swap it and measure. This is usually the biggest single surprise in the lab.

### 2.4 FlashAttention, conceptually (15 min)

Naive attention materialises the `T×T` score matrix in HBM. At `T = 8192` that is 256 MB per head per layer in FP32 — and it is written and read three times (scores → softmax → weighted sum).

FlashAttention never writes it. It tiles over keys and maintains a **running** max and sum, rescaling the accumulator as new tiles arrive — the online softmax trick. Same result, `O(T)` memory instead of `O(T²)`, and 2–4× faster in practice purely from avoided memory traffic.

Ask the class: *given the roofline number from 2.2, would you expect a speedup from an algorithm that does slightly more arithmetic but far fewer memory round-trips?* They should now be able to answer yes, and say why. That is the moment the two sessions join up.

Closing honesty slide: **nobody ships these kernels.** cuBLAS, cuDNN and `torch.nn.functional.scaled_dot_product_attention` are hand-tuned per architecture and use tensor cores. We wrote them to understand the cost model, not to compete.

---

## Lab sheet

**Setup**

```bash
pip install numpy numba
python3 attention_numpy.py     # CPU only, always works
python3 attention_cuda.py      # needs an NVIDIA GPU + CUDA toolkit
```

No GPU in the room? Numba ships a simulator that executes kernels on the CPU. It is slow, so the script auto-shrinks to `T=32`:

```bash
NUMBA_ENABLE_CUDASIM=1 python3 attention_cuda.py
```

Logic, race conditions and shared-memory bugs all surface in the simulator; only timings do not. Verified working — it reproduces the NumPy result to ~2e-07.

**Tasks**

| # | Task | What it teaches |
|---|---|---|
| 1 | Verify score variance grows with `d_k`; remove the `√d_k` scaling and print the resulting softmax rows | why the scale factor exists |
| 2 | Replace the batched multi-head matmul with a Python loop over heads; time both | batching *is* the parallelism |
| 3 | Vary `TILE` ∈ {8, 16, 32}; plot GFLOP/s | occupancy vs shared-memory pressure; 32×32 = 1024 threads hits the block limit |
| 4 | Delete the second `cuda.syncthreads()` in `matmul_tiled`; run 10× | race conditions are non-deterministic |
| 5 | Change the softmax stride to contiguous chunks per thread; measure | memory coalescing |
| 6 | Plot naive-matmul time for `N` = 256…2048; fit the exponent | `O(N³)` empirically |
| 7 | Compute peak memory for the `T×T` scores at `T` = 512, 2048, 8192 in FP32 and FP16 | motivates FlashAttention |

**Assessment suggestion:** 40% working code, 40% a one-page roofline analysis of *their own* GPU (peak bandwidth, peak FLOP/s, measured intensity, verdict), 20% oral defence of one design choice.

---

## Session-1 exercises

1. **Permutation equivariance.** Let `P` be a permutation matrix. Show `Attention(PQ, PK, PV) = P · Attention(Q,K,V)`. Conclude why positional encoding is mandatory. *(One line if they use `softmax(PXPᵀ) = P softmax(X) Pᵀ`.)*
2. **Parameter count.** Compute total parameters for one block with `d_model=768`, `h=12`, `d_ff=3072`. Then recompute with `h=1`. Explain why the answer does not change.
3. **Cost crossover.** Attention is `O(T²·d)`, the FFN is `O(T·d·d_ff)` with `d_ff=4d`. Find the `T` at which attention overtakes the FFN in FLOPs. For `d=768`, verify it is around `T ≈ 3072` — and note that below that length, the FFN dominates. Most students assume attention always dominates.

---

## Errors to pre-empt

- Applying the causal mask *after* softmax → rows no longer sum to 1.
- Forgetting `√d_k` → saturated softmax, dead gradients.
- Believing multi-head attention adds parameters.
- Assuming a GPU is fast because of FLOPs. It is fast because of bandwidth *and* because there is enough independent work to hide latency.
- Confusing `d_k` with `d_model`. Insist on the shape table.
- Expecting the CUDA simulator to show speedups. It measures correctness only.
