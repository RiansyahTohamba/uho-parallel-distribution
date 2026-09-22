"""
attention_cuda.py — the same attention math, written as CUDA kernels with Numba.
Purpose: show WHERE the parallelism lives and WHY memory traffic, not FLOPs,
is the bottleneck.

Real GPU:      python3 attention_cuda.py
No GPU (logic check only, very slow, use tiny sizes):
               NUMBA_ENABLE_CUDASIM=1 python3 attention_cuda.py
"""
import math
import numpy as np
from numba import cuda, float32

TILE = 16          # tile edge; TILE*TILE = 256 threads per block


# ----------------------------------------------------------------------
# Kernel 1: naive matmul. One thread = one output element.
# Each thread reads a whole row of A and a whole column of B from GLOBAL
# memory -> every element of A is re-read N times. Bandwidth-bound.
# ----------------------------------------------------------------------
@cuda.jit
def matmul_naive(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        acc = float32(0.0)
        for k in range(A.shape[1]):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


# ----------------------------------------------------------------------
# Kernel 2: tiled matmul. A block cooperatively stages TILExTILE sub-blocks
# in SHARED memory (on-chip, ~100x lower latency than global), so each
# loaded value is reused TILE times. Same FLOPs, far less DRAM traffic.
# This is the single most important GPU optimisation to teach.
# ----------------------------------------------------------------------
@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    acc = float32(0.0)
    n_tiles = (A.shape[1] + TILE - 1) // TILE

    for t in range(n_tiles):
        k = t * TILE
        # cooperative load, with bounds guards for ragged edges
        sA[ty, tx] = A[row, k + tx] if (row < A.shape[0] and k + tx < A.shape[1]) else 0.0
        sB[ty, tx] = B[k + ty, col] if (k + ty < B.shape[0] and col < B.shape[1]) else 0.0
        cuda.syncthreads()                 # everyone must finish loading

        for i in range(TILE):              # now read from fast shared memory
            acc += sA[ty, i] * sB[i, tx]
        cuda.syncthreads()                 # before anyone overwrites the tile

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


# ----------------------------------------------------------------------
# Kernel 3: scale + causal mask + row-wise softmax.
# One BLOCK per query row. Threads inside the block cooperate on two
# reductions (max, then sum) via shared memory tree reduction.
# Teaching point: attention needs cross-thread communication, and the
# unit of communication is the block, not the grid.
# ----------------------------------------------------------------------
@cuda.jit
def softmax_rows_causal(S, scale, causal):
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    n = S.shape[1]

    buf = cuda.shared.array(256, dtype=float32)

    # ---- pass 1: row max (for numerical stability) ----
    local = float32(-1e30)
    i = tid
    while i < n:
        if causal and i > row:
            S[row, i] = -1e30              # block the future
        else:
            S[row, i] = S[row, i] * scale
        if S[row, i] > local:
            local = S[row, i]
        i += nthreads
    buf[tid] = local
    cuda.syncthreads()

    s = nthreads // 2
    while s > 0:
        if tid < s and buf[tid + s] > buf[tid]:
            buf[tid] = buf[tid + s]
        cuda.syncthreads()
        s //= 2
    row_max = buf[0]
    cuda.syncthreads()

    # ---- pass 2: exponentiate and sum ----
    local = float32(0.0)
    i = tid
    while i < n:
        e = math.exp(S[row, i] - row_max)
        S[row, i] = e
        local += e
        i += nthreads
    buf[tid] = local
    cuda.syncthreads()

    s = nthreads // 2
    while s > 0:
        if tid < s:
            buf[tid] += buf[tid + s]
        cuda.syncthreads()
        s //= 2
    total = buf[0]
    cuda.syncthreads()

    # ---- pass 3: normalise ----
    i = tid
    while i < n:
        S[row, i] = S[row, i] / total
        i += nthreads


# ----------------------------------------------------------------------
# Host-side driver: single-head attention entirely on the GPU
# ----------------------------------------------------------------------
def attention_gpu(Q, K, V, causal=True, threads_per_row=256):
    T, d_k = Q.shape
    d_v = V.shape[1]
    # the shared-memory tree reduction requires a power-of-two thread count
    # that fits the 256-element buffer declared inside the kernel
    assert threads_per_row in (32, 64, 128, 256), "threads_per_row must be 32/64/128/256"

    dQ, dK, dV = cuda.to_device(Q), cuda.to_device(K), cuda.to_device(V)
    dS = cuda.device_array((T, T), dtype=np.float32)
    dO = cuda.device_array((T, d_v), dtype=np.float32)

    grid = ((T + TILE - 1) // TILE, (T + TILE - 1) // TILE)
    matmul_tiled[grid, (TILE, TILE)](dQ, cuda.to_device(np.ascontiguousarray(K.T)), dS)

    softmax_rows_causal[T, threads_per_row](dS, np.float32(1.0 / math.sqrt(d_k)), causal)

    grid2 = ((d_v + TILE - 1) // TILE, (T + TILE - 1) // TILE)
    matmul_tiled[grid2, (TILE, TILE)](dS, dV, dO)

    return dO.copy_to_host(), dS.copy_to_host()


def attention_cpu(Q, K, V, causal=True):
    d_k = Q.shape[1]
    S = Q @ K.T / math.sqrt(d_k)
    if causal:
        S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    W = E / E.sum(-1, keepdims=True)
    return W @ V, W


if __name__ == "__main__":
    import os, time
    sim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
    T, d = (32, 16) if sim else (1024, 64)
    nthreads = 32 if sim else 256

    rng = np.random.default_rng(0)
    Q = rng.normal(size=(T, d)).astype(np.float32)
    K = rng.normal(size=(T, d)).astype(np.float32)
    V = rng.normal(size=(T, d)).astype(np.float32)

    O_gpu, W_gpu = attention_gpu(Q, K, V, causal=True, threads_per_row=nthreads)
    O_cpu, W_cpu = attention_cpu(Q, K, V, causal=True)

    print(f"T={T} d={d}   max |GPU-CPU| output diff: {np.abs(O_gpu-O_cpu).max():.2e}")
    print(f"                max |GPU-CPU| weight diff: {np.abs(W_gpu-W_cpu).max():.2e}")
    print("row sums (should be 1):", W_gpu.sum(-1)[:4])
    assert np.abs(np.triu(W_gpu, 1)).max() == 0.0, "causal mask leaked"
    print("causal mask holds on GPU: OK")

    if not sim:
        # naive vs tiled, same FLOPs, different memory traffic
        A = rng.normal(size=(1024, 1024)).astype(np.float32)
        B = rng.normal(size=(1024, 1024)).astype(np.float32)
        dA, dB = cuda.to_device(A), cuda.to_device(B)
        dC = cuda.device_array((1024, 1024), dtype=np.float32)
        grid = (64, 64)
        for name, kern in (("naive", matmul_naive), ("tiled", matmul_tiled)):
            kern[grid, (TILE, TILE)](dA, dB, dC); cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                kern[grid, (TILE, TILE)](dA, dB, dC)
            cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20
            gflops = 2 * 1024**3 / dt / 1e9
            print(f"{name:6s} {dt*1e3:7.2f} ms  {gflops:8.1f} GFLOP/s")
