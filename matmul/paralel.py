# matrix multiplication in parallel way for array 3D using CPU
def matmul_parallel_cpu(A, B):
    import numpy as np
    from joblib import Parallel, delayed

    def compute_row(i):
        return np.dot(A[i], B)

    num_rows = A.shape[0]
    result = Parallel(n_jobs=-1)(delayed(compute_row)(i) for i in range(num_rows))
    return np.array(result)
# matrix multiplication in parallel way for array 3D using GPU
def matmul_parallel_gpu(A, B):
    import cupy as cp
    # check whether device using GPU or not
    if not cp.cuda.is_available():
        raise RuntimeError("GPU is not available")

    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    result_gpu = cp.dot(A_gpu, B_gpu)
    return cp.asnumpy(result_gpu)

