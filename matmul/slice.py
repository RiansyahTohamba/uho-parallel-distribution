import numpy as np

slice_a = np.random.rand(1,10)   # 2 matriks ukuran 2 x 3
print("full_a =\n", slice_a[0, :])
print("slice_a =\n", slice_a[0, 1:3])
