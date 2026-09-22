# matrix multiplication for array 3D
import numpy as np

# Array 3D = tumpukan (stack) beberapa matriks 2D.
# Bentuknya (b, N, M) = b buah matriks berukuran N x M.
# Pada citra digital, (N, M, b) juga array 3D, tetapi layer warna ada di sumbu terakhir.


A = np.arange(1, 13).reshape(2, 2, 3)   # 2 matriks ukuran 2 x 3
B = np.arange(1, 13).reshape(2, 3, 2)   # 2 matriks ukuran 3 x 2

print("A =\n", A)
print("shape A =", A.shape)
print("B =\n", B)
print("shape B =", B.shape)

# ---------------------------------------------------------------
# 1. Perkalian matriks (batch) dengan numpy
# ---------------------------------------------------------------
# numpy memperlakukan dua sumbu(index) terakhir sebagai matriks,
# sumbu(index) pertama dianggap "batch" dan dikerjakan satu per satu.
# (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
C = A @ B                 # sama dengan np.matmul(A, B)
print("\nA @ B =\n", C)
print("shape hasil =", C.shape)

# Syarat perkalian tetap berlaku: kolom A harus sama dengan baris B
print("kolom A =", A.shape[-1], ", baris B =", B.shape[-2])

# ---------------------------------------------------------------
# 2. Perhitungan manual (tiga perulangan bersarang per layer)
# ---------------------------------------------------------------
b, N, K = A.shape
M = B.shape[-1]
manual = np.zeros((b, N, M), dtype=A.dtype)

for layer in range(b):            # untuk setiap matriks pada tumpukan
    for i in range(N):            # baris
        for j in range(M):        # kolom
            total = 0
            for k in range(K):    # penjumlahan hasil kali baris x kolom
                total += A[layer, i, k] * B[layer, k, j]
            manual[layer, i, j] = total

print("\nhasil manual =\n", manual)
print("manual sama dengan numpy?", np.array_equal(manual, C))

# ---------------------------------------------------------------
# 3. Beda perkalian matriks dengan perkalian element-wise
# ---------------------------------------------------------------
# A * B tidak bisa karena bentuknya berbeda, jadi pakai B ditranspose
Bt = B.transpose(0, 2, 1)         # (2, 3, 2) -> (2, 2, 3)
print("\nA * Bt (element-wise) =\n", A * Bt)
print("A @ B  (perkalian matriks) =\n", C)

# ---------------------------------------------------------------
# 4. Penerapan pada citra: transformasi warna 3 x 3
# ---------------------------------------------------------------
# Citra BGR berbentuk (N, M, 3). Setiap piksel adalah vektor [B, G, R]
# yang dikalikan dengan matriks transformasi 3 x 3.
citra = np.array([
    [[255,   0,   0], [  0, 255,   0]],
    [[  0,   0, 255], [128, 128, 128]],
], dtype=np.float32)              # citra kecil 2 x 2 piksel, format BGR
print("\ncitra =\n", citra)
print("shape citra =", citra.shape)

# Bobot grayscale (BGR): 0.114 B + 0.587 G + 0.299 R
bobot_gray = np.array([0.114, 0.587, 0.299], dtype=np.float32)
gray = citra @ bobot_gray         # (2, 2, 3) @ (3,) -> (2, 2)
print("\ngrayscale =\n", gray)
print("shape grayscale =", gray.shape)

# Matriks sepia untuk urutan BGR (baris = keluaran, kolom = masukan)
sepia = np.array([
    [0.131, 0.534, 0.272],        # kanal B keluaran
    [0.168, 0.686, 0.349],        # kanal G keluaran
    [0.189, 0.769, 0.393],        # kanal R keluaran
], dtype=np.float32)

# Setiap piksel: sepia @ [B, G, R]. Karena piksel tersimpan sebagai baris,
# matriks sepia ditranspose lalu dikalikan dari kanan.
citra_sepia = np.clip(citra @ sepia.T, 0, 255).astype(np.uint8)
print("\ncitra sepia =\n", citra_sepia)
print("shape citra sepia =", citra_sepia.shape)
