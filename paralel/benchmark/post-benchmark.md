Betul. Ini konsep penting dalam GPU computing: **GPU tidak bekerja sendirian**. CPU bertindak sebagai *host* yang menyiapkan data, menjalankan program, dan terutama **meluncurkan kernel GPU**.

Namun istilahnya lebih tepat bukan "CPU harus mendukung GPU", melainkan **CPU + PCIe + memory subsystem + software stack harus mampu memberi makan GPU dengan cukup cepat**.

### 1. Arsitektur sederhananya

```text
             CPU (Host)
                │
       ┌────────┴────────┐
       │                 │
    System RAM        PCIe
                         │
                         ▼
                    GPU (Device)
                    ┌───────────┐
                    │ CUDA      │
                    │ Thousands │
                    │ of cores  │
                    └───────────┘
```

CPU melakukan:

1. menyiapkan input
2. mengalokasikan/mengatur memory
3. meluncurkan CUDA kernel
4. mengatur execution
5. menerima/mengecek hasil

GPU kemudian melakukan bagian komputasi paralel yang berat.

---

## 2. Kalau ingin "menunjukkan otot GPU", konfigurasi idealnya bagaimana?

Untuk benchmark training/inference, prinsipnya:

### CPU: jangan menjadi bottleneck

Yang penting bukan sekadar CPU mahal, tetapi:

* cukup banyak core/thread
* single-core performance bagus
* memory bandwidth cukup
* PCIe yang memadai
* tidak dibatasi oleh virtualisasi/WSL secara ekstrem

Misalnya GPU kelas tinggi dipasangkan dengan CPU yang sangat lemah:

```text
CPU
████████████████████  ← sibuk launch kernel

GPU
██                  ← banyak waktu menunggu
```

Hasil benchmark akan terlihat buruk.

Dengan CPU yang cukup kuat:

```text
CPU
████                ← launch/prepare

GPU
████████████████████████████
```

GPU lebih terutilisasi.

---

## 3. Dalam kasus benchmark Anda, masalah utamanya sangat jelas

Ada kalimat:

> **WSL exposes only 3 cores to launch kernels**

Ini berarti benchmark tersebut bukan hanya menguji GPU.

Ia juga menguji:

```text
GPU + CPU host + WSL configuration
```

Dengan hanya 3 CPU cores, CPU host dapat menjadi bottleneck untuk workload yang terdiri dari banyak kernel kecil.

Ini menjelaskan mengapa:

```text
MLP batch 128
GPU ≈ CPU
```

tetapi:

```text
MLP batch 8192
GPU ≈ 5× CPU
```

Batch besar membuat setiap kernel melakukan pekerjaan lebih banyak sehingga overhead CPU relatif semakin kecil.

---

# 4. Faktor yang menentukan "otot" GPU

Untuk benchmark CUDA, saya akan melihat setidaknya ini:

### CPU

```text
Core/thread       → cukup banyak
Single-core speed  → tinggi
Memory bandwidth   → cukup tinggi
PCIe               → sesuai GPU
```

### GPU

Yang sangat penting:

```text
CUDA cores / SM
Tensor Cores
VRAM capacity
VRAM bandwidth
FP32 throughput
BF16/FP16 throughput
FP8 throughput (jika relevan)
```

Misalnya untuk deep learning modern, **Tensor Core throughput** bisa jauh lebih penting daripada sekadar jumlah CUDA cores.

---

# 5. Tetapi CPU tidak selalu harus "sangat kuat"

Ini bagian yang sering disalahpahami.

Misalnya Anda melakukan:

```text
GEMM 4096 × 4096
```

Satu operasi matrix multiplication besar.

GPU mendapat pekerjaan besar:

```text
CPU
 │
 │ launch
 ▼
GPU
████████████████████████████████
```

CPU relatif santai setelah kernel diluncurkan.

Sebaliknya:

```text
MLP kecil
batch kecil
banyak operasi kecil
```

bisa menjadi:

```text
CPU
████████████████████████████

GPU
██ ██ ██ ██ ██ ██
↑
sering menunggu
```

Jadi **GPU yang sangat kuat tidak otomatis memberikan speedup besar**.

---

# 6. Ada konsep penting: GPU utilization

Untuk melihat apakah GPU benar-benar "menunjukkan otot", jangan hanya melihat speedup.

Lihat juga:

```text
GPU utilization
GPU memory utilization
GPU memory bandwidth
SM utilization
Tensor Core utilization
CPU utilization
```

Misalnya:

```text
GPU utilization = 95%
CPU utilization = 40%
```

biasanya GPU memang sedang bekerja keras.

Tetapi:

```text
GPU utilization = 25%
CPU utilization = 100%
```

bisa menunjukkan **CPU menjadi bottleneck**.

---

# 7. Untuk benchmark training, konfigurasi yang lebih ideal

Kalau tujuan Anda adalah membandingkan **CPU vs GPU secara fair**, saya akan membuat beberapa kategori workload:

```text
             Workload size
                  ↑
                  │
       Large      │   GPU territory
                  │
                  │
       Medium     │   transition
                  │
                  │
       Small      │   CPU territory
                  │
                  └──────────────────→
                         batch size
```

Kemudian ukur:

```text
batch = 32
batch = 128
batch = 512
batch = 2048
batch = 8192
```

dan beberapa model:

```text
GEMM
MLP
CNN
Transformer
```

Itu jauh lebih informatif daripada hanya mengatakan:

> "GPU 5× lebih cepat daripada CPU."

Karena Anda bisa menemukan **titik crossover**:

```text
              CPU lebih cepat
                    │
                    ▼
batch 32 ───────────●────────────
                   crossover
                         │
                         ▼
batch 8192 ──────────────── GPU lebih cepat
```

---

## 8. Dan ada satu hal lagi: `steps_per_execution=32`

Hasil benchmark Anda:

> `steps_per_execution=32` → **1.7× GPU**

Ini sangat menarik karena secara konsep ia **mengurangi frekuensi interaksi Python/CPU dengan GPU**.

Daripada:

```text
CPU → GPU
CPU → GPU
CPU → GPU
CPU → GPU
...
```

bisa dibuat lebih seperti:

```text
CPU
 │
 └──────→ GPU
          │
          ├─ step 1
          ├─ step 2
          ├─ step 3
          ├─ ...
          └─ step 32
```

Sehingga overhead host-side menjadi lebih kecil.

Jadi sebenarnya benchmark Anda sedang memperlihatkan prinsip yang sangat fundamental dalam GPU computing:

> **GPU paling efektif ketika computation-to-overhead ratio tinggi.**

Atau secara intuitif:

> **Beri GPU pekerjaan yang besar, paralel, dan cukup lama sehingga biaya CPU untuk mengaturnya menjadi kecil.**

Itulah kondisi ketika GPU benar-benar bisa "menunjukkan ototnya".
