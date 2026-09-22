"""
Benchmark CPU vs GPU (CUDA) untuk beban kerja AI: training dan inference.
Ditulis dengan Keras 3 (backend TensorFlow).

Disesuaikan untuk perangkat pada check_gpu/gpu_spec.md:
  GPU : RTX 3070 Ti Laptop, 8 GB VRAM, compute 8.6 (Ampere) -> bf16 + TF32, tanpa FP8
  CPU : i9-12900H, hanya 6 logical / 3 core yang terlihat di WSL
  RAM : 19 GiB total (~6 GiB bebas), swap 8 GiB
Semua kasus uji dibatasi agar muat di <4 GB VRAM dan <3 GB RAM, sehingga sisi CPU
tetap selesai dalam waktu yang wajar meski hanya punya 3 core.

Cara pakai
----------
    python cpu_vs_gpu.py                      # jalankan semua kasus di CPU lalu GPU
    python cpu_vs_gpu.py --preset quick       # versi cepat (~2-3 menit)
    python cpu_vs_gpu.py --preset full        # versi panjang, angka paling stabil
    python cpu_vs_gpu.py --only gemm,cnn      # pilih kasus tertentu
    python cpu_vs_gpu.py --device gpu         # satu device saja
    python cpu_vs_gpu.py --report hasil/      # tulis JSON + PNG ke folder lain

Catatan metodologi (ini yang membuat angkanya layak dikutip)
------------------------------------------------------------
1. CPU dan GPU dijalankan di *proses terpisah*. TensorFlow menempatkan op ke GPU
   sejak inisialisasi, jadi `with tf.device('/CPU:0')` di proses yang sama masih
   bocor ke GPU. Proses anak untuk CPU dijalankan dengan CUDA_VISIBLE_DEVICES=-1.
2. Setiap pengukuran didahului warm-up yang dibuang: iterasi pertama berisi
   tracing tf.function, autotune cuDNN, dan alokasi memori -- bukan biaya komputasi.
3. Eksekusi GPU bersifat asinkron. Timer selalu ditutup dengan sinkronisasi device
   eksplisit, kalau tidak yang terukur hanyalah kecepatan Python melempar kernel.
4. Yang dilaporkan adalah median, bukan rata-rata: satu iterasi yang kena thermal
   throttling atau scheduler WSL tidak boleh menggeser hasil.
5. Data dibangkitkan secara sintetis dengan seed tetap. Untuk mengukur *waktu*,
   isi data tidak berpengaruh -- yang penting bentuk tensor dan jumlah operasinya
   identik di kedua device, dan tidak ada dependensi unduhan dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Preset ukuran beban kerja
# ---------------------------------------------------------------------------

PRESETS = {
    "quick": {
        "gemm_sizes": [1024, 2048],
        "gemm_iters": 10,
        "mlp_samples": 20_000,
        "mlp_epochs": 2,
        "cnn_samples": 5_000,
        "cnn_epochs": 1,
        "trf_samples": 2_000,
        "trf_epochs": 1,
        "infer_batches": [1, 32, 256],
        "infer_iters": 30,
        "scaling_batches": [128, 1024, 4096],
    },
    "standard": {
        "gemm_sizes": [1024, 2048, 4096],
        "gemm_iters": 20,
        "mlp_samples": 60_000,
        "mlp_epochs": 3,
        "cnn_samples": 10_000,
        "cnn_epochs": 2,
        "trf_samples": 4_000,
        "trf_epochs": 1,
        "infer_batches": [1, 8, 64, 512],
        "infer_iters": 50,
        "scaling_batches": [128, 512, 2048, 8192],
    },
    "full": {
        "gemm_sizes": [1024, 2048, 4096, 8192],
        "gemm_iters": 30,
        "mlp_samples": 60_000,
        "mlp_epochs": 5,
        "cnn_samples": 20_000,
        "cnn_epochs": 3,
        "trf_samples": 8_000,
        "trf_epochs": 2,
        "infer_batches": [1, 8, 64, 512, 1024],
        "infer_iters": 100,
        "scaling_batches": [128, 512, 2048, 8192, 16384],
    },
}

ALL_CASES = ["gemm", "scaling", "overhead", "xla", "mlp", "cnn", "transformer",
             "inference", "bf16"]

SEED = 1337
MLP_BATCH = 128
CNN_BATCH = 128
TRF_BATCH = 32

# Hyperparameter transformer encoder kecil -- sengaja dibuat mirip bentuk beban
# kerja klasifikasi kalimat (seq pendek, encoder-only), bukan LLM generatif.
TRF_VOCAB = 20_000
TRF_SEQLEN = 128
TRF_DMODEL = 256
TRF_HEADS = 4
TRF_LAYERS = 4
TRF_FF = 1024
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Utilitas timing
# ---------------------------------------------------------------------------

def sync_device():
    """Blokir sampai semua kernel yang antre di device selesai.

    Wajib sebelum menutup timer: eksekusi GPU asinkron, tanpa ini yang terukur
    adalah waktu enqueue, bukan waktu komputasi.
    """
    import tensorflow as tf

    if hasattr(tf.test.experimental, "sync_devices"):
        tf.test.experimental.sync_devices()
    else:  # fallback: paksa transfer device->host
        tf.constant(0.0).numpy()


def timed_median(fn, iters, warmup=3):
    """Jalankan fn() beberapa kali, buang warm-up, kembalikan statistik detik."""
    for _ in range(warmup):
        fn()
    sync_device()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync_device()
        samples.append(time.perf_counter() - t0)

    samples.sort()
    return {
        "median_s": statistics.median(samples),
        "min_s": samples[0],
        "p90_s": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "iters": iters,
    }


def peak_vram_mb():
    import tensorflow as tf

    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return round(info["peak"] / 1024**2, 1)
    except Exception:
        return None


def reset_vram_stats():
    import tensorflow as tf

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data sintetis
# ---------------------------------------------------------------------------

def make_tabular(n, dim):
    import numpy as np

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((n, dim), dtype=np.float32)
    y = rng.integers(0, N_CLASSES, size=n).astype("int32")
    return x, y


def make_images(n):
    import numpy as np

    rng = np.random.default_rng(SEED)
    x = rng.random((n, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, N_CLASSES, size=n).astype("int32")
    return x, y


def make_tokens(n):
    import numpy as np

    rng = np.random.default_rng(SEED)
    x = rng.integers(0, TRF_VOCAB, size=(n, TRF_SEQLEN)).astype("int32")
    y = rng.integers(0, N_CLASSES, size=n).astype("int32")
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_mlp():
    import keras

    return keras.Sequential(
        [
            keras.Input(shape=(784,)),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(N_CLASSES, activation="softmax", dtype="float32"),
        ],
        name="mlp",
    )


def build_cnn():
    import keras

    L = keras.layers

    def block(x, filters):
        for _ in range(2):
            x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)
        return L.MaxPooling2D()(x)

    inp = keras.Input(shape=(32, 32, 3))
    x = block(inp, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(256, activation="relu")(x)
    out = L.Dense(N_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out, name="cnn")


def build_transformer():
    import keras
    import tensorflow as tf

    L = keras.layers
    inp = keras.Input(shape=(TRF_SEQLEN,), dtype="int32")

    tok = L.Embedding(TRF_VOCAB, TRF_DMODEL)(inp)
    pos = L.Embedding(TRF_SEQLEN, TRF_DMODEL)(tf.range(TRF_SEQLEN))
    x = tok + pos

    for _ in range(TRF_LAYERS):
        attn = L.MultiHeadAttention(num_heads=TRF_HEADS, key_dim=TRF_DMODEL // TRF_HEADS)(x, x)
        x = L.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = L.Dense(TRF_FF, activation="gelu")(x)
        ff = L.Dense(TRF_DMODEL)(ff)
        x = L.LayerNormalization(epsilon=1e-6)(x + ff)

    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(N_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out, name="transformer_encoder")


def compile_model(model, steps_per_execution=1):
    """steps_per_execution=1 adalah default Keras: setiap step kembali ke Python
    dan menyinkronkan device. Menaikkannya mengamortisasi overhead itu -- lihat
    kasus `overhead`."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[],
        steps_per_execution=steps_per_execution,
    )
    return model


# ---------------------------------------------------------------------------
# Kasus uji
# ---------------------------------------------------------------------------

def case_gemm(cfg, log):
    """Microbenchmark GEMM: batas atas throughput aritmetika device.

    Perkalian matriks N x N butuh 2*N^3 FLOP. Ini kasus paling menguntungkan GPU
    (compute-bound, akses memori teratur) sehingga berguna sebagai plafon: tidak
    ada beban kerja nyata yang speedup-nya melebihi angka di sini.
    """
    import tensorflow as tf

    results = []
    for n in cfg["gemm_sizes"]:
        a = tf.random.normal((n, n), dtype=tf.float32)
        b = tf.random.normal((n, n), dtype=tf.float32)

        @tf.function(reduce_retracing=True)
        def step():
            return tf.matmul(a, b)

        iters = max(3, cfg["gemm_iters"] // (1 + n // 2048))
        stat = timed_median(step, iters, warmup=2)
        gflops = (2.0 * n**3) / stat["median_s"] / 1e9
        log(f"  GEMM {n}x{n}: {stat['median_s']*1e3:8.2f} ms   {gflops:8.1f} GFLOP/s")
        results.append({"n": n, "gflops": round(gflops, 1), **stat})

        del a, b

    return {"case": "gemm", "unit": "GFLOP/s", "runs": results}


def _train_case(name, model, x, y, batch, epochs, log):
    """Ukur waktu training per epoch dengan model.fit.

    Satu epoch warm-up dijalankan pada potongan kecil terlebih dahulu supaya biaya
    tracing graph dan autotune cuDNN tidak masuk ke hasil.
    """
    reset_vram_stats()
    model.fit(x[: batch * 3], y[: batch * 3], batch_size=batch, epochs=1, verbose=0)
    sync_device()

    t0 = time.perf_counter()
    model.fit(x, y, batch_size=batch, epochs=epochs, verbose=0, shuffle=False)
    sync_device()
    total = time.perf_counter() - t0

    per_epoch = total / epochs
    samples_s = len(x) / per_epoch
    steps = -(-len(x) // batch)
    log(
        f"  {name}: {per_epoch:7.2f} s/epoch   {samples_s:9.1f} sampel/s   "
        f"{per_epoch / steps * 1e3:7.2f} ms/step"
    )
    return {
        "case": name,
        "unit": "sampel/s",
        "params": int(model.count_params()),
        "samples": len(x),
        "batch": batch,
        "epochs": epochs,
        "s_per_epoch": round(per_epoch, 3),
        "samples_per_s": round(samples_s, 1),
        "ms_per_step": round(per_epoch / steps * 1e3, 3),
        "peak_vram_mb": peak_vram_mb(),
    }


def case_scaling(cfg, log):
    """Sapuan ukuran batch pada satu langkah training MLP (tf.function murni).

    Ini kasus paling penting untuk perangkat ini. GPU punya biaya tetap per step
    -- peluncuran kernel, sinkronisasi, dan dispatch dari Python -- yang di WSL
    dengan 3 core terasa besar. Biaya itu tidak bergantung ukuran batch, sedangkan
    waktu komputasi CPU naik linear. Jadi ada titik potong: di bawahnya GPU kalah,
    di atasnya GPU menang, dan jaraknya melebar terus. Grafik dari kasus ini
    menjawab pertanyaan "kenapa GPU saya tidak terasa lebih cepat?".
    """
    import keras
    import tensorflow as tf

    model = build_mlp()
    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    runs = []
    for b in cfg["scaling_batches"]:
        xb, yb = make_tabular(b, 784)
        xb, yb = tf.constant(xb), tf.constant(yb)
        reset_vram_stats()
        iters = max(5, cfg["gemm_iters"])
        stat = timed_median(lambda a=xb, c=yb: train_step(a, c), iters, warmup=5)
        ms = stat["median_s"] * 1e3
        log(f"  batch={b:<6d} {ms:8.3f} ms/step   {b / stat['median_s']:10.0f} sampel/s")
        runs.append({
            "batch": b,
            "ms_per_step": round(ms, 4),
            "samples_per_s": round(b / stat["median_s"], 1),
            "peak_vram_mb": peak_vram_mb(),
        })
        del xb, yb

    return {"case": "scaling", "unit": "sampel/s", "params": int(model.count_params()),
            "runs": runs}


def case_overhead(cfg, log):
    """Efek `steps_per_execution` pada model.fit.

    Default Keras adalah 1: tiap step kembali ke Python dan menyinkronkan device
    untuk agregasi loss. Pada GPU ongkos itu bisa melebihi komputasinya sendiri.
    Menaikkan ke 32 membuat 32 step dijalankan dalam satu panggilan graph.
    Ini optimasi satu baris yang sering terlewat.
    """
    x, y = make_tabular(cfg["mlp_samples"], 784)
    runs = []
    for spe in (1, 32):
        model = compile_model(build_mlp(), steps_per_execution=spe)
        r = _train_case(f"mlp_fit_spe{spe}", model, x, y, MLP_BATCH, cfg["mlp_epochs"], log)
        r["steps_per_execution"] = spe
        runs.append(r)
        del model
    return {"case": "overhead", "unit": "sampel/s", "runs": runs}


def case_xla(cfg, log):
    """Efek jit_compile=True (XLA) pada training CNN.

    XLA sering direkomendasikan begitu saja. Kasus ini mengukurnya alih-alih
    mengasumsikannya: XLA mengompilasi ulang graph menjadi kernel yang menyatu,
    yang menolong bila beban kerja didominasi banyak op kecil, tetapi menambah
    waktu kompilasi dan bisa memilih kernel konvolusi yang lebih lambat daripada
    pilihan cuDNN. Hasilnya perlu diverifikasi per model dan per perangkat.
    """
    x, y = make_images(cfg["cnn_samples"])
    runs = []
    for jit in (False, True):
        model = build_cnn()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=[], steps_per_execution=32, jit_compile=jit)
        r = _train_case(f"cnn_fit_xla{int(jit)}", model, x, y, CNN_BATCH,
                        cfg["cnn_epochs"], log)
        r["jit_compile"] = jit
        runs.append(r)
        del model
    return {"case": "xla", "unit": "sampel/s", "runs": runs}


def case_mlp(cfg, log):
    """Training MLP dense. Kecil, murni matmul, tanpa conv -- speedup GPU nyata
    tapi moderat karena tiap step terlalu pendek untuk menutupi overhead launch."""
    x, y = make_tabular(cfg["mlp_samples"], 784)
    model = compile_model(build_mlp())
    return _train_case("mlp_train", model, x, y, MLP_BATCH, cfg["mlp_epochs"], log)


def case_cnn(cfg, log):
    """Training CNN pada citra 32x32. Konvolusi adalah beban kerja yang paling
    diuntungkan cuDNN + Tensor Core; di sinilah selisih CPU-GPU paling ekstrem."""
    x, y = make_images(cfg["cnn_samples"])
    model = compile_model(build_cnn())
    return _train_case("cnn_train", model, x, y, CNN_BATCH, cfg["cnn_epochs"], log)


def case_transformer(cfg, log):
    """Training encoder transformer kecil (seq 128). Bentuk beban kerjanya sama
    dengan fine-tuning BERT/DeBERTa untuk klasifikasi kalimat, hanya lebih kecil."""
    x, y = make_tokens(cfg["trf_samples"])
    model = compile_model(build_transformer())
    return _train_case("transformer_train", model, x, y, TRF_BATCH, cfg["trf_epochs"], log)


def case_inference(cfg, log):
    """Inference murni (forward pass) pada beberapa ukuran batch.

    Ini kasus yang paling sering disalahpahami. Pada batch 1 GPU sering kalah atau
    hanya imbang: waktunya didominasi latensi peluncuran kernel dan transfer
    host<->device, bukan komputasi. Keunggulan GPU baru muncul saat batch besar,
    ketika 46 SM benar-benar terisi penuh.
    """
    import tensorflow as tf

    out = {"case": "inference", "unit": "ms/batch", "models": {}}

    for label, builder, maker in (
        ("cnn", build_cnn, make_images),
        ("transformer", build_transformer, make_tokens),
    ):
        model = builder()
        max_b = max(cfg["infer_batches"])
        x, _ = maker(max_b)

        @tf.function(reduce_retracing=True)
        def forward(batch):
            return model(batch, training=False)

        runs = []
        for b in cfg["infer_batches"]:
            xb = tf.constant(x[:b])
            reset_vram_stats()
            stat = timed_median(lambda: forward(xb), cfg["infer_iters"], warmup=5)
            lat_ms = stat["median_s"] * 1e3
            thr = b / stat["median_s"]
            log(
                f"  {label} infer batch={b:<5d} {lat_ms:8.3f} ms/batch   "
                f"{thr:10.1f} sampel/s"
            )
            runs.append(
                {
                    "batch": b,
                    "latency_ms": round(lat_ms, 4),
                    "throughput_samples_s": round(thr, 1),
                    "peak_vram_mb": peak_vram_mb(),
                }
            )
            del xb

        out["models"][label] = {"params": int(model.count_params()), "runs": runs}
        del model, x

    return out


def case_bf16(cfg, log, on_gpu):
    """Training CNN + transformer dengan mixed precision bf16.

    Hanya bermakna di GPU: compute 8.6 punya Tensor Core bf16. Di CPU bf16
    diemulasikan dan justru lebih lambat, jadi kasus ini dilewati di sisi CPU.
    Dibandingkan terhadap hasil fp32 dari kasus cnn/transformer.
    """
    if not on_gpu:
        log("  dilewati: bf16 hanya diukur di GPU (di CPU bf16 diemulasi, bukan dipercepat)")
        return {"case": "bf16", "skipped": "cpu"}

    import keras

    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        xi, yi = make_images(cfg["cnn_samples"])
        r_cnn = _train_case("cnn_train_bf16", compile_model(build_cnn()), xi, yi,
                            CNN_BATCH, cfg["cnn_epochs"], log)
        del xi, yi

        xt, yt = make_tokens(cfg["trf_samples"])
        r_trf = _train_case("transformer_train_bf16", compile_model(build_transformer()),
                            xt, yt, TRF_BATCH, cfg["trf_epochs"], log)
    finally:
        keras.mixed_precision.set_global_policy("float32")

    return {"case": "bf16", "unit": "sampel/s", "runs": {"cnn": r_cnn, "transformer": r_trf}}


CASE_FN = {
    "gemm": case_gemm,
    "scaling": case_scaling,
    "overhead": case_overhead,
    "xla": case_xla,
    "mlp": case_mlp,
    "cnn": case_cnn,
    "transformer": case_transformer,
    "inference": case_inference,
    "bf16": case_bf16,
}


# ---------------------------------------------------------------------------
# Proses anak: menjalankan seluruh kasus pada SATU device
# ---------------------------------------------------------------------------

def run_child(device, cases, cfg, out_path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    on_gpu = bool(gpus)
    if device == "gpu" and not on_gpu:
        raise SystemExit("GPU diminta tetapi tidak terdeteksi TensorFlow.")
    if device == "cpu" and on_gpu:
        raise SystemExit("Isolasi CPU gagal: GPU masih terlihat oleh proses ini.")

    if on_gpu:
        # Tanpa ini TF mengambil seluruh VRAM di awal, sehingga peak memory
        # tidak terbaca dan proses lain di 8 GB yang sama ikut tercekik.
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

    def log(msg):
        print(msg, flush=True)

    header = {
        "device": device,
        "device_name": (
            tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
            if on_gpu
            else platform.processor() or "CPU"
        ),
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "cpu_threads_visible": os.cpu_count(),
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }
    log(f"\n=== DEVICE: {device.upper()} ({header['device_name']}) ===")

    results = {"meta": header, "cases": {}}
    for name in cases:
        log(f"\n[{device}] {name}")
        t0 = time.perf_counter()
        fn = CASE_FN[name]
        res = fn(cfg, log, on_gpu) if name == "bf16" else fn(cfg, log)
        res["wall_s"] = round(time.perf_counter() - t0, 2)
        results["cases"][name] = res

    Path(out_path).write_text(json.dumps(results, indent=2))
    log(f"\n[{device}] selesai -> {out_path}")


# ---------------------------------------------------------------------------
# Pelaporan
# ---------------------------------------------------------------------------

def fmt_speedup(gpu_val, cpu_val, higher_is_better=True):
    if not gpu_val or not cpu_val:
        return "-"
    ratio = gpu_val / cpu_val if higher_is_better else cpu_val / gpu_val
    return f"{ratio:.1f}x"


def build_report(cpu, gpu):
    """Kembalikan baris-baris tabel perbandingan (list of tuple)."""
    rows = []

    def c(name):
        return (cpu or {}).get("cases", {}).get(name)

    def g(name):
        return (gpu or {}).get("cases", {}).get(name)

    if c("gemm") and g("gemm"):
        cmap = {r["n"]: r["gflops"] for r in c("gemm")["runs"]}
        for r in g("gemm")["runs"]:
            n = r["n"]
            if n in cmap:
                rows.append(
                    ("GEMM", f"{n}x{n} fp32", "GFLOP/s",
                     f"{cmap[n]:.1f}", f"{r['gflops']:.1f}",
                     fmt_speedup(r["gflops"], cmap[n]))
                )

    if c("scaling") and g("scaling"):
        cmap = {r["batch"]: r for r in c("scaling")["runs"]}
        for r in g("scaling")["runs"]:
            b = r["batch"]
            if b in cmap:
                rows.append(
                    ("Skala batch (MLP)", f"batch {b}", "sampel/s",
                     f"{cmap[b]['samples_per_s']:.0f}", f"{r['samples_per_s']:.0f}",
                     fmt_speedup(r["samples_per_s"], cmap[b]["samples_per_s"]))
                )

    if c("overhead") and g("overhead"):
        cmap = {r["steps_per_execution"]: r for r in c("overhead")["runs"]}
        for r in g("overhead")["runs"]:
            spe = r["steps_per_execution"]
            if spe in cmap:
                rows.append(
                    ("Overhead fit", f"steps_per_execution={spe}", "sampel/s",
                     f"{cmap[spe]['samples_per_s']:.0f}", f"{r['samples_per_s']:.0f}",
                     fmt_speedup(r["samples_per_s"], cmap[spe]["samples_per_s"]))
                )

    if c("xla") and g("xla"):
        cmap = {r["jit_compile"]: r for r in c("xla")["runs"]}
        for r in g("xla")["runs"]:
            j = r["jit_compile"]
            if j in cmap:
                rows.append(
                    ("XLA (CNN)", f"jit_compile={j}", "sampel/s",
                     f"{cmap[j]['samples_per_s']:.0f}", f"{r['samples_per_s']:.0f}",
                     fmt_speedup(r["samples_per_s"], cmap[j]["samples_per_s"]))
                )

    for key, label in (("mlp", "Training MLP"), ("cnn", "Training CNN"),
                       ("transformer", "Training Transformer")):
        cc, gg = c(key), g(key)
        if cc and gg and "samples_per_s" in cc:
            rows.append(
                (label, f"batch {cc['batch']}, {cc['params']/1e6:.1f}M param", "sampel/s",
                 f"{cc['samples_per_s']:.1f}", f"{gg['samples_per_s']:.1f}",
                 fmt_speedup(gg["samples_per_s"], cc["samples_per_s"]))
            )

    cc, gg = c("inference"), g("inference")
    if cc and gg:
        for mname in cc.get("models", {}):
            if mname not in gg.get("models", {}):
                continue
            cmap = {r["batch"]: r for r in cc["models"][mname]["runs"]}
            for r in gg["models"][mname]["runs"]:
                b = r["batch"]
                if b not in cmap:
                    continue
                rows.append(
                    (f"Inference {mname}", f"batch {b}", "ms/batch",
                     f"{cmap[b]['latency_ms']:.3f}", f"{r['latency_ms']:.3f}",
                     fmt_speedup(r["latency_ms"], cmap[b]["latency_ms"], higher_is_better=False))
                )

    gb = g("bf16")
    if gb and "runs" in gb:
        for mname, run in gb["runs"].items():
            base = g(mname)
            if base and "samples_per_s" in base:
                rows.append(
                    (f"bf16 vs fp32 ({mname})", "GPU saja", "sampel/s",
                     f"{base['samples_per_s']:.1f} (fp32)",
                     f"{run['samples_per_s']:.1f} (bf16)",
                     fmt_speedup(run["samples_per_s"], base["samples_per_s"]))
                )

    return rows


def print_report(rows):
    if not rows:
        print("\nTidak ada pasangan hasil CPU+GPU untuk dibandingkan.")
        return
    head = ("Kasus", "Konfigurasi", "Satuan", "CPU", "GPU", "Speedup")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(6)]
    line = "  ".join("-" * w for w in widths)
    print("\n" + "=" * len(line))
    print("RINGKASAN CPU vs GPU")
    print("=" * len(line))
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    print(line)
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(line)
    print("Speedup > 1 berarti GPU lebih unggul. Untuk ms/batch, speedup dihitung "
          "sebagai CPU/GPU\nkarena pada satuan itu nilai yang lebih kecil lebih baik.")


def write_chart(cpu, gpu, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(grafik dilewati: {e})")
        return None

    def case_of(src, name):
        return (src or {}).get("cases", {}).get(name)

    cinf, ginf = case_of(cpu, "inference"), case_of(gpu, "inference")
    cscl, gscl = case_of(cpu, "scaling"), case_of(gpu, "scaling")

    labels, cvals, gvals = [], [], []
    for key, lbl in (("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "Transformer")):
        cc, gg = case_of(cpu, key), case_of(gpu, key)
        if cc and gg and "samples_per_s" in cc:
            labels.append(lbl)
            cvals.append(cc["samples_per_s"])
            gvals.append(gg["samples_per_s"])

    panels = [bool(labels), bool(cscl and gscl), bool(cinf and ginf)]
    ncols = sum(panels)
    if ncols == 0:
        return None
    fig, axes = plt.subplots(ncols, 1, figsize=(5.4, 3.1 * ncols), dpi=170)
    axes = [axes] if ncols == 1 else list(axes)
    ax_iter = iter(axes)

    if labels:
        ax = next(ax_iter)
        idx = range(len(labels))
        ax.bar([i - 0.2 for i in idx], cvals, width=0.4, label="CPU", color="#7a8b99")
        ax.bar([i + 0.2 for i in idx], gvals, width=0.4, label="GPU", color="#3b7dd8")
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylabel("sampel/detik (skala log)")
        ax.set_title("Training throughput (model.fit)")
        ax.legend(fontsize=8)
        for i, (cv, gv) in enumerate(zip(cvals, gvals)):
            ax.text(i + 0.2, gv, f"{gv / cv:.1f}x", ha="center", va="bottom", fontsize=8)

    if cscl and gscl:
        ax = next(ax_iter)
        cm = {r["batch"]: r["samples_per_s"] for r in cscl["runs"]}
        gm = {r["batch"]: r["samples_per_s"] for r in gscl["runs"]}
        bs = sorted(set(cm) & set(gm))
        ax.plot(bs, [cm[b] for b in bs], "-o", label="CPU", color="#7a8b99")
        ax.plot(bs, [gm[b] for b in bs], "-o", label="GPU", color="#3b7dd8")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("ukuran batch")
        ax.set_ylabel("sampel/detik")
        ax.set_title("Titik potong CPU/GPU (training step)")
        ax.legend(fontsize=8)

    if cinf and ginf:
        ax = next(ax_iter)
        for mname, style in (("cnn", "-o"), ("transformer", "-s")):
            if mname not in cinf["models"] or mname not in ginf["models"]:
                continue
            cm = {r["batch"]: r["latency_ms"] for r in cinf["models"][mname]["runs"]}
            gm = {r["batch"]: r["latency_ms"] for r in ginf["models"][mname]["runs"]}
            bs = sorted(set(cm) & set(gm))
            ax.plot(bs, [cm[b] / gm[b] for b in bs], style, label=mname)
        ax.axhline(1.0, color="crimson", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("ukuran batch")
        ax.set_ylabel("speedup GPU (CPU ms / GPU ms)")
        ax.set_title("Inference: speedup vs batch")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orkestrator
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark AI CPU vs GPU (Keras 3 / TensorFlow)")
    p.add_argument("--device", choices=["cpu", "gpu", "both"], default="both")
    p.add_argument("--preset", choices=list(PRESETS), default="standard")
    p.add_argument("--only", default=",".join(ALL_CASES),
                   help=f"daftar kasus dipisah koma dari: {','.join(ALL_CASES)}")
    p.add_argument("--report", default="results", help="folder keluaran JSON/PNG")
    p.add_argument("--no-chart", action="store_true")
    p.add_argument("--chart-only", action="store_true",
                   help="bangun ulang PNG dari summary.json yang sudah ada")
    p.add_argument("--no-isolate", action="store_true",
                   help="jalankan semua kasus dalam satu proses per device (lebih cepat, "
                        "tetapi hasil antar-kasus saling mencemari)")
    # argumen internal untuk proses anak
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", help=argparse.SUPPRESS)
    args = p.parse_args()

    cases = [c.strip() for c in args.only.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASE_FN]
    if unknown:
        p.error(f"kasus tidak dikenal: {unknown}. Pilihan: {ALL_CASES}")
    cfg = PRESETS[args.preset]

    if args.child:
        run_child(args.device, cases, cfg, args.json)
        return

    outdir = Path(args.report)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.chart_only:
        data = json.loads((outdir / "summary.json").read_text())
        png = write_chart(data.get("cpu"), data.get("gpu"), outdir / "cpu_vs_gpu.png")
        print(f"Grafik -> {png}")
        return
    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]

    print(f"Preset   : {args.preset}")
    print(f"Kasus    : {', '.join(cases)}")
    print(f"Device   : {', '.join(devices)}")
    print(f"Keluaran : {outdir.resolve()}")

    # Secara default setiap kasus dijalankan di prosesnya sendiri. Ini bukan
    # kehati-hatian berlebihan: menjalankan seluruh kasus dalam satu proses
    # membuat kasus belakangan mewarisi fragmentasi VRAM dan kondisi termal dari
    # kasus sebelumnya. Pada mesin ini selisihnya terukur sampai ~35% untuk CNN.
    payload = {}
    for dev in devices:
        env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
        if dev == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        groups = [[c] for c in cases] if not args.no_isolate else [cases]
        merged = None
        for grp in groups:
            jpath = outdir / (f"{dev}.{grp[0]}.json" if len(grp) == 1 else f"{dev}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--child", "--device", dev,
                   "--preset", args.preset, "--only", ",".join(grp), "--json", str(jpath)]
            if subprocess.call(cmd, env=env) != 0:
                print(f"\n[!] {dev}/{','.join(grp)} gagal; dilanjutkan tanpa kasus itu.")
                continue
            part = json.loads(jpath.read_text())
            if merged is None:
                merged = part
            else:
                merged["cases"].update(part["cases"])

        if merged is not None:
            (outdir / f"{dev}.json").write_text(json.dumps(merged, indent=2))
            payload[dev] = merged

    rows = build_report(payload.get("cpu"), payload.get("gpu"))
    print_report(rows)

    summary = outdir / "summary.json"
    summary.write_text(json.dumps(
        {"preset": args.preset, "rows": rows,
         "cpu": payload.get("cpu"), "gpu": payload.get("gpu")}, indent=2))
    print(f"\nJSON gabungan : {summary}")

    if not args.no_chart and "cpu" in payload and "gpu" in payload:
        png = write_chart(payload["cpu"], payload["gpu"], outdir / "cpu_vs_gpu.png")
        if png:
            print(f"Grafik        : {png}")


if __name__ == "__main__":
    main()
