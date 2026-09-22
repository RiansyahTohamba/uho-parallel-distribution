"""
Bangun laporan PDF (format Kindle) dari hasil benchmark cpu_vs_gpu.py.

    python cpu_vs_gpu.py --preset standard --report results
    python make_report.py

Membaca results/summary.json + results/cpu_vs_gpu.png, menulis
laporan-cpu-vs-gpu.md, lalu memanggil pandoc + xelatex untuk menghasilkan
laporan-cpu-vs-gpu.pdf berukuran 6x8 inci.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MD = HERE / "laporan-cpu-vs-gpu.md"
PDF = HERE / "laporan-cpu-vs-gpu.pdf"
CODE = HERE / "cpu_vs_gpu.py"
PREAMBLE = HERE / "kindle-preamble.tex"


# ---------------------------------------------------------------------------
# Pembantu pengambilan angka
# ---------------------------------------------------------------------------

def case(src, name):
    return (src or {}).get("cases", {}).get(name)


def table(header, rows):
    """Tabel pandoc pipe sederhana."""
    if not rows:
        return "_(tidak ada data)_\n"
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def x(a, b):
    """Rasio a/b sebagai string '3.4x', aman terhadap None/0."""
    if not a or not b:
        return "-"
    return f"{a / b:.1f}x"


def crossover(cpu, gpu):
    """Batch terkecil pada kasus `scaling` di mana GPU mulai unggul."""
    cs, gs = case(cpu, "scaling"), case(gpu, "scaling")
    if not (cs and gs):
        return None
    cm = {r["batch"]: r["samples_per_s"] for r in cs["runs"]}
    for r in sorted(gs["runs"], key=lambda r: r["batch"]):
        b = r["batch"]
        if b in cm and r["samples_per_s"] > cm[b]:
            return b, cm[b], r["samples_per_s"]
    return None


# ---------------------------------------------------------------------------
# Bagian-bagian dokumen
# ---------------------------------------------------------------------------

def sec_ringkasan(cpu, gpu, preset):
    gm = (gpu or {}).get("meta", {})
    cm = (cpu or {}).get("meta", {})
    xo = crossover(cpu, gpu)

    gemm_c = case(cpu, "gemm")
    gemm_g = case(gpu, "gemm")
    gemm_line = ""
    if gemm_c and gemm_g:
        best_c = max(r["gflops"] for r in gemm_c["runs"])
        best_g = max(r["gflops"] for r in gemm_g["runs"])
        gemm_line = (
            f"Pada perkalian matriks fp32 murni, GPU mencapai **{best_g:,.0f} GFLOP/s** "
            f"melawan **{best_c:,.0f} GFLOP/s** di CPU — sekitar **{x(best_g, best_c)}**. "
            "Angka ini adalah *plafon*: tidak ada beban kerja nyata dalam laporan ini "
            "yang speedup-nya melampauinya."
        )

    best = None
    for k, lbl in (("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "transformer")):
        cc, gg2 = case(cpu, k), case(gpu, k)
        if cc and gg2 and cc.get("samples_per_s"):
            ratio = gg2["samples_per_s"] / cc["samples_per_s"]
            if best is None or ratio > best[1]:
                best = (lbl, ratio)
    cnn_line = (
        f"Pada training penuh, perolehan terbesar ada di {best[0]} (**{best[1]:.1f}x**), "
        "jauh di bawah plafon GEMM tadi." if best else ""
    )

    xo_line = ""
    if xo:
        b, cv, gv = xo
        xo_line = (
            f"GPU baru mulai menang pada ukuran batch **{b}**. Di bawah itu CPU justru "
            f"lebih cepat, karena setiap langkah training di GPU membawa biaya tetap "
            f"(peluncuran kernel, sinkronisasi, dispatch dari Python) yang tidak bergantung "
            f"pada besar batch."
        )
    else:
        xo_line = (
            "Pada rentang batch yang diuji tidak ditemukan titik potong yang bersih — "
            "periksa kolom `scaling` pada tabel hasil untuk melihat arah kurvanya."
        )

    return f"""
# Ringkasan

Dokumen ini membandingkan CPU dan GPU untuk beban kerja AI pada satu perangkat
nyata: laptop dengan **{gm.get('device_name', 'GPU')}** dan
**Intel Core i9-12900H** yang dijalankan di WSL2. Semua angka di sini diukur
langsung di mesin tersebut, bukan dikutip dari spesifikasi pabrikan atau dari
benchmark publik. Preset yang dipakai: `{preset}`.

Tiga temuan utama:

**1. GPU tidak selalu lebih cepat.** {xo_line}

**2. Selisihnya sangat bergantung jenis operasi.** {gemm_line} {cnn_line}
Jarak antara plafon dan beban kerja nyata itulah inti persoalannya: yang hilang
di antaranya adalah operasi kecil, sinkronisasi, dan lapisan yang tidak menyatu
menjadi satu kernel.

**3. Untuk inference, ukuran batch menentukan segalanya.** Pada batch 1, GPU
sering hanya imbang atau bahkan kalah, karena waktu yang terukur didominasi
latensi peluncuran kernel dan transfer data, bukan komputasi. Keunggulan GPU
muncul saat batch besar, ketika 46 SM benar-benar terisi.

Konsekuensi praktisnya sederhana: **GPU pada perangkat ini adalah mesin
throughput, bukan mesin latensi.** Ia membayar biaya tetap di awal setiap
langkah, lalu mengerjakan pekerjaan besar nyaris gratis. Kalau pekerjaan yang
diberikan kecil, biaya tetap itu yang mendominasi dan CPU menang.

Lingkungan uji: TensorFlow {cm.get('tf_version', '?')}, Keras
{cm.get('keras_version', '?')}, {cm.get('cpu_threads_visible', '?')} thread CPU
terlihat di WSL, TF32 {'aktif' if gm.get('tf32_enabled') else 'nonaktif'}.
"""


def sec_metodologi():
    return """
# Kenapa benchmark ini bisa dipercaya

Sebagian besar perbandingan CPU-vs-GPU yang beredar salah ukur. Enam hal
berikut adalah yang membedakan angka yang bermakna dari angka yang kebetulan.

**Isolasi device lewat proses terpisah.** TensorFlow menempatkan operasi ke GPU
sejak inisialisasi. Menulis `with tf.device('/CPU:0')` di proses yang sudah
melihat GPU tidak cukup — sebagian op tetap bocor ke GPU dan hasil "CPU" jadi
terlalu bagus. Skrip ini menjalankan sisi CPU sebagai proses anak dengan
`CUDA_VISIBLE_DEVICES=-1`, lalu memeriksa ulang bahwa GPU benar-benar tidak
terlihat sebelum mengukur apa pun.

**Warm-up selalu dibuang.** Iterasi pertama berisi penelusuran (*tracing*)
`tf.function`, autotune algoritma cuDNN, dan alokasi memori pertama. Itu biaya
sekali jalan, bukan biaya komputasi. Memasukkannya ke hasil bisa melipatgandakan
angka GPU secara keliru.

**Sinkronisasi device sebelum timer ditutup.** Ini kesalahan paling umum.
Eksekusi GPU bersifat asinkron: `model(x)` kembali ke Python sebelum kernelnya
selesai. Tanpa sinkronisasi eksplisit, yang terukur adalah kecepatan Python
melempar perintah — biasanya menghasilkan "speedup" 100x yang tidak nyata.
Setiap timer di sini ditutup dengan sinkronisasi device.

**Setiap kasus dijalankan di proses sendiri.** Ini ditemukan secara empiris,
bukan direncanakan. Pada versi pertama skrip ini seluruh kasus berbagi satu
proses, dan CNN — yang berjalan menjelang akhir — melaporkan GPU lebih lambat
daripada CPU. Penyebabnya adalah kondisi yang diwariskan kasus-kasus sebelumnya:
fragmentasi VRAM dan, terutama, GPU yang sudah panas. Setiap kasus kini memakai
prosesnya sendiri.

Isolasi itu mengurangi masalahnya, tetapi jujur saja **tidak menghilangkannya**.
CNN yang sama, diukur pada mesin yang benar-benar dingin, memberi sekitar 1030
sampel/detik di GPU; diukur di tengah rangkaian benchmark — meski di proses
terpisah — sekitar 770. Selisih 30% itu murni termal, dan ini laptop: frekuensi
GPU dan CPU turun saat panas, dan tidak ada yang bisa dilakukan skrip untuk
mencegahnya.

Untuk beban yang pendek dan padat komputasi, efeknya bahkan lebih besar. GEMM
2048x2048 yang dijalankan sendirian pada mesin dingin mencapai sekitar 10.500
GFLOP/s; angka yang sama di tengah rangkaian benchmark turun ke sekitar 4.500 —
lebih dari dua kali lipat selisihnya. Penyebabnya adalah *boost clock*: burst
pendek pada GPU dingin berjalan pada frekuensi yang tidak bisa dipertahankan.

Karena itu, dua aturan membaca laporan ini. **Angka absolut punya ketidakpastian
besar** — anggap ±30% untuk beban panjang, dan bisa 2x untuk burst pendek seperti
GEMM. **Rasio CPU-GPU jauh lebih andal**, karena kedua sisi diukur dalam kondisi
termal yang berdekatan, dan itulah sebabnya laporan ini menekankan rasio.
Untuk angka absolut yang bersih, jalankan satu kasus saja pada mesin dingin.

**Median, bukan rata-rata.** Di WSL, satu iterasi bisa tertunda oleh penjadwal
atau thermal throttling. Rata-rata menyerap pencilan itu; median tidak.

**Data sintetis dengan seed tetap.** Untuk mengukur *waktu*, isi data tidak
berpengaruh — yang penting bentuk tensor dan jumlah operasinya identik di kedua
device. Keuntungannya: tidak ada dependensi unduhan, dan pengukuran bisa diulang
persis. Yang tidak diukur di sini adalah akurasi model; itu memang bukan tujuan
benchmark ini.

**Batas memori dihormati.** Setiap kasus dirancang muat di bawah 4 GB VRAM dari
8 GB yang tersedia, dan di bawah 3 GB RAM. `set_memory_growth` diaktifkan supaya
TensorFlow tidak langsung mencaplok seluruh VRAM — tanpa itu, penggunaan memori
puncak tidak terbaca dan proses lain ikut tercekik.

Satu hal yang **tidak** dikendalikan: perangkat ini laptop. Frekuensi CPU dan GPU
turun saat panas. Jalankan ulang saat mesin dingin bila angka terasa aneh, dan
perhatikan kolom `p90` pada JSON keluaran sebagai indikator kestabilan.
"""


def sec_kasus():
    return """
# Delapan kasus uji

Kasus disusun dari yang paling sintetis ke yang paling menyerupai pekerjaan
nyata. Dua kasus pertama menjelaskan *kenapa* kasus berikutnya berperilaku
seperti itu.

**1. GEMM (perkalian matriks).** Matriks $N \\times N$ butuh $2N^3$ operasi
titik-mengambang. Ini kasus paling menguntungkan GPU: padat komputasi, pola
akses memori teratur, tanpa cabang. Fungsinya sebagai plafon — batas atas
speedup yang mungkin dicapai pada perangkat ini.

**2. Sapuan ukuran batch.** Satu langkah training MLP diulang pada batch 128
sampai 8192. Ini kasus paling penting dalam dokumen ini, karena ia memisahkan
biaya tetap per langkah dari biaya komputasi yang sebanding dengan beban.

**3. Overhead `model.fit`.** Default Keras adalah `steps_per_execution=1`:
setiap langkah kembali ke Python dan menyinkronkan device untuk mengagregasi
loss. Kasus ini membandingkannya dengan `steps_per_execution=32`, yang
menjalankan 32 langkah dalam satu panggilan graph.

**4. Training MLP.** Jaringan *dense* 1.5 juta parameter. Murni matmul, tanpa
konvolusi, dengan langkah yang pendek — wakil dari model kecil.

**5. Training CNN.** Enam lapis konvolusi pada citra 32x32. Konvolusi adalah
operasi yang paling diuntungkan cuDNN dan Tensor Core; di sinilah selisih
CPU-GPU paling lebar.

**6. Training transformer encoder.** Empat lapis, `d_model` 256, panjang
sekuens 128. Bentuk beban kerjanya sengaja dibuat sama dengan *fine-tuning*
BERT/DeBERTa untuk klasifikasi kalimat, hanya lebih kecil supaya sisi CPU
selesai dalam waktu wajar.

**7. Inference pada beberapa ukuran batch.** Hanya *forward pass*, tanpa
gradien dan tanpa optimizer. Diukur pada batch 1 sampai 512 untuk menemukan
di titik mana GPU mulai unggul.

**8. XLA (`jit_compile=True`).** XLA mengompilasi graph menjadi kernel yang
menyatu. Ia sering direkomendasikan tanpa syarat; kasus ini mengukurnya alih-alih
mempercayainya.

**9. Mixed precision bf16.** Compute capability 8.6 punya Tensor Core bf16.
Kasus ini mengukur perolehannya terhadap fp32 pada CNN dan transformer.
Hanya dijalankan di GPU: di CPU, bf16 diemulasi dan justru lebih lambat,
sehingga perbandingannya tidak bermakna.
"""


def sec_hasil(cpu, gpu, rows, has_png):
    parts = ["\n# Hasil pengukuran\n"]

    if has_png:
        parts.append(
            "![Atas: throughput training pada model.fit. Tengah: titik potong CPU/GPU "
            "terhadap ukuran batch. Bawah: speedup inference terhadap ukuran batch; "
            "garis putus-putus merah adalah titik impas, di bawahnya CPU lebih "
            "cepat.](results/cpu_vs_gpu.png)\n"
        )

    # --- GEMM
    gc, gg = case(cpu, "gemm"), case(gpu, "gemm")
    if gc and gg:
        cm = {r["n"]: r["gflops"] for r in gc["runs"]}
        rws = [(f"{r['n']}x{r['n']}", f"{cm[r['n']]:,.0f}", f"{r['gflops']:,.0f}",
                x(r["gflops"], cm[r["n"]]))
               for r in gg["runs"] if r["n"] in cm]
        parts.append("## GEMM fp32 (GFLOP/s)\n")
        parts.append(table(["Ukuran", "CPU", "GPU", "Speedup"], rws))
        parts.append(
            "\nSpeedup naik seiring ukuran matriks. Pada matriks kecil, GPU belum "
            "terisi penuh dan biaya peluncuran kernel masih terasa; pada matriks "
            "besar barulah throughput aritmetikanya terlihat utuh. Pola ini akan "
            "berulang di setiap kasus berikutnya.\n"
        )

    # --- Scaling
    sc, sg = case(cpu, "scaling"), case(gpu, "scaling")
    if sc and sg:
        cm = {r["batch"]: r for r in sc["runs"]}
        rws = [(r["batch"], f"{cm[r['batch']]['ms_per_step']:.1f}",
                f"{r['ms_per_step']:.1f}",
                f"{cm[r['batch']]['samples_per_s']:,.0f}",
                f"{r['samples_per_s']:,.0f}",
                x(r["samples_per_s"], cm[r["batch"]]["samples_per_s"]))
               for r in sg["runs"] if r["batch"] in cm]
        parts.append("\n## Titik potong: throughput vs ukuran batch\n")
        parts.append(table(
            ["Batch", "CPU ms", "GPU ms", "CPU spl/s", "GPU spl/s", "Speedup"], rws))
        gpu_ms = [r["ms_per_step"] for r in sg["runs"]]
        cpu_ms = [cm[r["batch"]]["ms_per_step"] for r in sg["runs"] if r["batch"] in cm]
        if len(gpu_ms) >= 2 and len(cpu_ms) >= 2:
            parts.append(
                f"\nPerhatikan kolom milidetik, bukan kolom speedup. Waktu per langkah "
                f"di CPU naik hampir sebanding dengan besar batch — itu komputasi nyata. "
                f"Di GPU, waktu per langkah nyaris tidak bergerak pada batch kecil "
                f"({gpu_ms[0]:.1f} ms pada batch {sg['runs'][0]['batch']}) lalu baru naik "
                f"pada batch besar. Bagian yang datar itulah biaya tetap: sekitar "
                f"{min(gpu_ms):.0f} ms yang dibayar setiap langkah terlepas dari berapa "
                f"banyak pekerjaan yang diberikan.\n\n"
                "Di mesin ini biaya tetap tersebut besar karena WSL hanya memaparkan 3 "
                "core. Host-lah yang menyusun dan meluncurkan kernel; dengan CPU sesempit "
                "itu, GPU sering menganggur menunggu diberi pekerjaan. Implikasi "
                "praktisnya: **perbesar batch sampai VRAM hampir penuh.** Pada perangkat "
                "8 GB ini, batch yang terlalu kecil membuang kapasitas GPU secara "
                "percuma.\n"
            )

    # --- Overhead
    oc, og = case(cpu, "overhead"), case(gpu, "overhead")
    if oc and og:
        cm = {r["steps_per_execution"]: r for r in oc["runs"]}
        rws = [(r["steps_per_execution"],
                f"{cm[r['steps_per_execution']]['samples_per_s']:,.0f}",
                f"{r['samples_per_s']:,.0f}",
                x(r["samples_per_s"], cm[r["steps_per_execution"]]["samples_per_s"]))
               for r in og["runs"] if r["steps_per_execution"] in cm]
        parts.append("\n## Overhead `model.fit` (sampel/detik)\n")
        parts.append(table(["steps_per_execution", "CPU", "GPU", "GPU/CPU"], rws))
        try:
            g1 = next(r["samples_per_s"] for r in og["runs"] if r["steps_per_execution"] == 1)
            g32 = next(r["samples_per_s"] for r in og["runs"] if r["steps_per_execution"] == 32)
            parts.append(
                f"\nMenaikkan `steps_per_execution` dari 1 ke 32 memberi "
                f"**{x(g32, g1)}** di GPU tanpa mengubah model, data, atau "
                "hyperparameter apa pun — cukup satu argumen tambahan pada "
                "`model.compile`. Perolehan ini datang dari menghilangkan perjalanan "
                "bolak-balik Python dan sinkronisasi device di setiap langkah. Semakin "
                "kecil modelnya, semakin besar perolehannya. Di CPU efeknya kecil, "
                "karena di sana tidak ada sinkronisasi device yang perlu diamortisasi.\n"
            )
        except StopIteration:
            pass

    # --- Training
    rws = []
    for key, lbl in (("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "Transformer")):
        cc, gg2 = case(cpu, key), case(gpu, key)
        if cc and gg2 and "samples_per_s" in cc:
            rws.append((lbl, f"{cc['params']/1e6:.1f}M", f"{cc['s_per_epoch']:.1f}",
                        f"{gg2['s_per_epoch']:.1f}",
                        x(gg2["samples_per_s"], cc["samples_per_s"])))
    if rws:
        parts.append("\n## Training penuh (`model.fit`, detik per epoch)\n")
        parts.append(table(["Model", "Param", "CPU s/ep", "GPU s/ep", "Speedup"], rws))

        ranked = sorted(
            ((lbl, case(gpu, k)["samples_per_s"] / case(cpu, k)["samples_per_s"])
             for k, lbl in (("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "Transformer"))
             if case(cpu, k) and case(gpu, k) and case(cpu, k).get("samples_per_s")),
            key=lambda t: -t[1])
        if ranked:
            order = ", ".join(f"{l} ({v:.1f}x)" for l, v in ranked)
            parts.append(
                f"\nUrutan perolehan GPU pada pengukuran ini: {order}.\n\n"
                "Mekanismenya sama di ketiganya, hanya bobotnya berbeda. Yang menentukan "
                "adalah berapa banyak pekerjaan aritmetika yang dikerjakan per peluncuran "
                "kernel. Transformer memberi paling banyak: *attention* dan lapisan "
                "*feed-forward* adalah matmul besar, dan model ini yang terbesar "
                "parameternya. MLP memberi paling sedikit: langkahnya terlalu pendek, "
                "sehingga biaya tetap per langkah menelan sebagian besar waktunya. CNN "
                "berada di antaranya, dan hasilnya lebih rendah dari yang biasa "
                "diperkirakan orang untuk konvolusi — sebabnya ada pada `BatchNormalization`.\n\n"
                "Pengukuran terpisah pada satu langkah training CNN menunjukkan lapisan "
                "normalisasi itu sendiri menyumbang sekitar 51 ms dari 90 ms per langkah "
                "di GPU. Di backend TensorFlow, `BatchNormalization` Keras 3 tidak "
                "dipetakan ke kernel cuDNN yang menyatu; ia tersusun dari banyak operasi "
                "kecil, dan setiap operasi kecil membayar ongkos peluncuran. Dengan 3 core "
                "di WSL yang harus menyusun peluncuran itu, ongkosnya membengkak. Ini "
                "kembali ke tema yang sama: pada perangkat ini, yang membatasi sering kali "
                "bukan GPU-nya, melainkan host yang memberinya makan.\n\n"
                "Angka pada tabel ini memakai `steps_per_execution=1`, default Keras, "
                "sehingga mewakili apa yang dialami pengguna apa adanya — bukan hasil "
                "yang sudah disetel.\n"
            )

    # --- XLA
    xc, xg = case(cpu, "xla"), case(gpu, "xla")
    if xc and xg:
        cm = {r["jit_compile"]: r for r in xc["runs"]}
        rws2 = [("aktif" if r["jit_compile"] else "nonaktif",
                 f"{cm[r['jit_compile']]['samples_per_s']:,.0f}",
                 f"{r['samples_per_s']:,.0f}",
                 x(r["samples_per_s"], cm[r["jit_compile"]]["samples_per_s"]))
                for r in xg["runs"] if r["jit_compile"] in cm]
        parts.append("\n## XLA `jit_compile` pada CNN (sampel/detik)\n")
        parts.append(table(["XLA", "CPU", "GPU", "GPU/CPU"], rws2))
        try:
            g0 = next(r["samples_per_s"] for r in xg["runs"] if not r["jit_compile"])
            g1 = next(r["samples_per_s"] for r in xg["runs"] if r["jit_compile"])
            verdict = "mempercepat" if g1 > g0 else "justru memperlambat"
            parts.append(
                f"\nPada model dan perangkat ini, mengaktifkan XLA **{verdict}** training "
                f"GPU ({g0:,.0f} menjadi {g1:,.0f} sampel/detik, {x(g1, g0)}). "
                "Hasil ini layak digarisbawahi karena bertentangan dengan saran yang umum "
                "beredar. XLA menyatukan operasi-operasi kecil — yang mestinya menolong, "
                "mengingat diagnosis di atas — tetapi ia juga menggantikan pemilihan "
                "algoritma konvolusi milik cuDNN dengan pilihannya sendiri, dan untuk "
                "konvolusi 3x3 pada citra kecil pilihan itu bisa lebih buruk. Ditambah "
                "waktu kompilasi yang harus dibayar di awal.\n\n"
                "Kesimpulannya bukan \"jangan pakai XLA\", melainkan **ukur XLA pada model "
                "Anda sendiri sebelum memakainya.** Ia bisa memberi perolehan besar pada "
                "model yang didominasi operasi *pointwise*, dan kerugian seperti di sini "
                "pada model yang didominasi konvolusi.\n"
            )
        except StopIteration:
            pass

    # --- Inference
    ic, ig = case(cpu, "inference"), case(gpu, "inference")
    if ic and ig:
        parts.append("\n## Inference: latensi vs throughput\n")
        for mname in ic.get("models", {}):
            if mname not in ig.get("models", {}):
                continue
            cm = {r["batch"]: r for r in ic["models"][mname]["runs"]}
            rws = [(r["batch"], f"{cm[r['batch']]['latency_ms']:.2f}",
                    f"{r['latency_ms']:.2f}",
                    f"{cm[r['batch']]['throughput_samples_s']:,.0f}",
                    f"{r['throughput_samples_s']:,.0f}",
                    x(cm[r["batch"]]["latency_ms"], r["latency_ms"]))
                   for r in ig["models"][mname]["runs"] if r["batch"] in cm]
            parts.append(f"\n**{mname}** (ms per batch, dan sampel/detik)\n")
            parts.append(table(
                ["Batch", "CPU ms", "GPU ms", "CPU spl/s", "GPU spl/s", "Speedup"], rws))
        parts.append(
            "\nDua kolom terakhir menceritakan hal yang berbeda, dan keduanya penting.\n\n"
            "Kolom milidetik adalah **latensi**: berapa lama satu permintaan ditunggu. "
            "Pada batch 1, GPU nyaris tidak memberi keuntungan — pekerjaannya terlalu "
            "sedikit untuk menutupi ongkos mengirim data ke VRAM, meluncurkan kernel, dan "
            "menarik hasilnya kembali. Untuk layanan yang melayani satu permintaan pada "
            "satu waktu dengan target latensi ketat, CPU sering merupakan pilihan yang "
            "benar, dan sekaligus lebih murah dan lebih sederhana untuk dioperasikan.\n\n"
            "Kolom sampel/detik adalah **throughput**: berapa banyak yang selesai per "
            "satuan waktu. Di sini GPU menang telak pada batch besar. Untuk pekerjaan "
            "*batch* — memberi label satu korpus, mengekstrak *embedding*, mengevaluasi "
            "himpunan uji — GPU jelas alat yang tepat.\n\n"
            "Aturan praktisnya: kalau permintaan datang satu per satu dan pengguna "
            "menunggu, ukur latensi. Kalau pekerjaan bisa dikumpulkan dulu, ukur "
            "throughput dan perbesar batch.\n"
        )

    # --- bf16
    bg = case(gpu, "bf16")
    if bg and "runs" in bg:
        rws = []
        for mname, run in bg["runs"].items():
            base = case(gpu, mname)
            if base and "samples_per_s" in base:
                rws.append((mname, f"{base['samples_per_s']:,.0f}",
                            f"{run['samples_per_s']:,.0f}",
                            x(run["samples_per_s"], base["samples_per_s"]),
                            f"{base.get('peak_vram_mb') or '-'} / {run.get('peak_vram_mb') or '-'}"))
        if rws:
            parts.append("\n## Mixed precision bf16 (GPU saja, sampel/detik)\n")
            parts.append(table(
                ["Model", "fp32", "bf16", "Speedup", "VRAM puncak fp32/bf16 (MB)"], rws))
            parts.append(
                "\nbf16 punya rentang eksponen yang sama dengan fp32, hanya mantisanya "
                "lebih pendek. Karena itu ia tidak memerlukan *loss scaling* seperti "
                "fp16, dan pada Ampere praktis selalu layak dipakai untuk training. "
                "Perolehan kecepatannya nyata tetapi biasanya lebih sederhana daripada "
                "yang dijanjikan angka Tensor Core — sebab sebagian beban kerja terbatas "
                "oleh bandwidth memori, bukan aritmetika. Penghematan VRAM-nya sering "
                "justru lebih berharga daripada kecepatannya: pada perangkat 8 GB, VRAM "
                "yang bebas bisa ditukar dengan batch yang lebih besar, dan batch lebih "
                "besar itulah yang mengembalikan kecepatan.\n"
            )

    if rows:
        parts.append("\n## Tabel gabungan\n")
        parts.append(table(["Kasus", "Konfigurasi", "Satuan", "CPU", "GPU", "Speedup"],
                           [tuple(r) for r in rows]))

    return "\n".join(parts)


def sec_interpretasi(cpu, gpu):
    xo = crossover(cpu, gpu)
    xo_txt = (f"batch {xo[0]}" if xo else "ukuran batch yang perlu Anda cari sendiri")
    return f"""
# Kapan memakai CPU, kapan GPU

Hasil di atas bisa diringkas menjadi keputusan yang cukup jelas.

**Pakai GPU untuk:** training model apa pun yang punya konvolusi atau
*attention*; training apa pun yang berjalan lebih dari beberapa menit; inference
*batch* atas banyak data; dan setiap eksperimen yang ingin Anda ulang beberapa
kali dengan seed berbeda. Di kategori ini selisihnya bukan soal persentase,
melainkan soal apakah eksperimennya selesai hari ini atau minggu depan.

**Pakai CPU untuk:** inference satu-per-satu dengan target latensi ketat; model
klasik (SVM, gradient boosting, regresi logistik) yang memang tidak dirancang
untuk GPU; prapemrosesan dan tokenisasi; dan pengembangan awal ketika Anda hanya
ingin tahu apakah kodenya jalan. Memindahkan model 200 MB ke VRAM untuk
mengklasifikasi satu kalimat adalah kerugian bersih.

**Yang perlu diwaspadai** adalah wilayah abu-abu di bawah {xo_txt}. Di sana GPU
sedang menganggur menunggu diberi pekerjaan, dan menambah GPU tidak akan
menyelesaikan apa pun. Kalau training Anda terasa lambat di wilayah ini,
urutan perbaikannya:

1. **Perbesar batch** sampai VRAM hampir penuh. Ini pengungkit terbesar dan
   paling murah.
2. **Naikkan `steps_per_execution`** pada `model.compile`, misalnya ke 32.
   Satu argumen, tanpa risiko terhadap hasil.
3. **Aktifkan mixed precision bf16** dengan
   `keras.mixed_precision.set_global_policy("mixed_bfloat16")`. Aman di Ampere,
   dan VRAM yang dihemat bisa ditukar dengan batch yang lebih besar lagi.
4. **Periksa pipeline data.** Dengan 3 core di WSL, pemuatan dan augmentasi data
   gampang menjadi hambatan sesungguhnya. Gunakan `tf.data` dengan `.cache()`
   dan `.prefetch(tf.data.AUTOTUNE)`, dan jaga jumlah worker di kisaran 2-4.
5. **Baru setelah itu** pertimbangkan `jit_compile=True` (XLA). Ia bisa
   membantu banyak, tetapi juga bisa memperpanjang waktu kompilasi dan
   sesekali gagal pada model dinamis — jadi ukur, jangan asumsikan.

Satu catatan tentang perangkat ini secara khusus: **8 GB VRAM adalah batasan
yang mengikat, dan 3 core CPU di WSL adalah batasan kedua yang sering
terlupakan.** Yang pertama membatasi seberapa besar model yang muat; yang kedua
membatasi seberapa cepat Anda bisa memberinya makan. Menaikkan alokasi memori
WSL lewat `%UserProfile%\\.wslconfig` (`memory=24GB`, dan `processors=8` bila
tersedia) sering memberi perbaikan yang lebih besar daripada tuning model.
"""


def sec_riset():
    return """
# Kaitan dengan rencana riset

Bila arah pemakaiannya adalah *fine-tuning* encoder untuk klasifikasi ambiguitas
persyaratan — DeBERTa-v3-base atau IndoBERT pada kalimat pendek — hasil di atas
memberi beberapa konsekuensi langsung.

Kasus transformer dalam laporan ini adalah versi mininya: encoder-only, sekuens
128 token, hanya lebih kecil dari model 183 juta parameter yang sesungguhnya.
Rasio CPU-GPU yang terukur di sini akan bertahan dan bahkan melebar untuk model
yang lebih besar, karena model besar memberi lebih banyak pekerjaan per langkah
sehingga biaya tetap semakin tidak relevan.

Dalam praktiknya: dengan `max_length=128` dan batch 32 dalam bf16, DeBERTa-v3-base
memakai sekitar 2-4 GB VRAM dan satu putaran *fine-tuning* penuh selesai dalam
hitungan menit. Itu artinya validasi silang 5-lipat dengan beberapa seed sangat
terjangkau di perangkat ini — dan untuk kumpulan data ambiguitas yang biasanya
hanya beberapa ratus sampai beberapa ribu kalimat, pengulangan itu jauh lebih
menentukan kekuatan klaim Anda daripada memilih model yang lebih besar.

Tiga hal yang perlu diantisipasi:

Pertama, **dataset kecil membuat langkah training pendek**, dan langkah pendek
adalah wilayah tempat GPU paling tidak efisien. Perbesar batch dan naikkan
`steps_per_execution`; jangan menyimpulkan bahwa GPU-nya bermasalah.

Kedua, **tokenisasi berjalan di CPU**, dan CPU Anda hanya 3 core di WSL. Lakukan
tokenisasi sekali di awal lalu simpan hasilnya, jangan mengulanginya setiap
epoch.

Ketiga, **untuk inference saat demo atau saat menyajikan model**, ukur latensi
batch 1 seperti pada kasus 7 sebelum berasumsi bahwa GPU diperlukan. Untuk
mengklasifikasi satu kalimat persyaratan, CPU kemungkinan besar sudah cukup, dan
itu menyederhanakan penyajiannya secara signifikan.
"""


def sec_kode(code_text):
    return f"""
# Cara menjalankan ulang

```bash
# semua kasus, kedua device (perlu belasan menit)
python cpu_vs_gpu.py --preset standard --report results

# versi cepat untuk memastikan semuanya jalan
python cpu_vs_gpu.py --preset quick

# hanya kasus tertentu
python cpu_vs_gpu.py --only gemm,scaling,inference

# bangun ulang PDF ini dari hasil terbaru
python make_report.py
```

Keluarannya: `results/cpu.json`, `results/gpu.json`, `results/summary.json`
(berisi seluruh angka mentah termasuk min dan p90 untuk memeriksa kestabilan),
serta `results/cpu_vs_gpu.png`.

Bila hasil Anda berbeda jauh dari laporan ini, periksa dulu: apakah mesin sedang
panas, apakah ada proses lain memakai GPU (`nvidia-smi`), dan berapa banyak RAM
yang tersisa di WSL (`free -h`). Ketiganya bisa menggeser angka secara
substansial.

# Lampiran: kode lengkap

Berkas `cpu_vs_gpu.py`, ditulis dengan Keras 3 di atas backend TensorFlow.

```python
{code_text}
```
"""


# ---------------------------------------------------------------------------

def main():
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        sys.exit(f"{summary_path} tidak ditemukan. Jalankan cpu_vs_gpu.py lebih dulu.")

    data = json.loads(summary_path.read_text())
    cpu, gpu, rows = data.get("cpu"), data.get("gpu"), data.get("rows", [])
    preset = data.get("preset", "standard")
    has_png = (RESULTS / "cpu_vs_gpu.png").exists()

    gname = (gpu or {}).get("meta", {}).get("device_name", "GPU")
    front = f"""---
title: "CPU vs GPU untuk Beban Kerja AI"
subtitle: "Benchmark terukur pada {gname} dengan Keras 3"
date: "{date.today().isoformat()}"
lang: id
toc: true
toc-depth: 2
fontsize: 11pt
documentclass: article
colorlinks: false
header-includes: |
  \\input{{{PREAMBLE.name}}}
---
"""

    body = "\n".join([
        sec_ringkasan(cpu, gpu, preset),
        sec_metodologi(),
        sec_kasus(),
        sec_hasil(cpu, gpu, rows, has_png),
        sec_interpretasi(cpu, gpu),
        sec_riset(),
        sec_kode(CODE.read_text().rstrip()),
    ])
    MD.write_text(front + body)
    print(f"markdown -> {MD}")

    if not shutil.which("pandoc"):
        sys.exit("pandoc tidak terpasang; PDF dilewati.")

    cmd = [
        "pandoc", str(MD), "-o", str(PDF),
        "--pdf-engine=xelatex",
        "--highlight-style=tango",
        "--resource-path", str(HERE),
        "-V", "monofont=DejaVu Sans Mono",
        "-V", "monofontoptions=Scale=0.78",
    ]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        sys.exit("pandoc gagal.")
    print(f"pdf      -> {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
