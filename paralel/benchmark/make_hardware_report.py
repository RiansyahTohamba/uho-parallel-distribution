"""
Build the English companion report: "Neural Networks and the Machine".

    python make_hardware_report.py

Reads results/summary.json (produced by cpu_vs_gpu.py), draws the diagrams into
figures/, writes neural-networks-and-hardware.md, then calls pandoc + xelatex to
produce neural-networks-and-hardware.pdf in the same 6x8 inch format as the
Indonesian benchmark report it accompanies.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
MD = HERE / "neural-networks-and-hardware.md"
PDF = HERE / "neural-networks-and-hardware.pdf"
PREAMBLE = HERE / "hardware-preamble.tex"

# --- palette (dataviz reference instance, light surface, validated) ----------
GPU_C = "#2a78d6"   # categorical slot 1
CPU_C = "#eb6834"   # categorical slot 2
MEM_C = "#4a3aa7"   # slot 7, used only for the memory-hierarchy diagram
INK   = "#0b0b0b"
INK2  = "#52514e"
MUTED = "#8a8983"
GRID  = "#dcdbd6"
SURF  = "#fcfcfb"
FILL  = "#eceae4"

PAGE_W = 5.05   # inches of usable text width in the 6x8 layout

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})


def clean(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.set_axisbelow(True)


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, fc=FILL, ec=GRID, lw=0.8, r=0.012, alpha=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy0, xy1, color=INK2, lw=1.0, style="-|>", ms=6, z=5, ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  figure -> {p.name}")
    return p


# ---------------------------------------------------------------------------
# Data + derivations
# ---------------------------------------------------------------------------

def load():
    p = RESULTS / "summary.json"
    if not p.exists():
        sys.exit(f"{p} not found. Run cpu_vs_gpu.py --report results first.")
    return json.loads(p.read_text())


def case(side, name):
    return (side or {}).get("cases", {}).get(name)


# The MLP under test: 784 -> 1024 -> 512 -> 256 -> 10 (Dense, ReLU).
MLP_DIMS = [784, 1024, 512, 256, 10]


def mlp_layers():
    return list(zip(MLP_DIMS[:-1], MLP_DIMS[1:]))


def mlp_params():
    return sum(i * o + o for i, o in mlp_layers())


def mlp_fwd_flops_per_sample():
    """2*I*O per Dense layer: one multiply and one add per weight."""
    return sum(2 * i * o for i, o in mlp_layers())


# Backward pass costs roughly 2x the forward pass: one GEMM for the input
# gradient and one for the weight gradient, each the same shape as forward.
TRAIN_FACTOR = 3.0


def fit_cost(runs):
    """Least-squares fit of ms_per_step = fixed + marginal * batch.

    Returns (fixed_ms, marginal_ms_per_sample).
    """
    b = np.array([r["batch"] for r in runs], dtype=float)
    t = np.array([r["ms_per_step"] for r in runs], dtype=float)
    marg, fixed = np.polyfit(b, t, 1)
    return float(fixed), float(marg)


def crossover_batch(cf, cm, gf, gm):
    """Batch where the fitted GPU line drops below the fitted CPU line."""
    if gm >= cm:
        return None
    return (gf - cf) / (cm - gm)


# ---------------------------------------------------------------------------
# Figure 1 -- the stack, and who owns each floor of it
# ---------------------------------------------------------------------------

def fig_stack():
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.5))
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    layers = [
        ("model.fit(x, y, batch_size=128)", "what you wrote: a loop over batches", "cpu"),
        ("Keras 3  /  TensorFlow", "layers become a graph of tensor operations", "cpu"),
        ("Runtime: kernel selection + allocator", "picks an algorithm per op, allocates, queues the work", "cpu"),
        ("cuBLAS  /  cuDNN kernels", "one GEMM, one convolution, one elementwise pass", "gpu"),
        ("Thread blocks on 46 SMs", "128 fp32 lanes + 4 Tensor Cores per SM", "gpu"),
        ("VRAM  8 GB", "every tensor the kernels touch must already be here", "gpu"),
    ]

    x0, w = 0.15, 0.82
    h, gap = 0.128, 0.026
    top = 0.965
    ys = []
    for i, (title, sub, who) in enumerate(layers):
        y = top - (i + 1) * h - i * gap
        ys.append(y)
        c = CPU_C if who == "cpu" else GPU_C
        box(ax, x0, y, w, h, fc=SURF, ec=c, lw=1.1)
        box(ax, x0, y, 0.012, h, fc=c, ec=c, lw=0)
        ax.text(x0 + 0.035, y + h * 0.66, title, fontsize=7.2, fontweight="bold",
                color=INK, va="center", family="DejaVu Sans")
        ax.text(x0 + 0.035, y + h * 0.27, sub, fontsize=6.0, color=INK2,
                va="center", linespacing=1.35)

    # the device boundary
    yb = (ys[2] + ys[3] + h) / 2
    ax.plot([x0 - 0.015, x0 + w + 0.015], [yb, yb], ls=(0, (3, 2)), lw=0.9, color=MUTED)
    ax.text(x0 + w * 0.5, yb, "  PCIe  ", fontsize=6, color=MUTED, va="center",
            ha="center", style="italic", zorder=6,
            bbox=dict(fc=SURF, ec="none", pad=0.6))

    # side brackets
    def bracket(y_lo, y_hi, label, color):
        xb = x0 - 0.045
        ax.plot([xb, xb], [y_lo, y_hi], lw=1.2, color=color)
        ax.plot([xb, xb + 0.018], [y_lo, y_lo], lw=1.2, color=color)
        ax.plot([xb, xb + 0.018], [y_hi, y_hi], lw=1.2, color=color)
        ax.text(xb - 0.022, (y_lo + y_hi) / 2, label, fontsize=6.6, color=color,
                rotation=90, va="center", ha="center", fontweight="bold")

    bracket(ys[2], ys[0] + h, "HOST  (CPU)", CPU_C)
    bracket(ys[5], ys[3] + h, "DEVICE  (GPU)", GPU_C)

    ax.text(0.5, 0.008,
            "Software above the dashed line, silicon below it — and every training step "
            "travels all the way down and back.",
            fontsize=6.0, color=MUTED, ha="center", style="italic")
    return save(fig, "fig-stack.png")


# ---------------------------------------------------------------------------
# Figure 2 -- where each chip spends its transistors
# ---------------------------------------------------------------------------

def fig_die():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.35))

    def panel(ax, kind):
        blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        c = CPU_C if kind == "cpu" else GPU_C
        box(ax, 0.02, 0.20, 0.96, 0.72, fc=SURF, ec=c, lw=1.2)

        if kind == "cpu":
            ax.set_title("CPU — 3 cores, 6 threads", color=CPU_C, pad=5, fontsize=7.4)
            # a few fat cores: big control, big cache, few wide ALUs
            for k in range(3):
                cx = 0.06 + k * 0.312
                box(ax, cx, 0.42, 0.28, 0.46, fc=SURF, ec=GRID, lw=0.7)
                box(ax, cx + 0.02, 0.70, 0.24, 0.15, fc=c, ec="none", alpha=0.30)
                ax.text(cx + 0.14, 0.775, "control", fontsize=5.4, ha="center", color=INK2)
                for a in range(2):
                    box(ax, cx + 0.025 + a * 0.125, 0.475, 0.105, 0.19, fc=c, ec="none", alpha=0.85)
                    ax.text(cx + 0.0775 + a * 0.125, 0.57, "ALU", fontsize=4.8,
                            ha="center", va="center", color="white", fontweight="bold")
            box(ax, 0.06, 0.245, 0.876, 0.15, fc=c, ec="none", alpha=0.16)
            ax.text(0.5, 0.32, "large cache  (tens of MB)", fontsize=5.8, ha="center", color=INK2)
            facts = ("~5 GHz, deep out-of-order pipelines\n"
                     "built to finish one hard instruction fast\n"
                     "measured ceiling: 398 GFLOP/s (fp32 GEMM)")
        else:
            ax.set_title("GPU — 46 SMs, 5,888 lanes", color=GPU_C, pad=5, fontsize=7.4)
            # many thin lanes, thin control, thin cache
            box(ax, 0.06, 0.80, 0.876, 0.075, fc=c, ec="none", alpha=0.30)
            ax.text(0.5, 0.8375, "control  (shared across 32 lanes at a time)",
                    fontsize=5.4, ha="center", color=INK2, va="center")
            for r in range(6):
                for k in range(24):
                    box(ax, 0.062 + k * 0.0365, 0.435 + r * 0.058, 0.030, 0.046,
                        fc=c, ec="none", alpha=0.85, r=0.004)
            ax.text(0.5, 0.405, "ALU lanes", fontsize=5.4, ha="center", color=INK2, va="top")
            box(ax, 0.06, 0.245, 0.876, 0.075, fc=c, ec="none", alpha=0.16)
            ax.text(0.5, 0.2825, "small cache  (a few MB)", fontsize=5.8, ha="center",
                    color=INK2, va="center")
            facts = ("~1.5 GHz, in-order, no speculation\n"
                     "built to finish 10,000 easy ones at once\n"
                     "measured ceiling: 4,545 GFLOP/s (fp32 GEMM)")

        ax.text(0.5, 0.16, facts, fontsize=5.7, ha="center", va="top", color=INK2,
                linespacing=1.5)

    panel(axes[0], "cpu")
    panel(axes[1], "gpu")
    fig.subplots_adjust(wspace=0.06)
    return save(fig, "fig-die.png")


# ---------------------------------------------------------------------------
# Figure 3 -- the handshake: what actually happens in one training step
# ---------------------------------------------------------------------------

def fig_handshake():
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.2))
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    box(ax, 0.015, 0.283, 0.40, 0.637, fc=SURF, ec=CPU_C, lw=1.2)
    ax.text(0.215, 0.955, "HOST  —  CPU", fontsize=7.4, fontweight="bold",
            color=CPU_C, ha="center")
    box(ax, 0.585, 0.283, 0.40, 0.637, fc=SURF, ec=GPU_C, lw=1.2)
    ax.text(0.785, 0.955, "DEVICE  —  GPU", fontsize=7.4, fontweight="bold",
            color=GPU_C, ha="center")

    host = ["read the next batch",
            "trace the graph once, then find\na kernel for each op",
            "allocate / reuse device buffers",
            "enqueue kernels on the stream",
            "wait for the result, then decide\nwhat happens next"]
    dev = ["copy the batch into VRAM",
           "forward: GEMM, conv, activate",
           "backward: two GEMMs a layer",
           "optimiser update, in place",
           "signal the stream is empty"]

    hy = []
    for i, t in enumerate(host):
        y = 0.80 - i * 0.115
        hy.append(y)
        box(ax, 0.045, y - 0.045, 0.34, 0.09, fc=FILL, ec="none")
        ax.text(0.055, y, f"{i+1}", fontsize=6, fontweight="bold", color=CPU_C, va="center")
        ax.text(0.082, y, t, fontsize=5.9, color=INK, va="center", linespacing=1.3)
    dy = []
    for i, t in enumerate(dev):
        y = 0.80 - i * 0.115
        dy.append(y)
        box(ax, 0.615, y - 0.045, 0.34, 0.09, fc=FILL, ec="none")
        ax.text(0.625, y, f"{i+1}", fontsize=6, fontweight="bold", color=GPU_C, va="center")
        ax.text(0.652, y, t, fontsize=5.7, color=INK, va="center", linespacing=1.3)

    for i in range(5):
        arrow(ax, (0.421, hy[i]), (0.579, dy[i]), color=MUTED, lw=0.8, ms=5)
    # the return edge
    arrow(ax, (0.579, 0.272), (0.421, 0.272), color=INK2, lw=0.9, ms=5)

    ax.text(0.5, 0.955, "asynchronous —\nthe host does not wait", fontsize=5.4,
            color=MUTED, ha="center", va="center", style="italic", linespacing=1.3)
    ax.text(0.5, 0.225, "synchronisation point — the only place the host truly blocks",
            fontsize=5.9, color=INK2, ha="center", va="top", style="italic")

    # memory path
    box(ax, 0.045, 0.045, 0.34, 0.115, fc=SURF, ec=CPU_C, lw=0.9)
    ax.text(0.215, 0.125, "Host RAM — 19 GB", fontsize=6.2, ha="center", color=INK,
            fontweight="bold")
    ax.text(0.215, 0.075, "dataset, Python objects", fontsize=5.5, ha="center", color=INK2)
    box(ax, 0.615, 0.045, 0.34, 0.115, fc=SURF, ec=GPU_C, lw=0.9)
    ax.text(0.785, 0.125, "VRAM — 8 GB", fontsize=6.2, ha="center", color=INK,
            fontweight="bold")
    ax.text(0.785, 0.075, "weights, activations, gradients", fontsize=5.5, ha="center", color=INK2)
    arrow(ax, (0.392, 0.103), (0.608, 0.103), color=INK2, lw=1.0, ms=6, style="<|-|>")
    ax.text(0.5, 0.145, "PCIe", fontsize=5.8, color=INK2, ha="center")
    ax.text(0.5, 0.048, "the narrow\nbridge", fontsize=5.2, color=MUTED, ha="center",
            va="center", linespacing=1.3)
    return save(fig, "fig-handshake.png")


# ---------------------------------------------------------------------------
# Figure 4 -- an MLP layer is a matrix multiply
# ---------------------------------------------------------------------------

def fig_mlp_matmul():
    fig = plt.figure(figsize=(PAGE_W, 4.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.22)

    # --- top: the anatomy of one layer -------------------------------------
    ax = fig.add_subplot(gs[0]); blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("One Dense layer, one GEMM", loc="left", pad=6)

    ax_x, ax_y, ax_w, ax_h = 0.045, 0.10, 0.26, 0.50   # X : B x 784
    w_x, w_y, w_w, w_h = 0.40, 0.66, 0.32, 0.26        # W : 784 x 1024
    h_x, h_y, h_w, h_h = 0.40, 0.10, 0.32, 0.50        # H : B x 1024

    box(ax, ax_x, ax_y, ax_w, ax_h, fc=FILL, ec=MUTED, lw=0.7)
    box(ax, w_x, w_y, w_w, w_h, fc=FILL, ec=MUTED, lw=0.7)
    box(ax, h_x, h_y, h_w, h_h, fc=GPU_C, ec=GPU_C, lw=0.7, alpha=0.16)

    ax.text(ax_x + ax_w / 2, ax_y + ax_h + 0.055, "X   activations in", fontsize=6.4,
            ha="center", color=INK, fontweight="bold")
    ax.text(ax_x + ax_w / 2, ax_y + ax_h / 2, "batch × 784", fontsize=6.2, ha="center",
            va="center", color=INK2)
    ax.text(w_x + w_w / 2, w_y + w_h + 0.055, "W   weights", fontsize=6.4, ha="center",
            color=INK, fontweight="bold")
    ax.text(w_x + w_w / 2, w_y + w_h / 2, "784 × 1024", fontsize=6.2, ha="center",
            va="center", color=INK2)
    ax.text(h_x + h_w / 2, h_y + 0.145, "H = X · W", fontsize=7, ha="center",
            va="center", color=INK, fontweight="bold")
    ax.text(h_x + h_w / 2, h_y + 0.075, "batch × 1024", fontsize=6.2,
            ha="center", va="center", color=INK2)

    # the row / column that produce one output element
    ry = ax_y + ax_h * 0.62
    ax.plot([ax_x, ax_x + ax_w], [ry, ry], lw=2.0, color=CPU_C, zorder=6)
    cx = w_x + w_w * 0.42
    ax.plot([cx, cx], [w_y, w_y + w_h], lw=2.0, color=CPU_C, zorder=6)
    ax.plot([cx], [ry], marker="s", ms=4.5, color=CPU_C, zorder=7)
    ax.plot([ax_x + ax_w, cx], [ry, ry], lw=0.6, ls=(0, (2, 2)), color=CPU_C, zorder=5)
    ax.plot([cx, cx], [w_y, ry], lw=0.6, ls=(0, (2, 2)), color=CPU_C, zorder=5)

    ax.text(0.755, 0.60,
            "one output element =\none dot product of 784 terms",
            fontsize=6.0, color=CPU_C, va="center", linespacing=1.4)
    ax.text(0.755, 0.32,
            "at batch 128 there are\n131,072 of them, and no\ntwo depend on each other",
            fontsize=6.0, color=INK2, va="center", linespacing=1.4)
    ax.text(0.045, 0.02, "The batch dimension is not a loop the GPU has to run — it is the "
                         "parallelism itself.",
            fontsize=6.0, color=INK2, style="italic")

    # --- bottom: the whole network as a chain of GEMMs ---------------------
    ax2 = fig.add_subplot(gs[1]); blank(ax2); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_title("The whole network: four GEMMs and three cheap passes",
                  loc="left", pad=6)

    dims = MLP_DIMS
    xs = np.linspace(0.06, 0.92, len(dims))
    hmax, ymid = 0.30, 0.38
    for i, d in enumerate(dims):
        hh = hmax * (d / max(dims)) ** 0.42
        c = GPU_C if 0 < i < len(dims) - 1 else MUTED
        box(ax2, xs[i] - 0.019, ymid - hh / 2, 0.038, hh, fc=c, ec="none", alpha=0.75)
        if i:
            ax2.text(xs[i], 0.155, str(d), fontsize=6.0, ha="center", color=INK2)
        if 0 < i < len(dims) - 1:
            ax2.text(xs[i], ymid + hh / 2 + 0.03, "ReLU", fontsize=5.2, ha="center",
                     color=CPU_C)
    ax2.text(xs[0], 0.155, "784", fontsize=6.0, ha="center", color=INK2)

    for i, (a, b) in enumerate(mlp_layers()):
        xm = (xs[i] + xs[i + 1]) / 2
        arrow(ax2, (xs[i] + 0.023, ymid), (xs[i + 1] - 0.023, ymid), color=INK2,
              lw=0.9, ms=5)
        ax2.text(xm, 0.93, f"{a}×{b}", fontsize=5.9, ha="center", color=INK,
                 fontweight="bold")
        ax2.text(xm, 0.845, f"{2*a*b/1e6:.2f} MFLOP", fontsize=5.5, ha="center", color=INK2)

    fw = mlp_fwd_flops_per_sample()
    ax2.text(0.06, 0.015,
             f"Per sample: {fw/1e6:.2f} MFLOP forward, about {fw*TRAIN_FACTOR/1e6:.1f} "
             f"MFLOP once the backward pass is counted.",
             fontsize=6.0, color=INK2)
    return save(fig, "fig-mlp-matmul.png")


# ---------------------------------------------------------------------------
# Figure 5 -- how that GEMM is cut up and handed to 46 SMs
# ---------------------------------------------------------------------------

def fig_tiling():
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.85))
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # output matrix cut into 30 tiles
    gx, gy, gw, gh = 0.045, 0.26, 0.30, 0.56
    nc, nr = 6, 5
    for r in range(nr):
        for c in range(nc):
            box(ax, gx + c * gw / nc + 0.004, gy + r * gh / nr + 0.004,
                gw / nc - 0.008, gh / nr - 0.008,
                fc=GPU_C, ec=GPU_C, lw=0.5, alpha=0.75, r=0.004)
    ax.text(gx + gw / 2, gy + gh + 0.065, "30 output tiles of H", fontsize=6.4,
            ha="center", color=INK, fontweight="bold")
    ax.text(gx + gw / 2, gy - 0.065, "128×128 each — one thread block", fontsize=5.6,
            ha="center", color=INK2)

    arrow(ax, (gx + gw + 0.02, 0.54), (0.44, 0.54), color=INK2, lw=1.0, ms=6)
    ax.text((gx + gw + 0.02 + 0.44) / 2, 0.585, "scheduled\nonto", fontsize=5.6,
            ha="center", va="bottom", color=INK2, linespacing=1.3)

    # the SMs
    sx, sy, sw, sh = 0.455, 0.26, 0.24, 0.56
    box(ax, sx - 0.012, sy - 0.018, sw + 0.024, sh + 0.036, fc=SURF, ec=GPU_C, lw=1.0)
    k = 0
    for r in range(8):
        for c in range(6):
            if k >= 46:
                break
            box(ax, sx + c * 0.040, sy + 0.010 + r * 0.0665, 0.031, 0.050,
                fc=GPU_C, ec="none", alpha=0.85 if k < 30 else 0.20, r=0.004)
            k += 1
    ax.text(sx + sw / 2, sy + sh + 0.065, "46 SMs", fontsize=6.4, ha="center",
            color=INK, fontweight="bold")
    ax.text(sx + sw / 2, sy - 0.065, "30 busy, 16 idle", fontsize=5.6, ha="center",
            color=INK2)

    arrow(ax, (sx + sw + 0.03, 0.54), (0.745, 0.54), color=INK2, lw=1.0, ms=6)

    # inside one SM
    ix, iy, iw, ih = 0.755, 0.26, 0.235, 0.56
    box(ax, ix, iy, iw, ih, fc=SURF, ec=GPU_C, lw=1.0)
    ax.text(ix + iw / 2, iy + ih + 0.065, "inside one SM", fontsize=6.4, ha="center",
            color=INK, fontweight="bold")
    rows = [("registers", "the running sum", 0.80),
            ("shared memory", "a strip of X and W, loaded\nonce, reused 128 times", 0.48),
            ("Tensor Cores", "16×16 blocks fused into\none instruction", 0.16)]
    for label, sub, y in rows:
        ax.text(ix + 0.014, iy + ih * y + 0.055, label, fontsize=5.9, color=GPU_C,
                fontweight="bold")
        ax.text(ix + 0.014, iy + ih * y + 0.022, sub, fontsize=5.1, color=INK2,
                va="top", linespacing=1.35)

    ax.text(0.5, 0.06,
            "A tile is sized so its operands fit in shared memory. That reuse is the whole "
            "trick: data is read\nfrom VRAM once and multiplied against many times — and "
            "30 tiles leave 16 SMs with nothing to do.",
            fontsize=5.8, ha="center", va="top", color=INK2, style="italic",
            linespacing=1.4)
    return save(fig, "fig-tiling.png")


# ---------------------------------------------------------------------------
# Figure 6 -- the memory hierarchy the tensors travel through
# ---------------------------------------------------------------------------

def fig_memory():
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.15))
    blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    levels = [
        ("Registers",       "~11 MB total",  "~20,000 GB/s", "the running sum of a dot product"),
        ("Shared mem / L1", "100 KB per SM", "~5,000 GB/s",  "the tile of X and W being reused"),
        ("L2 cache",        "~4 MB",         "~1,500 GB/s",  "spill-over between thread blocks"),
        ("VRAM (GDDR6)",    "8 GB",          "~450 GB/s",    "weights, activations, gradients"),
        ("PCIe 4.0 x16",    "the bus",       "~25 GB/s",     "the batch, on its way in"),
        ("Host RAM",        "19 GB in WSL",  "~75 GB/s",     "the dataset, Python, the tokenizer"),
    ]

    x, w, h = 0.075, 0.40, 0.112
    ax.text(x, 0.955, "level", fontsize=5.6, color=MUTED, va="center")
    ax.text(0.505, 0.955, "bandwidth", fontsize=5.6, color=MUTED, va="center", ha="right")
    ax.text(0.545, 0.955, "what lives here during a training step", fontsize=5.6,
            color=MUTED, va="center")
    y = 0.925
    for name, cap, bw, use in levels:
        c = GPU_C if name not in ("PCIe 4.0 x16", "Host RAM") else CPU_C
        box(ax, x, y - h + 0.010, w, h - 0.020, fc=c, ec="none", alpha=0.18)
        box(ax, x, y - h + 0.010, 0.008, h - 0.020, fc=c, ec="none")
        ax.text(x + 0.020, y - 0.032, name, fontsize=6.4, color=INK, fontweight="bold",
                va="center")
        ax.text(x + 0.020, y - 0.076, cap, fontsize=5.4, color=INK2, va="center")
        ax.text(0.505, y - 0.054, bw, fontsize=6.2, color=c, fontweight="bold",
                va="center", ha="right")
        ax.text(0.545, y - 0.054, use, fontsize=5.5, color=INK2, va="center")
        y -= h

    ax.plot([0.048, 0.048], [y + 0.02, 0.925], lw=1.0, color=MUTED)
    arrow(ax, (0.048, 0.55), (0.048, 0.915), color=MUTED, lw=1.0, ms=6)
    ax.text(0.033, 0.55, "faster, smaller, closer to the ALUs", fontsize=5.5,
            color=MUTED, rotation=90, va="center", ha="center")

    ax.text(0.048, 0.175,
            "Each rung down costs roughly an order of magnitude in bandwidth. A kernel that "
            "reads its operands\nfrom VRAM once and reuses them in shared memory runs at the "
            "top of this ladder; one that streams a\ntensor in and straight back out runs at "
            "the bottom, no matter how many ALUs are waiting for it.",
            fontsize=5.75, color=INK2, va="top", linespacing=1.5)
    ax.text(0.048, 0.012, "Capacities and bandwidths are architecture figures for this class "
                          "of part, not measurements.",
            fontsize=5.2, color=MUTED, style="italic")
    return save(fig, "fig-memory.png")


# ---------------------------------------------------------------------------
# Figure 7 -- arithmetic intensity: which ops the hardware can actually feed
# ---------------------------------------------------------------------------

VRAM_BW = 450.0   # GB/s, architecture figure for this class of GPU


def gemm_intensity(m, k, n, bytes_per_el=4):
    flops = 2 * m * k * n
    traffic = bytes_per_el * (m * k + k * n + m * n)
    return flops / traffic


def fig_intensity(peak_gflops):
    ridge = peak_gflops / VRAM_BW

    ops = [
        ("Dense 784→1024, batch 8192", gemm_intensity(8192, 784, 1024)),
        ("Dense 784→1024, batch 128",  gemm_intensity(128, 784, 1024)),
        ("Dense 512→256, batch 128",   gemm_intensity(128, 512, 256)),
        ("Dense 256→10, batch 128",    gemm_intensity(128, 256, 10)),
        ("Bias add",                   0.5 / 4),
        ("ReLU",                       1.0 / 8),
        ("BatchNorm (inference form)", 2.0 / 8),
    ]
    labels = [o[0] for o in ops][::-1]
    vals = [o[1] for o in ops][::-1]

    fig, ax = plt.subplots(figsize=(PAGE_W, 2.5))
    clean(ax)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 600)
    ax.axvspan(0.05, ridge, color=GRID, alpha=0.55, lw=0, zorder=0)

    ys = np.arange(len(vals))
    ax.hlines(ys, 0.05, vals, color=GPU_C, lw=1.6, alpha=0.45, zorder=2)
    ax.plot(vals, ys, "o", ms=5.5, color=GPU_C, mec=SURF, mew=1.0, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_ylim(-0.75, len(vals) - 0.35)
    for y, v in zip(ys, vals):
        lbl = f"{v:,.2f}" if v < 1 else (f"{v:,.1f}" if v < 10 else f"{v:,.0f}")
        ax.annotate(lbl, (v, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=5.8, color=INK2)

    ax.axvline(ridge, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.annotate(f"machine balance\n{ridge:.0f} FLOP/byte", (ridge, 1.5),
                xytext=(7, 0), textcoords="offset points", fontsize=5.8, color=INK2,
                va="center", linespacing=1.35, zorder=6)
    ax.text(0.07, -0.62, "memory-bound", fontsize=6.0, color=INK2, style="italic")
    ax.text(ridge * 1.35, -0.62, "compute-bound", fontsize=6.0, color=INK2, style="italic")
    ax.set_xlabel("arithmetic intensity — FLOP per byte of traffic (log scale)", fontsize=6.4)
    ax.set_title("What a GPU can be fast at", loc="left", pad=6)
    ax.grid(axis="y", visible=False)
    return save(fig, "fig-intensity.png")


# ---------------------------------------------------------------------------
# Figure 8 -- measured step time, and the fixed cost hiding inside it
# ---------------------------------------------------------------------------

def fig_cost(cpu_runs, gpu_runs, cf, cm, gf, gm, xo):
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.6))
    clean(ax)

    grid = np.linspace(0, 9000, 300)
    for runs, f, m, c, name in ((cpu_runs, cf, cm, CPU_C, "CPU"),
                                (gpu_runs, gf, gm, GPU_C, "GPU")):
        b = [r["batch"] for r in runs]
        t = [r["ms_per_step"] for r in runs]
        ax.plot(grid, f + m * grid, color=c, lw=1.2, alpha=0.55, zorder=2)
        ax.plot(b, t, "o", ms=5.5, color=c, mec=SURF, mew=1.0, zorder=4, label=name)

    ax.axhline(gf, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(8650, gf + 3.5, f"fixed cost ≈ {gf:.0f} ms, both devices", fontsize=5.9,
            color=INK2, ha="right", va="bottom")
    ax.annotate("CPU", (8192, cf + cm * 8192), xytext=(-6, -13), textcoords="offset points",
                fontsize=6.6, color=CPU_C, fontweight="bold", ha="right")
    ax.annotate("GPU", (8192, gf + gm * 8192), xytext=(-6, 9), textcoords="offset points",
                fontsize=6.6, color=GPU_C, fontweight="bold", ha="right")

    b0 = gpu_runs[0]["batch"]
    ax.annotate(f"at batch {b0} the GPU's own arithmetic is only {gm * b0:.1f} ms\n"
                f"of the {gf + gm * b0:.1f} ms step — all the rest is toll",
                xy=(430, 148), xytext=(430, 148), textcoords="data",
                fontsize=5.9, color=GPU_C, linespacing=1.4)

    ax.set_xlim(-260, 8800)
    ax.set_ylim(0, 200)
    ax.set_xticks([0, 2048, 4096, 6144, 8192])
    ax.set_xlabel("batch size (samples per step)", fontsize=6.4)
    ax.set_ylabel("milliseconds per training step", fontsize=6.4)
    ax.set_title("One MLP training step: a fixed toll plus work", loc="left", pad=6)
    ax.legend(loc="upper center", fontsize=6.2, ncols=2)
    return save(fig, "fig-cost.png")


# ---------------------------------------------------------------------------
# Figure 9 -- how much of each device's own ceiling the MLP actually reaches
# ---------------------------------------------------------------------------

def fig_efficiency(cpu_runs, gpu_runs, cpu_peak, gpu_peak):
    per_sample = mlp_fwd_flops_per_sample() * TRAIN_FACTOR

    def achieved(runs):
        b = np.array([r["batch"] for r in runs], dtype=float)
        t = np.array([r["ms_per_step"] for r in runs], dtype=float) / 1e3
        return b, b * per_sample / t / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.35), sharey=True)
    for ax, runs, peak, c, name in ((axes[0], cpu_runs, cpu_peak, CPU_C, "CPU"),
                                    (axes[1], gpu_runs, gpu_peak, GPU_C, "GPU")):
        clean(ax)
        b, g = achieved(runs)
        ax.plot(b, g, "-o", ms=5, lw=1.6, color=c, mec=SURF, mew=1.0, zorder=4)
        ax.axhline(peak, color=c, lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.annotate(f"GEMM ceiling {peak:,.0f}", (b[0], peak), xytext=(0, 4),
                    textcoords="offset points", fontsize=5.8, color=c)
        for bi, gi in zip(b, g):
            pct = gi / peak * 100
            ax.annotate(f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%", (bi, gi), xytext=(0, -11),
                        textcoords="offset points", fontsize=5.6, color=INK2, ha="center")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([128, 512, 2048, 8192])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(90, 13000)
        ax.set_xlabel("batch size", fontsize=6.4)
        ax.set_title(name, loc="left", color=c, pad=5)
    axes[0].set_ylabel("GFLOP/s actually achieved", fontsize=6.4)
    axes[0].set_ylim(60, 9000)
    fig.suptitle("The same MLP, against each device's own measured ceiling",
                 fontsize=8, fontweight="bold", color=INK, x=0.005, ha="left", y=1.06)
    fig.subplots_adjust(wspace=0.08)
    return save(fig, "fig-efficiency.png")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def build_markdown(d, n, figs):
    gname = d["gpu"]["meta"]["device_name"]
    tf = d["gpu"]["meta"]["tf_version"]
    kr = d["gpu"]["meta"]["keras_version"]

    def fig(name, caption, width="100%"):
        return f'\n![{caption}](figures/{name}){{width={width}}}\n'

    front = f"""---
title: "Neural Networks and the Machine"
subtitle: "How a multilayer perceptron, a CPU, a GPU and their memories divide one training step"
date: "{date.today().isoformat()}"
lang: en
toc: true
toc-depth: 2
fontsize: 11pt
documentclass: article
colorlinks: false
header-includes: |
  \\input{{{PREAMBLE.name}}}
---
"""

    body = f"""
# What this document is for

Its companion, *CPU vs GPU untuk Beban Kerja AI*, measured what happens when the
same neural network is trained on a CPU and on a GPU. It answers the question
*what are the numbers*. This document answers the other one: *why are they
those numbers*.

The subject is a single machine — a laptop with an **{gname}** and an
**Intel Core i9-12900H**, running TensorFlow {tf} and Keras {kr} under WSL2 —
and a single idea: that a neural network and the hardware underneath it are not
two separate topics that happen to meet at an API. They are the same object
described twice. A layer is a matrix multiply; a matrix multiply is a grid of
independent dot products; a grid of independent dot products is exactly the
shape of work a GPU was built to swallow. Every place that correspondence holds,
the hardware runs near its limit. Every place it breaks, performance falls off a
cliff — and the measurements in the companion report are a catalogue of both.

Every number quoted here is either measured on that machine or derived from
those measurements by arithmetic that is shown. Where a figure comes from the
architecture rather than the stopwatch, it says so.

{fig(figs['stack'], 'The stack a single training step passes through. Everything above the dashed line is software running on the CPU; everything below is silicon on the GPU. Neither half can do the job alone.')}

The picture above is the map for the rest of the document. Read it downward and
it is a call stack. Read it as a loop — down, then back up, sixty thousand
times per epoch — and it is a training run.

Part I follows the hardware: what the two chips are actually good at, how they
cooperate, and what the cooperation costs. Part II follows the software: what a
multilayer perceptron does, why that work is matrix multiplication, and why
matrix multiplication is the one operation this silicon executes near its peak.
Part III puts the two halves against each other and reads the measurements.


# Part I — The hardware

## Two chips that made opposite bets

A CPU and a GPU are both piles of transistors that multiply numbers. The
difference is what fraction of those transistors do the multiplying.

{fig(figs['die'], 'Transistor budget, schematically. The CPU spends most of its area on control logic and cache so that a few instruction streams can run as fast as possible. The GPU spends almost all of its area on arithmetic lanes and shares one control unit across many of them.')}

A CPU core is built to make **one** instruction stream finish quickly. It
predicts branches, reorders instructions, speculates past cache misses, and
keeps tens of megabytes of cache nearby so that unpredictable memory access
patterns are survivable. All of that machinery is overhead in the strict sense:
it computes nothing. It exists so that the small number of ALUs are never idle
waiting for the next instruction to be decided.

A GPU makes the opposite bet. It assumes the work is **already known to be
parallel**, so it does not need to discover parallelism at runtime. One control
unit issues the same instruction to 32 lanes at once. There is no speculation,
no out-of-order execution, and comparatively little cache. What that buys is
lanes: {n['sm']} streaming multiprocessors, {n['lanes']:,} fp32 lanes,
{n['tc']} Tensor Cores. What it costs is that any work which is *not* already
parallel — a branch that diverges, a dependency chain, a small tensor — leaves
most of those lanes doing nothing.

The measured ceilings on this machine make the trade concrete. On a pure fp32
matrix multiply, the CPU reached **{n['cpu_peak']:,.0f} GFLOP/s** and the GPU
**{n['gpu_peak']:,.0f} GFLOP/s** — a factor of {n['peak_ratio']:.1f}. That
number is the *best case*, the ceiling nothing else in the companion report
comes close to. Holding onto it is the entire engineering problem.

## The division of labour

The GPU is not a computer. It cannot read a file, run the Python interpreter,
decide when an epoch has ended, or decide which kernel to launch next. It is an
accelerator on the far side of a bus, and it does exactly what the CPU tells it
to, when the CPU tells it to.

{fig(figs['handshake'], 'One training step. The host builds and dispatches work; the device executes it. The two run concurrently — until the host needs a result back, which is where the concurrency ends.')}

The important detail is step 4 and step 5. When the host enqueues a kernel, the
call **returns immediately**: the kernel has been placed on a stream, not run.
The host is free to enqueue the next one. This is what allows a slow, single-threaded
Python loop to drive a device with thousands of lanes — the host runs
ahead, building a queue the device eats through.

The moment that breaks is synchronisation. Anything that needs a *value* back
on the host — printing a loss, evaluating a metric, a Python `if` on a tensor —
forces the host to wait for the stream to drain. Keras does this once per step
by default (`steps_per_execution=1`). The companion report measures the cost of
that default directly: raising `steps_per_execution` to 32, so that thirty-two
steps are dispatched as one unit, took the GPU from
{n['spe1_gpu']:,.0f} to {n['spe32_gpu']:,.0f} samples/s
({n['spe_gain_gpu']:.1f}x) with no change to the model, the data, or a single
hyperparameter.

That is a useful result on its own. It is a more useful one when you notice that
the same change also sped up the **CPU** run, from {n['spe1_cpu']:,.0f} to
{n['spe32_cpu']:,.0f} samples/s. A device-synchronisation argument cannot
explain a speedup on the device that was never being synchronised with. What
both runs share is the Python-side dispatch loop — and on this machine that
loop has only 3 cores to run on.

## Memory is the hierarchy that matters

Arithmetic is not the scarce resource. Moving the operands to the arithmetic is.

{fig(figs['memory'], 'The memory hierarchy under one training step. Each rung down costs roughly an order of magnitude in bandwidth. The bottom two rungs are on the far side of the bus.')}

Two consequences of that ladder shape are worth stating plainly.

**The 8 GB at the VRAM rung is a hard wall, not a soft one.** A tensor a kernel
touches must be in VRAM. Not "should be" — must. Weights, activations kept for
the backward pass, gradients, and the optimiser's own state all live there
simultaneously, which is why memory use during training is several times the
size of the model. When it does not fit, nothing gets slower; the process dies.

**The PCIe rung is roughly twenty times narrower than the VRAM rung.** Anything
that crosses it during the step — a batch that was not prefetched, a metric
pulled back to Python, a `.numpy()` call in a callback — is paid for at the
slowest rate in the system. This is why the input pipeline is a performance
topic and not a plumbing detail, and why `.cache()` and
`.prefetch(tf.data.AUTOTUNE)` are load-bearing.

## What the handshake costs, measured

All of the above predicts a specific shape: every training step should carry a
**fixed cost** that does not depend on how much work the step contains, plus a
**marginal cost** proportional to the work. Fitting a straight line to the
measured step times confirms it, and does so unusually cleanly.

{fig(figs['cost'], 'Measured milliseconds per MLP training step against batch size, with a least-squares line through each device. The intercept is the fixed cost per step; the slope is the cost of one more sample.')}

The fit gives:

{n['cost_table']}

Two things stand out.

**The marginal costs differ by {n['marg_ratio']:.1f}x.** This is the GPU
advantage, and it is the only place the GPU advantage lives. Each additional
sample costs the GPU {n['gpu_marg_us']:.1f} microseconds against the CPU's
{n['cpu_marg_us']:.1f}.

**The fixed costs are nearly identical — about {n['gpu_fixed']:.1f} ms on
both.** If that toll were a device cost, it would not be the same on a device
that is not being used. Read together with the `steps_per_execution` result
above, the most economical explanation is that most of the per-step floor on
this machine is spent on the *host*: tracing, kernel lookup, allocator
bookkeeping, and the Python round trip, on 3 available cores. The GPU inherits
the CPU's dispatch bottleneck along with its own.

The arithmetic that follows is stark. At batch 128, the GPU's own work is
{n['gpu_work_128']:.2f} ms inside a {n['gpu_step_128']:.2f} ms step: about
{n['gpu_util_128']:.0f}% of the elapsed time is arithmetic and the rest is
toll. It takes a batch of roughly **{n['breakeven']:,.0f} samples** before the
computation is merely *equal* to the fixed cost. At batch 8192 the balance has
finally inverted — {n['gpu_work_8192']:.1f} ms of work inside a
{n['gpu_step_8192']:.1f} ms step, or {n['gpu_util_8192']:.0f}% arithmetic.

This is the mechanism behind the companion report's headline finding. A GPU is
not "faster". A GPU is a machine that charges a large entry fee and then sells
arithmetic cheaply. Whether that is a good deal depends entirely on how much
arithmetic you buy.
"""

    body += f"""

# Part II — The software

## A multilayer perceptron, written out

The network measured in the companion report is a plain MLP: an input of
{MLP_DIMS[0]} features, then Dense layers of
{', '.join(str(x) for x in MLP_DIMS[1:-1])} and {MLP_DIMS[-1]} units, ReLU
between them, softmax at the end, {n['mlp_params']:,} parameters. Nothing in it
is exotic, which is the point: it is the simplest thing that still exercises the
whole machine.

A single Dense layer is defined by

$$ H = f(X W + b) $$

and that is the whole story. $X$ holds the batch, one sample per row. $W$ holds
the weights. The product $XW$ is a matrix multiply — a **GEMM**, in the
vocabulary of the libraries that implement it. $b$ is a broadcast add, and $f$
is applied elementwise. Stack four of those and you have the network.

{fig(figs['mlp'], 'Top: the anatomy of one Dense layer as a GEMM. Bottom: the four GEMMs of the network under test, with the arithmetic each one costs per sample.')}

Three properties of that picture are what make the operation suit a GPU, and it
is worth naming them separately because a great many operations in a neural
network have only one or two.

**Every output element is independent.** $H_{{ij}}$ depends on row $i$ of $X$
and column $j$ of $W$ and on nothing else — not on $H_{{i,j-1}}$, not on
anything computed earlier in the same layer. There is no dependency chain to
discover and no order to respect. At batch 128 the first layer produces
131,072 output elements that could, in principle, be computed in 131,072
different places at the same moment.

**Every element is computed the same way.** The same instruction sequence — a
chain of fused multiply-adds — produces all of them. Nothing branches on the
data. This is precisely the contract the GPU's shared control unit requires: one
instruction, many lanes, no divergence.

**Each operand is used many times.** Row $i$ of $X$ participates in every one of
the 1024 outputs of row $i$. Column $j$ of $W$ participates in every one of the
batch's rows. That reuse is what makes it possible to read data from slow memory
once and do a great deal of arithmetic on it — and, as the next two sections
show, it is the property that actually decides the outcome.

The batch dimension deserves one more sentence. It is easy to read `batch_size`
as a memory-management convenience — how many samples we can afford to hold at
once. On this hardware it is better read as the opposite: **the batch dimension
is where the parallelism comes from.** A batch of 1 gives the GPU a matrix-vector
product, which has almost no reuse and leaves most of {n['sm']} SMs empty. A
batch of 8192 gives it a large, square-ish GEMM. Same model, same code, two
completely different pieces of hardware being exercised.

## How a GEMM becomes 46 SMs of work

Knowing that a matrix multiply is parallel does not make it fast. The
implementation in cuBLAS has to arrange the work so that the memory hierarchy in
Part I is used the right way round.

{fig(figs['tiling'], 'A GEMM is decomposed into tiles of the output matrix. Each tile becomes a thread block on one SM; inside the SM, strips of the operands are staged in shared memory and reused across the tile.')}

The output matrix is cut into tiles. Each tile is assigned to a thread block,
each thread block runs on one SM, and each thread accumulates a few output
elements in registers. The k-dimension is walked in strips: a strip of $X$ and a
strip of $W$ are loaded from VRAM into shared memory once, and then every thread
in the block multiplies against them repeatedly.

That last clause is where the performance comes from. A tile of 128x128 outputs
loads $2 \\times 128 \\times k$ operand values and performs
$2 \\times 128 \\times 128 \\times k$ FLOPs on them: each value that crossed the
VRAM rung is used 128 times before it is discarded. Without the staging, the
same multiply would re-read its operands from VRAM for every FLOP and run at
memory speed — roughly two orders of magnitude slower — no matter how many lanes
were available.

The Tensor Cores sit one level below this. On this GPU (compute capability 8.6)
each SM has {n['tc_per_sm']} of them, and each executes a small fixed-size
matrix multiply-accumulate as a single instruction rather than as a loop of
scalar FMAs. They are why `tf32` and `bf16` matter: those formats let the same
matrices go through the fast path. They are also why Tensor Core throughput is
so often unreachable in practice — feeding them requires the staging above to
keep up, and the memory hierarchy usually cannot.

## Arithmetic intensity: the number that decides everything

Both halves of this document now reduce to one ratio. For any operation, divide
the arithmetic it performs by the bytes it must move:

$$ I = \\frac{{\\text{{FLOPs}}}}{{\\text{{bytes of memory traffic}}}} $$

The hardware has the same ratio: its peak arithmetic rate divided by its memory
bandwidth. On this GPU that is roughly {n['gpu_peak']:,.0f} GFLOP/s over
~{VRAM_BW:.0f} GB/s, or about **{n['ridge']:.0f} FLOP per byte**. An operation
above that number can be fed fast enough to keep the lanes busy; one below it
starves, and its speed is set by memory bandwidth alone — the ALU count is
irrelevant.

{fig(figs['intensity'], "Arithmetic intensity of the operations in the network under test, against this GPU's balance point (the measured GEMM ceiling divided by VRAM bandwidth). Anything in the shaded region is limited by memory bandwidth, not by arithmetic.")}

The spread on that axis is three orders of magnitude, and it explains almost
every result in the companion report:

- The first Dense layer at batch 8192 sits at **{n['ai_big']:.0f} FLOP/byte**,
  twenty times above the balance point. This is the regime the GPU was designed
  for.
- The same layer at batch 128 sits at **{n['ai_small']:.0f}** — still
  compute-bound, but with four times less headroom. The tiles are smaller, the
  reuse is lower, and fewer SMs have anything to do.
- The last Dense layer, 256 to 10, sits at **{n['ai_tiny']:.1f}**, below the
  line. It is a memory-bound operation regardless of batch size, because a
  10-column weight matrix simply does not offer enough reuse.
- ReLU, bias add, and normalisation sit at **0.1 to 0.3**. They read a tensor,
  do one or two arithmetic operations per element, and write it back. On this
  GPU they run about **eighty times** below the balance point. They are pure
  memory traffic wearing the costume of a computation.

## The layers between the GEMMs

That last bullet is not a footnote. A network is not only its matrix multiplies,
and the cheap-looking operations between them are exactly where GPU advantage
leaks away.

The companion report measures this on the CNN case. A separate profile of one
training step found that `BatchNormalization` alone accounted for roughly
**51 ms of the 90 ms** GPU step — a layer with a negligible FLOP count consuming
more than half the time. In Keras 3 on the TensorFlow backend it was not fused
into the cuDNN convolution kernel, so it became a sequence of small,
memory-bound passes, each paying its own launch cost, dispatched by a host with
3 cores.

The measured training results line up with that reading exactly:

{n['train_table']}

The ordering is not about how "modern" each architecture is. It is about how
much arithmetic each one does per kernel launch. The transformer's attention and
feed-forward blocks are large GEMMs — it gains {n['trf_speedup']:.1f}x. The MLP's
GEMMs are large too, but its steps are so short that the fixed cost from Part I
swallows the gain, and the GPU actually *loses*. The CNN sits between them and is
dragged down by its normalisation layers.

This is also the honest answer to why kernel fusion exists. `jit_compile=True`
(XLA) tries to merge chains of small operations into one kernel so that the
intermediate tensors never round-trip to VRAM. When it works, it converts a
column of memory-bound operations into one compute-bound one. When it does not —
and on the CNN measured here it did not, costing
{n['xla_penalty']:.0%} of throughput — it is because XLA also replaced cuDNN's
tuned convolution algorithm with its own. Fusion is a real mechanism with real
limits, not a switch that makes things faster.
"""

    body += f"""

# Part III — Reading the measurements

## The same network against each device's own ceiling

The cleanest way to see the argument land is to stop comparing the two devices
to each other and compare each one to *itself*: to the fp32 GEMM ceiling
measured on that same chip, minutes apart, in the same session.

The MLP does {n['fwd_mflop']:.2f} MFLOP of forward arithmetic per sample. The
backward pass costs about twice the forward pass — one GEMM for the input
gradient and one for the weight gradient per layer — so a training step is about
{n['train_mflop']:.1f} MFLOP per sample. Multiplying by batch size and dividing
by the measured step time gives the rate actually achieved:

{fig(figs['efficiency'], "Achieved fp32 throughput on the MLP training step as a percentage of the same device's measured GEMM ceiling. Note the log scale on both axes.")}

The result is the opposite of the intuition most people bring to this comparison.

**The CPU reaches essentially all of its ceiling.** At batch 8192 it computes at
{n['cpu_ach_8192']:,.0f} GFLOP/s against a measured GEMM ceiling of
{n['cpu_peak']:,.0f} — around {n['cpu_pct_8192']:.0f}%, which is to say the MLP
training step runs as fast as the pure benchmark does. (Values slightly above
100% are run-to-run variation; the companion report documents roughly +/-30%
spread on absolute figures from thermal and clock behaviour.) The CPU has few
lanes, but it fills them.

**The GPU reaches {n['gpu_pct_8192']:.0f}% at its best and
{n['gpu_pct_128']:.1f}% at batch 128.** Everything in Parts I and II is visible
in that one number. The fixed dispatch cost is charged whether or not there is
work. The activations, the bias adds, and the softmax run at memory speed. The
last layer is too narrow to fill the machine. The batch dimension — the only
source of parallelism the GPU has — is too small to occupy {n['sm']} SMs.

The gap between {n['gpu_pct_128']:.1f}% and {n['gpu_pct_8192']:.0f}% is not a
defect in the GPU. It is the distance between "an operation that is theoretically
parallel" and "an operation shaped so that this specific memory hierarchy can
feed this specific number of lanes." Closing that distance is what batch size,
`steps_per_execution`, mixed precision, and fusion all do. None of them changes
the model.

## What follows, in order of leverage

Every item below is a direct consequence of a mechanism above, applied to this
machine.

1. **Increase the batch until VRAM is nearly full.** It raises arithmetic
   intensity, fills more SMs, and amortises the {n['gpu_fixed']:.1f} ms fixed
   cost across more samples. Nothing else on this list moves the number as far
   for as little effort.
2. **Set `steps_per_execution` to 32 in `model.compile`.** One argument, no
   effect on results, measured at {n['spe_gain_gpu']:.1f}x here. It attacks the
   host-side dispatch cost identified in Part I directly.
3. **Enable bf16 mixed precision.** Compute capability 8.6 has bf16 Tensor
   Cores, and bf16 carries the same exponent range as fp32, so no loss scaling
   is needed. On the transformer it gave {n['bf16_trf']:.1f}x. The VRAM it frees
   is often worth more than the arithmetic, because that VRAM buys a larger
   batch, and item 1 is the bigger lever anyway.
4. **Feed the device.** With 3 cores, host-side loading and augmentation are a
   genuine bottleneck. `tf.data` with `.cache()` and
   `.prefetch(tf.data.AUTOTUNE)`, 2 to 4 workers.
5. **Then, and only then, try `jit_compile=True`.** Measure it. On the CNN
   measured here it cost {n['xla_penalty']:.0%} of throughput.

And the two rules that decide whether to reach for the GPU at all:

**Use the GPU** when the work per step is large — convolutions, attention, big
dense layers, long runs, batched inference over a corpus. Above the balance
point, the machine does what it says on the box.

**Use the CPU** when requests arrive one at a time with a latency target, for
classical models, for tokenisation and preprocessing, and for the first hours of
development when you only want to know whether the code runs. Moving a 200 MB
model into VRAM to classify one sentence is a net loss, and the measurements in
the companion report show it: at batch 1 the GPU's advantage on the CNN was
{n['inf_b1_cnn']:.1f}x, against {n['inf_b64_cnn']:.1f}x at batch 64.

## One sentence

A neural network is a chain of matrix multiplications interrupted by cheap
elementwise work; a GPU is an enormous quantity of arithmetic behind a narrow
memory system and an expensive front door. Performance is whatever survives the
meeting of those two facts.


# Notes on the numbers

**Measured on this machine**, by `cpu_vs_gpu.py` with the `standard` preset,
each case in its own process, warm-up discarded, device synchronised before the
timer stops, medians rather than means: all GFLOP/s figures, all step times, all
samples/s figures, all speedups.

**Derived here**, by arithmetic shown in the text: the FLOP counts for the MLP,
the fixed and marginal costs from a least-squares fit to the four measured step
times per device, the achieved-throughput percentages, and the arithmetic
intensities.

**Architecture figures, not measured here**: {n['sm']} SMs,
{n['lanes']:,} fp32 lanes, {n['tc']} Tensor Cores, ~{VRAM_BW:.0f} GB/s of VRAM
bandwidth, ~4 MB of L2, 100 KB of shared memory per SM, PCIe 4.0 x16 bandwidth,
and host DRAM bandwidth. These describe the class of part; they are used here
only to give the memory hierarchy a scale, and no conclusion in this document
depends on their precision.

**Not controlled**: this is a laptop. CPU and GPU clocks fall under thermal
load, and the companion report measures the effect at up to 30% on long runs and
more on short bursts. Ratios between CPU and GPU are far more stable than
absolute numbers, because both sides are measured under similar thermal
conditions.

To regenerate everything in this document:

```bash
python cpu_vs_gpu.py --preset standard --report results
python make_hardware_report.py
```
"""
    MD.write_text(front + body)
    print(f"markdown -> {MD.name}")


# ---------------------------------------------------------------------------

# Architecture facts for this part (GA104, compute capability 8.6). Quoted, not
# measured -- see "Notes on the numbers" in the report.
SM_COUNT = 46
LANES_PER_SM = 128
TC_PER_SM = 4


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def main():
    d = load()
    cpu, gpu = d["cpu"], d["gpu"]

    cs = case(cpu, "scaling")["runs"]
    gs = case(gpu, "scaling")["runs"]
    cf, cmarg = fit_cost(cs)
    gf, gmarg = fit_cost(gs)

    cpu_peak = max(r["gflops"] for r in case(cpu, "gemm")["runs"])
    gpu_peak = max(r["gflops"] for r in case(gpu, "gemm")["runs"])

    print("figures:")
    figs = {
        "stack": fig_stack().name,
        "die": fig_die().name,
        "handshake": fig_handshake().name,
        "memory": fig_memory().name,
        "cost": fig_cost(cs, gs, cf, cmarg, gf, gmarg,
                         crossover_batch(cf, cmarg, gf, gmarg)).name,
        "mlp": fig_mlp_matmul().name,
        "tiling": fig_tiling().name,
        "intensity": fig_intensity(gpu_peak).name,
        "efficiency": fig_efficiency(cs, gs, cpu_peak, gpu_peak).name,
    }

    ov_c = {r["steps_per_execution"]: r for r in case(cpu, "overhead")["runs"]}
    ov_g = {r["steps_per_execution"]: r for r in case(gpu, "overhead")["runs"]}
    xl_c = {r["jit_compile"]: r for r in case(cpu, "xla")["runs"]}
    xl_g = {r["jit_compile"]: r for r in case(gpu, "xla")["runs"]}

    per_sample = mlp_fwd_flops_per_sample() * TRAIN_FACTOR
    step = {r["batch"]: r["ms_per_step"] for r in gs}

    def achieved(runs, b):
        t = {r["batch"]: r["ms_per_step"] for r in runs}[b] / 1e3
        return b * per_sample / t / 1e9

    train_rows = []
    for key, label in (("mlp", "MLP"), ("cnn", "CNN"), ("transformer", "Transformer")):
        c, g = case(cpu, key), case(gpu, key)
        train_rows.append([label, f"{c['params']/1e6:.1f}M", f"{c['s_per_epoch']:.1f}",
                           f"{g['s_per_epoch']:.1f}",
                           f"{g['samples_per_s']/c['samples_per_s']:.1f}x"])

    inf_c = case(cpu, "inference")["models"]["cnn"]["runs"]
    inf_g = case(gpu, "inference")["models"]["cnn"]["runs"]
    lat_c = {r["batch"]: r["latency_ms"] for r in inf_c}
    lat_g = {r["batch"]: r["latency_ms"] for r in inf_g}

    bf = case(gpu, "bf16")["runs"]["transformer"]

    n = {
        "sm": SM_COUNT,
        "lanes": SM_COUNT * LANES_PER_SM,
        "tc": SM_COUNT * TC_PER_SM,
        "tc_per_sm": TC_PER_SM,
        "cpu_peak": cpu_peak,
        "gpu_peak": gpu_peak,
        "peak_ratio": gpu_peak / cpu_peak,
        "ridge": gpu_peak / VRAM_BW,

        "spe1_gpu": ov_g[1]["samples_per_s"],
        "spe32_gpu": ov_g[32]["samples_per_s"],
        "spe_gain_gpu": ov_g[32]["samples_per_s"] / ov_g[1]["samples_per_s"],
        "spe1_cpu": ov_c[1]["samples_per_s"],
        "spe32_cpu": ov_c[32]["samples_per_s"],

        "cost_table": md_table(
            ["Device", "Fixed cost per step", "Marginal cost per sample"],
            [["CPU", f"{cf:.1f} ms", f"{cmarg*1e3:.1f} us"],
             ["GPU", f"{gf:.1f} ms", f"{gmarg*1e3:.1f} us"]]),
        "marg_ratio": cmarg / gmarg,
        "gpu_marg_us": gmarg * 1e3,
        "cpu_marg_us": cmarg * 1e3,
        "gpu_fixed": gf,
        "gpu_work_128": gmarg * 128,
        "gpu_step_128": step[128],
        "gpu_util_128": gmarg * 128 / step[128] * 100,
        "gpu_work_8192": gmarg * 8192,
        "gpu_step_8192": step[8192],
        "gpu_util_8192": gmarg * 8192 / step[8192] * 100,
        "breakeven": gf / gmarg,

        "mlp_params": mlp_params(),
        "fwd_mflop": mlp_fwd_flops_per_sample() / 1e6,
        "train_mflop": per_sample / 1e6,
        "ai_big": gemm_intensity(8192, 784, 1024),
        "ai_small": gemm_intensity(128, 784, 1024),
        "ai_tiny": gemm_intensity(128, 256, 10),

        "train_table": md_table(
            ["Model", "Params", "CPU s/epoch", "GPU s/epoch", "GPU speedup"], train_rows),
        "trf_speedup": (case(gpu, "transformer")["samples_per_s"]
                        / case(cpu, "transformer")["samples_per_s"]),
        "xla_penalty": 1 - xl_g[True]["samples_per_s"] / xl_g[False]["samples_per_s"],

        "cpu_ach_8192": achieved(cs, 8192),
        "cpu_pct_8192": achieved(cs, 8192) / cpu_peak * 100,
        "gpu_pct_8192": achieved(gs, 8192) / gpu_peak * 100,
        "gpu_pct_128": achieved(gs, 128) / gpu_peak * 100,

        "bf16_trf": bf["samples_per_s"] / case(gpu, "transformer")["samples_per_s"],
        "inf_b1_cnn": lat_c[1] / lat_g[1],
        "inf_b64_cnn": lat_c[64] / lat_g[64],
    }

    build_markdown(d, n, figs)

    if not shutil.which("pandoc"):
        sys.exit("pandoc not installed; skipping PDF.")

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
        sys.exit("pandoc failed.")
    print(f"pdf      -> {PDF.name}  ({PDF.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
