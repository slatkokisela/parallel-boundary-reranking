"""Generate the README figures from result directories (no numbers typed by hand).

    python scripts/make_figures.py

Writes docs/figures/{trec_covid_budget,batching_quality,latency_cpu_vs_gpu}.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TREC_EVAL = Path("results/trec_covid_bge-reranker-v2-m3_eval_ff82cbda5a")
NF_TEST = Path("results/final_test_8b63c00e50")
LAT_CPU = Path("results/latency_dev_663e55587d")
LAT_GPU = Path("results/latency_gpu_dev_70a9ca67c3")
OUT = Path("docs/figures")

SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (light surface)
CAL = '{"batch_size": 1, "prior": "calibrated"}'

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _b(params: str):
    return json.loads(params).get("batch_size")


def fig_trec_budget():
    s = pd.read_csv(TREC_EVAL / "summary.csv")
    s = s[s.subset == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, k in zip(axes, [20, 50, 100]):
        sk = s[s.k == k]
        mips = sk[sk.method == "mips"].ncg_pool.iloc[0] * 100
        full = sk[sk.method == "full_ce"].ncg_pool.iloc[0] * 100
        for y, lab in [(full, "Full-CE"), (mips, "MIPS")]:
            ax.axhline(y, color=REF, lw=1.2, ls="--", zorder=1)
            ax.text(74, y + 0.4, lab, color=INK2, fontsize=8, va="bottom", ha="right")
        for meth, params, color, lab in [("morph", CAL, BLUE, "BoundaryMORPH (calibrated)"),
                                         ("single_stage", "{}", ORANGE, "SingleStage")]:
            d = sk[(sk.method == meth) & (sk.params == params)].sort_values("B")
            ax.errorbar(d.B, d.ncg_pool * 100, yerr=[(d.ncg_pool - d.ncg_pool_lo) * 100,
                                                     (d.ncg_pool_hi - d.ncg_pool) * 100],
                        color=color, lw=2, marker="o", ms=6, capsize=3, elinewidth=1,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab, zorder=3)
        ax.set_title(f"k = {k}", loc="left")
        ax.set_xticks([10, 25, 50]); ax.set_xlim(5, 76)
        ax.set_xlabel("CE budget B (calls per query)")
    axes[0].set_ylabel("human nCG@k (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("TREC-COVID eval (30 held-out queries): 10–50 CE calls ≈ reranking all 1000 docs",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.14, "CE: bge-reranker-v2-m3. Dashed: Full-CE = CE scores all 1000 pool docs; MIPS = dense retrieval "
             "only (0 calls). Error bars: 95% bootstrap CI of the mean over queries.",
             color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "trec_covid_budget.png", dpi=160, bbox_inches="tight")


def fig_batching():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, (run, title) in zip(axes, [(NF_TEST, "NFCorpus test (323 queries)"),
                                       (TREC_EVAL, "TREC-COVID eval (30 queries)")]):
        a = pd.read_csv(run / "analysis_ncg_pool_all.csv")
        a = a[(a.k == 50) & (a.B == 25)]
        ax.axhline(0, color=REF, lw=1.2, ls="--", zorder=1)
        for i, (prefix, color, lab) in enumerate([("morph_topb[b=", BLUE, "top-b"),
                                                  ("morph_kb[b=", ORANGE, "kriging believer")]):
            rows = a[a.method.str.startswith(prefix) & ~a.method.str.contains("fantasy")]
            bs = rows.method.str.extract(r"\[b=(\d+)\]")[0].astype(int)
            x = bs + (i - 0.5) * 0.25
            ax.errorbar(x, rows.delta * 100, yerr=[(rows.delta - rows.ci_lo) * 100, (rows.ci_hi - rows.delta) * 100],
                        fmt="o", color=color, ms=6, capsize=3, elinewidth=1.2,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab, zorder=3)
        ax.set_xticks([2, 4, 8]); ax.set_xlabel("documents selected per round (b)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Δ human nCG@50 vs sequential (pp)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Batch selection costs no detectable quality (k = 50, B = 25)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.02, "Paired difference vs calibrated sequential BoundaryMORPH; 95% bootstrap CI.",
             color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "batching_quality.png", dpi=160, bbox_inches="tight")


def fig_latency():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhline(1.0, color=REF, lw=1.2, ls="--", zorder=1)
    for run, color, lab in [(LAT_CPU, ORANGE, "CPU (laptop, 4 threads)"), (LAT_GPU, BLUE, "GPU (Kaggle T4)")]:
        s = pd.read_csv(run / "summary.csv")
        s = s[s.B == 50]
        s["b"] = s.params.map(_b)
        seq = s[(s.method == "morph") & (s.b == 1)].total_s_median.iloc[0]
        kb = s[s.method == "morph_kb"].sort_values("b")
        xs = [1] + kb.b.tolist()
        ys = [1.0] + (kb.total_s_median / seq).tolist()
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=lab, zorder=3)
        ax.text(8.25, ys[-1], f"{ys[-1]:.2f}×", color=INK, fontsize=9, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1\n(sequential)", "2", "4", "8"])
    ax.set_xlim(0.85, 10)
    ax.set_xlabel("documents selected per round (b), kriging believer")
    ax.set_ylabel("per-query latency\nrelative to sequential")
    ax.set_title("Batching only pays off on a GPU (B = 50, median per query)", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "latency_cpu_vs_gpu.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_trec_budget(); fig_batching(); fig_latency()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
