"""Generate publication-quality figures for the ML4PS @ NeurIPS 2026 paper.

Reads results/paper_results.json produced by scripts/experiments_paper.py.

Figures produced:
  1. figures/fig1_convergence.pdf  — violations vs epoch (convergence curve)
  2. figures/fig2_ablation.pdf     — NMSE_long + violations bar chart (all 3 systems)
  3. figures/fig3_lambda_sweep.pdf — log λ vs violations / MSE (dual axis)

All figures use NeurIPS-compatible styling:
  - 10pt font (Computer Modern via LaTeX rendering if available, else DejaVu Sans)
  - Single-column: 3.25 in wide; double-column: 6.75 in wide
  - Colour-blind-safe palette (Wong 2011)

Run: python3 scripts/generate_figures.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# NeurIPS-compatible style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

# Wong (2011) colour-blind-safe palette + extras for new conditions
COLORS = {
    "none":               "#E69F00",   # orange
    "soft":               "#56B4E9",   # sky blue
    "soft_high":          "#009E73",   # teal
    "hard":               "#D55E00",   # vermillion
    "log_param":          "#CC79A7",   # mauve
    "stoich_param":       "#0072B2",   # blue
    "hard_inference_only": "#F0E442",  # yellow (dark border for visibility)
    "hard_relu":           "#882255",  # wine
}
MARKERS = {
    "none":               "o",
    "soft":               "s",
    "soft_high":          "^",
    "hard":               "D",
    "log_param":          "P",
    "stoich_param":       "X",
    "hard_inference_only": "v",
    "hard_relu":          "d",
}
LABELS = {
    "none":               "No constraint",
    "soft":               r"Soft $(\lambda=1)$",
    "soft_high":          r"Soft $(\lambda=10)$",
    "hard":               "Hard (softplus)",
    "log_param":          "Log-param",
    "stoich_param":       "Stoich-param",
    "hard_inference_only": "Hard (infer only)",
    "hard_relu":          "Hard (ReLU)",
}
LINESTYLES = {
    "none":               "--",
    "soft":               ":",
    "soft_high":          "-.",
    "hard":               "-",
    "log_param":          (0, (3, 1, 1, 1)),   # dash-dot-dot
    "stoich_param":       (0, (5, 1)),          # long dash
    "hard_inference_only": (0, (1, 1)),         # dense dots
    "hard_relu":          (0, (5, 2, 1, 2)),    # dash-dot
}

# Base conditions shown in convergence curve (keeps figure readable)
BASE_CONDITIONS = ["none", "soft", "soft_high", "hard"]
# Full ablation display order
DISPLAY_ORDER = [
    "none", "soft", "soft_high", "hard", "hard_relu",
    "log_param", "stoich_param", "hard_inference_only",
]

RESULTS_PATH = Path(__file__).parent.parent / "results" / "paper_results.json"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

SYSTEM_NAMES = {
    "exothermic_cstr": "Exothermic CSTR",
    "vdv_cstr": "Van de Vusse CSTR",
    "batch_abc": r"Batch A$\to$B$\to$C",
}
VIOL_LABEL = {
    "exothermic_cstr": "Positivity viol. (%)",
    "vdv_cstr": "Positivity viol. (%)",
    "batch_abc": r"Mass drift $|\Delta\Sigma C|$",
}

CONDITIONS = BASE_CONDITIONS  # kept for backward-compat; prefer DISPLAY_ORDER


def _ms(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Centered rolling mean with edge correction."""
    kernel = np.ones(window) / window
    out = np.convolve(arr, kernel, mode="same")
    half = window // 2
    for i in range(half):
        out[i] = arr[: 2 * i + 1].mean()
        out[-(i + 1)] = arr[-(2 * i + 1) :].mean()
    return out


def _despine(ax: matplotlib.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load() -> dict[str, Any]:
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Convergence curve
# ---------------------------------------------------------------------------

def fig1_convergence(data: dict[str, Any]) -> None:
    """Violations vs epoch for all 4 conditions on exothermic CSTR.

    Hard = exactly 0 from epoch 1.  Soft plateaus above 0 with high seed variance.
    This is the core visual argument for the training-inference gap.
    """
    conv = data.get("convergence")
    if conv is None:
        print("  [skip] No convergence data in results. Run experiments first.")
        return

    fig, (ax_viol, ax_nmse) = plt.subplots(
        1, 2, figsize=(3.25, 2.0), sharey=False
    )

    for cond in CONDITIONS:
        runs = conv.get(cond, [])
        if not runs:
            continue
        epochs = runs[0]["epochs"]
        viol_matrix = np.array([r["viol_curve"] for r in runs])   # (seeds, T)
        nmse_matrix = np.array([r["nmse_curve"] for r in runs])

        viol_mean = _smooth(viol_matrix.mean(axis=0))
        viol_min  = _smooth(viol_matrix.min(axis=0))
        viol_max  = _smooth(viol_matrix.max(axis=0))
        nmse_mean = _smooth(nmse_matrix.mean(axis=0))
        nmse_min  = _smooth(np.maximum(nmse_matrix.min(axis=0), 1e-2))
        nmse_max  = _smooth(nmse_matrix.max(axis=0))

        color = COLORS[cond]
        ls = LINESTYLES[cond]
        label = LABELS[cond]

        ax_viol.plot(epochs, viol_mean, color=color, ls=ls, lw=1.6, label=label)
        ax_viol.fill_between(
            epochs, viol_min, viol_max,
            alpha=0.12, color=color,
        )

        ax_nmse.plot(epochs, nmse_mean, color=color, ls=ls, lw=1.6, label=label)
        ax_nmse.fill_between(
            epochs, nmse_min, nmse_max,
            alpha=0.12, color=color,
        )

    ax_viol.set_xlabel("Training epoch")
    ax_viol.set_ylabel("Positivity violation (%)")
    ax_viol.set_title("(a) Constraint violation vs. epoch")
    ax_viol.legend(loc="upper right", framealpha=0.9, frameon=True)
    ax_viol.set_ylim(bottom=0)
    ax_viol.grid(True, alpha=0.3)
    _despine(ax_viol)

    ax_viol.axhline(0, color=COLORS["hard"], lw=0.8, ls="-", alpha=0.5)
    ax_viol.text(
        0.98, 0.04, "Hard: 0.00% (all epochs)",
        transform=ax_viol.transAxes,
        ha="right", va="bottom", fontsize=7,
        color=COLORS["hard"],
    )

    ax_nmse.set_xlabel("Training epoch")
    ax_nmse.set_ylabel(r"Normalized MSE (log scale)")
    ax_nmse.set_title("(b) Accuracy vs. epoch")
    ax_nmse.set_yscale("log")
    ax_nmse.set_ylim(bottom=1e-1)
    ax_nmse.grid(True, which="both", alpha=0.3)
    _despine(ax_nmse)

    fig.suptitle("Exothermic CSTR: constraint violation and accuracy over training", y=1.01)
    fig.tight_layout()

    out = FIGURES_DIR / "fig1_convergence.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Main ablation bar chart
# ---------------------------------------------------------------------------

def fig2_ablation(data: dict[str, Any]) -> None:
    """Per-system subplots: one panel per benchmark, bars for all conditions, log NMSE.

    Violation fraction annotated inside bars (white text).
    Architecturally-zero-violation conditions get a checkmark.
    """
    ablation = data.get("ablation", {})
    systems = [k for k in SYSTEM_NAMES if k in ablation]
    if not systems:
        print("  [skip] No ablation data in results.")
        return

    n_sys = len(systems)
    fig, axes = plt.subplots(1, n_sys, figsize=(6.75, 2.8), sharey=False)
    if n_sys == 1:
        axes = [axes]

    SHORT = {
        "none": "None",
        "soft": r"Soft$_1$",
        "soft_high": r"Soft$_{10}$",
        "hard": "Hard\nsoftplus",
        "log_param": "Log\nparam",
        "stoich_param": "Stoich\nparam",
        "hard_inference_only": "Hard\ninfer",
        "hard_relu": "Hard\nReLU",
    }

    for ax, sys_key in zip(axes, systems):
        sys_res = ablation[sys_key]
        present = [c for c in DISPLAY_ORDER if c in sys_res]
        n_cond = len(present)
        bar_width = 0.62

        for ci, cond in enumerate(present):
            runs = sys_res[cond]
            if not runs:
                continue
            nmse_m, nmse_s = _ms([r["nmse_long"] for r in runs])
            pv_m, _pv_s = _ms([r["physics_viol"] for r in runs])
            color = COLORS[cond]

            ax.bar(
                ci, nmse_m,
                width=bar_width,
                color=color, alpha=0.85,
                yerr=nmse_s, capsize=2, error_kw={"elinewidth": 0.8},
                zorder=3,
            )
            # Violation annotation: "0%" if zero, else percentage
            if pv_m < 1e-4:
                ax.text(
                    ci, nmse_m * 1.6, "0%",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color, fontweight="bold",
                )
            else:
                label_y = max(nmse_m * 0.35, 3e-3)
                ax.text(
                    ci, label_y, f"{pv_m:.0f}%",
                    ha="center", va="bottom", fontsize=6,
                    color="white", fontweight="bold", rotation=90,
                )

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels([SHORT[c] for c in present], fontsize=7, rotation=0)
        ax.set_title(SYSTEM_NAMES[sys_key], fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("NMSE (log scale)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
        ax.grid(axis="y", which="both", alpha=0.3)
        _despine(ax)

    # Legend from colour patches only
    from matplotlib.patches import Patch  # noqa: PLC0415
    handles = [Patch(facecolor=COLORS[c], alpha=0.85, label=LABELS[c])
               for c in DISPLAY_ORDER]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        fontsize=7,
        framealpha=0.9,
    )
    fig.suptitle(
        "Main ablation: NMSE (bars, log scale); \"0%\" = zero violations",
        fontsize=8, y=1.22,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_ablation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Lambda sweep
# ---------------------------------------------------------------------------

def fig3_lambda_sweep(data: dict[str, Any]) -> None:
    """Two-panel: (a) violation vs log-lambda, (b) MSE vs log-lambda.

    Cleaner than dual y-axis. Hard reference on each panel independently.
    """
    sweep = data.get("lambda_sweep", {})
    if not sweep:
        print("  [skip] No lambda sweep data in results.")
        return

    ablation = data.get("ablation", {})
    hard_runs = ablation.get("exothermic_cstr", {}).get("hard", [])
    hard_mse  = _ms([r["mse_long"] for r in hard_runs])[0] if hard_runs else None
    hard_viol = 0.0

    lambdas = sorted(float(k) for k in sweep)
    viol_means, viol_stds = [], []
    mse_means,  mse_stds  = [], []

    for lam in lambdas:
        runs = sweep[str(lam)]
        vm, vs = _ms([r["physics_viol"] for r in runs])
        mm, ms = _ms([r["mse_long"] for r in runs])
        viol_means.append(vm); viol_stds.append(vs)
        mse_means.append(mm);  mse_stds.append(ms)

    x = np.log10(lambdas)
    xtick_labels = [f"$10^{{{int(v)}}}$" for v in x]

    fig, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(6.75, 2.4))

    # --- panel (a): violations ---
    ax_v.errorbar(
        x, viol_means, yerr=viol_stds,
        color=COLORS["soft"], marker="s", capsize=3, lw=1.5,
        label=r"Soft (mean $\pm$ std)",
    )
    ax_v.axhline(hard_viol, color=COLORS["hard"], ls="--", lw=1.2, alpha=0.9)
    ax_v.text(
        x[-1], hard_viol + max(viol_means) * 0.04,
        "Hard: 0.00%", fontsize=7.5, color=COLORS["hard"], ha="right",
    )
    ax_v.set_xlabel(r"$\lambda$")
    ax_v.set_ylabel("Positivity violation (%)")
    ax_v.set_title("(a) Violations vs. penalty weight")
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(xtick_labels)
    ax_v.set_ylim(bottom=0)
    ax_v.legend(fontsize=7.5, framealpha=0.9)
    ax_v.grid(True, alpha=0.3)
    _despine(ax_v)

    # --- panel (b): MSE ---
    ax_m.errorbar(
        x, mse_means, yerr=mse_stds,
        color=COLORS["none"], marker="o", ls="--", capsize=3, lw=1.5,
        label=r"Soft (mean $\pm$ std)",
    )
    if hard_mse is not None:
        ax_m.axhline(hard_mse, color=COLORS["hard"], ls="--", lw=1.2, alpha=0.9)
        ax_m.text(
            x[0], hard_mse * 1.015,
            f"Hard: {hard_mse:.1f}", fontsize=7.5, color=COLORS["hard"], ha="left",
        )
    ax_m.set_xlabel(r"$\lambda$")
    ax_m.set_ylabel(r"Long-rollout MSE $\downarrow$")
    ax_m.set_title("(b) MSE vs. penalty weight")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(xtick_labels)
    ax_m.set_ylim(bottom=0)
    ax_m.legend(fontsize=7.5, framealpha=0.9)
    ax_m.grid(True, alpha=0.3)
    _despine(ax_m)

    fig.suptitle(r"$\lambda$ sweep — Exothermic CSTR (soft positivity constraint)", y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "fig3_lambda_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Seed variance (boxplot)
# ---------------------------------------------------------------------------

def fig4_seed_variance(data: dict[str, Any]) -> None:
    """Physics violations across seeds, all conditions, for each system.

    Bars = mean; dots = individual seeds.
    Hard variants (hard, hard_inference_only) collapse to zero.
    Architectural baselines (log_param, stoich_param) shown alongside.
    """
    ablation = data.get("ablation", {})
    systems = [k for k in SYSTEM_NAMES if k in ablation]
    if not systems:
        return

    n_sys = len(systems)
    fig, axes = plt.subplots(1, n_sys, figsize=(6.75, 2.6), sharey=False)
    if n_sys == 1:
        axes = [axes]

    for ax, sys_key in zip(axes, systems):
        sys_res = ablation[sys_key]

        # Only show conditions present in this system's results
        present = [c for c in DISPLAY_ORDER if c in sys_res]
        n_cond = len(present)
        bar_width = 0.55

        for ci, cond in enumerate(present):
            runs = sys_res[cond]
            viols = [r["physics_viol"] for r in runs]
            color = COLORS[cond]
            mean_v = float(np.mean(viols)) if viols else 0.0

            ax.bar(
                ci, mean_v, width=bar_width,
                color=color, alpha=0.65, zorder=2,
                edgecolor="none",
            )
            jitter = np.linspace(-0.1, 0.1, len(viols))
            for xj, val in zip(jitter, viols):
                ax.scatter(
                    ci + xj, val,
                    color=color, s=20, zorder=4,
                    edgecolors="white", linewidths=0.4,
                )
            # Label hard-mode conditions that reach exactly zero
            if mean_v < 1e-4:
                ax.text(
                    ci, 0.0, "0",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color, fontweight="bold",
                )

        ax.set_xticks(range(n_cond))
        short_labels = {
            "none": "None",
            "soft": r"Soft$_1$",
            "soft_high": r"Soft$_{10}$",
            "hard": "Hard\nsoftplus",
            "log_param": "Log\nparam",
            "stoich_param": "Stoich\nparam",
            "hard_inference_only": "Hard\ninfer",
            "hard_relu": "Hard\nReLU",
        }
        ax.set_xticklabels(
            [short_labels[c] for c in present],
            fontsize=7, rotation=0, ha="center",
        )
        ax.set_title(SYSTEM_NAMES[sys_key], fontsize=8)
        ax.set_ylabel("Physics violation" if ax is axes[0] else "")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        _despine(ax)

    fig.suptitle(
        "Physics violations per condition (bars = mean; dots = individual seeds)",
        fontsize=8, y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_seed_variance.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure — Combined: convergence + lambda sweep (top) + seed variance (bottom)
# ---------------------------------------------------------------------------

def fig_combined(data: dict[str, Any]) -> None:
    """Two-row combined figure for §4.3.

    Top row  (4 panels): convergence on exothermic CSTR (a-b) + λ sweep (c-d).
    Bottom row (3 panels): per-seed violation bars for all three benchmarks (e-g).
    """
    conv    = data.get("convergence")
    sweep   = data.get("lambda_sweep", {})
    ablation = data.get("ablation", {})
    systems = [k for k in SYSTEM_NAMES if k in ablation]

    fig = plt.figure(figsize=(6.75, 3.3))
    gs_top = fig.add_gridspec(1, 4, left=0.07, right=0.98,
                               top=0.96, bottom=0.52, wspace=0.46)
    gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.98,
                               top=0.44, bottom=0.09, wspace=0.40)

    ax_cv = fig.add_subplot(gs_top[0])   # (a) convergence: violations
    ax_cn = fig.add_subplot(gs_top[1])   # (b) convergence: NMSE
    ax_lv = fig.add_subplot(gs_top[2])   # (c) λ sweep: violations
    ax_lm = fig.add_subplot(gs_top[3])   # (d) λ sweep: MSE
    seed_axes = [fig.add_subplot(gs_bot[i]) for i in range(3)]   # (e-g)

    # ---- (a-b) Convergence -----------------------------------------------
    if conv is not None:
        for cond in BASE_CONDITIONS:
            runs = conv.get(cond, [])
            if not runs:
                continue
            epochs = runs[0]["epochs"]
            viol_matrix = np.array([r["viol_curve"] for r in runs])
            nmse_matrix = np.array([r["nmse_curve"] for r in runs])
            viol_mean = _smooth(viol_matrix.mean(axis=0))
            viol_min  = _smooth(viol_matrix.min(axis=0))
            viol_max  = _smooth(viol_matrix.max(axis=0))
            nmse_mean = _smooth(nmse_matrix.mean(axis=0))
            nmse_min  = _smooth(np.maximum(nmse_matrix.min(axis=0), 1e-2))
            nmse_max  = _smooth(nmse_matrix.max(axis=0))
            color = COLORS[cond]; ls = LINESTYLES[cond]; label = LABELS[cond]
            ax_cv.plot(epochs, viol_mean, color=color, ls=ls, lw=1.4, label=label)
            ax_cv.fill_between(epochs, viol_min, viol_max, alpha=0.12, color=color)
            ax_cn.plot(epochs, nmse_mean, color=color, ls=ls, lw=1.4)
            ax_cn.fill_between(epochs, nmse_min, nmse_max, alpha=0.12, color=color)

        ax_cv.axhline(0, color=COLORS["hard"], lw=0.8, ls="-", alpha=0.5)
        ax_cv.set_xlabel("Epoch", fontsize=7.5)
        ax_cv.set_ylabel("Viol. (%)", fontsize=7.5)
        ax_cv.set_title("(a) Violations vs. epoch", fontsize=7.5)
        ax_cv.legend(fontsize=5.5, framealpha=0.9, loc="upper right")
        ax_cv.set_ylim(bottom=0); ax_cv.grid(True, alpha=0.3); _despine(ax_cv)

        ax_cn.set_xlabel("Epoch", fontsize=7.5)
        ax_cn.set_ylabel("NMSE (log)", fontsize=7.5)
        ax_cn.set_title("(b) Accuracy vs. epoch", fontsize=7.5)
        ax_cn.set_yscale("log"); ax_cn.set_ylim(bottom=1e-1)
        ax_cn.grid(True, which="both", alpha=0.3); _despine(ax_cn)
    else:
        for ax in (ax_cv, ax_cn):
            ax.text(0.5, 0.5, "no convergence data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

    # ---- (c-d) Lambda sweep ----------------------------------------------
    if sweep:
        hard_runs = ablation.get("exothermic_cstr", {}).get("hard", [])
        hard_mse  = _ms([r["mse_long"] for r in hard_runs])[0] if hard_runs else None

        lambdas = sorted(float(k) for k in sweep)
        viol_means, viol_stds, mse_means, mse_stds = [], [], [], []
        for lam in lambdas:
            runs = sweep[str(lam)]
            vm, vs = _ms([r["physics_viol"] for r in runs])
            mm, ms = _ms([r["mse_long"] for r in runs])
            viol_means.append(vm); viol_stds.append(vs)
            mse_means.append(mm);  mse_stds.append(ms)

        x = np.log10(lambdas)
        xtick_labels = [f"$10^{{{int(v)}}}$" for v in x]

        ax_lv.errorbar(x, viol_means, yerr=viol_stds,
                       color=COLORS["soft"], marker="s", capsize=3, lw=1.4,
                       label=r"Soft (mean$\pm$std)")
        ax_lv.axhline(0.0, color=COLORS["hard"], ls="--", lw=1.1, alpha=0.9)
        ax_lv.text(x[-1], max(viol_means) * 0.06, "Hard: 0%",
                   fontsize=6, color=COLORS["hard"], ha="right")
        ax_lv.set_xlabel(r"$\lambda$", fontsize=7.5)
        ax_lv.set_ylabel("Viol. (%)", fontsize=7.5)
        ax_lv.set_title(r"(c) Violations vs. $\lambda$", fontsize=7.5)
        ax_lv.set_xticks(x); ax_lv.set_xticklabels(xtick_labels, fontsize=6.5)
        ax_lv.set_ylim(bottom=0); ax_lv.grid(True, alpha=0.3); _despine(ax_lv)

        ax_lm.errorbar(x, mse_means, yerr=mse_stds,
                       color=COLORS["none"], marker="o", ls="--", capsize=3, lw=1.4,
                       label=r"Soft (mean$\pm$std)")
        if hard_mse is not None:
            ax_lm.axhline(hard_mse, color=COLORS["hard"], ls="--", lw=1.1, alpha=0.9)
            ax_lm.text(x[0], hard_mse * 1.02, f"Hard: {hard_mse:.1f}",
                       fontsize=6, color=COLORS["hard"], ha="left")
        ax_lm.set_xlabel(r"$\lambda$", fontsize=7.5)
        ax_lm.set_ylabel(r"Long-rollout MSE", fontsize=7.5)
        ax_lm.set_title(r"(d) MSE vs. $\lambda$", fontsize=7.5)
        ax_lm.set_xticks(x); ax_lm.set_xticklabels(xtick_labels, fontsize=6.5)
        ax_lm.set_ylim(bottom=0); ax_lm.grid(True, alpha=0.3); _despine(ax_lm)
    else:
        for ax in (ax_lv, ax_lm):
            ax.text(0.5, 0.5, "no λ-sweep data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

    # ---- (e-g) Seed variance ---------------------------------------------
    SHORT = {
        "none": "None", "soft": r"S$_1$", "soft_high": r"S$_{10}$",
        "hard": "Hard", "hard_relu": "H-ReLU", "log_param": "Log",
        "stoich_param": "Stoich", "hard_inference_only": "HIO",
    }
    panel_labels = ["(e)", "(f)", "(g)"]
    for idx, (ax, sys_key) in enumerate(zip(seed_axes, systems)):
        sys_res = ablation[sys_key]
        present = [c for c in DISPLAY_ORDER if c in sys_res]
        bar_width = 0.58
        for ci, cond in enumerate(present):
            runs = sys_res[cond]
            viols = [r["physics_viol"] for r in runs]
            color = COLORS[cond]
            mean_v = float(np.mean(viols))
            ax.bar(ci, mean_v, width=bar_width, color=color, alpha=0.65, zorder=2)
            jitter = np.linspace(-0.12, 0.12, len(viols))
            for xj, val in zip(jitter, viols):
                ax.scatter(ci + xj, val, color=color, s=14, zorder=4,
                           edgecolors="white", linewidths=0.3)
            if mean_v < 1e-4:
                ax.text(ci, 0, "0", ha="center", va="bottom",
                        fontsize=6, color=color, fontweight="bold")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([SHORT[c] for c in present], fontsize=6, rotation=0)
        ax.set_title(f"{panel_labels[idx]} {SYSTEM_NAMES[sys_key]}", fontsize=7)
        if idx == 0:
            ax.set_ylabel("Physics violation", fontsize=7.5)
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=0.3); _despine(ax)

    out = FIGURES_DIR / "fig_combined.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run: python3 scripts/experiments_paper.py")
        return

    print(f"Loading results from {RESULTS_PATH}")
    data = _load()
    print(f"  ablation systems: {list(data.get('ablation', {}).keys())}")
    print(f"  lambda sweep entries: {len(data.get('lambda_sweep', {}))}")
    print(f"  convergence data: {'yes' if data.get('convergence') else 'no'}")
    print()

    print("Figure 1: convergence curve")
    fig1_convergence(data)

    print("Figure 2 (paper): combined lambda sweep + seed variance")
    fig_combined(data)

    print("Figure 2 (supplementary): main ablation bar chart")
    fig2_ablation(data)

    print("Figure 3 (supplementary): lambda sweep standalone")
    fig3_lambda_sweep(data)

    print("Figure 4 (supplementary): seed variance standalone")
    fig4_seed_variance(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Paper uses: fig1_convergence.pdf, fig_combined.pdf")
    print("Supplementary: fig2_ablation.pdf, fig3_lambda_sweep.pdf, fig4_seed_variance.pdf")


if __name__ == "__main__":
    main()
