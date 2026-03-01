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

# Wong (2011) colour-blind-safe palette
COLORS = {
    "none":      "#E69F00",   # orange
    "soft":      "#56B4E9",   # sky blue
    "soft_high": "#009E73",   # teal
    "hard":      "#D55E00",   # vermillion
}
MARKERS = {
    "none": "o",
    "soft": "s",
    "soft_high": "^",
    "hard": "D",
}
LABELS = {
    "none":      "No constraint",
    "soft":      r"Soft $(\lambda=1)$",
    "soft_high": r"Soft $(\lambda=10)$",
    "hard":      "Hard (projection)",
}
LINESTYLES = {
    "none":      "--",
    "soft":      ":",
    "soft_high": "-.",
    "hard":      "-",
}

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

CONDITIONS = ["none", "soft", "soft_high", "hard"]


def _ms(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


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
        1, 2, figsize=(6.75, 2.4), sharey=False
    )

    for cond in CONDITIONS:
        runs = conv.get(cond, [])
        if not runs:
            continue
        epochs = runs[0]["epochs"]
        viol_matrix = np.array([r["viol_curve"] for r in runs])   # (seeds, T)
        nmse_matrix = np.array([r["nmse_curve"] for r in runs])

        viol_mean = viol_matrix.mean(axis=0)
        viol_std = viol_matrix.std(axis=0)
        nmse_mean = nmse_matrix.mean(axis=0)
        nmse_std = nmse_matrix.std(axis=0)

        color = COLORS[cond]
        ls = LINESTYLES[cond]
        label = LABELS[cond]

        ax_viol.plot(epochs, viol_mean, color=color, ls=ls, label=label)
        ax_viol.fill_between(
            epochs,
            np.maximum(viol_mean - viol_std, 0),
            viol_mean + viol_std,
            alpha=0.15, color=color,
        )

        ax_nmse.plot(epochs, nmse_mean, color=color, ls=ls, label=label)
        ax_nmse.fill_between(
            epochs,
            np.maximum(nmse_mean - nmse_std, 0),
            nmse_mean + nmse_std,
            alpha=0.15, color=color,
        )

    ax_viol.set_xlabel("Training epoch")
    ax_viol.set_ylabel("Positivity violation (%)")
    ax_viol.set_title("(a) Constraint violation vs. epoch")
    ax_viol.legend(loc="upper right", framealpha=0.9)
    ax_viol.set_ylim(bottom=0)
    ax_viol.grid(True)

    # Annotate the hard=0 line
    ax_viol.axhline(0, color=COLORS["hard"], lw=0.8, ls="-", alpha=0.5)
    ax_viol.text(
        0.98, 0.04, "Hard: 0.00% (all epochs)",
        transform=ax_viol.transAxes,
        ha="right", va="bottom", fontsize=7,
        color=COLORS["hard"],
    )

    ax_nmse.set_xlabel("Training epoch")
    ax_nmse.set_ylabel(r"Normalized MSE (long horizon)")
    ax_nmse.set_title("(b) Accuracy vs. epoch")
    ax_nmse.set_ylim(bottom=0)
    ax_nmse.grid(True)

    fig.suptitle("Exothermic CSTR: hard projection vs. soft penalty over training", y=1.01)
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
    """Grouped bar chart: NMSE_long (left axis) + physics violation (right axis).

    Three systems side by side. Shows hard = zero violation, lowest NMSE.
    """
    ablation = data.get("ablation", {})
    systems = [k for k in SYSTEM_NAMES if k in ablation]
    if not systems:
        print("  [skip] No ablation data in results.")
        return

    n_sys = len(systems)
    n_cond = len(CONDITIONS)
    bar_width = 0.18
    group_gap = 0.1
    x_centers = np.arange(n_sys) * (n_cond * bar_width + group_gap)

    fig, ax1 = plt.subplots(figsize=(6.75, 2.8))
    ax2 = ax1.twinx()

    for ci, cond in enumerate(CONDITIONS):
        nmse_means, nmse_stds = [], []
        viol_means, viol_stds = [], []

        for sys_key in systems:
            runs = ablation[sys_key].get(cond, [])
            if not runs:
                nmse_means.append(0.0); nmse_stds.append(0.0)
                viol_means.append(0.0); viol_stds.append(0.0)
                continue
            nm_m, nm_s = _ms([r["nmse_long"] for r in runs])
            pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
            nmse_means.append(nm_m); nmse_stds.append(nm_s)
            viol_means.append(pv_m); viol_stds.append(pv_s)

        x_pos = x_centers + ci * bar_width - (n_cond - 1) * bar_width / 2
        color = COLORS[cond]

        # NMSE bars (solid)
        ax1.bar(
            x_pos, nmse_means,
            width=bar_width * 0.85,
            color=color, alpha=0.85,
            yerr=nmse_stds, capsize=2, error_kw={"elinewidth": 0.8},
            label=LABELS[cond],
            zorder=3,
        )
        # Physics violation dots on ax2
        for xi, (vm, vs) in zip(x_pos, zip(viol_means, viol_stds)):
            if vm < 1e-6:
                ax2.annotate(
                    "✓ 0",
                    xy=(xi, 0.0),
                    fontsize=6, ha="center", va="bottom",
                    color=color, fontweight="bold",
                )
            else:
                ax2.errorbar(
                    xi, vm, yerr=vs,
                    fmt=MARKERS[cond], color=color,
                    markersize=4, capsize=2, elinewidth=0.8,
                    zorder=5,
                )

    ax1.set_xticks(x_centers)
    ax1.set_xticklabels([SYSTEM_NAMES[k] for k in systems], fontsize=8)
    ax1.set_ylabel("Normalized MSE (long horizon)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax1.grid(axis="y", zorder=0)
    ax1.set_title("Main ablation: accuracy (bars) and physics violations (markers)")

    ax2.set_ylabel("Physics violation")
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", labelcolor="gray")

    # Note on right axis
    ax2.yaxis.label.set_color("gray")

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
    """Dual-axis: log λ vs positivity violation (left) and MSE_long (right).

    Shows no λ value eliminates violations without MSE collapse.
    Hard reference line shown on both axes.
    """
    sweep = data.get("lambda_sweep", {})
    if not sweep:
        print("  [skip] No lambda sweep data in results.")
        return

    ablation = data.get("ablation", {})
    hard_runs = ablation.get("exothermic_cstr", {}).get("hard", [])
    hard_mse = _ms([r["mse_long"] for r in hard_runs])[0] if hard_runs else None
    hard_viol = _ms([r["physics_viol"] for r in hard_runs])[0] if hard_runs else 0.0

    lambdas = sorted(float(k) for k in sweep)
    viol_means, viol_stds = [], []
    mse_means, mse_stds = [], []

    for lam in lambdas:
        runs = sweep[str(lam)]
        viol_m, viol_s = _ms([r["physics_viol"] for r in runs])
        mse_m, mse_s = _ms([r["mse_long"] for r in runs])
        viol_means.append(viol_m); viol_stds.append(viol_s)
        mse_means.append(mse_m); mse_stds.append(mse_s)

    fig, ax1 = plt.subplots(figsize=(3.25, 2.5))
    ax2 = ax1.twinx()

    x = np.log10(lambdas)

    # Violation line
    ax1.errorbar(
        x, viol_means, yerr=viol_stds,
        color=COLORS["soft"], marker="s", capsize=3, lw=1.4,
        label=r"Pos. violation % (soft)",
    )
    ax1.set_xlabel(r"$\log_{10}(\lambda)$")
    ax1.set_ylabel("Positivity violation (%)", color=COLORS["soft"])
    ax1.tick_params(axis="y", labelcolor=COLORS["soft"])
    ax1.set_ylim(bottom=0)

    # MSE line
    ax2.errorbar(
        x, mse_means, yerr=mse_stds,
        color=COLORS["none"], marker="o", ls="--", capsize=3, lw=1.4,
        label=r"MSE$_\mathrm{long}$ (soft)",
    )
    ax2.set_ylabel(r"MSE$_\mathrm{long}$", color=COLORS["none"])
    ax2.tick_params(axis="y", labelcolor=COLORS["none"])
    ax2.set_ylim(bottom=0)

    # Hard reference lines
    if hard_mse is not None:
        ax2.axhline(
            hard_mse, color=COLORS["hard"], ls="-", lw=1.2, alpha=0.8,
            label=f"Hard MSE = {hard_mse:.1f}",
        )
    ax1.axhline(
        hard_viol, color=COLORS["hard"], ls="-", lw=1.2, alpha=0.8,
        label=f"Hard viol. = {hard_viol:.2f}%",
    )
    ax1.text(
        x[-1], hard_viol + 0.5, "Hard: 0.00%",
        fontsize=7, color=COLORS["hard"], ha="right",
    )

    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"$10^{{{int(v)}}}$")
    )
    ax1.set_title(r"$\lambda$ sweep — Exothermic CSTR (soft positivity)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

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
    """Boxplot of physics violations across 3 seeds for each condition.

    Makes high-variance of soft constraints visually obvious.
    Hard has zero variance by construction.
    """
    ablation = data.get("ablation", {})
    systems = [k for k in SYSTEM_NAMES if k in ablation]
    if not systems:
        return

    n_sys = len(systems)
    fig, axes = plt.subplots(1, n_sys, figsize=(6.75, 2.4), sharey=False)
    if n_sys == 1:
        axes = [axes]

    for ax, sys_key in zip(axes, systems):
        sys_res = ablation[sys_key]
        box_data = []
        tick_labels = []
        box_colors = []

        for cond in CONDITIONS:
            runs = sys_res.get(cond, [])
            viols = [r["physics_viol"] for r in runs]
            box_data.append(viols)
            tick_labels.append(LABELS[cond])
            box_colors.append(COLORS[cond])

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"markersize": 4},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(CONDITIONS) + 1))
        ax.set_xticklabels(
            [LABELS[c].replace(r"$", "").replace("\\", "").replace("lambda", "λ")
             for c in CONDITIONS],
            fontsize=7, rotation=15, ha="right",
        )
        ax.set_title(SYSTEM_NAMES[sys_key], fontsize=8)
        ax.set_ylabel("Physics violation" if ax == axes[0] else "")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.4)

        # Annotate hard with "0 ± 0"
        hard_idx = CONDITIONS.index("hard") + 1
        ax.text(
            hard_idx, ax.get_ylim()[1] * 0.05,
            "0.00\n±0",
            ha="center", va="bottom", fontsize=7,
            color=COLORS["hard"], fontweight="bold",
        )

    fig.suptitle(
        "Seed variance of physics violations (3 seeds each).\n"
        "Hard: zero by construction; soft: high variance across seeds.",
        fontsize=8, y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_seed_variance.pdf"
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

    print("Figure 2: main ablation bar chart")
    fig2_ablation(data)

    print("Figure 3: lambda sweep")
    fig3_lambda_sweep(data)

    print("Figure 4: seed variance boxplot")
    fig4_seed_variance(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("PDF + PNG versions produced for each figure.")


if __name__ == "__main__":
    main()
