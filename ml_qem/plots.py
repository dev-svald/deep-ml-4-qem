"""
All matplotlib visualisations for the ML-QEM Classical Shadows experiment.

Functions
---------
plot_dataset_stats   — bias, error distribution, noisy vs ideal scatter
plot_benchmark       — saves 5 individual panels + one combined benchmark figure:
                         MAE bar, % improved, L1RC box, per-observable MAE,
                         best-model Pred vs Ideal
plot_final_summary   — condensed two-panel summary figure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from .benchmark import MethodResult

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4", "#17becf"]

# Consistent colours used across ALL scatter plots for noisy vs model data
NOISY_COLOR = "#e8372b"   # bright tomato-red  — noisy raw estimates
IDEAL_COLOR = "#1a3a8c"   # deep navy          — best-model predictions

# Axis tick format: at most 2 decimal places
FMT2 = FormatStrFormatter("%.2f")

# ── Short display names ───────────────────────────────────────────────────────
DISPLAY_NAMES: dict[str, str] = {
    "MLP-Correction": "MLP-C",
    "MLP-Prediction": "MLP-P",
    "Attention-MLP":  "A-MLP",
}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _circuit_label(circuit_type: str) -> str:
    labels = {
        "pauli":     "Pauli Random Circuits",
        "brickwall": "Brick-wall Circuits",
        "combined":  "Combined Circuits (Pauli + Brick-wall)",
    }
    return labels.get(circuit_type, circuit_type.title())


def _brighten_legend(ax: plt.Axes) -> None:
    """Make every legend handle fully opaque so colours read clearly."""
    leg = ax.get_legend()
    if leg is None:
        return
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)


def _subpath(save_path: str | Path | None, fname: str) -> Path | None:
    """Resolve an individual-file save path inside a directory."""
    if save_path is None:
        return None
    p = Path(save_path)
    return (p / fname) if p.is_dir() else p.parent / fname


# ---------------------------------------------------------------------------
# 1. Dataset statistics
# ---------------------------------------------------------------------------

def plot_dataset_stats(
    y_ideal:      np.ndarray,
    y_noisy:      np.ndarray,
    save_path:    str | Path | None = None,
    circuit_type: str = "combined",
) -> None:
    """
    Three-panel figure showing the noise bias before any mitigation.

    Left   : per-observable mean bias
    Centre : distribution of estimation errors
    Right  : noisy vs ideal scatter  (uses NOISY_COLOR for consistency)
    """
    baseline_mae = float(np.mean(np.abs(y_noisy - y_ideal)))
    per_obs_bias = np.mean(y_noisy - y_ideal, axis=0)
    all_errors   = (y_noisy - y_ideal).flatten()
    M_obs        = y_ideal.shape[1]
    circ_label   = _circuit_label(circuit_type)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ── Per-observable bias ────────────────────────────────────────────────
    axes[0].bar(range(M_obs), per_obs_bias, alpha=0.75, color="steelblue")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_xlabel("Observable index")
    axes[0].set_ylabel("Mean bias")
    axes[0].set_title("Per-observable noise bias")
    axes[0].yaxis.set_major_formatter(FMT2)

    # ── Error distribution ─────────────────────────────────────────────────
    axes[1].hist(all_errors, bins=50, color="salmon", alpha=0.85,
                 edgecolor="k", lw=0.4)
    axes[1].axvline(0, color="k", ls="--")
    axes[1].set_xlabel("Estimation error")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error distribution (raw shadow)\nMAE={baseline_mae:.2f}")
    axes[1].xaxis.set_major_formatter(FMT2)

    # ── Noisy vs ideal scatter (NOISY_COLOR for consistency) ───────────────
    axes[2].scatter(y_ideal.flatten(), y_noisy.flatten(),
                    alpha=0.12, s=5, color=NOISY_COLOR, label="Noisy")
    lim = [-1.1, 1.1]
    axes[2].plot(lim, lim, "k--", lw=1.5, label="y = x (perfect)")
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel(r"Ideal $\langle O\rangle$")
    axes[2].set_ylabel("Noisy estimate")
    axes[2].set_title("Noisy vs ideal observables")
    axes[2].xaxis.set_major_formatter(FMT2)
    axes[2].yaxis.set_major_formatter(FMT2)
    axes[2].legend(fontsize=9)
    _brighten_legend(axes[2])

    plt.suptitle(
        f"Dataset Statistics — Before Mitigation\n{circ_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_path, "dataset_stats.png")


# ---------------------------------------------------------------------------
# 2. Full benchmark: 5 individual plots + 1 combined figure
# ---------------------------------------------------------------------------

def plot_benchmark(
    results:      dict[str, MethodResult],
    y_ideal:      np.ndarray,
    y_noisy:      np.ndarray,
    observables:  list[dict],
    save_path:    str | Path | None = None,
    circuit_type: str = "combined",
) -> None:
    """
    Save five individual benchmark plots, then one combined 5-panel figure.

    Individual files saved (inside save_path directory):
        benchmark_mae.png
        benchmark_pct_improved.png
        benchmark_l1rc.png
        benchmark_per_obs_mae.png
        benchmark_pred_vs_ideal.png

    Combined file:
        benchmark_results.png
    """
    method_names = list(results.keys())
    disp_names   = [_display(m) for m in method_names]
    M_obs        = y_ideal.shape[1]
    colors       = PALETTE[: len(method_names)]
    circ_label   = _circuit_label(circuit_type)

    maes     = [results[m].mae_val     for m in method_names]
    pct_imp  = [results[m].pct_improved for m in method_names]
    l1rc_data = [results[m].l1rc       for m in method_names]

    # Identify the best ML model for the Pred vs Ideal panel
    ml_names  = [n for n in method_names
                 if n not in ("No Mitigation", "PEC Shadows")]
    best_name = min(ml_names, key=lambda m: results[m].mae_val) if ml_names else ""
    y_best    = results[best_name].y_pred if best_name else np.zeros_like(y_noisy)
    lim       = [-1.1, 1.1]
    x_obs     = np.arange(M_obs)
    bar_width  = 0.13

    # ── (a) MAE bar chart ──────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    bars = ax1.barh(disp_names, maes, color=colors, alpha=0.85)
    ax1.set_xlabel("Mean Absolute Error")
    ax1.set_title(f"MAE — lower is better\n{circ_label}", fontweight="bold")
    ax1.xaxis.set_major_formatter(FMT2)
    for bar, val in zip(bars, maes):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    _save_or_show(fig1, _subpath(save_path, "benchmark_mae.png"), "benchmark_mae.png")

    # ── (b) % improved bar chart ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.barh(disp_names, pct_imp, color=colors, alpha=0.85)
    ax2.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("% Circuits Improved")
    ax2.set_title(f"% Improved (L1RC < 0)\n{circ_label}", fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.xaxis.set_major_formatter(FMT2)
    plt.tight_layout()
    _save_or_show(fig2, _subpath(save_path, "benchmark_pct_improved.png"),
                  "benchmark_pct_improved.png")

    # ── (c) L1RC box plot ──────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    bp = ax3.boxplot(l1rc_data, vert=True, patch_artist=True,
                     medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(0, color="red", ls="--", lw=1.5, label="No improvement")
    ax3.set_xticks(range(1, len(method_names) + 1))
    ax3.set_xticklabels(disp_names, fontsize=9, rotation=15, ha="right")
    ax3.set_ylabel("L1 Relative Change")
    ax3.set_title(f"L1RC Distribution\n{circ_label}", fontweight="bold")
    ax3.yaxis.set_major_formatter(FMT2)
    ax3.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig3, _subpath(save_path, "benchmark_l1rc.png"), "benchmark_l1rc.png")

    # ── (d) Per-observable MAE ─────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    for idx, (name, dname) in enumerate(zip(method_names, disp_names)):
        err    = np.mean(np.abs(results[name].y_pred - y_ideal), axis=0)
        offset = (idx - len(method_names) / 2) * bar_width
        ax4.bar(x_obs + offset, err, width=bar_width, label=dname,
                color=colors[idx], alpha=0.75)
    ax4.set_xlabel("Observable index")
    ax4.set_ylabel("MAE")
    ax4.set_title(f"Per-Observable MAE by Method\n{circ_label}", fontweight="bold")
    ax4.yaxis.set_major_formatter(FMT2)
    ax4.legend(fontsize=8, ncol=3)
    ax4.set_xticks(range(0, M_obs, max(1, M_obs // 10)))
    plt.tight_layout()
    _save_or_show(fig4, _subpath(save_path, "benchmark_per_obs_mae.png"),
                  "benchmark_per_obs_mae.png")

    # ── (e) Pred vs Ideal scatter ──────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(5, 5))
    if best_name:
        ax5.scatter(y_ideal.flatten(), y_best.flatten(),
                    alpha=0.09, s=4, color=IDEAL_COLOR,
                    label=_display(best_name))
        ax5.scatter(y_ideal.flatten(), y_noisy.flatten(),
                    alpha=0.06, s=4, color=NOISY_COLOR, label="Noisy (raw)")
    ax5.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    ax5.set_xlim(lim)
    ax5.set_ylim(lim)
    ax5.set_xlabel(r"Ideal $\langle O\rangle$")
    ax5.set_ylabel("Predicted")
    ax5.set_title(
        f"Pred. vs Ideal ({_display(best_name) if best_name else ''})\n{circ_label}",
        fontweight="bold",
    )
    ax5.xaxis.set_major_formatter(FMT2)
    ax5.yaxis.set_major_formatter(FMT2)
    ax5.legend(fontsize=8)
    _brighten_legend(ax5)
    plt.tight_layout()
    _save_or_show(fig5, _subpath(save_path, "benchmark_pred_vs_ideal.png"),
                  "benchmark_pred_vs_ideal.png")

    # ── Combined 5-panel figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # MAE bars
    axA = fig.add_subplot(gs[0, 0])
    bars = axA.barh(disp_names, maes, color=colors, alpha=0.85)
    axA.set_xlabel("Mean Absolute Error")
    axA.set_title("MAE (lower is better)")
    axA.xaxis.set_major_formatter(FMT2)
    for bar, val in zip(bars, maes):
        axA.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

    # % improved
    axB = fig.add_subplot(gs[0, 1])
    axB.barh(disp_names, pct_imp, color=colors, alpha=0.85)
    axB.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
    axB.set_xlabel("% Circuits Improved")
    axB.set_title("% Improved (L1RC < 0)")
    axB.set_xlim(0, 105)
    axB.xaxis.set_major_formatter(FMT2)

    # L1RC box plot
    axC = fig.add_subplot(gs[0, 2])
    bp = axC.boxplot(l1rc_data, vert=True, patch_artist=True,
                     medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axC.axhline(0, color="red", ls="--", lw=1.5, label="No improvement")
    axC.set_xticks(range(1, len(method_names) + 1))
    axC.set_xticklabels(disp_names, fontsize=9, rotation=15, ha="right")
    axC.set_ylabel("L1 Relative Change")
    axC.set_title("L1RC Distribution")
    axC.yaxis.set_major_formatter(FMT2)
    axC.legend(fontsize=8)

    # Per-observable MAE
    axD = fig.add_subplot(gs[1, 0:2])
    for idx, (name, dname) in enumerate(zip(method_names, disp_names)):
        err    = np.mean(np.abs(results[name].y_pred - y_ideal), axis=0)
        offset = (idx - len(method_names) / 2) * bar_width
        axD.bar(x_obs + offset, err, width=bar_width, label=dname,
                color=colors[idx], alpha=0.75)
    axD.set_xlabel("Observable index")
    axD.set_ylabel("MAE")
    axD.set_title("Per-Observable MAE by Method")
    axD.yaxis.set_major_formatter(FMT2)
    axD.legend(fontsize=7, ncol=3)
    axD.set_xticks(range(0, M_obs, max(1, M_obs // 10)))

    # Pred vs Ideal
    axE = fig.add_subplot(gs[1, 2])
    if best_name:
        axE.scatter(y_ideal.flatten(), y_best.flatten(),
                    alpha=0.09, s=4, color=IDEAL_COLOR,
                    label=_display(best_name))
        axE.scatter(y_ideal.flatten(), y_noisy.flatten(),
                    alpha=0.06, s=4, color=NOISY_COLOR, label="Noisy (raw)")
    axE.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    axE.set_xlim(lim)
    axE.set_ylim(lim)
    axE.set_xlabel(r"Ideal $\langle O\rangle$")
    axE.set_ylabel("Predicted")
    axE.set_title(f"Pred. vs Ideal ({_display(best_name) if best_name else ''})")
    axE.xaxis.set_major_formatter(FMT2)
    axE.yaxis.set_major_formatter(FMT2)
    axE.legend(fontsize=8)
    _brighten_legend(axE)

    plt.suptitle(
        f"ML-QEM Classical Shadows — Benchmark Results\n{circ_label}",
        fontsize=14, fontweight="bold",
    )
    _save_or_show(fig, save_path, "benchmark_results.png")


# ---------------------------------------------------------------------------
# 3. Final summary (2-panel)
# ---------------------------------------------------------------------------

def plot_final_summary(
    results:      dict[str, MethodResult],
    y_ideal:      np.ndarray,
    y_noisy:      np.ndarray,
    observables:  list[dict],
    save_path:    str | Path | None = None,
    circuit_type: str = "combined",
) -> None:
    """
    Condensed two-panel summary: MAE bar + per-observable error profile for
    selected methods (No Mitigation, PEC, MLP-P, A-MLP).
    """
    method_names = list(results.keys())
    baseline_mae = results.get(
        "No Mitigation", next(iter(results.values()))
    ).mae_val
    M_obs      = y_ideal.shape[1]
    colors     = PALETTE[: len(method_names)]
    circ_label = _circuit_label(circuit_type)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: MAE bar chart with improvement percentages ───────────────────
    maes_v = [results[m].mae_val for m in method_names]
    imprs  = [100.0 * (baseline_mae - results[m].mae_val) / baseline_mae
              for m in method_names]
    disp   = [_display(m) for m in method_names]
    bars   = axes[0].bar(range(len(method_names)), maes_v,
                          color=colors, alpha=0.85, edgecolor="k", lw=0.5)
    axes[0].set_xticks(range(len(method_names)))
    axes[0].set_xticklabels(disp, fontsize=9)
    axes[0].set_ylabel("MAE on Pauli observables")
    axes[0].set_title(f"Observable Estimation Error by Method\n{circ_label}")
    axes[0].yaxis.set_major_formatter(FMT2)
    for bar, imp in zip(bars, imprs):
        label = f"+{imp:.0f}%" if imp > 0 else f"{imp:.0f}%"
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Right: per-observable error profile for key methods ────────────────
    focus_keys   = [n for n in ["No Mitigation", "PEC Shadows",
                                 "MLP-Prediction", "Attention-MLP"]
                    if n in results]
    focus_colors = [PALETTE[method_names.index(n)] for n in focus_keys]
    x_obs = np.arange(M_obs)
    n_f   = len(focus_keys)
    w     = 0.8 / max(n_f, 1)
    for k, (name, col) in enumerate(zip(focus_keys, focus_colors)):
        err = np.mean(np.abs(results[name].y_pred - y_ideal), axis=0)
        axes[1].bar(x_obs + (k - n_f / 2 + 0.5) * w, err,
                    w, label=_display(name), color=col, alpha=0.75)
    axes[1].set_xlabel("Observable")
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"Per-Observable Error Profile\n{circ_label}")
    axes[1].yaxis.set_major_formatter(FMT2)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(range(0, M_obs, max(1, M_obs // 6)))

    plt.suptitle(
        f"ML-QEM Classical Shadows — Final Summary\n{circ_label}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_path, "final_summary.png")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure,
                   save_path: str | Path | None,
                   default_name: str) -> None:
    if save_path is not None:
        path = Path(save_path)
        if path.is_dir():
            path = path / default_name
        fig.savefig(path, bbox_inches="tight", dpi=120)
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
