"""
Benchmark metrics, comparative evaluation table, and ablation study.

Metrics
-------
    MAE        mean absolute error on observable expectations
    RMSE       root mean squared error
    L1RC       L1 Relative Change (Placidi et al. 2026, Eq. 21)
               < 0 means the method successfully mitigated the noise
    % improved fraction of circuits where L1RC < 0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def l1_relative_change(y_mit:   np.ndarray,
                        y_ideal: np.ndarray,
                        y_noisy: np.ndarray) -> np.ndarray:
    """
    Per-circuit L1 Relative Change from Placidi et al. (2026) Eq. (21).

        ℛ(circuit) = (‖y_ideal − y_mit‖₁ − ‖y_ideal − y_noisy‖₁)
                     / ‖y_ideal − y_noisy‖₁

    Values < 0 indicate successful mitigation.

    Parameters
    ----------
    y_mit   : (N, M) mitigated predictions
    y_ideal : (N, M) ground-truth ideal expectations
    y_noisy : (N, M) raw noisy shadow estimates (baseline)

    Returns
    -------
    (N,) array of L1RC values
    """
    numer = np.sum(np.abs(y_ideal - y_mit),   axis=1)
    denom = np.sum(np.abs(y_ideal - y_noisy), axis=1)
    denom = np.maximum(denom, 1e-10)
    return (numer - denom) / denom


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name:        str
    y_pred:      np.ndarray
    mae_val:     float
    rmse_val:    float
    l1rc:        np.ndarray
    l1rc_median: float
    pct_improved: float

    def __str__(self) -> str:
        return (f"{self.name:<22} "
                f"MAE={self.mae_val:.4f}  "
                f"RMSE={self.rmse_val:.4f}  "
                f"L1RC_med={self.l1rc_median:+.4f}  "
                f"%Improved={self.pct_improved:.1f}%")


# ---------------------------------------------------------------------------
# Run full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    predictions:  dict[str, np.ndarray],
    y_ideal:      np.ndarray,
    y_noisy:      np.ndarray,
    print_table:  bool = True,
) -> dict[str, MethodResult]:
    """
    Evaluate all mitigation methods and return structured results.

    Parameters
    ----------
    predictions : {method_name: (N, M_obs) predictions}
                  The 'No Mitigation' key should map to `y_noisy`.
    y_ideal     : (N, M_obs) ground-truth ideal expectations
    y_noisy     : (N, M_obs) raw noisy shadow estimates
    print_table : whether to print a formatted benchmark table

    Returns
    -------
    dict of method_name → MethodResult
    """
    results: dict[str, MethodResult] = {}

    if print_table:
        sep = "=" * 75
        print(sep)
        print(f"{'Method':<22} {'MAE':>8} {'RMSE':>8} {'L1RC med':>10} {'%Improved':>10}")
        print("-" * 75)

    for name, y_pred in predictions.items():
        l1rc   = l1_relative_change(y_pred, y_ideal, y_noisy)
        result = MethodResult(
            name         = name,
            y_pred       = y_pred,
            mae_val      = mae(y_pred, y_ideal),
            rmse_val     = rmse(y_pred, y_ideal),
            l1rc         = l1rc,
            l1rc_median  = float(np.median(l1rc)),
            pct_improved = float(100.0 * np.mean(l1rc < 0)),
        )
        results[name] = result
        if print_table:
            print(f"{name:<22} {result.mae_val:>8.4f} {result.rmse_val:>8.4f} "
                  f"{result.l1rc_median:>10.4f} {result.pct_improved:>9.1f}%")

    if print_table:
        print(sep)
        print("Note: L1RC < 0 → improvement over raw noisy shadow.")

    return results


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def _ablate(model,
            X_test:    np.ndarray,
            y_ideal:   np.ndarray,
            m_obs:     int,
            ablation:  str) -> float:
    """
    Randomise or zero-out part of the input and measure the MAE degradation.

    ablation options
    ----------------
    'random_noisy'  — shuffle the noisy shadow estimates across samples
    'random_meta'   — replace noise metadata with standard-normal noise
    'zero_noisy'    — set all noisy estimates to 0 (only metadata remains)
    """
    X_abl = X_test.copy()
    if ablation == "random_noisy":
        idx = np.random.permutation(len(X_abl))
        X_abl[:, :m_obs] = X_abl[idx, :m_obs]
    elif ablation == "random_meta":
        X_abl[:, m_obs:] = np.random.randn(*X_abl[:, m_obs:].shape)
    elif ablation == "zero_noisy":
        X_abl[:, :m_obs] = 0.0

    y_pred = model.predict(X_abl)
    return float(np.mean(np.abs(y_pred - y_ideal)))


def run_ablation(
    models:   dict[str, object],
    X_test:   np.ndarray,
    y_ideal:  np.ndarray,
    m_obs:    int,
    print_table: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Ablation study: measure MAE under different input perturbations for each model.

    Parameters
    ----------
    models     : {name: fitted model with .predict(X) method}
    X_test     : (N, M_obs + 3) test features
    y_ideal    : (N, M_obs) ideal labels
    m_obs      : number of observables
    print_table: whether to print results

    Returns
    -------
    Nested dict {model_name: {ablation_label: mae}}
    """
    ablation_map = {
        "Standard":            None,
        "Randomise Noisy Est.": "random_noisy",
        "Randomise Metadata":  "random_meta",
        "Zero Noisy Est.":     "zero_noisy",
    }
    all_results: dict[str, dict[str, float]] = {}

    if print_table:
        print("\nAblation study: impact of input perturbation")
        print("=" * 60)

    for model_name, model in models.items():
        row: dict[str, float] = {}
        if print_table:
            print(f"\n  {model_name}:")

        for label, abl_type in ablation_map.items():
            if abl_type is None:
                y_pred = model.predict(X_test)  # type: ignore[union-attr]
                m      = float(np.mean(np.abs(y_pred - y_ideal)))
            else:
                m = _ablate(model, X_test, y_ideal, m_obs, abl_type)

            row[label] = m
            if print_table:
                std_mae = row.get("Standard", m)
                delta   = m - std_mae
                print(f"    {label:<28}: MAE={m:.4f}  (Δ={delta:+.4f})")

        all_results[model_name] = row

    return all_results
