"""
Traditional quantum error mitigation baselines.

    ┌──────────────────┬──────────────────────────────────────────────────────┐
    │ PEC Shadows      │ Analytical quasi-probability inversion of the noise  │
    │                  │ channel. Requires exact knowledge of the noise model. │
    ├──────────────────┼──────────────────────────────────────────────────────┤
    │ ZNE Shadows      │ Zero-noise extrapolation: estimate observables at    │
    │                  │ noise levels {λ, 2λ, 3λ} and fit a linear model.     │
    └──────────────────┴──────────────────────────────────────────────────────┘

References
----------
- Temme et al. (2017) — probabilistic error cancellation (PEC)
- Li & Benjamin (2017) — zero-noise extrapolation
- Jnane et al. (2023) — PEC applied to classical shadows
"""

from __future__ import annotations

import time
import numpy as np

from .config import CFG
from .noise import apply_depolarising
from .data import collect_shadow_from_density_matrix
from .utils import build_observable_set, get_shadow_expectations


# ---------------------------------------------------------------------------
# PEC — Probabilistic Error Cancellation
# ---------------------------------------------------------------------------

def pec_gamma_norm(p_gate: float) -> float:
    """
    Single-gate quasiprobability norm for a depolarising channel with
    error rate p_gate.

        ‖γ‖₁ = (1 + p) / (1 − p)

    This follows from the quasi-probability representation of the inverse
    of the depolarising channel.
    """
    return (1.0 + p_gate) / (1.0 - p_gate)


def pec_circuit_norm(n_1q: int, p_1q: float,
                      n_2q: int, p_2q: float) -> float:
    """
    Total PEC quasiprobability norm for a circuit.

        ‖g‖₁ = ∏_k ‖γ^(k)‖₁

    The sample overhead for PEC is ‖g‖₁².
    """
    return pec_gamma_norm(p_1q) ** n_1q * pec_gamma_norm(p_2q) ** n_2q


def pec_correct_observables(noisy_exp_vec: np.ndarray,
                              observables:  list[dict],
                              cfg:          dict | None = None) -> np.ndarray:
    """
    Apply analytical PEC correction to a vector of shadow observable estimates.

    For a k-local observable under a single-qubit depolarising channel D_p,
    the expected shrinkage is:

        ⟨O⟩_noisy = (1 − 4p/3)^k · (1 − 2α)^k · ⟨O⟩_ideal

    We analytically invert this per observable.  In practice (unknown noise)
    one would sample quasi-probabilities; here we use exact inversion since
    the simulation parameters are known.

    Parameters
    ----------
    noisy_exp_vec : (M_obs,) noisy shadow estimates
    observables   : list of observable dicts (from utils.build_observable_set)
    cfg           : experiment config (defaults to global CFG)

    Returns
    -------
    (M_obs,) PEC-corrected estimates
    """
    c = cfg or CFG
    p_eff = c["p_depol"]
    alpha = c["readout_err"]

    corrected = np.empty_like(noisy_exp_vec)
    for i, obs in enumerate(observables):
        k = len(obs["qubits"])
        shrink = ((1.0 - 4.0 * p_eff / 3.0) ** k) * ((1.0 - 2.0 * alpha) ** k)
        corrected[i] = (noisy_exp_vec[i] / shrink
                        if abs(shrink) > 1e-9
                        else noisy_exp_vec[i])
    return corrected


def run_pec(X_test: np.ndarray,
             observables: list[dict],
             cfg: dict | None = None) -> np.ndarray:
    """
    Apply PEC correction to every sample in the test set.

    Parameters
    ----------
    X_test      : (N, M_obs + 3) feature matrix
    observables : list of observable dicts
    cfg         : experiment config

    Returns
    -------
    Y_pec : (N, M_obs) PEC-corrected predictions
    """
    M_obs = len(observables)
    return np.array([
        pec_correct_observables(X_test[i, :M_obs], observables, cfg)
        for i in range(len(X_test))
    ])


# ---------------------------------------------------------------------------
# ZNE — Zero-Noise Extrapolation
# ---------------------------------------------------------------------------

def _boost_density_matrix(rho_noisy: np.ndarray,
                            boost_factor: float,
                            cfg:          dict | None = None) -> np.ndarray:
    """
    Approximate a noise-boosted density matrix ρ(λ·p) from ρ(p) by
    inserting extra single-qubit depolarising channels.

        ρ(λ·p) ≈ D_{(λ−1)·p}(ρ(p))   for each qubit
    """
    c = cfg or CFG
    n_q     = c["n_qubits"]
    p_extra = (boost_factor - 1.0) * c["p_depol"]
    rho = rho_noisy.copy()
    if p_extra > 0.0:
        for q in range(n_q):
            rho = apply_depolarising(rho, q, n_q, p_extra)
    return rho


def _zne_linear_extrapolate(lambdas:    np.ndarray,
                              exp_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Linear ZNE extrapolation to λ = 0 for every (circuit, observable) pair.

    Fits  y = a + b·λ  per observable per circuit and returns the intercept.

    Parameters
    ----------
    lambdas    : (L,) noise-scaling factors, e.g. [1, 2, 3]
    exp_arrays : list of L arrays, each (N_circuits, M_obs)

    Returns
    -------
    (N_circuits, M_obs) zero-noise extrapolated estimates
    """
    lambdas = np.asarray(lambdas, dtype=float)
    n_circ  = exp_arrays[0].shape[0]
    M       = exp_arrays[0].shape[1]
    Y_zne   = np.zeros((n_circ, M))

    for i in range(n_circ):
        for m in range(M):
            y_vals = np.array([exp_arrays[j][i, m] for j in range(len(lambdas))])
            coeffs = np.polyfit(lambdas, y_vals, deg=1)  # [slope, intercept]
            Y_zne[i, m] = coeffs[1]  # intercept → λ = 0

    return Y_zne


def run_zne(rho_noisy_list: list[np.ndarray],
             observables:    list[dict],
             lambdas:        list[float] | None = None,
             cfg:            dict | None = None,
             verbose:        bool = True) -> np.ndarray:
    """
    Apply ZNE mitigation to a list of noisy density matrices.

    For each circuit, we generate shadows at noise levels λ·p (p from cfg),
    estimate observables at each level, and linearly extrapolate to λ = 0.

    Parameters
    ----------
    rho_noisy_list : list of (dim, dim) noisy density matrices
    observables    : list of observable dicts
    lambdas        : noise-scaling factors (default [1.0, 2.0, 3.0])
    cfg            : experiment config
    verbose        : print timing

    Returns
    -------
    Y_zne : (N, M_obs) ZNE-corrected estimates
    """
    if lambdas is None:
        lambdas = [1.0, 2.0, 3.0]
    c      = cfg or CFG
    n_q    = c["n_qubits"]
    N_snap = max(50, c["shadow_size"] // 2)   # fewer shots per boosted level
    alpha  = c["readout_err"]

    t0 = time.time()
    boosted_exps: list[np.ndarray] = []

    for lam in lambdas:
        level_exps = []
        for rho_nx in rho_noisy_list:
            rho_b   = _boost_density_matrix(rho_nx, lam, c)
            out_b, rec_b = collect_shadow_from_density_matrix(
                rho_b, N_snap, n_q, alpha_readout=alpha
            )
            exp_b = get_shadow_expectations(out_b, rec_b, observables)
            level_exps.append(exp_b)
        boosted_exps.append(np.array(level_exps))

    Y_zne = _zne_linear_extrapolate(lambdas, boosted_exps)

    if verbose:
        print(f"  ZNE (λ={lambdas}) completed in {time.time() - t0:.1f}s")

    return Y_zne
