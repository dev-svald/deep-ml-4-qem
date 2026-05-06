"""
Mathematical utilities: Pauli matrices, classical shadow snapshot formula,
density-matrix metrics, and observable set construction.

References
----------
- Huang et al. (2020) — classical shadows protocol
- Jnane et al. (2023) — quantum error mitigated classical shadows
- Placidi et al. (2026) — deep learning approaches to QEM
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
from itertools import combinations
from typing import Sequence

# ---------------------------------------------------------------------------
# Pauli matrices and measurement-basis unitaries
# ---------------------------------------------------------------------------

I2: np.ndarray = np.eye(2, dtype=complex)
X:  np.ndarray = np.array([[0, 1],  [1, 0]],   dtype=complex)
Y:  np.ndarray = np.array([[0, -1j],[1j, 0]],  dtype=complex)
Z:  np.ndarray = np.array([[1, 0],  [0, -1]],  dtype=complex)

H_GATE: np.ndarray = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDAG:   np.ndarray = np.array([[1, 0], [0, -1j]], dtype=complex)

# Unitaries that rotate each Pauli eigenbasis into the computational (Z) basis.
# Index: 0 → X-basis  (apply H),  1 → Y-basis  (apply H·S†),  2 → Z-basis  (identity)
PAULI_UNITARIES: list[np.ndarray] = [H_GATE, H_GATE @ SDAG, I2]
PAULI_OPS:       list[np.ndarray] = [X, Y, Z]

# Projectors onto computational-basis states
P0: np.ndarray = np.array([[1, 0], [0, 0]], dtype=complex)   # |0><0|
P1: np.ndarray = np.array([[0, 0], [0, 1]], dtype=complex)   # |1><1|

PAULI_NAMES: list[str] = ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Tensor product helper
# ---------------------------------------------------------------------------

def tensor(*ops: np.ndarray) -> np.ndarray:
    """Kronecker (tensor) product of a sequence of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# ---------------------------------------------------------------------------
# Classical shadow snapshot formula
# ---------------------------------------------------------------------------

def snapshot_state(b_list: Sequence[int], obs_list: Sequence[int]) -> np.ndarray:
    """
    Compute one Pauli-shadow snapshot matrix.

    Implements the per-qubit formula:
        rho_hat = ⊗_j ( 3 · U_j† |b_j><b_j| U_j  −  I )

    Parameters
    ----------
    b_list   : measurement outcomes per qubit — +1 (bit=0) or -1 (bit=1)
    obs_list : Pauli recipe per qubit — 0=X, 1=Y, 2=Z

    Returns
    -------
    rho_hat : (2^n, 2^n) complex ndarray
    """
    rho = np.array([[1.0]], dtype=complex)
    for b, recipe in zip(b_list, obs_list):
        Pb = P0 if b == 1 else P1
        U  = PAULI_UNITARIES[int(recipe)]
        local = 3.0 * (U.conj().T @ Pb @ U) - I2
        rho = np.kron(rho, local)
    return rho


def shadow_state_reconstruction(outcomes: np.ndarray,
                                 recipes:  np.ndarray) -> np.ndarray:
    """
    Reconstruct a density-matrix estimate by averaging snapshots.

    Parameters
    ----------
    outcomes : (N, n) array of ±1
    recipes  : (N, n) array of 0/1/2

    Returns
    -------
    rho_est : (2^n, 2^n) complex ndarray — empirical shadow estimate
    """
    N = len(outcomes)
    n_q = outcomes.shape[1]
    dim = 2 ** n_q
    rho_est = np.zeros((dim, dim), dtype=complex)
    for t in range(N):
        rho_est += snapshot_state(outcomes[t], recipes[t])
    return rho_est / N


# ---------------------------------------------------------------------------
# Efficient median-of-means observable estimator
# ---------------------------------------------------------------------------

def estimate_observable_from_shadow(
    outcomes:   np.ndarray,
    recipes:    np.ndarray,
    obs_qubits: Sequence[int],
    obs_paulis: Sequence[int],
    k:          int = 10,
) -> float:
    """
    Median-of-means estimator for Tr(O ρ) from a classical shadow.

    Only snapshots whose recipe matches the observable's support contribute.
    The efficient implementation avoids building full snapshot matrices.

    Parameters
    ----------
    outcomes   : (N, n) array of ±1
    recipes    : (N, n) array of 0/1/2
    obs_qubits : qubit indices where O acts non-trivially
    obs_paulis : Pauli index (0=X, 1=Y, 2=Z) at each support qubit
    k          : number of chunks for median-of-means (reduces variance)

    Returns
    -------
    Scalar estimate of Tr(O ρ)
    """
    N = len(outcomes)
    chunk = max(1, N // k)
    target_locs   = np.array(obs_qubits)
    target_paulis = np.array(obs_paulis)
    means = []

    for i in range(0, N, chunk):
        out_k = np.array(outcomes[i : i + chunk])
        rec_k = np.array(recipes[i  : i + chunk])
        # Keep only snapshots that measured in the correct Pauli bases
        mask = np.all(rec_k[:, target_locs] == target_paulis, axis=1)
        if mask.sum() > 0:
            prod = np.prod(out_k[mask][:, target_locs], axis=1)
            means.append(prod.mean())
        else:
            means.append(0.0)

    return float(np.median(means))


# ---------------------------------------------------------------------------
# Density-matrix distance metrics
# ---------------------------------------------------------------------------

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """½ · ‖ρ − σ‖₁  (trace distance)."""
    diff = rho - sigma
    eigs = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigs)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(ρ, σ) = (Tr √(√ρ σ √ρ))²."""
    sqrt_rho = la.sqrtm(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    return float(np.real(np.trace(la.sqrtm(M))) ** 2)


def operator_norm(A: np.ndarray) -> float:
    """Frobenius norm: √Tr(A† A)."""
    return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))


# ---------------------------------------------------------------------------
# Shadow sample-complexity bound
# ---------------------------------------------------------------------------

def shadow_bound(M: int, k: int = 3, epsilon: float = 0.1) -> int:
    """
    Minimum number of classical-shadow snapshots N needed to estimate M
    observables of locality k to additive error ε with high probability.

    Based on the median-of-means sample complexity (Huang et al. 2020):

        N = ⌈ log(M) · 4^k / ε² ⌉

    Parameters
    ----------
    M       : number of observables to estimate simultaneously
    k       : maximum locality of the observables (default 3)
    epsilon : maximum allowed estimation error  (default 0.05)

    Returns
    -------
    N : int — minimum shadow size (number of snapshots)

    Examples
    --------
    >>> shadow_bound(M=100, k=2, epsilon=0.1)
    >>> shadow_bound(M=500, k=3, epsilon=0.05)
    """
    import math
    if M <= 0:
        raise ValueError(f"M must be a positive integer, got {M}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

    N = math.ceil(math.log(M) * (4 ** k) / (epsilon ** 2))
    return N


# ---------------------------------------------------------------------------
# Observable set construction
# ---------------------------------------------------------------------------

def build_observable_set(n_qubits: int, max_locality: int = 3) -> list[dict]:
    """
    Build the full Pauli observable set up to locality `max_locality`.

    Includes all k-local Pauli strings for k = 1, 2, …, max_locality,
    over every combination of qubit support.

    Each entry is a dict with keys:
        qubits    : list of qubit indices where O acts non-trivially
        paulis    : list of Pauli indices (0=X, 1=Y, 2=Z) at those qubits
        locality  : int — number of non-identity factors (k)
        matrix    : (2^n, 2^n) operator matrix embedded in full Hilbert space
        label     : human-readable string, e.g. 'X0', 'Y1Z2', 'X0Y1Z2'

    Observable counts (for n qubits)
    ----------------------------------
        1-local : 3·n
        2-local : 9·C(n,2)
        3-local : 27·C(n,3)

    Parameters
    ----------
    n_qubits    : number of qubits in the system
    max_locality: highest Pauli weight to include (default 3)

    Returns
    -------
    list of observable dicts
    """
    from itertools import product as iproduct

    observables: list[dict] = []

    for k in range(1, max_locality + 1):
        for qubits in combinations(range(n_qubits), k):
            # Enumerate all 3^k Pauli strings on these k qubits (skip all-I)
            for pauli_indices in iproduct(range(3), repeat=k):
                ops = [I2] * n_qubits
                label_parts = []
                for qubit, p_idx in zip(qubits, pauli_indices):
                    ops[qubit] = PAULI_OPS[p_idx]
                    label_parts.append(f"{PAULI_NAMES[p_idx]}{qubit}")
                mat = tensor(*ops)
                observables.append({
                    "qubits":   list(qubits),
                    "paulis":   list(pauli_indices),
                    "locality": k,
                    "matrix":   mat,
                    "label":    "".join(label_parts),
                })

    return observables


def get_exact_expectations(rho: np.ndarray,
                            observables: list[dict]) -> np.ndarray:
    """Compute Tr(O ρ) exactly for every observable in the set."""
    return np.array([np.real(np.trace(obs["matrix"] @ rho))
                     for obs in observables])


def get_shadow_expectations(outcomes:    np.ndarray,
                             recipes:     np.ndarray,
                             observables: list[dict],
                             k:           int = 10) -> np.ndarray:
    """Estimate Tr(O ρ) from a classical shadow for all observables."""
    return np.array([
        estimate_observable_from_shadow(
            outcomes, recipes,
            obs["qubits"], obs["paulis"], k=k,
        )
        for obs in observables
    ])


# ---------------------------------------------------------------------------
# Self-test (run with:  python -m ml_qem.utils)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    # Snapshot of |0><0| under Z measurement → diag(2, -1)
    snap = snapshot_state([1], [2])
    assert np.allclose(snap, np.diag([2.0, -1.0])), f"Snapshot error: {snap}"
    print("Snapshot formula:                    PASS")

    # Two-qubit snapshot |00> all-Z → Tr should be 4
    tr = np.trace(snapshot_state([1, 1], [2, 2])).real
    assert abs(tr - 4.0) < 1e-10, f"Trace error: {tr}"
    print("Two-qubit snapshot trace (=4):       PASS")

    # Observable set: 1+2+3-local for n=3
    obs_set = build_observable_set(3, max_locality=3)
    # 1-local: 3*3=9  |  2-local: 9*3=27  |  3-local: 27*1=27  →  total=63
    expected = 3 * 3 + 9 * 3 + 27 * 1
    assert len(obs_set) == expected, \
        f"Observable count: got {len(obs_set)}, expected {expected}"
    print(f"Observable set (n=3, k≤3, M={len(obs_set)}): PASS")

    # Check locality field is correctly set
    localities = {obs["locality"] for obs in obs_set}
    assert localities == {1, 2, 3}, f"Locality set wrong: {localities}"
    print(f"Locality field {{1,2,3}}:              PASS")

    # shadow_bound: formula check
    M_test  = len(obs_set)
    N       = shadow_bound(M_test, k=3, epsilon=0.05)
    N_check = math.ceil(math.log(M_test) * (4 ** 3) / (0.05 ** 2))
    assert N == N_check, f"shadow_bound mismatch: {N} vs {N_check}"
    print(f"shadow_bound(M={M_test}, k=3, ε=0.05) = {N}: PASS")
