"""
Noise simulation engine (density-matrix level, pure NumPy).

Three physically motivated noise sources:

    Source              Model                                       Parameter
    ──────────────────  ──────────────────────────────────────────  ─────────
    Single-qubit gate   Depolarising  D_p(ρ) = (1−p)ρ + p/3 ΣPρP  p = 0.03
    Two-qubit gate      Local depolarising (each qubit)             p = 0.05
    Readout (SPAM)      Symmetric bit-flip α on each qubit          α = 0.02
"""

from __future__ import annotations

import numpy as np
from .utils import I2, X, Y, Z, PAULI_UNITARIES, tensor


# ---------------------------------------------------------------------------
# Quantum channels (density matrix → density matrix)
# ---------------------------------------------------------------------------

def apply_depolarising(rho: np.ndarray,
                        qubit_idx: int,
                        n_qubits:  int,
                        p:         float) -> np.ndarray:
    """
    Single-qubit depolarising channel on qubit `qubit_idx` of an n-qubit system.

        D_p(ρ) = (1−p)ρ + (p/3)(XρX + YρY + ZρZ)   on the target qubit.

    Implemented via Kraus operators embedded in the full Hilbert space.
    """
    if p == 0.0:
        return rho
    result = (1.0 - p) * rho
    for P_op in [X, Y, Z]:
        ops = [I2] * n_qubits
        ops[qubit_idx] = P_op
        K = tensor(*ops)
        result += (p / 3.0) * (K @ rho @ K.conj().T)
    return result


def apply_2q_depolarising(rho:      np.ndarray,
                           qubit_i:  int,
                           qubit_j:  int,
                           n_qubits: int,
                           p:        float) -> np.ndarray:
    """
    Local depolarising on *both* qubits involved in a two-qubit gate.
    Applies single-qubit depolarising sequentially to qubit_i and qubit_j.
    """
    rho = apply_depolarising(rho, qubit_i, n_qubits, p)
    rho = apply_depolarising(rho, qubit_j, n_qubits, p)
    return rho


# ---------------------------------------------------------------------------
# Readout (SPAM) error
# ---------------------------------------------------------------------------

def apply_readout_error(probs: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply a symmetric readout (SPAM) error with bit-flip probability α
    independently on each qubit, operating on a probability vector.

    For n qubits:
        P_noisy[b] = Σ_{b'} ∏_j M[b_j, b'_j] · P_ideal[b']
    where  M = [[1−α, α], [α, 1−α]]

    Parameters
    ----------
    probs : (2^n,) probability vector (must sum to 1)
    alpha : single-qubit readout bit-flip probability

    Returns
    -------
    (2^n,) noisy probability vector
    """
    if alpha == 0.0:
        return probs
    n_q = int(np.log2(len(probs)))
    M1 = np.array([[1.0 - alpha, alpha],
                   [alpha,       1.0 - alpha]])
    M_full = M1
    for _ in range(n_q - 1):
        M_full = np.kron(M_full, M1)
    return M_full @ probs


# ---------------------------------------------------------------------------
# Combined shadow snapshot with readout error
# ---------------------------------------------------------------------------

def snap_with_readout(rho:      np.ndarray,
                       alpha:    float,
                       n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single Pauli-basis shadow snapshot with readout (SPAM) error.

    Steps:
        1. Sample a uniformly random Pauli recipe (X/Y/Z per qubit).
        2. Rotate ρ into the measurement basis U_meas.
        3. Compute ideal measurement probabilities from the diagonal of U ρ U†.
        4. Apply readout confusion matrix to obtain noisy probabilities.
        5. Sample a bitstring from the noisy distribution.

    Parameters
    ----------
    rho      : (2^n, 2^n) density matrix
    alpha    : readout bit-flip probability
    n_qubits : number of qubits

    Returns
    -------
    outcomes : (n,) array of ±1  (+1 ↔ bit 0,  −1 ↔ bit 1)
    recipe   : (n,) array of 0/1/2  (X/Y/Z per qubit)
    """
    recipe = np.random.randint(0, 3, size=n_qubits)

    # Rotate ρ to measurement basis
    U_meas = tensor(*[PAULI_UNITARIES[r] for r in recipe])
    rho_rot = U_meas @ rho @ U_meas.conj().T

    # Ideal measurement probabilities (diagonal of rotated ρ)
    probs_ideal = np.real(np.diag(rho_rot))
    probs_ideal = np.maximum(probs_ideal, 0.0)
    probs_ideal /= probs_ideal.sum()

    # Apply readout error
    probs_noisy = apply_readout_error(probs_ideal, alpha)
    probs_noisy = np.maximum(probs_noisy, 0.0)
    probs_noisy /= probs_noisy.sum()

    # Sample outcome index
    dim = 2 ** n_qubits
    outcome_idx = np.random.choice(dim, p=probs_noisy)

    # Decode bitstring (Qiskit convention: qubit 0 is the most-significant bit here)
    bits = [(outcome_idx >> (n_qubits - 1 - j)) & 1 for j in range(n_qubits)]
    outcomes = np.array([1 if b == 0 else -1 for b in bits])

    return outcomes, recipe


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rho0 = np.array([[0.8, 0.1 + 0.1j],
                      [0.1 - 0.1j, 0.2]])
    rho_noisy = apply_depolarising(rho0, 0, 1, 0.05)
    assert abs(np.trace(rho_noisy).real - 1.0) < 1e-10
    print("Depolarising: trace preserved —   PASS")

    probs = np.array([0.5, 0.3, 0.1, 0.1])
    probs_ro = apply_readout_error(probs, 0.02)
    assert abs(probs_ro.sum() - 1.0) < 1e-10
    print("Readout error: probabilities sum to 1 — PASS")
