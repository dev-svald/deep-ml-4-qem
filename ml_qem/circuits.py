"""
PennyLane circuit families and density-matrix helpers.

Two circuit families are supported (following Placidi et al. 2026):
    1. Pauli gadget circuits  — e^{−iα P} layers for random Pauli strings P
    2. Brick-wall circuits    — alternating RY/RZ layers + CNOT entanglers

Each family provides an *ideal* density matrix (no noise) and a *noisy*
density matrix (single-qubit depolarising after every gate layer).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .config import CFG


# ---------------------------------------------------------------------------
# PennyLane device setup (created lazily from config)
# ---------------------------------------------------------------------------

def _make_mixed_device(n_qubits: int) -> qml.Device:
    return qml.device("default.mixed", wires=n_qubits)


# ---------------------------------------------------------------------------
# Circuit family 1: Random Pauli gadget circuit
# ---------------------------------------------------------------------------

def make_pauli_circuit(params: np.ndarray, n_qubits: int) -> None:
    """
    PennyLane tape: apply T Pauli-gadget layers  e^{−i α_t P_t}.

    Parameters
    ----------
    params   : (T, n+1) array — row t is [alpha_t, pauli_0, ..., pauli_{n-1}]
               pauli values: 0=I, 1=X, 2=Y, 3=Z
    n_qubits : number of qubits (must match params shape)
    """
    pauli_map = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    T = params.shape[0]
    for t in range(T):
        alpha  = float(params[t, 0])
        paulis = params[t, 1:].astype(int)
        ops    = [pauli_map[p](w) for w, p in enumerate(paulis) if p > 0]
        if ops:
            op = qml.prod(*ops) if len(ops) > 1 else ops[0]
            qml.exp(op, -1j * alpha)


def random_pauli_params(n_qubits: int,
                         T_range:  tuple[int, int] = (3, 7)) -> np.ndarray:
    """Sample random parameters for a Pauli gadget circuit."""
    T = np.random.randint(*T_range)
    params = np.zeros((T, n_qubits + 1))
    params[:, 0]  = np.random.uniform(0, np.pi, T)
    params[:, 1:] = np.random.randint(0, 4, (T, n_qubits))
    return params


# ---------------------------------------------------------------------------
# Circuit family 2: Brick-wall random circuit
# ---------------------------------------------------------------------------

def make_brickwall_circuit(params: np.ndarray, n_qubits: int) -> None:
    """
    PennyLane tape: L brick-wall layers of RY/RZ + CNOT.

    Parameters
    ----------
    params   : (L, 2*n) array — row ℓ is [ry_0,..,ry_{n-1}, rz_0,..,rz_{n-1}]
    n_qubits : number of qubits
    """
    L = params.shape[0]
    for layer in range(L):
        for q in range(n_qubits):
            qml.RY(params[layer, q],          wires=q)
            qml.RZ(params[layer, n_qubits + q], wires=q)
        for q in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        if layer % 2 == 1:
            for q in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])


def random_brickwall_params(n_qubits: int,
                              L_range:  tuple[int, int] = (2, 5)) -> np.ndarray:
    """Sample random parameters for a brick-wall circuit."""
    L = np.random.randint(*L_range)
    return np.random.uniform(0, 2 * np.pi, (L, 2 * n_qubits))


# ---------------------------------------------------------------------------
# Density-matrix evaluation
# ---------------------------------------------------------------------------

def get_ideal_density_matrix(circuit_fn,
                               params:   np.ndarray | None,
                               n_qubits: int) -> np.ndarray:
    """
    Run `circuit_fn(params, n_qubits)` on a noiseless mixed device and
    return the resulting density matrix.
    """
    dev = _make_mixed_device(n_qubits)

    @qml.qnode(dev)
    def _circuit():
        if params is not None:
            circuit_fn(params, n_qubits)
        return qml.density_matrix(wires=range(n_qubits))

    return np.array(_circuit())


def get_noisy_density_matrix(circuit_fn,
                               params:   np.ndarray | None,
                               n_qubits: int,
                               p_1q:     float,
                               p_2q:     float | None = None) -> np.ndarray:
    """
    Run `circuit_fn(params, n_qubits)` on a PennyLane mixed device with
    single-qubit DepolarizingChannel appended after the circuit layers.

    This is an *approximation*: noise is injected as a global per-qubit
    depolarising channel at the end rather than after every gate.
    For a more gate-level model, use the NumPy engine in noise.py.
    """
    dev = _make_mixed_device(n_qubits)

    @qml.qnode(dev)
    def _circuit():
        if params is not None:
            circuit_fn(params, n_qubits)
        for q in range(n_qubits):
            qml.DepolarizingChannel(p_1q, wires=q)
        return qml.density_matrix(wires=range(n_qubits))

    return np.array(_circuit())


# ---------------------------------------------------------------------------
# Convenience: sample one random circuit (either family)
# ---------------------------------------------------------------------------

def sample_random_circuit(n_qubits: int,
                            circuit_type: str = "mixed",
                            circuit_index: int = 0
                            ) -> tuple[callable, np.ndarray]:
    """
    Return (circuit_fn, params) for a randomly sampled circuit.

    circuit_type : 'pauli' | 'brickwall' | 'mixed'
        'mixed' alternates between Pauli (even index) and brick-wall (odd index).
    """
    use_pauli = (
        circuit_type == "pauli"
        or (circuit_type == "mixed" and circuit_index % 2 == 0)
    )
    if use_pauli:
        return make_pauli_circuit, random_pauli_params(n_qubits)
    else:
        return make_brickwall_circuit, random_brickwall_params(n_qubits)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = CFG["n_qubits"]

    def bell(params, n_q):
        qml.Hadamard(0)
        for q in range(n_q - 1):
            qml.CNOT(wires=[0, q + 1])

    rho_id = get_ideal_density_matrix(bell, None, n)
    rho_nx = get_noisy_density_matrix(bell, None, n, CFG["p_depol"])

    from .utils import trace_distance
    td = trace_distance(rho_id, rho_nx)
    print(f"Bell state trace distance ideal↔noisy: {td:.4f}  (expected >0)")
    assert td > 0, "Noise had no effect!"
    print("Circuit builders: PASS")
