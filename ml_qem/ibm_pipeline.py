"""
IBM Quantum / Qiskit backend pipeline for collecting classical shadows
from real or simulated hardware.

Default mode: Qiskit AerSimulator with a noise model matching the experiment
              config — use this for local testing without IBM credentials.

Real hardware mode: uncomment the IBM Runtime section at the bottom of this
                    file and follow the printed instructions.

Shadow collection strategy
--------------------------
Each snapshot = one Qiskit circuit (state_prep + basis_rotation + measure_all)
submitted with shots=1.  Circuits are batched (default 50 per job) to minimise
scheduler overhead on real hardware.
"""

from __future__ import annotations

import time
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from .config import CFG
from .utils import build_observable_set, get_shadow_expectations
from .benchmark import l1_relative_change


# ---------------------------------------------------------------------------
# Qiskit circuit builders
# ---------------------------------------------------------------------------

def build_brickwall_qiskit(n_qubits: int, params: np.ndarray) -> QuantumCircuit:
    """
    Build a brick-wall ansatz as a Qiskit QuantumCircuit.

    Parameters
    ----------
    n_qubits : number of qubits
    params   : (L, 2·n) angles — RY and RZ angles per qubit per layer
    """
    L  = params.shape[0]
    qc = QuantumCircuit(n_qubits)
    for layer in range(L):
        for q in range(n_qubits):
            qc.ry(params[layer, q],              q)
            qc.rz(params[layer, n_qubits + q],   q)
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
        if layer % 2 == 1:
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)
    return qc


def add_shadow_measurement(qc_state: QuantumCircuit,
                            recipe:   np.ndarray) -> QuantumCircuit:
    """
    Append a Pauli-basis rotation followed by measure_all to a circuit copy.

    recipe : (n,) int array — 0=X (apply H), 1=Y (apply Sdg then H), 2=Z (no-op)
    """
    qc = qc_state.copy()
    n_q = qc.num_qubits
    for q in range(n_q):
        r = int(recipe[q])
        if r == 0:       # X basis
            qc.h(q)
        elif r == 1:     # Y basis
            qc.sdg(q)
            qc.h(q)
        # r == 2: Z basis — no rotation needed
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Noise model builder
# ---------------------------------------------------------------------------

def build_aer_noise_model(n_qubits:    int,
                           p_1q:       float,
                           p_2q:       float,
                           readout_err: float) -> NoiseModel:
    """
    Build a Qiskit AerSimulator noise model that mirrors the NumPy simulation.

    Single-qubit gates : depolarising(p_1q)
    Two-qubit gates    : depolarising(p_2q)
    Readout            : symmetric bit-flip(readout_err) on every qubit
    """
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_1q, 1),
        ["h", "rx", "ry", "rz", "s", "sdg"],
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_2q, 2),
        ["cx"],
    )
    ro = ReadoutError([[1.0 - readout_err, readout_err],
                       [readout_err,       1.0 - readout_err]])
    for q in range(n_qubits):
        nm.add_readout_error(ro, [q])
    return nm


# ---------------------------------------------------------------------------
# Shadow collection from a Qiskit backend
# ---------------------------------------------------------------------------

def collect_shadow_qiskit(
    qc_state: QuantumCircuit,
    n_shots:  int,
    backend:  AerSimulator,
    recipes:  np.ndarray | None = None,
    batch_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a classical shadow from a Qiskit backend.

    Parameters
    ----------
    qc_state   : state-preparation circuit (no measurement)
    n_shots    : number of snapshots
    backend    : AerSimulator or IBM Runtime backend
    recipes    : optional preset Pauli recipes; randomly sampled if None
    batch_size : circuits per backend job (reduce for real hardware limits)

    Returns
    -------
    outcomes : (n_shots, n) array of ±1
    recipes  : (n_shots, n) array of 0/1/2
    """
    n_q = qc_state.num_qubits
    if recipes is None:
        recipes = np.random.randint(0, 3, size=(n_shots, n_q))

    outcomes = np.zeros((n_shots, n_q), dtype=float)

    all_circuits = [add_shadow_measurement(qc_state, recipes[t])
                    for t in range(n_shots)]

    for start in range(0, n_shots, batch_size):
        batch   = all_circuits[start : start + batch_size]
        job     = backend.run(batch, shots=1, memory=True)
        result  = job.result()

        for local_idx, t in enumerate(range(start, min(start + batch_size, n_shots))):
            mem = result.get_memory(local_idx)
            if mem:
                bitstring = mem[0]  # e.g. '0101'  (Qiskit: bit 0 = rightmost)
                bits      = [int(bitstring[-(q + 1)]) for q in range(n_q)]
                outcomes[t] = np.array([1 if b == 0 else -1 for b in bits])

    return outcomes, recipes


# ---------------------------------------------------------------------------
# End-to-end IBM / AerSimulator pipeline
# ---------------------------------------------------------------------------

def run_ibm_pipeline(
    n_circuits: int,
    ml_model,
    backend:    AerSimulator,
    ideal_backend: AerSimulator,
    n_shots:    int = 200,
    cfg:        dict | None = None,
    verbose:    bool = True,
) -> dict:
    """
    End-to-end pipeline for Qiskit-backend shadow collection + ML mitigation.

    Steps per circuit:
        1. Generate a random brick-wall circuit.
        2. Collect noisy shadow (from `backend`).
        3. Build feature vector and apply ML mitigation.
        4. Collect ideal shadow (from `ideal_backend`) for ground truth.

    Parameters
    ----------
    n_circuits    : number of random test circuits
    ml_model      : fitted QEM model with .predict(X) method
    backend       : noisy Aer backend (or IBM Runtime backend)
    ideal_backend : noiseless Aer backend for ground-truth evaluation
    n_shots       : snapshots per circuit
    cfg           : experiment config
    verbose       : print progress

    Returns
    -------
    dict with keys: noisy_exp, mit_exp, ideal_exp (each (n_circuits, M_obs))
    """
    c    = cfg or CFG
    n_q  = c["n_qubits"]
    observables = build_observable_set(n_q)
    M_obs       = len(observables)
    noise_meta  = np.array([c["p_depol"], c["p_2q"], c["readout_err"]])

    all_noisy: list[np.ndarray] = []
    all_mit:   list[np.ndarray] = []
    all_ideal: list[np.ndarray] = []

    t0 = time.time()
    for i in range(n_circuits):
        L      = np.random.randint(2, 4)
        params = np.random.uniform(0, 2 * np.pi, (L, 2 * n_q))
        qc     = build_brickwall_qiskit(n_q, params)

        # Noisy shadow
        out_n, rec_n = collect_shadow_qiskit(qc, n_shots, backend)
        noisy_exp    = get_shadow_expectations(out_n, rec_n, observables)

        # ML mitigation
        x_vec  = np.concatenate([noisy_exp, noise_meta]).reshape(1, -1)
        y_mit  = ml_model.predict(x_vec)[0]

        # Ideal shadow (ground truth)
        out_i, rec_i = collect_shadow_qiskit(qc, n_shots, ideal_backend)
        ideal_exp    = get_shadow_expectations(out_i, rec_i, observables)

        all_noisy.append(noisy_exp)
        all_mit.append(y_mit)
        all_ideal.append(ideal_exp)

        if verbose and (i + 1) % max(1, n_circuits // 3) == 0:
            n_done    = i + 1
            noisy_arr = np.array(all_noisy)
            ideal_arr = np.array(all_ideal)
            mit_arr   = np.array(all_mit)
            m_noisy   = float(np.mean(np.abs(noisy_arr - ideal_arr)))
            m_mit     = float(np.mean(np.abs(mit_arr   - ideal_arr)))
            print(f"  [{n_done}/{n_circuits}]  "
                  f"MAE noisy={m_noisy:.4f}  MAE mit={m_mit:.4f}")

    elapsed = time.time() - t0
    noisy_arr = np.array(all_noisy)
    mit_arr   = np.array(all_mit)
    ideal_arr = np.array(all_ideal)

    l1rc = l1_relative_change(mit_arr, ideal_arr, noisy_arr)

    if verbose:
        print(f"\nIBM pipeline completed in {elapsed:.1f}s")
        print(f"  MAE before mitigation: {float(np.mean(np.abs(noisy_arr - ideal_arr))):.4f}")
        print(f"  MAE after  mitigation: {float(np.mean(np.abs(mit_arr   - ideal_arr))):.4f}")
        print(f"  L1RC median: {float(np.median(l1rc)):.4f}")
        print(f"  % circuits improved: {100.0 * float(np.mean(l1rc < 0)):.1f}%")

    return dict(
        noisy_exp = noisy_arr,
        mit_exp   = mit_arr,
        ideal_exp = ideal_arr,
        l1rc      = l1rc,
        elapsed_s = elapsed,
    )


# ---------------------------------------------------------------------------
# Real IBM hardware instructions (template — not executed automatically)
# ---------------------------------------------------------------------------

REAL_HARDWARE_TEMPLATE = '''
# ── Real IBM Quantum Hardware Pipeline ────────────────────────────────────────
# 1. Install:  pip install qiskit-ibm-runtime
#
# 2. Authenticate (one-time):
#    from qiskit_ibm_runtime import QiskitRuntimeService
#    QiskitRuntimeService.save_account(channel="ibm_quantum",
#                                      token="YOUR_IBM_API_TOKEN",
#                                      overwrite=True)
#
# 3. Connect:
#    service      = QiskitRuntimeService(channel="ibm_quantum")
#    real_backend = service.backend("ibm_sherbrooke")   # or ibm_kyiv etc.
#
# 4. Read real device noise params for the feature vector:
#    props     = real_backend.properties()
#    p_1q_real = np.mean([props.gate_error("sx", q) for q in range(n)])
#    p_2q_real = np.mean([props.gate_error("ecr", [q, q+1])
#                         for q in range(n-1)])
#    ro_real   = np.mean([props.readout_error(q) for q in range(n)])
#    noise_meta_real = np.array([p_1q_real, p_2q_real, ro_real])
#
# 5. Transpile circuits to native gate set:
#    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#    pm      = generate_preset_pass_manager(backend=real_backend, optimization_level=1)
#    qc_isa  = pm.run(qc_shadow)
#
# 6. Submit via SamplerV2:
#    from qiskit_ibm_runtime import SamplerV2 as Sampler
#    sampler = Sampler(real_backend)
#    job     = sampler.run([qc_isa], shots=1)
#    counts  = job.result()[0].data.meas.get_counts()
#    bitstring = list(counts.keys())[0]
#
# 7. Call run_ibm_pipeline() with real_backend in place of AerSimulator.
# ─────────────────────────────────────────────────────────────────────────────
'''

if __name__ == "__main__":
    print("IBM Pipeline — real hardware instructions:")
    print(REAL_HARDWARE_TEMPLATE)
