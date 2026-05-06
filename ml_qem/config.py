"""
Global configuration for the ML-QEM Classical Shadows experiment.

All experiment hyperparameters live here. Adjust values to scale up or down.

shadow_size
-----------
The theoretical sample-complexity bound (Huang et al. 2020) is:

    N_bound = ⌈ log(M) · 4^k / ε² ⌉

This bound guarantees estimation error ≤ ε for all M observables of locality k.
For n=5, k=3, ε=0.05 this gives N_bound ≈ 151 730 snapshots per circuit —
computationally infeasible for ML training.

`shadow_size` is therefore capped at SHADOW_SIZE_CAP (default 2000).
The theoretical bound is still computed and stored as `shadow_size_bound`
for reference and reporting.  Raise the cap when running on a cluster.
"""

from .utils import build_observable_set, shadow_bound

# ── System ────────────────────────────────────────────────────────────────────
_N_QUBITS     = 5       # number of qubits
_MAX_LOCALITY = 3       # highest Pauli weight included in the observable set
_EPSILON      = 0.2    # maximum allowed estimation error for shadow_bound

# Practical cap: how many snapshots we actually generate per circuit for ML.
# Raise this (e.g. to 10_000) on a cluster.
SHADOW_SIZE_CAP = 2_000

# Derive M and the theoretical / practical shadow sizes
_OBSERVABLES       = build_observable_set(_N_QUBITS, max_locality=_MAX_LOCALITY)
_M                 = len(_OBSERVABLES)
_SHADOW_SIZE_BOUND = shadow_bound(_M, k=_MAX_LOCALITY, epsilon=_EPSILON)
_SHADOW_SIZE       = min(_SHADOW_SIZE_BOUND, SHADOW_SIZE_CAP)

# ── Full config dict ──────────────────────────────────────────────────────────
CFG: dict = dict(
    n_qubits           = _N_QUBITS,
    max_locality       = _MAX_LOCALITY,
    shadow_epsilon     = _EPSILON,
    shadow_size_bound  = _SHADOW_SIZE_BOUND,   # theoretical minimum (informational)
    shadow_size        = _SHADOW_SIZE,          # practical value used during generation
    n_train            = 1_000,               # number of training circuits
    n_test             = 2_00,                # number of test circuits
    p_depol            = 0.03,
    p_2q               = 0.05,
    readout_err        = 0.02,
    pec_norm           = None,                 # filled automatically during PEC setup
)

if __name__ == "__main__":
    print(f"n_qubits          = {CFG['n_qubits']}")
    print(f"max_locality      = {CFG['max_locality']}")
    print(f"M (obs. set)      = {_M}")
    print(f"shadow_size_bound = {CFG['shadow_size_bound']}  (theoretical, ε={_EPSILON})")
    print(f"shadow_size       = {CFG['shadow_size']}  (capped at {SHADOW_SIZE_CAP})")
    print(f"n_train           = {CFG['n_train']}")
    print(f"n_test            = {CFG['n_test']}")
