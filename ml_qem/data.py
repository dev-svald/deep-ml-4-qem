"""
Data generation, persistence, and combination pipeline.

Each training/test sample consists of:
    x  : [noisy_shadow_expectations (M_obs), noise_metadata (3)]
    y  : ideal_expectations (M_obs)

Datasets are split strictly by circuit family so models can be trained
on each family independently, then on a combined set:

    Families
    --------
    pauli     — Pauli gadget circuits  e^{-iα P}
    brickwall — Alternating RY/RZ layers + CNOT brick-wall

Disk layout  (default root: datasets/)
---------------------------------------
    datasets/
        n3/
            pauli_train.npz
            pauli_test.npz
            brickwall_train.npz
            brickwall_test.npz
            combined_train.npz
            combined_test.npz
        n5/
            ...

Each .npz file stores X, Y, rho_ideal, rho_noisy, and a JSON metadata
blob. The observable set is NOT stored — it is always rebuilt from
(n_qubits, max_locality) which are saved in metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .config import CFG
from .circuits import (
    get_ideal_density_matrix,
    get_noisy_density_matrix,
    make_pauli_circuit,
    make_brickwall_circuit,
    random_pauli_params,
    random_brickwall_params,
)
from .noise import snap_with_readout
from .utils import (
    build_observable_set,
    get_exact_expectations,
    get_shadow_expectations,
    shadow_bound,
)

CircuitFamily = Literal["pauli", "brickwall", "combined"]


# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------

@dataclass
class QEMDataset:
    """
    Container for a generated ML-QEM dataset.

    Attributes
    ----------
    X            : (N, M_obs + 3)  — noisy shadow estimates + noise metadata
    Y            : (N, M_obs)      — ideal observable expectations
    rho_ideal    : (N, dim, dim)   — ideal density matrices
    rho_noisy    : (N, dim, dim)   — noisy density matrices
    circuit_type : 'pauli' | 'brickwall' | 'combined'
    n_qubits     : number of qubits
    max_locality : Pauli weight used to build the observable set
    cfg_snapshot : copy of the config dict used during generation
    observables  : list of observable dicts (rebuilt from n_qubits/max_locality)
    """

    X:            np.ndarray
    Y:            np.ndarray
    rho_ideal:    np.ndarray
    rho_noisy:    np.ndarray
    circuit_type: CircuitFamily
    n_qubits:     int
    max_locality: int                      = 3
    cfg_snapshot: dict                     = field(default_factory=dict)
    observables:  list[dict]               = field(repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.observables:
            self.observables = build_observable_set(
                self.n_qubits, max_locality=self.max_locality
            )

    # ── convenience properties ────────────────────────────────────────────

    @property
    def n_circuits(self) -> int:
        return len(self.X)

    @property
    def m_obs(self) -> int:
        return self.Y.shape[1]

    @property
    def noisy_expectations(self) -> np.ndarray:
        """(N, M_obs) noisy shadow observable estimates."""
        return self.X[:, : self.m_obs]

    @property
    def noise_metadata(self) -> np.ndarray:
        """(N, 3) noise metadata vector (p_1q, p_2q, alpha)."""
        return self.X[:, self.m_obs :]

    def __repr__(self) -> str:
        return (
            f"QEMDataset(circuit_type='{self.circuit_type}', "
            f"n_qubits={self.n_qubits}, "
            f"n_circuits={self.n_circuits}, "
            f"m_obs={self.m_obs})"
        )

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """
        Save the dataset to a compressed .npz file.

        The observable matrices are NOT stored (they are large and can be
        rebuilt deterministically). All other data is saved.

        Parameters
        ----------
        path : file path, e.g. 'datasets/n3/pauli_train.npz'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "circuit_type": self.circuit_type,
            "n_qubits":     self.n_qubits,
            "max_locality": self.max_locality,
            "cfg_snapshot": self.cfg_snapshot,
        }
        np.savez_compressed(
            path,
            X          = self.X,
            Y          = self.Y,
            rho_ideal  = self.rho_ideal,
            rho_noisy  = self.rho_noisy,
            meta       = np.array(json.dumps(meta)),   # scalar string array
        )
        size_mb = path.stat().st_size / 1e6
        print(f"  Saved  → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path | str) -> "QEMDataset":
        """
        Load a dataset from a .npz file saved by QEMDataset.save().

        Parameters
        ----------
        path : file path, e.g. 'datasets/n3/pauli_train.npz'

        Returns
        -------
        QEMDataset with observables rebuilt from stored metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))

        ds = cls(
            X            = data["X"],
            Y            = data["Y"],
            rho_ideal    = data["rho_ideal"],
            rho_noisy    = data["rho_noisy"],
            circuit_type = meta["circuit_type"],
            n_qubits     = meta["n_qubits"],
            max_locality = meta["max_locality"],
            cfg_snapshot = meta["cfg_snapshot"],
        )
        print(f"  Loaded ← {path}  ({ds})")
        return ds


# ---------------------------------------------------------------------------
# Shadow collection from a density matrix
# ---------------------------------------------------------------------------

def collect_shadow_from_density_matrix(
    rho:           np.ndarray,
    n_shots:       int,
    n_qubits:      int,
    alpha_readout: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate `n_shots` Pauli-basis snapshots from density matrix `rho`.

    Returns
    -------
    outcomes : (n_shots, n) array of ±1
    recipes  : (n_shots, n) array of 0/1/2
    """
    outcomes = np.zeros((n_shots, n_qubits), dtype=float)
    recipes  = np.zeros((n_shots, n_qubits), dtype=int)
    for t in range(n_shots):
        outcomes[t], recipes[t] = snap_with_readout(rho, alpha_readout, n_qubits)
    return outcomes, recipes


# ---------------------------------------------------------------------------
# Core single-family dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_circuits:   int,
    circuit_type: CircuitFamily = "pauli",
    verbose:      bool          = True,
    cfg:          dict | None   = None,
) -> QEMDataset:
    """
    Generate a labelled dataset from a single circuit family.

    Parameters
    ----------
    n_circuits   : number of circuits to generate
    circuit_type : 'pauli' | 'brickwall'
                   ('combined' is not valid here — use combine_datasets()
                    or generate_all_datasets() instead)
    verbose      : print progress every 20 % of circuits
    cfg          : override global CFG (useful for scaling or tests)

    Returns
    -------
    QEMDataset
    """
    if circuit_type == "combined":
        raise ValueError(
            "circuit_type='combined' is not valid for generate_dataset(). "
            "Use combine_datasets() or generate_all_datasets() instead."
        )

    c           = cfg or CFG
    n_qubits    = c["n_qubits"]
    max_loc     = c.get("max_locality", 3)
    shadow_size = c["shadow_size"]
    p_1q        = c["p_depol"]
    p_2q_val    = c["p_2q"]
    alpha_ro    = c["readout_err"]

    observables = build_observable_set(n_qubits, max_locality=max_loc)
    M_obs       = len(observables)
    noise_meta  = np.array([p_1q, p_2q_val, alpha_ro])
    log_every   = max(1, n_circuits // 5)

    X_list:      list[np.ndarray] = []
    Y_list:      list[np.ndarray] = []
    rho_id_list: list[np.ndarray] = []
    rho_nx_list: list[np.ndarray] = []

    # Pick circuit builder for this family
    _circ_fn     = make_pauli_circuit   if circuit_type == "pauli"    \
                   else make_brickwall_circuit
    _params_fn   = random_pauli_params  if circuit_type == "pauli"    \
                   else random_brickwall_params

    for i in range(n_circuits):
        params = _params_fn(n_qubits)

        rho_id = get_ideal_density_matrix(_circ_fn, params, n_qubits)
        rho_nx = get_noisy_density_matrix(_circ_fn, params, n_qubits,
                                          p_1q, p_2q_val)

        rho_id_list.append(rho_id)
        rho_nx_list.append(rho_nx)

        outcomes, recipes = collect_shadow_from_density_matrix(
            rho_nx, shadow_size, n_qubits, alpha_readout=alpha_ro
        )

        noisy_exp = get_shadow_expectations(outcomes, recipes, observables)
        ideal_exp = get_exact_expectations(rho_id, observables)

        X_list.append(np.concatenate([noisy_exp, noise_meta]))
        Y_list.append(ideal_exp)

        if verbose and (i + 1) % log_every == 0:
            print(f"  [{i + 1:>{len(str(n_circuits))}}/{n_circuits}] circuits done")

    return QEMDataset(
        X            = np.array(X_list),
        Y            = np.array(Y_list),
        rho_ideal    = np.array(rho_id_list),
        rho_noisy    = np.array(rho_nx_list),
        circuit_type = circuit_type,
        n_qubits     = n_qubits,
        max_locality = max_loc,
        cfg_snapshot = {k: v for k, v in c.items() if v is not None},
        observables  = observables,
    )


# ---------------------------------------------------------------------------
# Dataset combination
# ---------------------------------------------------------------------------

def combine_datasets(*datasets: QEMDataset) -> QEMDataset:
    """
    Concatenate two or more QEMDatasets along the circuit axis.

    All datasets must share the same (n_qubits, max_locality, M_obs).
    The combined dataset is shuffled so circuit families are interleaved.

    Returns
    -------
    QEMDataset with circuit_type='combined'
    """
    if len(datasets) < 2:
        raise ValueError("Need at least two datasets to combine.")

    n_qubits     = datasets[0].n_qubits
    max_locality = datasets[0].max_locality
    m_obs        = datasets[0].m_obs

    for ds in datasets[1:]:
        if ds.n_qubits != n_qubits:
            raise ValueError(
                f"n_qubits mismatch: {n_qubits} vs {ds.n_qubits}"
            )
        if ds.max_locality != max_locality:
            raise ValueError(
                f"max_locality mismatch: {max_locality} vs {ds.max_locality}"
            )
        if ds.m_obs != m_obs:
            raise ValueError(
                f"m_obs mismatch: {m_obs} vs {ds.m_obs}"
            )

    X         = np.concatenate([ds.X         for ds in datasets], axis=0)
    Y         = np.concatenate([ds.Y         for ds in datasets], axis=0)
    rho_ideal = np.concatenate([ds.rho_ideal for ds in datasets], axis=0)
    rho_noisy = np.concatenate([ds.rho_noisy for ds in datasets], axis=0)

    # Shuffle so circuit families are randomly interleaved
    idx = np.random.permutation(len(X))
    X, Y, rho_ideal, rho_noisy = X[idx], Y[idx], rho_ideal[idx], rho_noisy[idx]

    return QEMDataset(
        X            = X,
        Y            = Y,
        rho_ideal    = rho_ideal,
        rho_noisy    = rho_noisy,
        circuit_type = "combined",
        n_qubits     = n_qubits,
        max_locality = max_locality,
        cfg_snapshot = datasets[0].cfg_snapshot,
        observables  = datasets[0].observables,
    )


# ---------------------------------------------------------------------------
# Convenience: generate and save all six datasets at once
# ---------------------------------------------------------------------------

def generate_all_datasets(
    data_dir:  Path | str = "datasets",
    cfg:       dict | None = None,
    verbose:   bool = True,
) -> dict[str, QEMDataset]:
    """
    Generate and save the full set of six datasets for a given config:

        pauli_train, pauli_test,
        brickwall_train, brickwall_test,
        combined_train, combined_test

    Files are saved under:
        <data_dir>/n<n_qubits>/

    Parameters
    ----------
    data_dir : root directory for all dataset files
    cfg      : experiment config (defaults to global CFG)
    verbose  : print progress

    Returns
    -------
    dict with keys: pauli_train, pauli_test, brickwall_train,
                    brickwall_test, combined_train, combined_test
    """
    c        = cfg or CFG
    n_qubits = c["n_qubits"]
    n_train  = c["n_train"]
    n_test   = c["n_test"]
    root     = Path(data_dir) / f"n{n_qubits}"

    datasets: dict[str, QEMDataset] = {}

    for family in ("pauli", "brickwall"):
        for split, n_circ in (("train", n_train), ("test", n_test)):
            key  = f"{family}_{split}"
            path = root / f"{key}.npz"

            print(f"\n── Generating {key} ({n_circ} circuits, "
                  f"n={n_qubits}) ──────────────")
            t0 = time.time()
            ds = generate_dataset(n_circ, circuit_type=family,     # type: ignore[arg-type]
                                   verbose=verbose, cfg=c)
            print(f"   Done in {time.time() - t0:.1f}s  →  saving …")
            ds.save(path)
            datasets[key] = ds

    # Build combined from freshly generated singles
    print("\n── Building combined_train ────────────────────────────────────")
    datasets["combined_train"] = combine_datasets(
        datasets["pauli_train"], datasets["brickwall_train"]
    )
    datasets["combined_train"].save(root / "combined_train.npz")

    print("\n── Building combined_test ─────────────────────────────────────")
    datasets["combined_test"] = combine_datasets(
        datasets["pauli_test"], datasets["brickwall_test"]
    )
    datasets["combined_test"].save(root / "combined_test.npz")

    return datasets


def load_all_datasets(
    data_dir:  Path | str = "datasets",
    n_qubits:  int | None = None,
    cfg:       dict | None = None,
) -> dict[str, QEMDataset]:
    """
    Load all six datasets from disk.

    Parameters
    ----------
    data_dir : root directory (same as used in generate_all_datasets)
    n_qubits : number of qubits; inferred from cfg if not given
    cfg      : experiment config

    Returns
    -------
    dict with keys: pauli_train, pauli_test, brickwall_train,
                    brickwall_test, combined_train, combined_test

    Raises
    ------
    FileNotFoundError if any dataset file is missing
    """
    c        = cfg or CFG
    n_q      = n_qubits or c["n_qubits"]
    root     = Path(data_dir) / f"n{n_q}"

    keys = [
        "pauli_train", "pauli_test",
        "brickwall_train", "brickwall_test",
        "combined_train", "combined_test",
    ]

    missing = [k for k in keys if not (root / f"{k}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {root}: {missing}\n"
            "Run generate_all_datasets() or use --regenerate in main.py."
        )

    return {key: QEMDataset.load(root / f"{key}.npz") for key in keys}


def datasets_exist(
    data_dir:  Path | str = "datasets",
    n_qubits:  int | None = None,
    cfg:       dict | None = None,
) -> bool:
    """Return True if all six dataset files are present on disk."""
    c    = cfg or CFG
    n_q  = n_qubits or c["n_qubits"]
    root = Path(data_dir) / f"n{n_q}"
    keys = ["pauli_train", "pauli_test", "brickwall_train",
            "brickwall_test", "combined_train", "combined_test"]
    return all((root / f"{k}.npz").exists() for k in keys)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _small_cfg = {**CFG, "shadow_size": 50, "n_train": 6, "n_test": 4}

    print("── Testing generate_dataset (pauli) ──")
    ds_p = generate_dataset(4, "pauli", verbose=True, cfg=_small_cfg)
    print(ds_p)
    assert ds_p.circuit_type == "pauli"
    assert ds_p.X.shape[0] == 4
    print("PASS\n")

    print("── Testing generate_dataset (brickwall) ──")
    ds_b = generate_dataset(4, "brickwall", verbose=True, cfg=_small_cfg)
    print(ds_b)
    assert ds_b.circuit_type == "brickwall"
    print("PASS\n")

    print("── Testing combine_datasets ──")
    ds_c = combine_datasets(ds_p, ds_b)
    print(ds_c)
    assert ds_c.circuit_type == "combined"
    assert ds_c.n_circuits == 8
    print("PASS\n")

    print("── Testing save / load ──")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "n3" / "pauli_train.npz"
        ds_p.save(save_path)
        ds_loaded = QEMDataset.load(save_path)
        assert np.allclose(ds_p.X, ds_loaded.X)
        assert ds_loaded.circuit_type == "pauli"
        assert ds_loaded.n_qubits == ds_p.n_qubits
    print("PASS\n")

    print("All data.py tests: PASS")
