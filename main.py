"""
ML-QEM Classical Shadows — end-to-end experiment runner.

Usage
-----
    python main.py                          # full run; generate datasets if missing
    python main.py --quick                  # smoke-test with small config
    python main.py --regenerate             # force re-generation of all datasets
    python main.py --circuit-type pauli     # train/evaluate on Pauli circuits only
    python main.py --circuit-type brickwall # train/evaluate on brick-wall circuits only
    python main.py --circuit-type combined  # (default) train on both families
    python main.py --data-dir my_data/      # custom dataset storage directory
    python main.py --no-ibm                 # skip IBM/Qiskit backend section
    python main.py --out-dir results/       # save figures to a directory
    python main.py --tune --n-trials 50     # Optuna HPO before training

Dataset layout
--------------
    <data-dir>/
        n<n_qubits>/
            pauli_train.npz       brickwall_train.npz       combined_train.npz
            pauli_test.npz        brickwall_test.npz        combined_test.npz

    Datasets are generated once and reused. Use --regenerate to overwrite.

TensorBoard
-----------
    tensorboard --logdir runs/       →  http://localhost:6006
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── local package ────────────────────────────────────────────────────────────
from ml_qem.config import CFG
from ml_qem.utils  import build_observable_set
from ml_qem.data   import (
    generate_all_datasets,
    load_all_datasets,
    datasets_exist,
    QEMDataset,
)
from ml_qem.baselines import run_pec
from ml_qem.models import (
    MLPCorrectionNet,
    MLPPredictionNet,
    AttentionMLPNet,
    QEMEstimator,
)
from ml_qem.trainer import Trainer
from ml_qem.benchmark import run_benchmark, run_ablation
from ml_qem.plots import (
    plot_dataset_stats,
    plot_benchmark,
    plot_final_summary,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ML-QEM Classical Shadows experiment"
    )
    p.add_argument("--quick",        action="store_true",
                   help="Small run (50 train / 20 test / 500 shadows) for smoke-testing")
    p.add_argument("--regenerate",   action="store_true",
                   help="Force re-generation of all datasets even if files exist")
    p.add_argument("--circuit-type", choices=["pauli", "brickwall", "combined"],
                   default="combined",
                   help="Circuit family to train/evaluate on (default: combined)")
    p.add_argument("--data-dir",     type=Path, default=Path("datasets"),
                   help="Root directory for dataset files (default: datasets/)")
    p.add_argument("--no-ibm",       action="store_true",
                   help="Skip the Qiskit IBM pipeline section")
    p.add_argument("--out-dir",      type=Path,
                   default=Path("Some results"),
                   help="Directory to save output figures (default: 'Some results/')")
    p.add_argument("--n-train",      type=int, default=None,
                   help="Override n_train from config (number of training circuits)")
    p.add_argument("--n-test",       type=int, default=None,
                   help="Override n_test from config (number of test circuits)")
    p.add_argument("--shadow-size",  type=int, default=None,
                   help="Override shadow_size from config (snapshots per circuit)")
    p.add_argument("--tune",         action="store_true",
                   help="Run Optuna HPO before training each model")
    p.add_argument("--n-trials",     type=int, default=30,
                   help="Number of Optuna trials per model (default: 30)")
    p.add_argument("--log-dir",      type=Path, default=Path("runs"),
                   help="Root directory for TensorBoard logs (default: runs/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 0: configuration ─────────────────────────────────────────────
    cfg = dict(CFG)
    # Apply CLI overrides (highest priority: --quick, then --n-train/--n-test/--shadow-size)
    if args.quick:
        cfg.update(n_train=50, n_test=20, shadow_size=500)
    if args.n_train is not None:
        cfg["n_train"] = args.n_train
    if args.n_test is not None:
        cfg["n_test"] = args.n_test
    if args.shadow_size is not None:
        cfg["shadow_size"] = args.shadow_size

    n_qubits     = cfg["n_qubits"]
    max_locality = cfg.get("max_locality", 3)
    observables  = build_observable_set(n_qubits, max_locality=max_locality)
    M_obs        = len(observables)
    circuit_type = args.circuit_type
    data_dir     = args.data_dir

    shadow_bound_info = (
        f"shadow_size={cfg['shadow_size']}"
        f"  (theoretical bound: {cfg.get('shadow_size_bound', 'n/a')})"
    )

    print("=" * 65)
    print("  ML-QEM Classical Shadows")
    print("=" * 65)
    print(f"  n_qubits={n_qubits}  |  M_obs={M_obs}")
    print(f"  {shadow_bound_info}" + (" [--quick]" if args.quick else ""))
    print(f"  n_train={cfg['n_train']}  |  n_test={cfg['n_test']}  |  "
          f"circuit_type='{circuit_type}'")
    print(f"  p_depol={cfg['p_depol']}  |  p_2q={cfg['p_2q']}  |  "
          f"readout_err={cfg['readout_err']}")
    print(f"  data_dir={data_dir.resolve()}")
    print()

    # ── Part 1: datasets (generate once, reload on subsequent runs) ───────
    print("── Part 1: Datasets ──────────────────────────────────────────")

    need_generate = args.regenerate or not datasets_exist(data_dir, cfg=cfg)

    if need_generate:
        reason = "forced by --regenerate" if args.regenerate else "not found on disk"
        print(f"  Generating all datasets ({reason}) …")
        all_ds = generate_all_datasets(data_dir=data_dir, cfg=cfg, verbose=True)
    else:
        print(f"  Loading datasets from {data_dir.resolve()} …")
        all_ds = load_all_datasets(data_dir=data_dir, cfg=cfg)

    # Select train/test split according to --circuit-type
    train_ds: QEMDataset = all_ds[f"{circuit_type}_train"]
    test_ds:  QEMDataset = all_ds[f"{circuit_type}_test"]

    print(f"\n  Training on : {train_ds}")
    print(f"  Testing on  : {test_ds}")
    print()

    X_train, Y_train = train_ds.X, train_ds.Y
    X_test,  Y_test  = test_ds.X,  test_ds.Y
    y_noisy          = X_test[:, :M_obs]

    baseline_mae = float(np.mean(np.abs(y_noisy - Y_test)))
    print(f"  Baseline MAE (no mitigation, {circuit_type}): {baseline_mae:.4f}\n")

    # ── Part 2: dataset statistics ────────────────────────────────────────
    print("── Part 2: Dataset statistics ────────────────────────────────")
    plot_dataset_stats(Y_test, y_noisy, save_path=out_dir, circuit_type=circuit_type)

    # ── Part 3: PEC baseline ──────────────────────────────────────────────
    print("── Part 3: PEC baseline ──────────────────────────────────────")
    Y_pec     = run_pec(X_test, test_ds.observables, cfg)
    pec_mae   = float(np.mean(np.abs(Y_pec - Y_test)))
    print(f"   PEC MAE: {pec_mae:.4f}  (improvement: "
          f"{100*(baseline_mae - pec_mae)/baseline_mae:.1f}%)\n")

    log_dir    = args.log_dir
    max_epochs = 200 if args.quick else 500
    n_heads    = 2 if M_obs % 2 == 0 else 1   # safe default for attn head count

    def _make_estimator(model_type: str, best_hp: dict | None) -> QEMEstimator:
        hp = best_hp or {}
        n_layers   = int(hp.get("n_layers", 3))
        hidden_dim = int(hp.get("hidden_dim", 256))
        hidden_dims = tuple([hidden_dim] * n_layers)
        dropout     = float(hp.get("dropout", 0.10))
        activation  = hp.get("activation", "tanh")
        d_model     = int(hp.get("d_model", 64))
        nh          = int(hp.get("n_heads", n_heads))
        while d_model % nh != 0:
            nh = nh // 2 or 1

        if model_type == "mlp_corr":
            net = MLPCorrectionNet(M_obs, hidden_dims, dropout, activation)
        elif model_type == "mlp_pred":
            net = MLPPredictionNet(X_train.shape[1], M_obs,
                                   hidden_dims, dropout, activation)
        else:
            net = AttentionMLPNet(M_obs, d_model, nh,
                                  hidden_dims, dropout, activation)
        return QEMEstimator(net, m_obs=M_obs)

    def _train(model_type: str, label: str) -> QEMEstimator:
        print(f"── {label} {'─'*(52-len(label))}")
        best_hp: dict | None = None

        if args.tune:
            from ml_qem.tuning import tune
            print(f"   Running HPO ({args.n_trials} trials) …")
            best_hp, _ = tune(
                model_type = model_type,
                X_train    = X_train, Y_train = Y_train,
                X_val      = X_test,  Y_val   = Y_test,
                n_trials   = args.n_trials,
                max_epochs = max(50, max_epochs // 3),
                log_dir    = log_dir / "hpo",
                verbose    = False,
            )
            print(f"   Best HP: { {k: round(v, 5) if isinstance(v, float) else v for k, v in best_hp.items()} }")

        est     = _make_estimator(model_type, best_hp)
        hp      = best_hp or {}
        trainer = Trainer(
            estimator    = est,
            run_name     = model_type,
            log_dir      = log_dir,
            hist_every   = 20,
            lr           = float(hp.get("lr", 1e-3)),
            weight_decay = float(hp.get("weight_decay", 1e-4)),
            batch_size   = int(hp.get("batch_size", 64)),
            max_epochs   = max_epochs,
            patience     = 40,
            scheduler    = hp.get("scheduler", "cosine"),
        )
        t0 = time.time()
        est._fit_normalise(X_train, Y_train)
        tr_ds, val_ds = est.get_datasets(X_train, Y_train, val_split=0.15)
        trainer.fit(tr_ds, val_ds)
        trainer.save_checkpoint(Path("checkpoints") / f"{model_type}.pt")
        print(f"   MAE: {est.score_mae(X_test, Y_test):.4f}  ({time.time()-t0:.1f}s)\n")
        return est

    # ── Part 5: MLP-Correction ────────────────────────────────────────────
    mlp_corr   = _train("mlp_corr", "Part 5: MLP-Correction")
    Y_mlp_corr = mlp_corr.predict(X_test)

    # ── Part 6: MLP-Prediction ────────────────────────────────────────────
    mlp_pred   = _train("mlp_pred", "Part 6: MLP-Prediction")
    Y_mlp_pred = mlp_pred.predict(X_test)

    # ── Part 7: Attention-MLP ─────────────────────────────────────────────
    attn_mlp = _train("attn_mlp", "Part 7: Attention-MLP")
    Y_attn   = attn_mlp.predict(X_test)

    # ── Part 8: Benchmark ─────────────────────────────────────────────────
    print("── Part 8: Benchmark ─────────────────────────────────────────")
    predictions = {
        "No Mitigation":  y_noisy,
        "PEC Shadows":    Y_pec,
        "MLP-Correction": Y_mlp_corr,
        "MLP-Prediction": Y_mlp_pred,
        "Attention-MLP":  Y_attn,
    }
    results = run_benchmark(predictions, Y_test, y_noisy, print_table=True)

    print()
    run_ablation(
        models   = {"MLP-Correction": mlp_corr,   # type: ignore[dict-item]
                    "MLP-Prediction": mlp_pred,
                    "Attention-MLP":  attn_mlp},
        X_test   = X_test,
        y_ideal  = Y_test,
        m_obs    = M_obs,
        print_table = True,
    )

    plot_benchmark(results, Y_test, y_noisy, test_ds.observables,
                   save_path=out_dir, circuit_type=circuit_type)

    # ── Part 9: IBM / Qiskit pipeline ────────────────────────────────────
    if not args.no_ibm:
        print("\n── Part 9: IBM Qiskit pipeline (AerSimulator) ────────────────")
        try:
            from qiskit_aer import AerSimulator
            from ml_qem.ibm_pipeline import build_aer_noise_model, run_ibm_pipeline

            nm_qiskit     = build_aer_noise_model(
                n_qubits    = n_qubits,
                p_1q        = cfg["p_depol"],
                p_2q        = cfg["p_2q"],
                readout_err = cfg["readout_err"],
            )
            backend_noisy = AerSimulator(noise_model=nm_qiskit)
            backend_ideal = AerSimulator()

            run_ibm_pipeline(
                n_circuits    = 10 if not args.quick else 3,
                ml_model      = mlp_pred,
                backend       = backend_noisy,
                ideal_backend = backend_ideal,
                n_shots       = 150 if not args.quick else 50,
                cfg           = cfg,
                verbose       = True,
            )
        except ImportError:
            print("   qiskit_aer not installed — skipping IBM pipeline.")
            print("   Install with:  pip install qiskit-aer")
    else:
        print("\n[--no-ibm] Skipping IBM pipeline.")

    # ── Part 10: final summary ────────────────────────────────────────────
    print("\n── Part 10: Final summary ─────────────────────────────────────")
    plot_final_summary(results, Y_test, y_noisy, test_ds.observables,
                       save_path=out_dir, circuit_type=circuit_type)

    best = min(results, key=lambda m: results[m].mae_val)
    print()
    print("=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Circuit type: {circuit_type}")
    print(f"  Best method : {best}")
    print(f"  MAE         : {results[best].mae_val:.4f}  "
          f"(baseline {baseline_mae:.4f})")
    print(f"  Improvement : {100*(baseline_mae - results[best].mae_val)/baseline_mae:.1f}%")
    if out_dir:
        print(f"  Figures saved to : {out_dir.resolve()}")
    print(f"  Datasets at      : {data_dir.resolve()}")
    print(f"  TensorBoard logs : {log_dir.resolve()}")
    print(f"  To view TB:  tensorboard --logdir {log_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
