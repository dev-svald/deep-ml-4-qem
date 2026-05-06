"""
Hyperparameter tuning with Optuna.

Each Optuna trial trains a fresh model with a sampled configuration,
logging to its own TensorBoard sub-directory so every trial's curves
can be compared side-by-side.

After tuning, the best hyperparameters are printed and can be passed
directly to the Trainer for a final full-budget training run.

Usage (standalone)
------------------
    python -m ml_qem.tuning --model mlp_pred --n-trials 50

Usage (from code)
-----------------
    from ml_qem.tuning import tune

    best_hp, study = tune(
        model_type = "mlp_pred",   # or "mlp_corr" | "attn_mlp"
        X_train    = X_train,
        Y_train    = Y_train,
        X_val      = X_val,
        Y_val      = Y_val,
        n_trials   = 40,
        log_dir    = "runs/hpo",
    )

Search spaces
-------------
    Shared across all models:
        lr              log-uniform  [1e-4, 5e-3]
        weight_decay    log-uniform  [1e-6, 1e-2]
        batch_size      categorical  [32, 64, 128, 256]
        dropout         uniform      [0.0, 0.3]
        activation      categorical  [relu, tanh, gelu, silu]
        scheduler       categorical  [cosine, plateau]

    mlp_corr / mlp_pred:
        n_layers        int          [2, 5]
        hidden_dim      categorical  [64, 128, 256, 512]

    attn_mlp:
        d_model         categorical  [32, 64, 128]
        n_heads         categorical  [2, 4, 8]
        n_layers        int          [2, 4]
        hidden_dim      categorical  [128, 256, 512]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .models import (
    MLPCorrectionNet,
    MLPPredictionNet,
    AttentionMLPNet,
    QEMEstimator,
)
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Per-trial objective
# ---------------------------------------------------------------------------

def _objective(
    trial:      optuna.Trial,
    model_type: str,
    X_train:    np.ndarray,
    Y_train:    np.ndarray,
    X_val:      np.ndarray,
    Y_val:      np.ndarray,
    m_obs:      int,
    log_dir:    Path,
    max_epochs: int,
) -> float:
    """
    Optuna objective: returns best validation loss for a single trial.
    Lower is better (MSE in normalised space).
    """
    # ── shared hyperparameters ───────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout      = trial.suggest_float("dropout", 0.0, 0.30)
    activation   = trial.suggest_categorical("activation",
                                              ["relu", "tanh", "gelu", "silu"])
    scheduler    = trial.suggest_categorical("scheduler",
                                              ["cosine", "plateau"])

    # ── model-specific hyperparameters ───────────────────────────────────
    in_dim = X_train.shape[1]

    if model_type == "mlp_corr":
        n_layers   = trial.suggest_int("n_layers", 2, 5)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        hidden_dims = tuple([hidden_dim] * n_layers)
        net = MLPCorrectionNet(
            m_obs       = m_obs,
            hidden_dims = hidden_dims,
            dropout     = dropout,
            activation  = activation,
        )

    elif model_type == "mlp_pred":
        n_layers   = trial.suggest_int("n_layers", 2, 5)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        hidden_dims = tuple([hidden_dim] * n_layers)
        net = MLPPredictionNet(
            in_dim      = in_dim,
            m_obs       = m_obs,
            hidden_dims = hidden_dims,
            dropout     = dropout,
            activation  = activation,
        )

    elif model_type == "attn_mlp":
        d_model    = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads    = trial.suggest_categorical("n_heads", [2, 4, 8])
        # n_heads must divide d_model
        while d_model % n_heads != 0:
            n_heads = n_heads // 2 if n_heads > 1 else 1
            trial.set_user_attr("n_heads_adjusted", n_heads)
        n_layers   = trial.suggest_int("n_layers", 2, 4)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        hidden_dims = tuple([hidden_dim] * n_layers)
        net = AttentionMLPNet(
            m_obs       = m_obs,
            d_model     = d_model,
            n_heads     = n_heads,
            hidden_dims = hidden_dims,
            dropout     = dropout,
            activation  = activation,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # ── estimator + trainer ──────────────────────────────────────────────
    est = QEMEstimator(net, m_obs=m_obs)
    est._fit_normalise(X_train, Y_train)

    # Combine train+val for dataset splitting (trainer handles the val split)
    X_all = np.concatenate([X_train, X_val], axis=0)
    Y_all = np.concatenate([Y_train, Y_val], axis=0)

    # Fixed val fraction relative to combined set
    val_frac = len(X_val) / len(X_all)
    tr_ds, val_ds = est.get_datasets(X_all, Y_all, val_split=val_frac)

    run_name = f"{model_type}/trial_{trial.number:04d}"
    trainer  = Trainer(
        estimator    = est,
        run_name     = run_name,
        log_dir      = log_dir,
        hist_every   = max(max_epochs, 9999),  # disable histograms during HPO
        lr           = lr,
        weight_decay = weight_decay,
        batch_size   = int(batch_size),
        max_epochs   = max_epochs,
        patience     = max(10, max_epochs // 10),
        scheduler    = scheduler,
        dropout      = dropout,
    )

    # Integrate Optuna pruning: report val_loss each epoch via a callback
    class _PruningCallback:
        def __init__(self, _trial: optuna.Trial) -> None:
            self._trial = _trial

        def __call__(self, epoch: int, val_loss: float) -> None:
            self._trial.report(val_loss, epoch)
            if self._trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    trainer.fit(tr_ds, val_ds)
    return trainer.best_val_loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune(
    model_type: str,
    X_train:    np.ndarray,
    Y_train:    np.ndarray,
    X_val:      np.ndarray,
    Y_val:      np.ndarray,
    n_trials:   int   = 40,
    max_epochs: int   = 150,     # reduced budget per trial
    log_dir:    Path | str = "runs/hpo",
    study_name: str   = "qem_hpo",
    direction:  str   = "minimize",
    seed:       int   = 42,
    verbose:    bool  = True,
) -> tuple[dict, optuna.Study]:
    """
    Run an Optuna hyperparameter search for the specified model type.

    Parameters
    ----------
    model_type  : "mlp_corr" | "mlp_pred" | "attn_mlp"
    X_train     : (N_tr, M_obs + 3) training features
    Y_train     : (N_tr, M_obs)     training labels
    X_val       : (N_v,  M_obs + 3) validation features
    Y_val       : (N_v,  M_obs)     validation labels
    n_trials    : number of Optuna trials
    max_epochs  : epochs per trial (use a fraction of full training budget)
    log_dir     : root directory for per-trial TensorBoard logs
    study_name  : Optuna study name (used for SQLite storage if extended)
    direction   : "minimize" (val loss)
    seed        : random seed for reproducibility
    verbose     : show Optuna progress bar and trial summaries

    Returns
    -------
    (best_params dict, optuna.Study)
    """
    log_dir = Path(log_dir)
    m_obs   = Y_train.shape[1]

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=seed)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study   = optuna.create_study(
        study_name = study_name,
        direction  = direction,
        sampler    = sampler,
        pruner     = pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        return _objective(
            trial      = trial,
            model_type = model_type,
            X_train    = X_train,
            Y_train    = Y_train,
            X_val      = X_val,
            Y_val      = Y_val,
            m_obs      = m_obs,
            log_dir    = log_dir,
            max_epochs = max_epochs,
        )

    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=verbose,
                   catch=(Exception,))

    best = study.best_params

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Best trial:  #{study.best_trial.number}")
        print(f"  Val loss  :  {study.best_value:.6f}")
        print("  Best hyperparameters:")
        for k, v in best.items():
            print(f"    {k:<18} = {v}")
        print(f"\n  TensorBoard logs → {log_dir.resolve()}")
        print(f"  Run:  tensorboard --logdir {log_dir.resolve()}")
        print("=" * 60)

    return best, study


def retrain_best(
    model_type: str,
    best_hp:    dict,
    X_train:    np.ndarray,
    Y_train:    np.ndarray,
    m_obs:      int,
    max_epochs: int   = 500,
    log_dir:    Path | str = "runs/final",
    ckpt_path:  Path | str | None = None,
) -> QEMEstimator:
    """
    Re-train from scratch using the best hyperparameters found by `tune()`,
    now with the full epoch budget and histogram logging enabled.

    Parameters
    ----------
    model_type : "mlp_corr" | "mlp_pred" | "attn_mlp"
    best_hp    : dict returned by tune()
    X_train    : (N, M_obs + 3) full training features
    Y_train    : (N, M_obs)     full training labels
    m_obs      : number of observables
    max_epochs : full training budget
    log_dir    : TensorBoard log directory for the final run
    ckpt_path  : if provided, save best model weights here

    Returns
    -------
    Trained QEMEstimator ready for .predict()
    """
    hp        = best_hp
    in_dim    = X_train.shape[1]
    dropout   = hp.get("dropout", 0.1)
    activation = hp.get("activation", "tanh")
    n_layers  = hp.get("n_layers", 3)
    hidden_dim = hp.get("hidden_dim", 256)
    hidden_dims = tuple([hidden_dim] * n_layers)

    if model_type == "mlp_corr":
        net = MLPCorrectionNet(m_obs, hidden_dims, dropout, activation)
    elif model_type == "mlp_pred":
        net = MLPPredictionNet(in_dim, m_obs, hidden_dims, dropout, activation)
    elif model_type == "attn_mlp":
        d_model = hp.get("d_model", 64)
        n_heads = hp.get("n_heads", 4)
        while d_model % n_heads != 0:
            n_heads = n_heads // 2 or 1
        net = AttentionMLPNet(m_obs, d_model, n_heads, hidden_dims, dropout, activation)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    est     = QEMEstimator(net, m_obs=m_obs)
    trainer = Trainer(
        estimator    = est,
        run_name     = f"{model_type}/best",
        log_dir      = Path(log_dir),
        hist_every   = 20,
        lr           = hp.get("lr", 1e-3),
        weight_decay = hp.get("weight_decay", 1e-4),
        batch_size   = int(hp.get("batch_size", 64)),
        max_epochs   = max_epochs,
        patience     = 40,
        scheduler    = hp.get("scheduler", "cosine"),
    )

    est._fit_normalise(X_train, Y_train)
    tr_ds, val_ds = est.get_datasets(X_train, Y_train, val_split=0.15)
    trainer.fit(tr_ds, val_ds)

    if ckpt_path is not None:
        trainer.save_checkpoint(ckpt_path)

    return est


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="QEM hyperparameter tuning with Optuna")
    p.add_argument("--model",    choices=["mlp_corr", "mlp_pred", "attn_mlp"],
                   default="mlp_pred")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--epochs",   type=int, default=150,
                   help="Epochs per trial (use reduced budget, e.g. 150)")
    p.add_argument("--log-dir",  type=Path, default=Path("runs/hpo"))
    args = p.parse_args()

    print("Generating a small dataset for tuning demo …")
    from .config import CFG
    from .data   import generate_dataset

    ds    = generate_dataset(80,  verbose=True, cfg={**CFG, "shadow_size": 100})
    ds_v  = generate_dataset(20,  verbose=False, cfg={**CFG, "shadow_size": 100})
    m_obs = ds.Y.shape[1]

    best_hp, study = tune(
        model_type = args.model,
        X_train    = ds.X,
        Y_train    = ds.Y,
        X_val      = ds_v.X,
        Y_val      = ds_v.Y,
        n_trials   = args.n_trials,
        max_epochs = args.epochs,
        log_dir    = args.log_dir,
    )

    print("\nRe-training with best hyperparameters …")
    retrain_best(
        model_type = args.model,
        best_hp    = best_hp,
        X_train    = ds.X,
        Y_train    = ds.Y,
        m_obs      = m_obs,
        log_dir    = args.log_dir.parent / "final",
        ckpt_path  = Path("checkpoints") / f"{args.model}_best.pt",
    )

    print(f"\nDone! View results:\n  tensorboard --logdir {args.log_dir.parent}")


if __name__ == "__main__":
    _cli()
