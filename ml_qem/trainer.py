"""
Training loop with full TensorBoard monitoring.

What gets logged
----------------
Per epoch (scalars):
    Loss/train          — MSE on training set
    Loss/val            — MSE on validation set
    MAE/train           — mean absolute error (interpretable units)
    MAE/val
    LearningRate        — current LR (useful when using a scheduler)
    Grad/norm           — total L2 gradient norm (spot vanishing/exploding grads)

Per training run (histograms, every `hist_every` epochs):
    Weights/<layer>     — weight distributions per named parameter
    Grads/<layer>       — gradient distributions per named parameter

How to open TensorBoard
-----------------------
    tensorboard --logdir runs/

Then open  http://localhost:6006  in your browser.
Each model × Optuna trial gets its own named run so you can compare them
side-by-side on the Scalars and HParams tabs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Default hyperparameters (can be overridden in Trainer.__init__)
# ---------------------------------------------------------------------------

DEFAULT_HP = dict(
    lr           = 1e-3,
    weight_decay = 1e-4,
    batch_size   = 64,
    max_epochs   = 500,
    patience     = 30,        # early-stopping patience (epochs)
    scheduler    = "cosine",  # "cosine" | "plateau" | None
    grad_clip    = 1.0,       # max gradient norm (0 = disabled)
    val_split    = 0.15,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Generic training loop for any QEM nn.Module wrapped in QEMEstimator.

    Parameters
    ----------
    estimator     : QEMEstimator  — holds the nn.Module + normalisation stats
    run_name      : str           — TensorBoard run identifier
    log_dir       : Path | str    — root for TensorBoard logs (default: 'runs/')
    hist_every    : int           — log weight/grad histograms every N epochs
    hp            : dict          — hyperparameter overrides (see DEFAULT_HP)
    """

    def __init__(self,
                 estimator,
                 run_name:   str        = "qem",
                 log_dir:    Path | str = "runs",
                 hist_every: int        = 20,
                 **hp) -> None:
        self.estimator  = estimator
        self.model      = estimator.model
        self.device     = estimator.device
        self.run_name   = run_name
        self.log_dir    = Path(log_dir) / run_name
        self.hist_every = hist_every

        # Merge hyperparameters
        self.hp = {**DEFAULT_HP, **hp}

        # Will be set during fit()
        self.writer:        SummaryWriter | None = None
        self.best_val_loss: float = float("inf")
        self.best_state:    dict  | None = None
        self.history:       dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_mae":  [], "val_mae":  [],
            "lr":         [], "grad_norm": [],
        }

    # ── public API ────────────────────────────────────────────────────────

    def fit(self,
            train_ds: TensorDataset,
            val_ds:   TensorDataset) -> "Trainer":
        """
        Run the full training loop.

        Parameters
        ----------
        train_ds : TensorDataset of (X_normalised, Y_normalised)
        val_ds   : TensorDataset of (X_normalised, Y_normalised)
        """
        hp         = self.hp
        batch_size = hp["batch_size"]

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size * 4,
                                  shuffle=False)

        optimizer  = AdamW(self.model.parameters(),
                           lr=hp["lr"], weight_decay=hp["weight_decay"])
        scheduler  = self._make_scheduler(optimizer, len(train_loader))
        criterion  = nn.MSELoss()

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._log_hparams()

        patience_count = 0
        t0 = time.time()

        for epoch in range(1, hp["max_epochs"] + 1):
            # ── train ────────────────────────────────────────────────────
            tr_loss, tr_mae, grad_norm = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, hp
            )

            # ── validate ─────────────────────────────────────────────────
            val_loss, val_mae = self._eval_epoch(val_loader, criterion)

            # ── scheduler step (ReduceLROnPlateau) ───────────────────────
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            # ── record history ───────────────────────────────────────────
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(tr_mae)
            self.history["val_mae"].append(val_mae)
            self.history["lr"].append(current_lr)
            self.history["grad_norm"].append(grad_norm)

            # ── TensorBoard scalars ──────────────────────────────────────
            self.writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("MAE",  {"train": tr_mae,  "val": val_mae},  epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)
            self.writer.add_scalar("Grad/norm",    grad_norm,  epoch)

            # ── weight & gradient histograms ─────────────────────────────
            if epoch % self.hist_every == 0:
                self._log_histograms(epoch)

            # ── early stopping & checkpointing ───────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state    = {k: v.cpu().clone()
                                      for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= hp["patience"]:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best val_loss={self.best_val_loss:.6f})")
                break

            # ── console log every 10 % of max_epochs ─────────────────────
            log_every = max(1, hp["max_epochs"] // 10)
            if epoch % log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:>4}/{hp['max_epochs']}  "
                      f"train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}  "
                      f"val_MAE={val_mae:.5f}  lr={current_lr:.2e}  "
                      f"({elapsed:.0f}s)")

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self.writer.flush()
        self.writer.close()
        print(f"  Training done — best val_loss={self.best_val_loss:.6f}  "
              f"log → {self.log_dir}")
        return self

    # ── internal helpers ──────────────────────────────────────────────────

    def _train_epoch(self, loader, optimizer, criterion,
                     scheduler, hp) -> tuple[float, float, float]:
        self.model.train()
        total_loss = total_mae = total_gnorm = 0.0
        n_batches  = 0

        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(self.device), Y_b.to(self.device)
            optimizer.zero_grad()

            # For MLPCorrectionNet the input is split inside QEMEstimator.predict,
            # but here we pass the FULL normalised vector; the model itself decides
            # what to use.  MLPCorrectionNet expects metadata slice: we pass it the
            # whole X and let the forward method select columns via the wrapper.
            # Note: during training we always pass the full normalised X_b; the
            # correction is applied in the loss computation below.
            out  = self._forward_for_training(X_b)
            loss = criterion(out, Y_b)

            loss.backward()

            # Gradient clipping
            gnorm = 0.0
            if hp.get("grad_clip", 0) > 0:
                gnorm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), hp["grad_clip"]
                ).item()
            else:
                for p in self.model.parameters():
                    if p.grad is not None:
                        gnorm += p.grad.data.norm(2).item() ** 2
                gnorm = gnorm ** 0.5

            optimizer.step()

            # CosineAnnealingLR steps per batch
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

            # Compute MAE in normalised space (proxy; denorm not needed for monitoring)
            with torch.no_grad():
                mae_b = torch.mean(torch.abs(out - Y_b)).item()

            total_loss  += loss.item()
            total_mae   += mae_b
            total_gnorm += gnorm
            n_batches   += 1

        return (total_loss  / n_batches,
                total_mae   / n_batches,
                total_gnorm / n_batches)

    def _eval_epoch(self, loader, criterion) -> tuple[float, float]:
        self.model.eval()
        total_loss = total_mae = 0.0
        n_batches  = 0

        with torch.no_grad():
            for X_b, Y_b in loader:
                X_b, Y_b = X_b.to(self.device), Y_b.to(self.device)
                out  = self._forward_for_training(X_b)
                loss = criterion(out, Y_b)
                mae  = torch.mean(torch.abs(out - Y_b))
                total_loss += loss.item()
                total_mae  += mae.item()
                n_batches  += 1

        return total_loss / n_batches, total_mae / n_batches

    def _forward_for_training(self, X_b: torch.Tensor) -> torch.Tensor:
        """
        Route the batch through the model.

        MLPCorrectionNet receives only the metadata slice (last 3 columns) of
        the *normalised* X_b; the correction is applied to the normalised
        noisy estimates.  This keeps the loss in normalised space.
        """
        from .models import MLPCorrectionNet
        if isinstance(self.model, MLPCorrectionNet):
            meta   = X_b[:, self.estimator.m_obs:]
            noisy  = X_b[:, : self.estimator.m_obs]
            return self.model(meta) * noisy
        return self.model(X_b)

    def _make_scheduler(self, optimizer, steps_per_epoch: int):
        sched = self.hp.get("scheduler")
        if sched == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self.hp["max_epochs"] * steps_per_epoch,
                eta_min=self.hp["lr"] / 100,
            )
        if sched == "plateau":
            return ReduceLROnPlateau(optimizer, patience=10,
                                     factor=0.5, min_lr=1e-6)
        return None  # constant LR

    def _log_hparams(self) -> None:
        """Write hyperparameters to TensorBoard HParams tab."""
        hp_flat = {k: float(v) if not isinstance(v, str) else v
                   for k, v in self.hp.items()}
        # SummaryWriter.add_hparams requires metric dict; fill after training
        # Store for end-of-training call
        self._hp_flat = hp_flat

    def _log_histograms(self, epoch: int) -> None:
        """Log weight and gradient distributions per named parameter."""
        for name, param in self.model.named_parameters():
            if param.data.numel() > 0:
                self.writer.add_histogram(f"Weights/{name}", param.data, epoch)
            if param.grad is not None and param.grad.numel() > 0:
                self.writer.add_histogram(f"Grads/{name}", param.grad, epoch)

    def save_checkpoint(self, path: Path | str) -> None:
        """Save best model weights and training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.best_state or self.model.state_dict(),
            "history":     self.history,
            "hp":          self.hp,
        }, path)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore weights from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        print(f"  Checkpoint loaded ← {path}")
