"""
PyTorch neural network models for ML-QEM Classical Shadows.

Three architectures in increasing sophistication:

    ┌──────────────────────┬─────────────────────────────────────────────────┐
    │ MLPCorrectionNet     │ Learns a multiplicative correction mask         │
    │                      │ c(noise_metadata) ∈ ℝ^M applied to noisy       │
    │                      │ shadow estimates.  Input: 3-dim noise metadata. │
    ├──────────────────────┼─────────────────────────────────────────────────┤
    │ MLPPredictionNet     │ Direct mapping (noisy_estimates + metadata) →   │
    │                      │ mitigated_estimates.  Full input vector.        │
    ├──────────────────────┼─────────────────────────────────────────────────┤
    │ AttentionMLPNet      │ Trainable multi-head self-attention over        │
    │                      │ observable tokens, followed by a MLP head.      │
    │                      │ Captures cross-observable correlations.         │
    └──────────────────────┴─────────────────────────────────────────────────┘

Each nn.Module is wrapped in a sklearn-compatible estimator (QEMEstimator)
so the rest of the pipeline (benchmark, plots, main.py) continues to work
without changes.

References
----------
- Placidi et al. (2026) §3 — MLP-CORRECTION and MLP-PREDICTION
- Vaswani et al. (2017)    — Attention is all you need
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp_block(in_dim: int,
               out_dim: int,
               dropout: float,
               activation: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        activation,
        nn.Dropout(dropout),
    )


def _build_mlp(dims: list[int], dropout: float,
               activation: nn.Module) -> nn.Sequential:
    """Stack of MLP blocks with a final linear projection (no norm/dropout)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(_mlp_block(dims[i], dims[i + 1], dropout, activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model A — MLPCorrectionNet
# ---------------------------------------------------------------------------

class MLPCorrectionNet(nn.Module):
    """
    Predicts a multiplicative correction mask c ∈ ℝ^M from noise metadata.

        ŷ_mit = c(x_meta) ⊙ ŷ_noisy

    Input  : (batch, 3)      — noise metadata (p_1q, p_2q, α)
    Output : (batch, M_obs)  — correction factors
    """

    def __init__(self,
                 m_obs:        int,
                 hidden_dims:  Sequence[int] = (256, 128, 64),
                 dropout:      float = 0.1,
                 activation:   str = "relu") -> None:
        super().__init__()
        self.m_obs = m_obs
        act = _resolve_activation(activation)
        dims = [3, *hidden_dims, m_obs]
        self.net = _build_mlp(dims, dropout, act)

    def forward(self, x_meta: torch.Tensor) -> torch.Tensor:
        """Return correction factors c(x_meta)."""
        return self.net(x_meta)


# ---------------------------------------------------------------------------
# Model B — MLPPredictionNet
# ---------------------------------------------------------------------------

class MLPPredictionNet(nn.Module):
    """
    Direct regression: (noisy_estimates, noise_metadata) → mitigated_estimates.

    Input  : (batch, M_obs + 3)  — full feature vector
    Output : (batch, M_obs)      — mitigated observable expectations
    """

    def __init__(self,
                 in_dim:       int,
                 m_obs:        int,
                 hidden_dims:  Sequence[int] = (512, 256, 128, 64),
                 dropout:      float = 0.15,
                 activation:   str = "tanh") -> None:
        super().__init__()
        act = _resolve_activation(activation)
        dims = [in_dim, *hidden_dims, m_obs]
        self.net = _build_mlp(dims, dropout, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Model C — AttentionMLPNet
# ---------------------------------------------------------------------------

class ObservableAttention(nn.Module):
    """
    Multi-head self-attention treating each observable feature as a token.

    The M_obs noisy estimates are reshaped into M_obs tokens of dim 1,
    projected to d_model, attended, then projected back.  Noise metadata
    is concatenated as a global context embedding.

    Input  : (batch, M_obs + 3)
    Output : (batch, M_obs + 3)   — residual-connected attended features
    """

    def __init__(self,
                 m_obs:    int,
                 d_model:  int = 64,
                 n_heads:  int = 4,
                 dropout:  float = 0.1) -> None:
        super().__init__()
        self.m_obs   = m_obs
        self.d_model = d_model

        # Project each scalar observable feature to d_model
        self.token_proj  = nn.Linear(1, d_model)
        # Learnable positional embedding per observable slot
        self.pos_embed   = nn.Embedding(m_obs, d_model)
        # Multi-head self-attention
        self.attn        = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.norm        = nn.LayerNorm(d_model)
        # Project back to scalar per observable
        self.out_proj    = nn.Linear(d_model, 1)
        # Metadata MLP (3 → M_obs + 3, kept for residual)
        self.meta_proj   = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, M_obs + 3)
        Returns enriched features of the same shape.
        """
        obs  = x[:, : self.m_obs]          # (batch, M_obs)
        meta = x[:, self.m_obs :]          # (batch, 3)

        # Tokenise: (batch, M_obs, 1) → (batch, M_obs, d_model)
        tokens = self.token_proj(obs.unsqueeze(-1))
        pos    = self.pos_embed(
            torch.arange(self.m_obs, device=x.device)
        ).unsqueeze(0)                      # (1, M_obs, d_model)
        tokens = tokens + pos

        # Self-attention with residual + layer norm
        attended, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attended)

        # Project back: (batch, M_obs, d_model) → (batch, M_obs)
        obs_out  = self.out_proj(tokens).squeeze(-1)
        meta_out = self.meta_proj(meta)

        return torch.cat([obs_out, meta_out], dim=-1)


class AttentionMLPNet(nn.Module):
    """
    Two-stage model:
        1. ObservableAttention — cross-observable context via MHA
        2. MLP head — regression to mitigated expectations

    Input  : (batch, M_obs + 3)
    Output : (batch, M_obs)
    """

    def __init__(self,
                 m_obs:       int,
                 d_model:     int = 64,
                 n_heads:     int = 4,
                 hidden_dims: Sequence[int] = (512, 256, 128),
                 dropout:     float = 0.15,
                 activation:  str = "tanh") -> None:
        super().__init__()
        in_dim = m_obs + 3
        act = _resolve_activation(activation)
        self.attention = ObservableAttention(m_obs, d_model, n_heads, dropout)
        dims = [in_dim, *hidden_dims, m_obs]
        self.mlp = _build_mlp(dims, dropout, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = self.attention(x)
        return self.mlp(x_attn)


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

def _resolve_activation(name: str) -> nn.Module:
    _map = {
        "relu":    nn.ReLU(),
        "tanh":    nn.Tanh(),
        "gelu":    nn.GELU(),
        "silu":    nn.SiLU(),
        "leaky":   nn.LeakyReLU(0.1),
        "elu":     nn.ELU(),
    }
    name = name.lower()
    if name not in _map:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_map)}")
    return _map[name]


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class QEMEstimator:
    """
    Thin sklearn-style wrapper around any QEM nn.Module.

    Handles:
      - Input/output normalisation (StandardScaler-equivalent via tensors)
      - Training delegation to Trainer (set externally)
      - predict() method compatible with benchmark.py and plots.py
    """

    def __init__(self,
                 model:  nn.Module,
                 m_obs:  int,
                 device: str | None = None) -> None:
        self.model  = model
        self.m_obs  = m_obs
        self.device = _get_device(device)

        # Normalisation statistics (filled during fit)
        self._x_mean:  torch.Tensor | None = None
        self._x_std:   torch.Tensor | None = None
        self._y_mean:  torch.Tensor | None = None
        self._y_std:   torch.Tensor | None = None

        self.model.to(self.device)

    # ── normalisation ────────────────────────────────────────────────────

    def _fit_normalise(self, X: np.ndarray, Y: np.ndarray) -> None:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32)
        self._x_mean = X_t.mean(0)
        self._x_std  = X_t.std(0).clamp(min=1e-8)
        self._y_mean = Y_t.mean(0)
        self._y_std  = Y_t.std(0).clamp(min=1e-8)

    def _normalise_X(self, X: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(X, dtype=torch.float32)
        return (t - self._x_mean) / self._x_std

    def _denormalise_Y(self, Y_norm: torch.Tensor) -> np.ndarray:
        return (Y_norm * self._y_std + self._y_mean).numpy()

    # ── public interface ─────────────────────────────────────────────────

    def get_datasets(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     val_split: float = 0.15,
                     ) -> tuple[TensorDataset, TensorDataset]:
        """
        Return normalised (train_dataset, val_dataset) ready for DataLoader.
        Call this *after* fit_normalise has been called.
        """
        X_norm = self._normalise_X(X)
        Y_t    = torch.as_tensor(Y, dtype=torch.float32)
        Y_norm = (Y_t - self._y_mean) / self._y_std

        n_val = max(1, int(len(X_norm) * val_split))
        idx   = torch.randperm(len(X_norm))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        tr_ds  = TensorDataset(X_norm[tr_idx],  Y_norm[tr_idx])
        val_ds = TensorDataset(X_norm[val_idx], Y_norm[val_idx])
        return tr_ds, val_ds

    def fit(self, X: np.ndarray, Y: np.ndarray,
            trainer: "Trainer | None" = None,  # imported lazily to avoid circular
            **trainer_kwargs) -> "QEMEstimator":
        """
        Convenience: fit normalisation stats then run the Trainer.
        If `trainer` is None, a default Trainer is created from trainer_kwargs.
        """
        from .trainer import Trainer

        self._fit_normalise(X, Y)
        tr_ds, val_ds = self.get_datasets(X, Y)

        if trainer is None:
            trainer = Trainer(self, **trainer_kwargs)
        trainer.fit(tr_ds, val_ds)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_norm = self._normalise_X(X).to(self.device)
            # For MLPCorrectionNet, forward returns correction factors;
            # we need noisy_exp to apply them.
            if isinstance(self.model, MLPCorrectionNet):
                noisy_exp = torch.as_tensor(
                    X[:, : self.m_obs], dtype=torch.float32
                ).to(self.device)
                correction = self.model(X_norm[:, self.m_obs:])
                out = correction * noisy_exp
            else:
                out = self.model(X_norm)
            return self._denormalise_Y(out.cpu())

    # ── sklearn score shims ──────────────────────────────────────────────

    def score_mae(self, X: np.ndarray, Y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(self.predict(X) - Y_true)))

    def score_rmse(self, X: np.ndarray, Y_true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((self.predict(X) - Y_true) ** 2)))
