"""
ml_qem — Machine-Learning for Quantum Error Mitigation via Classical Shadows.

Public API (import from here for convenience):

    from ml_qem import CFG
    from ml_qem.data import generate_dataset
    from ml_qem.models import MLPCorrectionModel, MLPPredictionModel, AttentionMLPModel
    from ml_qem.baselines import run_pec, run_zne
    from ml_qem.benchmark import run_benchmark, run_ablation
    from ml_qem.plots import plot_dataset_stats, plot_benchmark, plot_final_summary
"""

from .config import CFG

__all__ = ["CFG"]
