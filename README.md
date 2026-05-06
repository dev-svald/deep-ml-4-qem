# ML-QEM Classical Shadows

A Python project implementing a full research pipeline for **Machine-Learning Quantum Error Mitigation via Classical Shadows**.

Three ML models are benchmarked against physics-motivated baseline (PEC). Training is monitored in real time with **TensorBoard**, and hyperparameters are optimised with **Optuna**.

## References

- Huang et al. (2020) — *Predicting Many Properties of a Quantum System from Very Few Measurements*
- Jnane et al. (2023) — *Quantum Error Mitigated Classical Shadows*
- Placidi et al. (2026) — *Deep Learning Approaches to Quantum Error Mitigation*
- Liao et al. (2023) — *Machine Learning for Practical Quantum Error Mitigation*

---

## Project structure

```
ClassicalShadows/
├── main.py                   # End-to-end experiment runner (entry point)
├── requirements.txt
├── README.md
└── ml_qem/                   # Python package
    ├── __init__.py
    ├── config.py             # Global hyperparameters (CFG dict)
    ├── utils.py              # Pauli matrices, snapshot formula, observables, metrics
    ├── noise.py              # Depolarising channel, readout error, snap_with_readout
    ├── circuits.py           # PennyLane circuit builders (Pauli gadgets, brick-wall)
    ├── data.py               # Dataset generation, QEMDataset dataclass
    ├── baselines.py          # PEC shadow mitigation
    ├── models.py             # PyTorch nn.Module models + QEMEstimator wrapper
    ├── trainer.py            # Training loop with TensorBoard monitoring
    ├── tuning.py             # Optuna hyperparameter optimisation
    ├── benchmark.py          # Metrics (MAE, RMSE, L1RC), benchmark table, ablation
    ├── ibm_pipeline.py       # Qiskit circuit builders, AerSimulator + IBM hardware
    └── plots.py              # All matplotlib visualisations
```