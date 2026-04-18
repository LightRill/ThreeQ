# DThreeQ Two Moons Screen

First-stage feasibility screen for the DThreeQ idea described in
`Threeq_Dplus_VS_EP.pdf`.

This suite intentionally starts with a smaller but still comparable two moons
screen before launching MNIST/FashionMNIST and architecture ablations.

Screen settings:

- dataset: two moons, 2000 samples, 20% test split
- hidden sizes: `[128, 64]`
- batch size: 32
- epochs: 30
- inference steps: 15
- activation: tanh
- variants: Dplus direct/nudge/gradual, EP nudge/gradual, BP baseline
- learning rates: `1e-3`, `1e-4`, `5e-5`, `1e-5`
- seeds: `42`, `2025`, `7`

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate radas-3.11
cd /workspace/ThreeQ
python -u experiments/dthreeq_suite/run_experiment.py
```

Restore/materialize:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate radas-3.11
cd /workspace/ThreeQ
python -u experiments/dthreeq_suite/materialize_results.py
```

