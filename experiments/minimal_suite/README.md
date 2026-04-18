# Minimal Comparable Suite

This suite keeps the original ThreeQ directories intact and adds a shared
implementation for the smallest fair ablation on two moons.

Variants:

- `direct_cost_w10`: Base3Q-style direct weak objective, 10 weak steps.
- `direct_cost_w50`: Base3Q-style direct weak objective, 50 weak steps.
- `direct_linear_w10`: ClampWithLinear-style weak phase, 10 weak steps.
- `ep_cost_w2`: EP-style energy difference, 2 weak steps.
- `ep_cost_w10`: EP-style energy difference, 10 weak steps.
- `ep_cost_w50`: EP-style energy difference, 50 weak steps.
- `ep_tied_w10`: EP-style energy difference with strict transpose sharing.

Run with radas:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate radas-3.11
cd /workspace/ThreeQ/experiments/minimal_suite
python -u run_experiment.py
```

The script changes into the repository root before calling radas so the
canonical experiment id is stable:
`_workspace_ThreeQ_threeq_minimal_comparable_v1`.
It also pins the Ray job entrypoint to the cluster's `radas-cluster`
interpreter so the driver Python/Ray versions match the running cluster.

Restore/materialize:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate radas-3.11
cd /workspace/ThreeQ/experiments/minimal_suite
python -u materialize_results.py
```
