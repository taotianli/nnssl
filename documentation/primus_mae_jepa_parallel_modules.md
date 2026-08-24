# Parallel modules for MAE + JEPA + VICReg

These experiments test three independent mechanisms. They must not be stacked
in the first screen because each addresses a different source of negative
transfer.

| ID | Module | Intervention |
|---:|---|---|
| 0 | HOP-B8 | MAE reconstruction branches after encoder block 8; JEPA and VICReg use the final representation. |
| 1 | ELC-VIC | A flipped 3-D target view is aligned back to the original grid; local invariance, variance and covariance are applied with total weight 0.01. |
| 2 | DOA-R32 | Reconstruction and prediction receive separate zero-initialized rank-32 residual adapters. |

All three use the matched teacher-free recipe: public Primus-M MAE
initialization, terminal JEPA weight 0.005, VICReg weight 0.005, batch size 4,
and 200 epochs. As in the existing stabilized baseline, epochs 0–9 are
predictor-only with JEPA weight 1.0; epochs 10–29 linearly ramp the JEPA weight
to 0.005; later epochs keep 0.005. Extra modules are pretraining scaffolds;
downstream segmentation loads the unchanged `down_projection` and `eva`
encoder.

Submit all three jobs, requesting up to three GPUs:

```bash
sbatch --array=0-2 scripts/slurm/openmind_pretrain_mae_jepa_parallel_modules.slurm
```

Limit to two simultaneous GPUs:

```bash
sbatch --array=0-2%2 scripts/slurm/openmind_pretrain_mae_jepa_parallel_modules.slurm
```

The script uses 20 data-augmentation workers by default. Fresh runs load the
public MAE checkpoint; interrupted runs resume only from their isolated output
directory. ELC logs its total local loss plus the invariance, variance, and
covariance components separately for training and validation, which makes
projector collapse or an over-dominant covariance penalty visible.
