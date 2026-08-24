# Structure-conditioned latent bridge (SCLB)

SCLB keeps the paper's reconstruction + prediction story while making the two
objectives cooperate. The teacher-free online target is stabilized with the
current best VICReg setting (`0.005`). Masked patch predictions are pooled over
64 connected, crop-relative 3-D regions. A zero-initialized FiLM residual uses
the predicted region latent to condition the standard MAE decoder.

The main variant detaches the condition before FiLM. Consequently, the JEPA
loss teaches the predictor and the predicted semantics guide reconstruction,
but voxel loss cannot pull the predictor back toward low-level intensity. The
standard downstream adaptation plan is unchanged: segmentation loads only
`down_projection` and `eva`; predictor, regularizer, and bridge are discarded.

The first sweep is a causal ablation rather than five unrelated methods:

| ID | Variant | Question |
|---:|---|---|
| 0 | Parallel | Does region-level JEPA alone help without the bridge? |
| 1 | Detached SCLB | Does one-way semantic conditioning improve dense transfer? |
| 2 | Shuffled SCLB | Is spatially compact structure necessary, or only added capacity? |
| 3 | Bidirectional SCLB | Does allowing pixel gradients into the predictor hurt? |
| 4 | Cross-view SCLB | Does a weak aligned target intensity view add useful JEPA invariance while MAE stays unchanged? |

Submit all five one-GPU jobs (five GPUs when they can run concurrently):

```bash
sbatch --array=0-4 scripts/slurm/openmind_pretrain_mae_jepa_sclb_sweep.slurm
```

Limit concurrency, for example to two GPUs:

```bash
sbatch --array=0-4%2 scripts/slurm/openmind_pretrain_mae_jepa_sclb_sweep.slurm
```

Every new output directory is isolated by trainer name. A new run loads the
public Primus-M MAE checkpoint; an interrupted run resumes from its own latest,
best, or final checkpoint. Data augmentation uses 20 worker processes by
default and can be changed with `NNUNET_N_PROC_DA`.

Important interpretation limit: the regions are connected regular 3-D token
cells, not anatomical labels or SAM/SLIC proposals. The shuffled control tests
whether crop-relative spatial organization matters. If ID 1 beats IDs 0 and 2,
the next experiment should replace the cells with offline anatomical proposals.
