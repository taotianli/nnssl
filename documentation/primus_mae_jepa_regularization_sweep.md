# Primus MAE+JEPA regularization sweep

All trainers start from the public PrimusM MAE checkpoint supplied through
`-pretrained_weights`. Existing Wave1/Wave2 trainers are unchanged. First
measure the common encoder-gradient budget (seven independent one-GPU jobs):

```bash
sbatch --array=0-6 scripts/slurm/openmind_calibrate_mae_jepa_regularizers.slurm
```

Or submit calibration and Block A with an explicit dependency:

```bash
CAL_JOB=$(sbatch --parsable --array=0-6 \
  scripts/slurm/openmind_calibrate_mae_jepa_regularizers.slurm)
sbatch --dependency=afterok:${CAL_JOB} --array=0-6 \
  scripts/slurm/openmind_pretrain_mae_jepa_regularization_sweep.slurm
```

This measures 200 batches by default and writes `q_R` JSON files under
`/lus/lfs1aip2/projects/u6mn/openmind_jepa/regularizer_calibration`. The
training array reads these files and derives each actual lambda from the
VICReg-0.01 reference budget. It fails fast rather than silently falling back
to incomparable raw weights when calibration files are missing.

Run Block A first (seven independent one-GPU jobs). The middle VIC/SIG runs
are rerun under the matched seed and calibrated budget rather than assuming
the historical raw `.01/.02` checkpoints equal `rho_ref`:

```bash
sbatch --array=0-6 scripts/slurm/openmind_pretrain_mae_jepa_regularization_sweep.slurm
```

### Downstream screening from the interrupted 24-hour checkpoints

If the first 24-hour Block A jobs stop near epoch 100, their latest checkpoints
can be screened on Dataset201 before the 200-epoch continuation finishes:

```bash
sbatch scripts/slurm/openmind_downstream_block_a_latest.slurm
```

The seven array elements each request one GPU. Use `--array=0-6%2` to limit
execution to two GPUs at a time. On first use, every task atomically freezes its
mutable `checkpoint_latest.pth` as `checkpoint_24h_snapshot.pth`; later
pretraining continuation therefore cannot change the checkpoint associated
with the downstream result. Existing snapshots are deliberately reused on
resubmission so interrupted downstream fine-tuning remains reproducible.

The result run names contain `BlockA`, `24hLatest`, and the checkpoint's saved
`current_epoch` (for example `_E98`), and are separate from future 200-epoch
runs. These results compare the Block A methods at roughly 100 epochs. They
must not be reported as compute-matched comparisons against the 200-epoch Wave
1/Wave 2 checkpoints.

The equivalent early screening for Block B experiment IDs 7-14 is:

```bash
sbatch scripts/slurm/openmind_downstream_block_b_latest.slurm
```

Its downstream array uses indices 0-7, which map in order to pretraining IDs
7-14. It applies the same immutable 24-hour snapshot and epoch-suffixed run-name
rules as Block A.

After checking the NoEMA/no-regularizer control, run Block B (eight one-GPU jobs):

```bash
sbatch --array=7-14 scripts/slurm/openmind_pretrain_mae_jepa_regularization_sweep.slurm
```

To limit concurrency, append `%N`, for example `--array=7-14%4`. Paths and the
dataset can be overridden with environment variables accepted by the Slurm
script. Completed checkpoints are skipped through `--c` resume semantics.

The TSV manifest is the reusable source of task IDs, trainer names,
calibration keys, and rho multipliers. Compare the NoEMA control before
interpreting regularizer gains.

All array tasks default to `EXPERIMENT_SEED=20260821`, which seeds Python,
NumPy, torch, projector initialization, patch dropping, and region sampling.
Calibration keeps the deterministic single-thread augmenter. Full pretraining
now defaults to 20 augmentation processes (`NNUNET_N_PROC_DA=20`) so the H200
is not starved by 3-D CPU augmentation. Multi-process augmentation does not
guarantee bit-identical batch streams across separate jobs, so comparisons are
seed-matched but not bit-exact. Set `NNUNET_N_PROC_DA=0` explicitly only for a
small deterministic diagnostic. Regularizer random projections and region
offsets use isolated RNG streams and do not advance the model's
patch-drop/DropPath stream.
The framework resumes model/optimizer/logger state with `--c`, but its existing
data-loader checkpoint format does not preserve an exact worker/augmentation
cursor; resumed runs are optimization-equivalent rather than bit-exact.

VISReg follows the scale/center/sliced-Wasserstein construction in the
[paper](https://arxiv.org/abs/2606.02572) and its
[official implementation](https://github.com/HaiyuWu/visreg). The loss runs in
FP32 even when the surrounding MAE+JEPA forward uses AMP.
