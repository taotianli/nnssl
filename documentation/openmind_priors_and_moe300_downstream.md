# Dataset201 prior sweep and 300-epoch MoE benchmark

## Ten prior checkpoints, standard 150-epoch downstream protocol

```bash
sbatch --array=0-9 \
  scripts/slurm/openmind_downstream_priors_201.slurm
```

Each task requests one GPU. Use `--array=0-9%5` to limit the suite to five
simultaneous GPUs. The manifest contains the ten completed 200-epoch prior
pretraining checkpoints in the same order as the prior pretraining sweep.

## Best MoE checkpoint, 300 downstream epochs

```bash
sbatch scripts/slurm/openmind_downstream_moe_rc0p05_201_300ep.slurm
```

This initializes Dataset201 fine-tuning from the completed
`PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p05` pretraining checkpoint and changes
only downstream training length from 150 to 300 epochs. Its run name ends in
`_300ep`, so it cannot overwrite the existing 150-epoch benchmark.

Both suites reuse `openmind_downstream_checkpoint_array.slurm`: completed runs
are skipped, interrupted runs resume latest/best, and a final fine-tuning
checkpoint without a completed record enters full-volume validation.
