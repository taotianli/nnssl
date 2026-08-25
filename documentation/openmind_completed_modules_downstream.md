# Dataset201 benchmark for completed SCLB/HOP/DOA checkpoints

This three-model suite evaluates only the completed 200-epoch checkpoints:

| Array ID | Model | Pretraining checkpoint |
|---:|---|---|
| 0 | SCLB Parallel control | `PrimusJEPASCLBParallel_200ep_BS4` |
| 1 | HOP at encoder block 8 | `PrimusJEPAHOPB8_200ep_BS4` |
| 2 | rank-32 dual-objective adapters | `PrimusJEPADOAR32_200ep_BS4` |

The incomplete ELC checkpoint and the four SCLB variants that failed during
epoch 0 are deliberately excluded. Each array element requests one GPU and
reuses the standard 150-epoch Primus downstream benchmark on Dataset201.

```bash
cd /lus/lfs1aip2/projects/u6mn/nnssl
git pull origin feature/primus-jepa

sbatch --array=0-2 \
  scripts/slurm/openmind_downstream_completed_modules_201.slurm
```

To run no more than two tasks simultaneously:

```bash
sbatch --array=0-2%2 \
  scripts/slurm/openmind_downstream_completed_modules_201.slurm
```

The reusable downstream runner skips completed records, resumes interrupted
fine-tuning from latest/best checkpoints, and performs full-volume validation
when fine-tuning has already produced a final checkpoint.
