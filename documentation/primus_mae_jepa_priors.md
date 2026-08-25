# MRI prior sweep for MAE + JEPA + VICReg

This sweep adds five independent prior families to the selected matched
teacher-free baseline `PrimusJEPANoEMA_200ep_BS4_VICReg0p01`. The matched
baseline itself is deliberately **not** included, because its checkpoint and
Dataset201 result already exist.

All runs start from the same public PrimusM OpenMind MAE checkpoint and retain:

- MAE reconstruction and JEPA latent prediction;
- online/full-view JEPA targets (no EMA teacher);
- VICReg weight 0.01;
- batch size 4, 200 epochs, encoder LR `3e-5`, head LR `3e-4`;
- experiment seed `20260821`, matching the existing regularization baseline;
- 10 predictor-only epochs, then a 20-epoch JEPA/prior ramp to JEPA weight 0.005;
- the same downstream adaptation keys (`eva`, `down_projection`).

The design transfers the useful idea from the structured-prior paper
[arXiv:2606.18658](https://arxiv.org/abs/2606.18658)—adding geometric or
population structure instead of only matching individual embeddings—but uses
MRI-compatible priors that can run on the current OpenMind preprocessing.

## Ten configurations

| IDs | Family | Configuration A | Configuration B | Added information |
|---|---|---|---|---|
| 0–1 | Anatomy support/boundary | occupancy | occupancy + patch-grid signed-distance proxy | optional preprocessed anatomy mask; missing masks use low-confidence nonzero-image support |
| 2–3 | Spatial heat kernel | 6 neighbours | 26 neighbours | intensity-gated local manifold smoothness on masked latent tokens |
| 4–5 | Spectral reconstruction | 8 shells | 16 shells | normalized radial 3-D Fourier energy, weighted toward high frequencies |
| 6–7 | Acquisition/style | corruption strength 0.05 | 0.10 | clean latent target plus gain/bias/noise/blur context; known style parameters condition only the MAE decoder |
| 8–9 | Soft region prior | 3 intensity pseudo-tissues | foreground-weighted 2x2x2 regions | richer within-support patch identity without downstream labels |

The final family is intentionally a **pseudo-tissue/coarse-region prior**, not
SynthSeg, FastSurfer, or a registered atlas. No such external atlas labels are
present in Dataset746, and silently claiming an atlas prior would invalidate
the comparison. A later atlas experiment can reuse the same head once offline
soft posteriors are added to preprocessing.

Anatomy-mask polarity is inferred per volume from image support because older
OpenMind preprocessed collections use inconsistent foreground conventions.
The binary support and signed-distance target are never used at downstream
fine-tuning time.

## Submit

After pulling the branch on the cluster:

```bash
cd /lus/lfs1aip2/projects/u6mn/nnssl
git pull

# Ten array elements = ten independent one-GPU jobs.
sbatch --array=0-9 scripts/slurm/openmind_pretrain_mae_jepa_prior_sweep.slurm

# If the account should run at most five simultaneously:
sbatch --array=0-9%5 scripts/slurm/openmind_pretrain_mae_jepa_prior_sweep.slurm
```

Each task requests one GPU, 32 CPUs, and 24 hours. `nnUNet_n_proc_DA` defaults
to 20 so input augmentation remains multi-process. At the ten-minute signal,
the existing nnSSL signal handler writes `checkpoint_latest.pth`; resubmitting
the same array automatically resumes latest/best/final rather than loading the
public MAE again.

For a smaller screen:

```bash
# One representative from each family, five GPUs if all start together.
sbatch --array=0,2,4,6,8 scripts/slurm/openmind_pretrain_mae_jepa_prior_sweep.slurm
```

## Logged quantities

The standard MAE, JEPA, VICReg, total loss, and JEPA-weight histories remain.
The new trainer additionally logs train/validation prior loss, ramped prior
weight, mask coverage, occupancy/SDF, heat affinity/pair count, low/high
spectral error, pseudo-region entropy, and corruption magnitude. They are in:

```text
${nnssl_results}/Dataset746_OpenMind20000/<TRAINER>__nnsslPlans__onemmiso/fold_all/
```

Primary selection remains downstream Dataset201 Dice/NSD, not the pretraining
total loss. Any apparent winner should then be checked on a lesion task and
with at least two additional pretraining seeds before making a TMI claim.
