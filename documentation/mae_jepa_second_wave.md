# Primus MAE+JEPA second-wave ablations

The public OpenMind MAE remains the initialization and dense-transfer control.
These experiments test five ways to make reconstruction and prediction
complementary instead of merely adding two independently optimized losses.
None is assumed to improve Dice until the same downstream protocol has been
run; pretraining loss is not a model-selection substitute for downstream Dice.

| Array | Trainer | Single changed mechanism | Expected role |
|---:|---|---|---|
| 0 | `PrimusJEPASerialCAE_200ep_BS8_L0p005` | JEPA masked-latent predictions replace MAE decoder mask tokens | Pixel reconstruction also trains the predictor |
| 1 | `PrimusJEPAGatedSkip_200ep_BS8_G0p25` | Zero-initialized, bounded visible patch skip to decoder | Keep low-level detail out of the semantic encoder |
| 2 | `PrimusJEPAIndependentCrossView_200ep_BS4_Weak` | Independent MAE/JEPA masks plus aligned weak MRI target view | Decouple task difficulty and learn intensity invariance without breaking coordinates |
| 3 | `PrimusJEPAStructure_200ep_BS8_E0p05` | MAE pixel loss plus 0.05-weight smoothed 3-D finite-difference loss | Preserve anatomical boundaries useful for segmentation |
| 4 | `PrimusJEPAGradGuard_200ep_BS4_R0p25` | Cap JEPA encoder gradient at 25% of MAE and attenuate negative cosine | Prevent the auxiliary task from erasing dense MAE features |

All configurations use the existing stabilized recipe: public MAE
initialization, 10 predictor-only epochs, encoder LR `3e-5`, predictor LR
`3e-4`, and a 20-epoch ramp to JEPA weight `0.005`. The independent-cross-view
and gradient-guard variants use batch size 4 because they add a second online
pass or gradient probes, respectively. The other three use batch size 8.
Gradient guard intentionally rejects DDP because its pre-backward gradient
probe is not compatible with the PyTorch DDP reducer; the supplied jobs are
single-GPU tasks.

Run all five (five array elements request five one-GPU jobs):

```bash
cd /lus/lfs1aip2/projects/u6mn/nnssl
sbatch scripts/slurm/openmind_pretrain_mae_jepa_second_wave.slurm
```

Run selected methods, for example serial, structure and gradient guard:

```bash
sbatch --array=0,3,4 scripts/slurm/openmind_pretrain_mae_jepa_second_wave.slurm
```

The script accepts reusable environment overrides such as `DATASET_ID`,
`NNSSL_PREPROCESSED`, `NNSSL_RESULTS`, and `MAE_CHECKPOINT`. Existing output
automatically resumes with `--c`; a new output loads the public MAE checkpoint.
The corresponding downstream checkpoint manifest is
`scripts/configs/openmind_mae_jepa_second_wave_checkpoints.tsv`.

## Paper lineage and scope

- Serial latent coupling is adapted from [Context Autoencoder](https://arxiv.org/abs/2202.03026), not a line-by-line CAE reproduction.
- The gated decoder skip is adapted from [BootMAE](https://arxiv.org/abs/2207.07116).
- Independent task masks follow the context/target separation in [I-JEPA](https://arxiv.org/abs/2301.08243), while the aligned intensity target follows the masked-online/full-momentum-view design in [CMAE](https://arxiv.org/abs/2207.13532). Geometry is deliberately unchanged for exact 3-D patch correspondence.
- The structure target follows the structured reconstruction motivation of [MaskFeat](https://arxiv.org/abs/2112.09133) and the medical-frequency motivation of [Frepa](https://arxiv.org/abs/2407.14651), but uses inexpensive smoothed 3-D finite differences rather than HOG or a global FFT.
- Gradient guard is a project-specific asymmetric fusion rule motivated by [PCGrad](https://arxiv.org/abs/2001.06782) and [GradNorm](https://proceedings.mlr.press/v80/chen18a.html). It is not claimed as an exact reproduction of either algorithm.

MASS uses offline class-agnostic 3-D proposal masks and reference/query
in-context segmentation. Without that additional SAM2 proposal pipeline, a
random block-mask variant should not be called MASS. Proposal-guided targets
remain a later experiment after proposal preprocessing is available.

## Decision rule

Compare every checkpoint with the public MAE and matched MAE-only continuation
using identical Dataset201 splits and adaptation settings. Use full-volume Dice
and NSD, then confirm any gain on Dataset203 because organ-only improvement does
not establish lesion transfer. Inspect `jepa_loss_history.json`; the structure
variant adds `train_edge_losses`/`val_edge_losses`, and gradient guard adds
`encoder_grad_cosines`/`jepa_gradient_scales`.
