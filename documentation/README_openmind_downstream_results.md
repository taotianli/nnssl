# OpenMind downstream benchmark results

Last updated: 2026-08-22

This file records the downstream results reported so far for the PrimusM
OpenMind MAE and the MAE+JEPA experiments in this repository. Unless stated
otherwise, the numbers are **full-volume validation** metrics written to
`benchmark_run.json`; they are not patch-level pseudo Dice values.

## Evaluation protocol

- Downstream trainer: `PretrainedTrainer_Primus_150ep`
- Configuration: `3d_fullres`
- Fold: `0`
- Primary comparison dataset: `Dataset201_AMOS`
- Primary metric: foreground mean Dice
- Supporting metrics: foreground mean NSD and IoU
- Public baseline checkpoint:
  `/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth`

Results from different dataset versions, splits, preprocessing, or adaptation
plans are not directly comparable. The small differences near the public MAE
baseline are single-run observations and still need repeated pretraining and
downstream seeds before being treated as real improvements.

## Dataset201 AMOS summary

The public PrimusM OpenMind MAE result, `0.839504` Dice, is used as the
reference. `Delta Dice (pp)` is the absolute Dice difference multiplied by
100, so `+0.106` means an increase of 0.00106 Dice.

| Group | Model | Status | Dice | NSD | IoU | Delta Dice (pp) |
|---|---|---:|---:|---:|---:|---:|
| Public baseline | PrimusM OpenMind MAE | completed | 0.839504 | 0.742750 | 0.749736 | +0.000 |
| Initial fusion | MAE+JEPA, initial lambda 0.1 run | completed | 0.827629 | 0.723889 | 0.734789 | -1.187 |
| Matched continuation | MAE-only, LR 3e-4 | completed | 0.833664 | 0.732314 | 0.741884 | -0.584 |
| Matched continuation | MAE-only, LR 3e-5 | completed | 0.836602 | 0.737227 | 0.746173 | -0.290 |
| Matched continuation | MAE+JEPA, lambda 0.01 | completed | 0.824821 | 0.718482 | 0.730521 | -1.468 |
| Matched continuation | MAE+JEPA, lambda 0.005 | completed | 0.826687 | 0.720672 | 0.732740 | -1.282 |
| Wave 1 | JEPA staged, lambda 0.005 | completed | 0.840013 | 0.743544 | 0.750191 | +0.051 |
| Wave 1 | JEPA adaptive, ratio 0.05 | completed | 0.838665 | 0.740717 | 0.748607 | -0.084 |
| Wave 1 | JEPA block target, K=512 | completed | 0.839958 | 0.742057 | 0.750231 | +0.045 |
| Wave 1 | JEPA dual teacher, alpha 0.25 to 0.75 | completed | 0.838650 | 0.740720 | 0.748368 | -0.085 |
| Wave 1 | No-EMA JEPA + VICReg 0.01 | completed | **0.840563** | 0.744215 | 0.750877 | **+0.106** |
| Wave 1 | No-EMA JEPA + SIGReg 0.02 | completed | 0.839966 | **0.744552** | **0.750962** | +0.046 |
| Wave 2 | Serial CAE, lambda 0.005 | not found | - | - | - | - |
| Wave 2 | Gated skip, gate 0.25 | completed | 0.839425 | 0.742529 | 0.749774 | -0.008 |
| Wave 2 | Independent cross-view, weak target view | completed | 0.839978 | 0.744005 | 0.750355 | +0.047 |
| Wave 2 | Structure-aware reconstruction, edge 0.05 | completed | 0.835866 | 0.735011 | 0.744726 | -0.364 |
| Wave 2 | Gradient guard, ratio 0.25 | **invalid / investigate** | 0.000000 | 0.000000 | 0.000000 | - |

### Current ranking by Dataset201 Dice

Among valid completed runs, the highest observed results are:

1. No-EMA JEPA + VICReg 0.01: `0.840563`
2. JEPA staged, lambda 0.005: `0.840013`
3. Independent cross-view: `0.839978`
4. No-EMA JEPA + SIGReg 0.02: `0.839966`
5. JEPA block target K=512: `0.839958`
6. Public PrimusM MAE: `0.839504`

The best observed gain is only `+0.001059` Dice (`+0.106` percentage
points). This is encouraging for representation regularization, but it is too
small to establish superiority without matched repeated seeds and paired
case-level analysis.

## Public PrimusM MAE across Dataset201-209

These are the latest recorded full-volume Dice results for the downloaded
public PrimusM OpenMind MAE checkpoint under the local benchmark setup.

| ID | Dataset | Dice |
|---:|---|---:|
| 201 | Dataset201_AMOS | 0.839504 |
| 202 | Dataset202_KiTS19 | 0.718949 |
| 203 | Dataset203_ISLES22 | 0.738727 |
| 204 | Dataset204_HNTSMRG | 0.601380 |
| 205 | Dataset205_HaNSeg | 0.411039 |
| 206 | Dataset206_MSFLAIR | 0.515441 |
| 207 | Dataset207_TopCoW_MRA | 0.708598 |
| 208 | Dataset208_YaleBrainMets | 0.504172 |
| 209 | Dataset209_ACDC | 0.831864 |

These values reproduce the performance of this local data preparation and
split configuration, not necessarily the exact paper benchmark. In
particular, dataset release, case inventory, labels, split lists, and
preprocessing must be checked before comparing a number directly with a paper
table.

## Interpretation so far

- Naively continuing MAE training or adding a fixed JEPA loss reduced AMOS
  Dice. A smaller JEPA coefficient alone did not solve the degradation.
- Staging, block targets, independent task views, VICReg, and SIGReg recovered
  the public MAE baseline. This supports improving the interaction between
  reconstruction and prediction rather than simply increasing JEPA weight.
- No-EMA VICReg currently has the best Dice, while No-EMA SIGReg has the best
  observed NSD and IoU. Their margins over the public MAE are very small.
- Structure-aware reconstruction with edge weight 0.05 underperformed the
  public MAE in this run.
- The Gradient Guard result is not a valid negative result: exact zeros usually
  indicate failed weight transfer, collapsed predictions, label/plan mismatch,
  or a validation/checkpoint problem. Its log and `benchmark_run.json` must be
  inspected before drawing a method conclusion.
- Serial CAE has no `benchmark_run.json` at the expected Wave 2 output path and
  remains unevaluated or stored under a different run name.

## Next decision gate

Before adding more architecture changes:

1. Recover or rerun Serial CAE and diagnose Gradient Guard.
2. Run the calibrated regularization Block A to compare NoReg, VICReg, and
   SIGReg at matched encoder-gradient budgets.
3. Repeat the strongest candidates with at least 2-3 pretraining seeds and
   matched downstream seeds.
4. Use paired case-level bootstrap confidence intervals on Dataset201.
5. Confirm gains on a lesion-oriented task such as Dataset203; an AMOS-only
   improvement does not establish general dense-transfer superiority.

## Result provenance

Benchmark records are stored under:

```text
/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/
  DatasetXXX_NAME/
    PretrainedTrainer_Primus_150ep__RUN_NAME__3d_fullres/
      fold_0/benchmark_run.json
```

For Dataset201 Wave 2, the expected run names are defined in
`scripts/configs/openmind_mae_jepa_second_wave_checkpoints.tsv`.
