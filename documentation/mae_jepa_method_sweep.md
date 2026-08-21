# Primus MAE+JEPA dense-transfer method sweep

This sweep keeps the original Primus MAE reconstruction path and public-MAE
initialization.  All implementations use new trainer names; the original JEPA
code and completed checkpoints are unchanged.

## Why these experiments

The AMOS result indicates that simply continuing MAE training already causes a
small loss of dense-transfer quality, while fixed-weight JEPA increases that
drop.  The first priority is therefore controlling encoder drift, not increasing
JEPA capacity indiscriminately.

| Array | Family | Trainer | Single changed idea |
|---:|---|---|---|
| 0 | staged | `PrimusJEPAStaged_200ep_BS8_L0p005` | predictor-only 10 epochs, encoder LR 3e-5, JEPA ramp to .005 |
| 1 | staged | `PrimusJEPAStaged_200ep_BS8_L0p01` | same, ramp to .01 |
| 2 | adaptive | `PrimusJEPAAdaptive_200ep_BS8_R0p05` | normalize JEPA contribution to about 5% of MAE |
| 3 | adaptive | `PrimusJEPAAdaptive_200ep_BS8_R0p10` | normalize JEPA contribution to about 10% of MAE |
| 4 | block target | `PrimusJEPABlock_200ep_BS8_K512` | predict a compact 512-token 3-D region inside the MAE mask |
| 5 | block target | `PrimusJEPABlock_200ep_BS8_K1024` | compact 1,024-token target |
| 6 | dual teacher | `PrimusJEPADualTeacher_200ep_BS8_A0p25_0p75` | frozen public-MAE anchor + EMA teacher |
| 7 | dual teacher | `PrimusJEPADualTeacher_200ep_BS8_A0p50_0p90` | stronger public-MAE anchor |
| 8 | no EMA | `PrimusJEPANoEMA_200ep_BS4_VICReg0p01` | online target and VICReg variance/covariance |
| 9 | no EMA | `PrimusJEPANoEMA_200ep_BS4_SIGReg0p02` | online target and sliced Gaussian SIGReg |

The no-EMA variants use batch size 4 because gradients are retained through a
second full-token encoder pass.  Their throughput and memory should be reported
alongside accuracy.

## Mask semantics

The MAE mask remains the existing random 75% reconstruction mask.  In the block
variants, JEPA independently chooses the nearest K positions around a random 3-D
centre **within that hidden set**.  Therefore the predictor never sees a target
token, and MAE and JEPA no longer solve exactly the same spatial task.

This is inspired by I-JEPA's block targets and by MASS's region-guided objective,
but it is not a reproduction of MASS.  A faithful MASS experiment additionally
requires offline SAM2/SLIC/atlas proposal masks, matched reference/query crops,
and a task-conditioned segmentation decoder.

## Run order

Run a cheap screening wave before spending ten full jobs:

```bash
sbatch --array=0,2,4,6,8-9%2 scripts/slurm/openmind_pretrain_mae_jepa_method_sweep.slurm
```

Then run the paired parameter values for families that pass the screen:

```bash
sbatch --array=1,3,5,7%2 scripts/slurm/openmind_pretrain_mae_jepa_method_sweep.slurm
```

Every run starts from the public PrimusM MAE checkpoint.  Resubmitting the same
array index resumes its own latest checkpoint.

## Decision rule

Do not select a method from pretraining loss.  Evaluate the same checkpoint and
fine-tuning protocol first on AMOS (Dataset201), then confirm on at least one
lesion task such as ISLES22.  Compare against both the untouched public MAE and
the matched low-LR MAE-only continued-pretraining control.  A useful candidate
must recover the public-MAE dense baseline within repeated-run uncertainty before
semantic/classification benefits are considered.

## Reusable downstream array

The checkpoint manifest preserves the same zero-based indices as the pretraining
sweep. To fine-tune and full-volume validate screening entries 0, 2, 4, 6, 8,
and 9 on Dataset201, run:

```bash
sbatch --array=0,2,4,6,8-9 \
  --export=ALL,MODEL_SUITE=wave1,DATASET_ID=201 \
  scripts/slurm/openmind_downstream_checkpoint_array.slurm
```

The explicit Wave 1 array creates six tasks with no concurrency throttle, and
each task requests one GPU. Slurm therefore receives a request for six GPUs in
total, although individual tasks may wait until GPUs are available. The shared
script now defaults to the five Wave 2 models, so Wave 1 calls should retain the
`MODEL_SUITE=wave1` and `--array=0,2,4,6,8-9` overrides shown above. Completed
jobs are skipped, interrupted fine-tuning resumes from `checkpoint_latest.pth`,
and a completed fine-tune without a benchmark record runs validation only.

For another downstream dataset, change only `DATASET_ID`. For another collection
of checkpoints, create a tab-separated manifest with three columns—model label,
absolute checkpoint path, and unique run name—and pass it at submission time:

```bash
sbatch --array=0-3 \
  --export=ALL,DATASET_ID=203,MODEL_MANIFEST=/absolute/path/models.tsv \
  scripts/slurm/openmind_downstream_checkpoint_array.slurm
```

With no `%N` suffix, selecting N manifest rows requests N one-GPU tasks. Add a
suffix such as `%2` only when an explicit two-GPU concurrency cap is desired.

Blank lines and `#` comments in a manifest are ignored. The array index is the
zero-based order of the remaining rows.

## References

- I-JEPA: <https://arxiv.org/abs/2301.08243>
- VICReg: <https://arxiv.org/abs/2105.04906>
- LeJEPA / SIGReg: <https://arxiv.org/abs/2511.08544>
- MASS paper: <https://arxiv.org/abs/2603.13660>
- MASS code: <https://github.com/Stanford-AIMI/MASS>
