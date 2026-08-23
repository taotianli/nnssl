# Sparse residual MoE for Primus MAE+JEPA

This experiment adds conditional capacity to the teacher-free MAE+JEPA line
without replacing the public Primus-M representation. Every run starts from
the public OpenMind MAE checkpoint passed through `-pretrained_weights`.

## Design

Neuro-JEPA uses token-level sparse routing inside transformer FFNs: every token
passes through shared experts and top-k routed experts. This is not the same as
multi-view learning. Views specify which observations/objectives are compared;
MoE decides which parameters process each token.

Directly replacing Primus FFNs with randomly initialized experts would remove
the strongest property of the current baseline: a pretrained dense MAE encoder
that already transfers well to segmentation. The implementation therefore uses
the following residual form after upper EVA blocks 8, 10, 12, and 14:

```text
x_(l+1) = DenseEVA_l(x_l) + SparseResidualMoE_l(x_l)
```

- `DenseEVA` is the intact public MAE path and acts as the always-active shared
  expert.
- Eight residual experts use bottleneck width 192; top-2 experts are active for
  each token, with a conservative residual route scale of 0.25.
- Expert output layers and router weights start at zero. The initial function is
  exactly the public MAE function.
- Expert balance uses Neuro-JEPA-style non-gradient routing-bias updates, not an
  extra load-balancing loss that could introduce another gradient conflict.
- Experts and routers use the predictor LR (`3e-4`); the transferable dense
  encoder keeps the lower LR (`3e-5`).
- Downstream adaptation loads `eva.base`. Sparse experts are a pretraining
  scaffold and are discarded for the standard PrimusM segmentation benchmark.

The improved variant adds **routing consistency (RC)**. For each visible patch,
the router distribution from the masked context pass is matched to the router
distribution at the same coordinate in the full online target pass using
Jensen-Shannon divergence. It discourages experts from specializing to masking
artifacts rather than stable anatomical/contrast information. RC is disabled
during the 10-epoch predictor warm-up.

## Sweep

| ID | Trainer | Question |
|---:|---|---|
| 0 | `PrimusJEPAMoE_200ep_BS4_NoReg` | Does added conditional capacity work without collapse control? |
| 1 | `PrimusJEPAMoE_200ep_BS4_VICReg0p01` | Does MoE complement the best current VICReg setting? |
| 2 | `PrimusJEPAMoE_200ep_BS4_SIGReg0p02` | Is the effect regularizer-specific? |
| 3 | `PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p01` | Weak view-consistent routing |
| 4 | `PrimusJEPAMoE_200ep_BS4_VICReg0p01_RC0p05` | Strong view-consistent routing |
| 5 | `PrimusJEPAMoE16_200ep_BS4_VICReg0p01_RC0p01` | Capacity control: 16 experts, top-4 |

Submit all six independent single-GPU jobs:

```bash
cd /lus/lfs1aip2/projects/u6mn/nnssl
sbatch --array=0-5 \
  scripts/slurm/openmind_pretrain_mae_jepa_moe_sweep.slurm
```

Limit concurrency, if necessary, with `--array=0-5%2`. Each array element
requests one GPU. The script resumes from `checkpoint_latest.pth`,
`checkpoint_best.pth`, or `checkpoint_final.pth`; otherwise it loads the public
MAE checkpoint. Data augmentation uses 20 processes by default so a 24-hour run
does not inherit the slow single-threaded regularizer-calibration setting.

Monitor both representation and router behavior. In addition to MAE, JEPA, and
regularizer losses, logs contain routing consistency, routing entropy, and the
minimum/maximum expert utilization relative to the layer mean. A model with a
good reconstruction loss but collapsed utilization is not a successful MoE.

## Interpretation and next gate

The minimal evidence gate is AMOS 201 plus a lesion dataset such as ISLES 203.
First compare ID 1 against the matched non-MoE VICReg model. Then compare ID 3
against ID 1 to isolate routing consistency. ID 5 determines whether any gain is
conditional specialization or simply more parameters. Run at least three
downstream seeds for the best candidate before claiming a gain over MAE; the
current single-run margin is around one tenth of a Dice point and is not enough
for a TMI claim.

This implementation is inspired by, but is not a line-by-line reproduction of:

- Neuro-JEPA sparse MoE and auxiliary-loss-free routing:
  <https://arxiv.org/abs/2606.14957>
- Neuro-JEPA official implementation:
  <https://github.com/NYUMedML/Neuro-JEPA>
- I-JEPA context/target prediction:
  <https://arxiv.org/abs/2301.08243>
- VICRegL local correspondence regularization:
  <https://arxiv.org/abs/2210.01571>
- LeJEPA/SIGReg teacher-free regularization:
  <https://arxiv.org/abs/2511.08544>
