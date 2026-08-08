# OpenMind downstream segmentation benchmark

`scripts/openmind_downstream_benchmark.py` runs one public Primus-M OpenMind
checkpoint on one of the nine uploaded, preprocessed segmentation datasets. It
uses the OpenMind nnU-Net adaptation interface and performs full-volume
validation after training.

## Cluster setup

The defaults match the current Isambard layout:

```text
adaptation repo  /home/u6mn/taotl.u6mn/u6mn/nnUNet-openmind
preprocessed     /lus/lfs1aip2/projects/u6mn/datasets/OpenMind_downstream/segmentation_preprocessed
weights          /lus/lfs1aip2/projects/u6mn/openmind_jepa/weights
results          /lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark
```

Install the adaptation fork into the active environment once:

```bash
cd /home/u6mn/taotl.u6mn/u6mn/nnUNet-openmind
python -m pip install -e .
```

## Run one model/dataset pair

First run the input and protocol checks without allocating GPU memory:

```bash
cd /home/u6mn/taotl.u6mn/u6mn/nnssl
python scripts/openmind_downstream_benchmark.py \
  --model PrimusM-OpenMind-MAE \
  --dataset 203 \
  --dry-run
```

Then launch the benchmark on a GPU node:

```bash
python scripts/openmind_downstream_benchmark.py \
  --model PrimusM-OpenMind-MAE \
  --dataset 203
```

To run all nine datasets sequentially for the same checkpoint:

```bash
for dataset in 201 202 203 204 205 206 207 208 209; do
  python scripts/openmind_downstream_benchmark.py \
    --model PrimusM-OpenMind-MAE \
    --dataset "$dataset"
done
```

`--dataset` accepts `203`, `ISLES22`, or `Dataset203_ISLES22`. Public Primus-M
methods can be selected by changing `--model` to the corresponding directory
under the weights root, for example `PrimusM-OpenMind-VoCo`.

Resume an interrupted run or repeat only full-volume validation:

```bash
python scripts/openmind_downstream_benchmark.py --model PrimusM-OpenMind-MAE --dataset 203 --continue
python scripts/openmind_downstream_benchmark.py --model PrimusM-OpenMind-MAE --dataset 203 --validation-only
```

## Protocol and outputs

The default protocol matches the paper's main segmentation benchmark:

- 150 epochs, 250 training iterations per epoch;
- batch size 2 and a 160 x 160 x 160 patch;
- Primus-M learning rate `3e-5`;
- 15-epoch whole-network linear warm-up followed by polynomial decay;
- full-volume Dice after training, not patch-level pseudo Dice.

The source `PMPrep.json` is never overwritten. A model-specific plan such as
`OMBench_PrimusM_OpenMind_MAE.json` is created in the dataset folder. Results
are isolated by model, dataset, and fold. The two most useful files are:

```text
<results>/<dataset>/<trainer>__<plan>__3d_fullres/fold_0/validation/summary.json
<results>/<dataset>/<trainer>__<plan>__3d_fullres/fold_0/benchmark_run.json
```

`summary.json` is nnU-Net's per-case/per-class full-volume evaluation.
`benchmark_run.json` additionally records the resolved checkpoint, dataset,
protocol, GPU, status, runtime, foreground mean metrics, and exact output path.

The runner installs process-local compatibility adapters for the uploaded
downstream preprocessing artifacts. They accept foreground locations stored as
`(z, y, x)` even though the adaptation fork expects `(class, z, y, x)`. They
also map sparse source labels such as those in TopCoW to contiguous training
channels and restore the original label IDs when exporting full volumes. No
`.b2nd`, properties, source plans, or ground-truth files are rewritten.

Epoch count is implemented by the trainer class in the adaptation repository;
it is not stored in the Hugging Face `adaptation_plan.json`. That JSON describes
the pretrained architecture, input preprocessing, patch size, and weight-key
mapping. The downstream `PMPrep.json` describes the already preprocessed target
dataset and supplies `pretrain_info` to the adaptation trainer.

## Reproduction scope

The uploaded datasets contain a single predefined split, so the default is
fold 0. This reproduces one OpenMind-style train/validation run per pair. It is
not a five-fold result unless the dataset has five entries in
`splits_final.json` and all folds are run separately. Dataset conversion and
split differences from the authors' files can also cause Dice differences even
when optimization settings match.
