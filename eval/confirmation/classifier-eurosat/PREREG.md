# Pre-registration: ResNet-50 land-cover classifier on EuroSAT RGB

Status: FROZEN. The SHA-256 of this file is recorded in `PREREG.md.sha256`. Any later change goes in a
DEVIATIONS section appended below, is disclosed, and may never loosen a pass bar.

Registered: 2026-09-25, before any model has been evaluated on the test split.

## Question

Does a ResNet-50 fine-tuned on EuroSAT RGB classify held-out Sentinel-2 patches accurately enough to be used as
the land-cover step in Geo-VLA's workflows (flood exposure, deforestation, urban growth, crop health)?

## Data

- Dataset: EuroSAT RGB, 27,000 Sentinel-2 patches (64x64 px, 10 m), 10 classes.
  Helber, Bischke, Dengel, Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification", IEEE J-STARS 12(7):2217-2226, 2019. DOI 10.1109/JSTARS.2019.2918242 (verified via Crossref).
- Local copy: 27,000 files; SHA-256 of the sorted relative file list
  `75eda203f3467758e82a931d736c52b64a188dad65c8bd747d19bdfca84db94d`.
- Split: `training.datasets.stratified_split(labels, val_frac=0.1, test_frac=0.1, seed=42)` on the full dataset,
  giving 21,600 train / 2,700 validation / 2,700 test. Per-class test counts: AnnualCrop 300, Forest 300,
  HerbaceousVegetation 300, Highway 250, Industrial 250, Pasture 200, PermanentCrop 250, Residential 300,
  River 250, SeaLake 300.
- The test split is identified by the SHA-256 of its sorted file ids (`test_split_sha256` below). Training code
  never loads it; only `training.confirm` does, once.

## Development protocol (what may be iterated)

- Any number of training runs, hyperparameter changes and model comparisons are allowed, judged on the
  **validation split only**.
- Reference configuration: ResNet-50 with ImageNet weights, 224 px input, AdamW (max lr 3e-4, weight decay 1e-4),
  one-cycle schedule, cross-entropy with label smoothing 0.05, batch 64, 10 epochs, seed 42, augmentation =
  flips, 90-degree rotation, colour jitter; the checkpoint with the best validation accuracy is kept.
- The candidate for confirmation is chosen on validation metrics alone. Its checkpoint SHA-256 is written into
  `APPROVAL.md` together with this file's hash and the split hash, after the user explicitly approves.

## Hypotheses and pass bars (applied to the test split, one run)

| ID | Metric | Pass bar |
|---|---|---|
| H1 | Overall accuracy | >= 0.95 |
| H2 | Macro-averaged F1 over the 10 classes | >= 0.94 |
| H3 | Lowest per-class recall | >= 0.85 |

The model passes only if all three bars are met. Reported without a pass bar: the per-class precision, recall and
F1 table, the confusion matrix, and the gap between validation and test accuracy (a gap above 0.02 would suggest
overfitting to the validation split and is discussed if it occurs).

Rationale for the bars: H1/H2 require the network to be reliable enough that land-cover area figures in the
reports are dominated by real change rather than classification error; H3 guards against a good average hiding a
class the workflows depend on (for example PermanentCrop or River) being misclassified.

## Procedure

1. Freeze this file (`PREREG.md.sha256`).
2. Develop on validation. Choose the candidate version.
3. The user approves in writing; `APPROVAL.md` quotes the three hashes.
4. `python -m training.confirm --name classifier-eurosat --version <vN>` runs once, writes `results.json` and
   `SPENT.json`. The outcome is reported as it is, pass or fail, with its numbers.

```json prereg
{
  "model": "classifier",
  "dataset": "eurosat",
  "split_seed": 42,
  "img_size": 224,
  "dataset_file_list_sha256": "75eda203f3467758e82a931d736c52b64a188dad65c8bd747d19bdfca84db94d",
  "test_split_sha256": "c9484fa85378f4d38f5a3fe2b60fcfa9714856fec214562013139193c2fae2f1",
  "test_count": 2700,
  "pass_bars": {"accuracy": 0.95, "macro_f1": 0.94, "min_class_recall": 0.85}
}
```
