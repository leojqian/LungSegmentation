# research/

**Not part of the lungmap package.** These are the scripts that trained and
evaluated the model lungmap ships. You don't need them to use lungmap.

## Running them

The datasets aren't in this repository. Put them at the repo root:
- `MontgomerySet/` ([Montgomery County X-ray Set](https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33))
- `SampleDDR_August2026/` (Konica Minolta DDR sample, 20 cases)

Run every script from the repo root:

```bash
uv run --extra research python research/<folder>/<script>.py
```

Outputs go to `outputs/`, `data/` and model files at the repo root, all of
which are git-ignored.

## Pipeline that produced the shipped model

| step | script | does | writes |
|---|---|---|---|
| 1 | `montgomery/step01_load_data.py` | Montgomery images + masks → training arrays | `data/*.npy` |
| 2 | `montgomery/step02_unet_model.py` | U-Net architecture (imported by step 3) | — |
| 3 | `montgomery/step03_train_unet.py` | train the base U-Net | `best_model.h5` |
| 4 | `montgomery/step04_test_model.py` | sanity-check the base model on one image | (windows) |
| 5 | `montgomery/step05_map_diaphragm.py` | trace diaphragms across Montgomery; tuned `lungmap/geometry` | `outputs/` |
| 6 | `ddr/step06_finetune.py` | fine-tune on DDR, 5-fold by patient | **`best_model_ddr_finetuned_phase3.h5`** (the shipped model) |
| 7 | `ddr/step07_evaluate.py` | Dice + diaphragm-point error vs DDR ground truth | `outputs/ddr_eval/` |
| 8 | `ddr/step08_review_report.py` | HTML review of every annotated frame | `outputs/ddr_visualize/` |

The shipped download (`lungmap-model-v1.h5`) is step 6's model with its
training-only optimizer state removed. Its predictions are identical.

## archive/: superseded, not used

Earlier DDR experiments, kept as the record behind numbers in
`docs/design/2026-08-24-ddr-domain-adaptation.md`.

| script | was |
|---|---|
| `archive/ddr_validate.py` | first validation of the base model on DDR |
| `archive/ddr_calibrate_preprocessing.py` | preprocessing calibration (windowing, mask kernel, image source) |
| `archive/ddr_finetune_phase2.py` | phase-2 fine-tune, replaced by step 6 |

## Old file names

The design notes in `docs/design/` use the original names:

| old | new |
|---|---|
| `Step1LoadData.py` … `Step5MapDiaphragm.py` | `montgomery/step01_…` … `step05_…` |
| `Step6ValidateDDR.py` | `archive/ddr_validate.py` |
| `Step7CalibrateDDR.py` | `archive/ddr_calibrate_preprocessing.py` |
| `Step8FineTuneDDR.py` | `archive/ddr_finetune_phase2.py` |
| `submission/finetune_ddr.py` | `ddr/step06_finetune.py` |
| `submission/evaluate_ddr.py` | `ddr/step07_evaluate.py` |
| `submission/visualize_ddr.py` | `ddr/step08_review_report.py` |
