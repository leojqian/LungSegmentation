# DDR Domain Adaptation — Design

Date: 2026-08-24
Status: proposed

## Goal

Close the accuracy gap between the Montgomery-trained U-Net's in-domain performance
and its performance on the DDR dataset (`SampleDDR_August2026`), measured by
`Step6ValidateDDR.py`.

## Baseline (measured 2026-08-24, before any change here)

| | Dice (mean) | Dice (median) | n |
|---|---|---|---|
| Montgomery validation split | 0.980 | 0.984 | 14 |
| DDR (`Step6ValidateDDR.py`, current preprocessing) | 0.662 | 0.679 | 40 frames |

| | mean | median | n |
|---|---|---|---|
| DDR diaphragm point error | 107.7px / 43.1mm | 93.7px / 37.5mm | 160 points |

A visual check of one DDR overlay (`KaU001_PA_deep`, frame 119) showed the predicted
mask growing a spurious thin filament trailing into the abdomen — an artifact that
does not appear on Montgomery predictions, consistent with genuine domain shift
rather than a scoring bug.

**Pipeline-parity check.** The Montgomery number above comes from direct 256x256
model output (no upscale, no morphology); the DDR number goes through
`Step6ValidateDDR.py`'s full pipeline (upscale to native res, threshold, 5x5
morphological open). Verified this is not a confound: re-ran both pipelines on 15
Montgomery images and they agree within 0.001 Dice. The 0.98 -> 0.66 gap is
genuine domain shift, not a scoring artifact.

**Related finding, out of scope here.** `Step1LoadData.py` pairs image/left-mask/
right-mask via `zip(glob(), glob(), glob())` on three separate, unsorted directory
listings — correct only if the OS happens to return all three in matching order.
Verified 0/138 mismatches in this environment, so not currently broken, but it is
relying on unspecified `glob()` behavior rather than a guaranteed contract. A
one-line `sorted()` fix would make it robust rather than accidentally-correct.
Not fixed as part of this spec.

## Scope decisions

| Question | Decision |
|---|---|
| Order | Phase 1 (preprocessing, no retraining) first. Phase 2 (fine-tuning) is gated on Phase 1's results, not automatic. |
| Retrain from scratch | Out of scope — infeasible with 40 labeled DDR masks. |
| Diaphragm-point ground truth as a training signal | Out of scope. `DM-MODE_truth` remains evaluation-only, downstream of whatever mask the model/geometry pipeline produces — consistent with `diaphragm.py` operating on masks, not point annotations. |
| Split unit for any DDR-labeled evaluation | Case (patient), never frame — two frames from the same patient are correlated and a frame-level split would leak. |

## Phase 1 — Preprocessing calibration

No model weights change. A one-off comparison across candidate preprocessing
variants, evaluated with `Step6ValidateDDR.py`'s existing scoring across all 20
DDR cases (legitimate for calibrating constants, not learned parameters — same
methodology `CLIFF_STEP`/`RISE_FRACTION` were tuned with in
`2026-08-15-diaphragm-mapping-design.md`). The winning combination becomes the new
default in `dicom_io.py` / `Step6ValidateDDR.py` — no new permanent CLI flags for a
one-time calibration decision.

### Candidates

Fixed in advance, not expanded after seeing results — with only 20 cases,
searching until something looks good is how you fit noise instead of signal.

1. **Windowing**, 2 candidates: current per-frame min/max stretch, vs. the DICOM's
   own `WindowCenter`/`WindowWidth` tags (2048/4095 in this set). Per-frame min/max
   lets a single frame's outliers (e.g. a bright collimator edge) rescale the whole
   image; the window tags encode the device's own intended display range.
   `to_model_input` already supports both modes — this only wires `load_frames` to
   read and pass through the tags.
2. **Mask post-processing kernel size**, 4 candidates: 5 (current), 9, 15, 21. The
   filament artifact is exactly what a bigger morphological opening should remove.
3. **Image source**, 2 candidates: `images_pres` (current, log + Regius-processed)
   vs. `images` (raw, pre-log-conversion). Lower priority — raw sensor data likely
   needs its own processing to look film-like — but cheap to test since the
   pipeline already accepts either path.

### Evaluation

Coordinate ascent, not full factorial — 2x4x2=16 combinations against 20 cases
would multiply the multiple-comparisons risk for little benefit. In priority order,
holding everything else at its current default:

1. Test the 2 windowing candidates. Keep the winner.
2. With windowing fixed, test the 4 kernel candidates. Keep the winner.
3. With windowing and kernel fixed, test the 2 image-source candidates.

8 evaluations total, each run across all 20 DDR cases using the existing Dice/IoU
and diaphragm-point scoring. For every comparison, report **per-case paired
deltas alongside the mean/median** (e.g. "improved on 15/20 cases") — a mean-only
comparison can look like a win while actually being driven by one or two cases.

**Spot-check overlays**: a fixed rule, not picked per variant — the same 3 cases
every time (the 2 with the worst baseline Dice, plus 1 at the median), so overlays
are visually comparable across variants rather than cherry-picked.

### Output

A tuning table (variant, mean/median Dice, per-case win count, mean/median point
error in mm), following the style of the `--rise` tuning table in the
diaphragm-mapping design. Findings and the chosen defaults get written up as an
addendum to this document.

## Phase 2 — Fine-tuning (gated on Phase 1 results)

**Go/no-go**: decided from Phase 1's numbers, not a hardcoded threshold. Only
pursued if a residual gap remains that preprocessing did not close.

### Data and split

40 labeled masks from only 20 patients (2 frames each). Split at the **case**
level. A single held-out split is too noisy to trust with this little data, so:
**5-fold group cross-validation by case** (16 train / 4 val patients per fold),
fixed random seed 42 (matching `Step1LoadData.py`'s existing convention), fold
assignment logged to `outputs/ddr/fold_assignment.json` (case -> fold index) so
the split is auditable and rerunnable. Aggregate **out-of-fold** predictions
across all 5 folds, so every case is scored by a model that never saw it during
its fold — producing one Dice/point-error number directly comparable to the
Phase 1 baseline.

### Fine-tuning strategy

- Start from `best_model.h5`'s weights.
- Learning rate much lower than original training (1e-4 -> ~1e-5/1e-6).
- Light augmentation to fight overfitting on ~32 images/fold: small rotation,
  brightness/contrast jitter, slight translation.
- **No horizontal flip.** Chest anatomy is not left-right symmetric — the cardiac
  silhouette sits left of midline, which `diaphragm.py`'s own cardiac-border trims
  already rely on (`trim_cliffs` for the abrupt right-side border, `trim_rise` for
  the gradual left-side one). Flipping would teach the model a wrong prior.

### Guard against regressing Montgomery

Fine-tuning on 40 DDR images risks catastrophic forgetting of the 0.980 Montgomery
Dice. A 4-case fold's own validation loss is too noisy a signal to trust for
picking which epoch's weights to keep, so stopping and checkpoint selection are
decoupled:

- **When to stop training**: a generous early-stopping patience on the fold's own
  validation loss — compute-saving only, not used to pick the final checkpoint.
- **Which checkpoint to keep**: every epoch, also score the current weights against
  the existing 14-image Montgomery validation split (cheap — under a second).
  **Pre-registered regression floor: Montgomery Dice must stay >= 0.96** (from a
  0.980 baseline — roughly 2 points of acceptable drop). Among epochs meeting the
  floor, keep whichever has the lowest fold validation loss.
- **Fold failure is a valid, reported outcome**: if no epoch in a fold ever meets
  the floor, that fold's OOF predictions fall back to unmodified `best_model.h5`
  weights, and this is reported explicitly (not silently dropped) — "fine-tuning
  found no improvement within the regression budget on fold N" is itself a result.

The fine-tuned model is saved separately (`best_model_ddr_finetuned.h5`) —
`best_model.h5` is never overwritten.

### Deliverable

A new script implementing the 5-fold loop, producing:
1. Out-of-fold DDR Dice/point-error — the honest, leakage-free "after" number,
   directly comparable to Phase 1's "before".
2. Per-fold Montgomery Dice trajectory and the selected checkpoint's epoch — the
   regression guard, and evidence the floor rule was actually applied rather than
   asserted.

## Phase 1 addendum — results (run 2026-08-24)

`Step7CalibrateDDR.py`, full 20-case coordinate ascent:

| Stage | Winner | vs. baseline (paired) | Dice mean/median | Point error mean/median |
|---|---|---|---|---|
| baseline | — | — | 0.6617 / 0.6723 | 43.1mm / 42.5mm |
| windowing | min/max (current) | 0W/0L/20T vs itself; window tags: 2W/18L/0T | window tags: 0.6519 / 0.6650 | window tags: 48.0mm / 48.9mm |
| kernel size | 21 | 18W/2L/0T (same 18 cases win at 9, 15, and 21) | 0.6632 / 0.6747 | 42.6mm / 42.1mm |
| image source | images_pres (current) | raw: 0W/20L/0T | raw: 0.0195 / 0.0172 | raw: 223.8mm / 223.1mm |

**Conclusion: preprocessing is not the problem.** The kernel-size win is real —
18/20 cases improve consistently across three larger kernel sizes, binomial
p≈0.0002 against a no-effect null — but tiny (Dice +0.0015, point error
-0.5mm). Windowing makes things *worse*. Raw pre-log-conversion sensor data is
unusable outright (Dice collapses to 0.02) — normalizing it would need its own
processing pipeline, not a parameter swept here.

A visual check (`KaU007_PA_deep`, frame 10, one of the two fixed worst-baseline
spot-check cases) showed the baseline and kernel=21 overlays as visually
indistinguishable — the spurious trailing filament noted in the original
baseline check is still present at kernel=21. The predicted contour disagrees
with the true boundary broadly, not just at that one artifact, confirming this
is a distributed segmentation-quality problem rather than a cleanup-fixable
edge case.

**Action taken**: `pipeline.OPEN_KERNEL` default changed from 5 to 21 (the one
real, if small, win). Verified this does not regress Montgomery (Dice 0.9856
at both kernel=5 and kernel=21, identical to 4 decimal places, on the same
15-image check used for the original baseline). Windowing and image source
are left at their current defaults (both already the winners).

**Gate decision**: proceeds to Phase 2. The residual gap after calibration
(Dice 0.663 vs. Montgomery's 0.980; point error 42.6mm) is essentially
unchanged from the pre-calibration baseline — preprocessing has been ruled
out, not fixed, so weight adaptation is the remaining lever.

## Phase 2 addendum — results (run 2026-08-24)

`Step8FineTuneDDR.py`, 5-fold group cross-validation by case (seed 42), starting
from the Phase 1-calibrated `best_model.h5`, LR 1e-5, all layers trainable, up
to 30 epochs, early stopping patience 8, checkpoint gated on Montgomery Dice
≥ 0.96 (measured every epoch on a fixed 15-image Montgomery check set):

| Fold | Held-out cases | Epochs run | Montgomery Dice range (epoch 0 → last) | Checkpoint | Fold DDR Dice (unmodified model) |
|---|---|---|---|---|---|
| 0 | KaU001, 002, 016, 018 | 12 | 0.947 → 0.822 | none — floor never met | 0.691 |
| 1 | KaU004, 006, 009, 012 | 13 | 0.931 → 0.801 | none — floor never met | 0.711 |
| 2 | KaU003, 014, 017, 019 | 11 | 0.936 → 0.799 | none — floor never met | 0.719 |
| 3 | KaU005, 010, 013, 020 | 11 | 0.952 → 0.869 | none — floor never met | 0.637 |
| 4 | KaU007, 008, 011, 015 | 14 | 0.952 → 0.861 | none — floor never met | 0.559 |

**5/5 folds fell back to the unmodified model.** No fold's Montgomery Dice
reached 0.96 at *any* epoch, let alone stayed there — the closest any fold
got was 0.952, at epoch 0, before a single full pass of gradient updates had
converged. By epoch 1, every fold had already dropped below 0.94. This isn't
fold-specific noise: it's the same pattern in all five folds, meaning it's a
property of the recipe (LR 1e-5, all-layer fine-tuning, 16-case DDR training
set), not of any particular held-out split.

Contrast with the earlier 5-case smoke test, where all 5 (much smaller,
4-case-training) folds passed the floor comfortably: full training-set scale
converges each epoch's gradient toward DDR much faster per epoch than the
smoke test's handful of images did, and that speed is what blows past the
Montgomery floor before any single epoch is done adapting to DDR — i.e. more
DDR training data made forgetting *worse*, not better, at this LR.

Per-fold "Fold DDR Dice" above is scored with the unmodified model (since
every fold fell back) and varies 0.559–0.719 purely from which four cases
landed in that fold's held-out split — consistent with the wide per-case
spread already seen in the Phase 1 addendum, not a fine-tuning effect.

**Final out-of-fold result**: identical to the Phase 1 baseline, because
every fold's prediction comes from the same unmodified `best_model.h5`.

| Stage | Dice | Point error |
|---|---|---|
| Montgomery (in-domain) | 0.980 | — (no independent point ground truth) |
| Baseline (pre-calibration) | 0.662 | 107.7px / 43.1mm |
| Phase 1 (kernel=21, calibrated) | 0.663 | 106.5px / 42.6mm |
| Phase 2 (fine-tuned, OOF) | 0.663 | 106.5px / 42.6mm |

**Conclusion: fine-tuning, as configured, contributes nothing.** The domain
gap (0.980 → 0.663 Dice) is exactly as large after Phase 2 as after Phase 1.
This is an honest negative result, not a bug — the regression floor did its
job (Montgomery is unharmed), but at the cost of rejecting every candidate
checkpoint. The Montgomery-Dice trajectories above point at *why*: forgetting
happens within the first epoch, before the model has adapted much to DDR at
all, so there is no epoch where "enough DDR adaptation" and "not too much
forgetting" coincide at this LR/scale.

**Gate decision**: does not close the gap; two live options, not mutually
exclusive — (a) make the recipe more conservative (freeze the encoder and
fine-tune only the decoder, and/or drop LR by 10x+) so DDR adaptation happens
slowly enough for some epoch to land inside the floor, or (b) accept this as
Phase 2's result and report the domain gap as unclosed by fine-tuning at the
data scale available (20 cases). Neither was attempted in this run; the
choice is deferred to whoever picks this back up.

## Out of scope

- Retraining from scratch.
- Using diaphragm-point annotations as a training signal.
- Any change to `diaphragm.py`, `landmarks.py`, or `render.py` — Phase 1 and 2 only
  touch preprocessing and model weights; the geometry pipeline already tracks
  correctly when given a reasonable mask (confirmed visually in the baseline
  overlay check).
