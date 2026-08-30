# Diaphragm Mapping — Design

Date: 2026-08-15
Status: implemented 2026-08-16. The algorithm section below was revised during
implementation — the original assumption that the whole lower border is diaphragm
proved wrong. See "Cardiac border" below.

## Goal

Locate the diaphragm in each image of the Montgomery County chest X-ray set and
represent it as a curve: one per hemidiaphragm, left and right. Render the curves
onto the X-rays for visual inspection, and persist the curve geometry for later use.

## Scope decisions

| Question | Decision |
|---|---|
| What structure | Hemidiaphragm curves — the inferior border of each lung field. Not the mediastinum. |
| Input | The provided `ManualMask` PNGs. Not raw pixels, not U-Net output. |
| Output | Overlay images plus curve coordinates in `.npz`. |
| Viewing | Batch-to-disk by default; `--view` opens an interactive window instead. |

Deriving the curves from the manual masks makes this pure geometry — deterministic,
no training, runs on all 138 images today. Because the geometry consumes a binary
mask and does not care where the mask came from, the same `extract_diaphragm`
function will later accept U-Net predictions unchanged.

## Data findings

Measured against the actual dataset before designing:

- 138 CXRs, 138 left masks, 138 right masks. Sorted basenames match exactly across
  all three directories, so `sorted(glob(...))` pairs them correctly.
- CXR and mask shapes match in all 138 pairs.
- Masks are clean binary (`{0, 255}`), images are `uint8` grayscale.
- **No mask touches the bottom image edge.** The full diaphragm dome is inside the
  annotation; nothing is clipped.
- **Boundaries are already smooth** — median vertical step between adjacent columns
  is 1px.
- **274 of 276 masks are a single connected component.** `MCUCXR_0399_1.png` and
  `MCUCXR_0052_0.png` each contain a stray second blob that must be discarded.
- **Anatomy is upright in all 138.** 41 frames are wider than tall, but single-lung
  bounding boxes are taller than wide in every image (min h/w = 1.46), confirming
  those are wider films rather than rotated patients. A vertical per-column scan is
  therefore valid for the entire set.
- Images are large (~4020x4892, ~19.7 MP). Processing is one image at a time; only
  the render step downscales.

## Architecture

Two files, separating pure geometry from I/O.

### `diaphragm.py`

No file reads, no drawing, no global state. Arrays in, arrays out.

| Function | Contract |
|---|---|
| `largest_component(mask)` | binary mask -> same mask with only the largest blob retained |
| `bottom_profile(mask)` | -> `(xs, ys)`, the lowest set pixel per occupied column |
| `smooth(ys, k)` | median filter over `ys` with odd kernel `k` |
| `trim_cliffs(xs, ys, max_step)` | cut to the run around the deepest point, bounded by steep steps |
| `trim_rise(xs, ys, max_rise)` | cut where the profile climbs `max_rise` above its deepest point |
| `split_lungs(mask)` | combined two-lung mask -> the two lungs, in image order |
| `dome_apex(curve)` | highest point of a curve, tied columns averaged |
| `extract_diaphragm(mask)` | composes the above -> `(N, 2)` int array of `(x, y)` |

### `Step5MapDiaphragm.py`

The driver. Globs and pairs files, calls `extract_diaphragm`, renders, writes
outputs, and hosts the CLI. All real-world messiness lives here so the geometry
stays trivially testable.

Naming follows the existing `StepNName.py` convention, though this script does not
depend on Steps 2-4 — it reads manual masks, not model output.

## Algorithm

```
mask > 127
  -> largest connected component
  -> per column x containing any lung pixel: y = lowest set pixel
  -> median filter y, kernel ~= 0.5% of mask width, forced odd
  -> trim_cliffs:  cut at steps > 8 px/column
  -> trim_rise:    cut where the climb exceeds 15% of lung height
  -> (x, y) points
```

Given the measured 1px roughness, smoothing is close to a no-op on manual masks. It
is retained as insurance for ragged U-Net predictions later, where the same function
will be doing real work.

Both trims are anchored at the profile's deepest point, which is the costophrenic
angle and therefore always genuine diaphragm.

### Cardiac border (revision)

The original design assumed the entire per-column bottom profile was diaphragm.
Rendering the first overlays disproved it: near the midline the lung field sits
*above* the heart, so "lowest lung pixel in this column" jumps to the cardiac
border and the curve runs vertically up the mediastinum. Two different failures,
needing two different trims:

- **Right lung** — the lung's lower border meets the cardiac border as an abrupt
  step. Measured: ~2 px/column across the dome, 8+ at the transition, with only
  ~2% of columns exceeding 8. `trim_cliffs` cuts there.
- **Left lung** — the border merges into the cardiac silhouette as a *gradual
  diagonal* at 0.5-2.5 px/column, indistinguishable from the dome by slope alone.
  What separates them is total climb: the dome rises only ~10% of lung height
  above the costophrenic angle, while the cardiac border keeps climbing for ~80%.
  `trim_rise` cuts on that budget.

There is a third failure the two trims above cannot see: past the costophrenic
angle the border turns back up the **lateral chest wall**. That hook is too
gradual for `trim_cliffs` and fits inside `trim_rise`'s budget, yet being the
highest point of the curve it captures the apex — which is how a dome rule ends
up drawn on the chest wall (MCUCXR_0008_0). A dome's own flank rises too, so
length separates them: a wall hook is a small fraction of the curve, a dome flank
is most of it. `trim_terminal_upturn` drops runs shorter than 25%.

### Tuning `max_rise_frac`

Exposed as `--rise`. Measured across all 276 hand-drawn hemidiaphragms:

| `--rise` | traced span | apex actually measured | no dome fit |
|---|---|---|---|
| 0.15 | 37% | 116/276 (42%) | 6 |
| 0.20 | 52% | 204/276 (74%) | 3 |
| **0.25** | **66%** | **236/276 (86%)** | 4 |
| 0.30 | 68% | 233/276 | 12 |
| 0.40 | 72% | 228/276 | 18 |

An early setting of 0.15 was calibrated before `trim_cliffs` and
`trim_terminal_upturn` existed, so it was doing their job and doing it badly: it
amputated real diaphragm. On MCUCXR_0043_0 the dome rises 360px = **16.3% of lung
height**, just over that budget, so ~600 columns of *measured* border were
discarded and then re-invented by extrapolation. Abrupt cardiac borders are caught
by `trim_cliffs` regardless — 0043's is a 396px step.

At 0.25 the fraction of apexes sitting on measured rather than extrapolated
geometry rises from 42% to 86%. Past 0.30 the no-fit count climbs as the cardiac
border re-enters and the curve stops being dome-shaped.

**Independent check.** With 0.25, the image-left hemidiaphragm (the patient's
RIGHT, which the liver elevates) is higher in **121/138 images — 88%**, against a
textbook expectation of ~90%. At 0.15 it was 90/138 (65%). Nothing in the tuning
optimised for this, so the agreement is evidence the geometry improved rather
than merely got longer.

### Error handling

- Empty mask -> raise `ValueError`. Never return an empty curve silently.
- Mask with no positive pixels after component selection -> same.
- Missing CXR for a given mask -> skip with a warning naming the file; continue the
  batch. One bad pair must not abort a 138-image run.

## Outputs

```
outputs/
  overlays/MCUCXR_0001_0.png     downscaled CXR with left + right curves drawn
  contact_sheet_01..06.png       24 thumbnails each, filename captioned
  curves.npz                     {"<id>_left": (N,2), "<id>_right": (N,2),
                                  "<id>_shape": (h,w)}
```

Curve coordinates are stored in **full-resolution pixel space**, with the source
shape recorded alongside, so they remain valid regardless of any later resizing.
Overlays downscale to 1024px maximum dimension for viewing only.

Contact sheets exist because clicking through 138 individual images is how a
systematic failure gets missed. A curve that has gone wrong — cutting across the
lung, hugging an image edge, absent — is obvious at thumbnail size, since the defect
is one of shape rather than fine detail. Individual overlays are then opened only
for the ones that look wrong.

## CLI

```bash
python Step5MapDiaphragm.py                 # batch: overlays, contact sheets, npz
python Step5MapDiaphragm.py --view          # interactive window, writes nothing
python Step5MapDiaphragm.py --rise 0.2      # longer curves (see table above)
python Step5MapDiaphragm.py --limit 6       # quick iteration on a few images
python Step5MapDiaphragm.py --masks         # tint the lung regions under the curves
```

`--masks` blends each lung mask over the X-ray at 30% opacity in the same colour as
its curve, then draws the curve on top. This turns "is the curve right?" into a
single-image check: the curve should lie along the bottom edge of its own tint, and
wherever it stops short, the tint shows exactly how much of the lower border was
trimmed as cardiac. It is the fastest way to judge a `--rise` value.

Viewer keys: `space` next, `b` back, `esc` quit. Built on `cv2.imshow`; the
installed `opencv-python` 5.0.0.93 has Cocoa GUI support, so no new dependency.

## Dependencies

`cv2`, `numpy`, `tqdm` — exactly what `Step1LoadData.py` already imports. Drawing
uses `cv2.polylines`; matplotlib is not installed and is not needed. `pytest` must
be added to run the test suite.

## Testing

Synthetic masks with analytically known answers, so correctness does not depend on
eyeballing X-rays.

- Filled half-disc of known center and radius -> extracted profile matches the
  circle within 1px; apex lands at the known column.
- Two-blob mask -> `largest_component` retains the intended blob.
- Empty mask -> raises `ValueError`.
- Regression on `MCUCXR_0399_1` and `MCUCXR_0052_0` -> curve width matches the kept
  component rather than spanning both blobs.
- Smoke test across all 276 masks -> every curve non-empty, `x` strictly increasing.

## Dome height measurement

Each curve's apex is marked with a dot and a full-width horizontal rule, so the two
hemidiaphragm heights can be compared by eye, and the elevation difference is
reported numerically. Apexes are saved to `curves.npz` as `<id>_left_apex` /
`<id>_right_apex`.

Tied columns are averaged when locating the apex, following the same reasoning as
`averagePoint` in Step4TestTheModel: a shallow dome sits at the same integer height
for tens of columns, so a plain `argmin` returns whichever end of that plateau comes
first and jitters under small mask changes. Flattened diaphragms — the COPD case
this measurement is most useful for — are the flattest of all.

Across the 138 manual masks the image-left hemidiaphragm is higher in 79, median
difference +22px. That is the expected direction: image-left is the patient's RIGHT
lung, and the liver pushes the right hemidiaphragm up.

**Caveat.** When a curve is truncated at the cardiophrenic angle, its highest point
is its own medial endpoint rather than a true dome peak, so the apex inherits the
trim's sensitivity. Reported heights are pixel measurements in each image's own
resolution; without DICOM spacing they are not physical distances and are only
comparable within an image, not across images.

## Step 4 integration

`Step4TestTheModel.py` keeps its existing logic — model load, prediction, mask
overlay, corner landmarks — and adds a fourth window, "Diaphragm", carrying the
traced curves and apex rules on top of the predicted mask.

Two things had to be handled for model output, neither of which arises with the
manual masks:

1. **Splitting.** Step 1 merges left+right before training, so the model predicts
   both lungs in one image. `extract_diaphragm` keeps only the largest component,
   so a raw prediction silently loses a lung. `split_lungs` runs first.
2. **Cliff threshold scaling.** The model outputs 256x256; upscaled to display size
   one predicted pixel becomes an ~11px step, which always trips the 8px `max_step`
   tuned on 1px-smooth hand-drawn masks. A single such step mid-diaphragm truncated
   the right lung's curve to 19% of its width. Step 4 derives the threshold instead:
   `3 * (display_height / 256)`, which restores the curve to 95% while still
   catching the real cardiac border. At 50px the lateral chest wall gets swallowed.

The general lesson: `max_step` is an absolute pixel threshold, so it must scale with
the effective resolution of whatever produced the mask, not with the image size.

## Out of scope

- Clinical metrics (dome height, costophrenic angle, flattening index).
- Detecting the diaphragm from raw pixels without a mask.
- Any change to the U-Net pipeline in Steps 2-4.
