# lungmap

Diaphragm geometry from chest X-ray lung masks. Given a lung contour, it returns
the lung apex, both bottom corners, and the diaphragm's high point — plus the
diaphragm curve itself.

No model, no training, no network. Deterministic geometry over `numpy` and
`opencv-python`.

## Install

```bash
pip install git+https://github.com/<you>/lung-segmentation
```

## Use

```python
import cv2
from lungmap import measure_lung

image = cv2.imread("chest.png", cv2.IMREAD_GRAYSCALE)
mask  = your_lung_segmentation(image)          # binary, ONE lung

contours, _ = cv2.findContours(mask.astype("uint8"),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)

lung = measure_lung(contour)
```

`measure_lung(contour)` takes an `(N, 2)` or OpenCV `(N, 1, 2)` contour in image
coordinates. No image is needed — the contour is rasterised onto a canvas sized
from itself, so every point returned stays in the original image's coordinates.
It returns a `LungPoints` with four fields:

| field | meaning |
|---|---|
| `.top` | lung apex |
| `.lower_left` | bottom corner, image-left |
| `.lower_right` | bottom corner, image-right |
| `.dome` | the diaphragm's high point |

`.dome` can be a *lower bound* rather than a true height, when the dome hides
behind the mediastinum — see Limits below. If you need to know which, or need
the diaphragm curve itself (e.g. to draw it), call `diaphragm_of(mask)` on the
same lung instead of `measure_lung`; it returns `Diaphragm(measured, fitted)` and
pairs with `apex_of` and `apex_is_lower_bound`.

### Two lungs in one mask

If your segmenter outputs both lungs merged:

```python
from lungmap import split_lungs
left, right = split_lungs(combined_mask)     # ordered by image position
```

### Masks from a neural net

The default cliff threshold suits hand-drawn masks, whose edges are ~1px smooth.
A mask upscaled from a small prediction has staircase edges that trip it:

```python
from lungmap import cliff_step_for
lung = measure_lung(contour, max_step=cliff_step_for(image.shape[0]))
```

### Drawing

```python
from lungmap import apex_is_lower_bound, diaphragm_of, draw_diaphragm, draw_height_rule

curve = diaphragm_of(mask)
draw_diaphragm(canvas, curve, color=(80, 200, 255))
draw_height_rule(canvas, lung.dome, color=(80, 200, 255), label="R",
                 lower_bound=apex_is_lower_bound(curve.measured, lung.dome))
```

Two conventions carry meaning: **solid vs dashed** is measured vs inferred, and
**filled vs hollow** is a height vs a lower bound on one.

## How it works

The diaphragm is the lung's lower border — but only part of that border is
diaphragm. Near the midline it climbs the cardiac silhouette; laterally it climbs
the chest wall. So the method reads the border, then discards what isn't
diaphragm:

1. lowest lung pixel per column, median-smoothed
2. `trim_cliffs` — abrupt cardiac border
3. `trim_rise` — gradual cardiac border, which has the same slope as the dome
4. `trim_terminal_upturn` — lateral chest wall
5. `trim_to_span` — clip to the two bottom corners, found independently from the
   contour
6. `fit_dome` — quadratic through what survives, extended under the heart

Typically ~1700 columns in, ~790 out. Over half a lung's lower border is not
diaphragm.

## Limits — please read

**No ground truth.** This was developed against 138 images with lung masks but no
diaphragm annotations. Every threshold was fitted in-sample. It is a repeatable
estimate of a well-defined quantity, not a validated measurement of the
anatomical diaphragm. Don't build quantitative or clinical claims on it without
validating against annotated apexes.

**Lower bounds.** Where the dome hides behind the heart the measured curve is a
bare flank with no peak, so `.dome` is the highest *visible* point — a floor.
This was ~27% of hemidiaphragms in the development set, mostly the patient's
left. `measure_lung` does not report which; call `apex_is_lower_bound` on
`diaphragm_of(mask).measured` if that distinction matters to you.

**Upright PA films only.** The per-column scan assumes an upright frontal view. A
5° tilt halves the traced curve, and nothing warns you. Lateral views are
meaningless to it.

**It reads the mask, not the X-ray.** Where disease stops the lung mask above the
true diaphragm — effusion, consolidation — the curve confidently follows the top
of the disease instead. Nothing flags this.

**Two lungs required** by `split_lungs`; it raises rather than guessing.

## Requirements

`numpy >= 1.20`, `opencv-python >= 4.5`, Python >= 3.9.
