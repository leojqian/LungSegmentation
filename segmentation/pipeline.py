# One entry point: any supported lung image in, segmented masks + diaphragm
# curves + landmark points out. Ties together format-specific loading
# (dicom_io.py), the U-Net (segmentation), and the pure geometry
# (diaphragm.py/landmarks.py) that Step4/5/6 previously each reimplemented.

import os
from collections import namedtuple

import cv2
import numpy as np

from geometry.diaphragm import (CLIFF_STEP, RISE_FRACTION, cliff_step_for, diaphragm_of,
                                measure_lung, split_lungs)
from formats.dicom_io import load_frames, to_model_input, window_params
from geometry.landmarks import contour_of

MODEL_SIZE = 256
OPEN_KERNEL = 21    # calibrated 2026-08-24 against the DDR set — see the Phase 1
                    # addendum in docs/superpowers/specs/2026-08-24-ddr-domain-
                    # adaptation-design.md. Wins on 18/20 DDR cases but the gain
                    # is small (Dice +0.002); does not change Montgomery, where
                    # kernel choice was already shown not to matter (see the
                    # baseline pipeline-parity check in the same doc).
CLIFF_FACTOR = 3    # matches the factor cliff_step_for was tuned with in Step4/5/6

#: one segmented+measured image. masks/lungs/curves are keyed "R"/"L" (R = the
#: image-left lung, this codebase's convention for a PA film's patient-right
#: side). pixel_spacing_mm is None for formats with no physical calibration
#: (e.g. PNG); set for DICOM.
LungAnalysis = namedtuple("LungAnalysis",
                          "image mask masks lungs curves pixel_spacing_mm")


def load_image(path, frame=None, window=False):
    """Any supported image path -> (float32 HxW array in [0, 1], pixel_spacing_mm).

    Dispatches on file extension. `.dcm` files are multi-frame; `frame` selects
    which one (default 0). Other formats (png/jpg/...) are single images, where
    `frame` makes no sense and is rejected rather than silently ignored.

    window=True normalizes DICOM frames using the file's own WindowCenter/Width
    tags instead of that frame's min/max — see dicom_io.window_params. Falls
    back to min/max when a file has no window tags, same as window=False.
    """
    if os.path.splitext(path)[1].lower() == ".dcm":
        frames, spacing_mm, photometric = load_frames(path)
        wc, ww = window_params(path) if window else (None, None)
        image = to_model_input(frames[frame if frame is not None else 0], photometric, wc, ww)
        return image, spacing_mm

    if frame is not None:
        raise ValueError(f"{path} is not multi-frame DICOM; frame must be None")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image.astype(np.float32) / 255.0, None


def segment(image, model, open_kernel=OPEN_KERNEL):
    """Normalized image -> binary lung mask uint8, at the image's own resolution.

    Resizes down to the model's input size for prediction, then upscales the
    soft probabilities (not the threshold) back up before thresholding — see
    Step4's own note on why: thresholding at 256 first locks the boundary onto
    that grid and turns every mask pixel into a many-pixel staircase.
    """
    small = cv2.resize(image, (MODEL_SIZE, MODEL_SIZE))
    small = np.repeat(small[..., None], 3, axis=-1)
    soft = model.predict(np.expand_dims(small, 0), verbose=0)[0, ..., 0]

    full = cv2.resize(soft, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    opened = cv2.morphologyEx((full > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((open_kernel, open_kernel), np.uint8))
    return _keep_largest_components(opened, n=2)


def _keep_largest_components(mask, n):
    """Drop every connected component except the n largest.

    Clears spurious false-positive islands (soft-tissue/skin-fold activations
    seen on DDR) that survive morphological opening without touching the real
    lung blobs — exactly two are expected, so n=2 is a safe floor here.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels - 1 <= n:
        return mask
    keep = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:n]
    return np.isin(labels, keep).astype(np.uint8)


def measure_masks(masks, max_rise_frac=RISE_FRACTION, max_step=CLIFF_STEP):
    """{"R": mask, "L": mask} -> (lungs, curves), both keyed the same way.

    The no-model tail of analyze_frame, split out so callers who already have
    per-side masks in hand — Step5's manual-mask path never runs the U-Net —
    can reuse the same landmark/diaphragm logic instead of reimplementing it.

    Each side is measured independently: a mask too degenerate to trace (too
    few columns, empty after component selection) maps that side to None
    rather than losing the other side's already-successful measurement.
    """
    lungs, curves = {}, {}
    for side, mask in masks.items():
        try:
            lungs[side] = measure_lung(contour_of(mask), max_rise_frac, max_step)
        except ValueError:
            lungs[side] = None
        try:
            curves[side] = diaphragm_of(mask, max_rise_frac, max_step)
        except ValueError:
            curves[side] = None
    return lungs, curves


def analyze_frame(image, model, open_kernel=OPEN_KERNEL, max_rise_frac=RISE_FRACTION,
                  cliff_factor=CLIFF_FACTOR, pixel_spacing_mm=None):
    """An already-loaded, normalized image -> LungAnalysis.

    The building block `analyze` composes with `load_image`. Call this directly
    when frames are already in memory — e.g. looping over one multi-frame DICOM
    a caller loaded once — to avoid re-decoding the file per frame.

    Raises ValueError only when segmentation can't find two lungs at all; a
    side whose geometry can't be traced maps to None in .lungs/.curves rather
    than losing the other side (see measure_masks).
    """
    mask = segment(image, model, open_kernel)
    max_step = cliff_step_for(mask.shape[0], model_size=MODEL_SIZE, factor=cliff_factor)

    masks = dict(zip(["R", "L"], split_lungs(mask)))
    lungs, curves = measure_masks(masks, max_rise_frac, max_step)

    return LungAnalysis(image=image, mask=mask, masks=masks, lungs=lungs,
                        curves=curves, pixel_spacing_mm=pixel_spacing_mm)


def analyze(path, model, frame=None, open_kernel=OPEN_KERNEL,
           max_rise_frac=RISE_FRACTION, cliff_factor=CLIFF_FACTOR, window=False):
    """One image, any supported format -> LungAnalysis.

    path: image file. `.dcm` files are multi-frame; pass `frame` to pick which
    frame (default 0). Other formats (png/jpg/...) are single images. This is
    the one entry point for "segment, map the diaphragm, and get the landmark
    points" — Step4/5/6 all call this rather than reimplementing the pipeline.
    """
    image, spacing_mm = load_image(path, frame, window)
    return analyze_frame(image, model, open_kernel, max_rise_frac, cliff_factor, spacing_mm)
