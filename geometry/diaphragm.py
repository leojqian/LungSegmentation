# Diaphragm geometry: a binary lung mask in, a hemidiaphragm curve out.
#
# Pure array math — no file I/O, no drawing. The diaphragm is the lung's lower
# border, but only part of that border is diaphragm: near the midline it climbs
# the cardiac silhouette, and laterally it climbs the chest wall. Most of the
# work here is deciding which parts to discard.
#
# Rationale and measurements for every threshold live in
# docs/superpowers/specs/2026-08-15-diaphragm-mapping-design.md

from collections import namedtuple

import cv2
import numpy as np

from geometry.landmarks import bottom_corners, landmarks_of, top_point

#: measured = traced from the mask; fitted = dome extrapolated behind the heart
Diaphragm = namedtuple("Diaphragm", "measured fitted")

#: the four points derived from one lung contour. dome may be a lower bound
#: rather than a true height — call apex_is_lower_bound on the diaphragm curve
#: (from diaphragm_of) if that distinction matters to the caller.
LungPoints = namedtuple("LungPoints", "top lower_left lower_right dome")

CLIFF_STEP = 8          # px/column jump that marks the cardiac border
RISE_FRACTION = 0.20    # max climb above the costophrenic angle, as % lung height
DOME_EXTEND = 0.35      # how far the fitted dome may reach past measured data
APEX_TOLERANCE = 0.15   # how far past measured data a fitted apex is trusted


# --- mask preparation --------------------------------------------------------

def largest_component(mask):
    """Biggest connected blob only. Some masks carry a stray second blob."""
    m = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        raise ValueError("mask is empty — no lung pixels to trace")
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def split_lungs(mask):
    """Combined two-lung mask -> (image_left, image_right).

    The U-Net is trained on left+right merged, so a prediction holds both lungs
    in one image and must be split before either can be traced.
    """
    m = np.asarray(mask).astype(np.uint8)
    if m.max() > 1:
        m = (m > 127).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n - 1 < 2:
        raise ValueError(f"expected 2 lungs in the mask, found {max(n - 1, 0)}")
    biggest = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2]
    return tuple(labels == i for i in sorted(biggest, key=lambda i: centroids[i][0]))


def cliff_step_for(mask_height, model_size=256, factor=3):
    """Cliff threshold for a mask upscaled from a model_size-square prediction.

    CLIFF_STEP is an absolute pixel count, so it must track the mask's effective
    resolution: a 256px prediction blown up 11x turns every source pixel into an
    11px staircase that would trip the default.
    """
    return max(1, int(round(factor * mask_height / model_size)))


# --- tracing the lower border ------------------------------------------------

def bottom_profile(mask):
    """Lowest set pixel per column -> (xs, ys). Empty columns are omitted."""
    m = np.asarray(mask).astype(bool)
    xs = np.where(m.any(axis=0))[0]
    lowest = m.shape[0] - 1 - np.argmax(m[::-1], axis=0)
    return xs, lowest[xs]


def smooth(ys, k):
    """Median filter with edge padding. Even kernels round up to odd."""
    k = int(k)
    if k <= 1:
        return np.asarray(ys)
    k += k % 2 == 0
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(np.asarray(ys), k // 2, mode="edge"), k)
    return np.median(windows, axis=-1).astype(ys.dtype)


def trim_cliffs(xs, ys, max_step):
    """Cut at steps steeper than max_step — the abrupt cardiac border.

    Anchored at the deepest point, which is the costophrenic angle and so is
    always genuine diaphragm.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    cliffs = np.abs(np.diff(ys.astype(float))) > max_step
    return _run_around(xs, ys, int(np.argmax(ys)),
                       lambda i: not cliffs[i], lambda i: not cliffs[i - 1])


def trim_rise(xs, ys, max_rise):
    """Cut where the curve has climbed max_rise above its deepest point.

    On the patient's left the cardiac border merges as a gradual diagonal at the
    same slope as the dome, so only total climb separates them.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    anchor = int(np.argmax(ys))
    within = ys >= ys[anchor] - max_rise
    return _run_around(xs, ys, anchor, lambda i: within[i + 1], lambda i: within[i - 1])


def trim_terminal_upturn(xs, ys, max_frac=0.25):
    """Drop a short hook at either end where the border turns back upward.

    Past the costophrenic angle the border climbs the lateral chest wall. A dome
    flank rises too, so length separates them: a wall hook is a small fraction
    of the curve, a dome flank is most of it.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    n, limit = len(ys), max_frac * len(ys)

    lo = 0
    while lo + 1 < n and ys[lo + 1] > ys[lo]:
        lo += 1
    hi = n - 1
    while hi > 0 and ys[hi - 1] > ys[hi]:
        hi -= 1
    if lo > limit:
        lo = 0
    if (n - 1 - hi) > limit:
        hi = n - 1
    return (xs, ys) if lo >= hi else (xs[lo:hi + 1], ys[lo:hi + 1])


def trim_to_span(xs, ys, lo, hi, min_points=10):
    """Clip to [lo, hi] — the cardiophrenic and costophrenic angles.

    Unlike the slope-based trims, these corners come from the contour via
    landmarks.bottom_corners, so they are independent evidence rather than
    another threshold tuned on the same data. The original is kept if clipping
    would leave almost nothing.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    lo, hi = sorted((lo, hi))
    keep = (xs >= lo) & (xs <= hi)
    return (xs, ys) if keep.sum() < min_points else (xs[keep], ys[keep])


def _run_around(xs, ys, anchor, extend_right, extend_left):
    """Contiguous run around anchor, growing while each predicate holds."""
    lo = anchor
    while lo > 0 and extend_left(lo):
        lo -= 1
    hi = anchor
    while hi < len(ys) - 1 and extend_right(hi):
        hi += 1
    return xs[lo:hi + 1], ys[lo:hi + 1]


def trace_diaphragm(mask, kernel=None, max_step=CLIFF_STEP,
                    max_rise_frac=RISE_FRACTION, min_columns=10):
    """Lung mask -> (N, 2) measured points along the hemidiaphragm.

    kernel defaults to ~0.5% of the mask width: near a no-op on hand-drawn
    masks, real work on ragged predictions. Masks narrower than min_columns
    raise rather than returning a curve of a few pixels that would read as a
    real measurement.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    blob = largest_component(m > 127 if m.max() > 1 else m.astype(bool))

    xs, ys = bottom_profile(blob)
    if len(xs) < min_columns:
        raise ValueError(f"mask spans only {len(xs)} columns, need {min_columns}")

    ys = smooth(ys, kernel if kernel is not None
                else max(1, int(round(0.005 * len(xs)))))

    xs, ys = trim_cliffs(xs, ys, max_step)                       # cardiac, abrupt
    xs, ys = trim_rise(xs, ys, max_rise_frac * blob.any(axis=1).sum())  # cardiac, gradual
    xs, ys = trim_terminal_upturn(xs, ys)                        # lateral chest wall
    return np.stack([xs, ys], axis=1).astype(int)


# --- fitting the hidden dome -------------------------------------------------

def fit_dome(curve, x0, x1, min_points=10, max_extend=None):
    """Fit a quadratic through the measured points and evaluate over [x0, x1].

    The medial dome hides behind the heart with no lung above it to trace. A
    quadratic is the simplest shape with a single peak; in image coordinates it
    domes upward only when its leading coefficient is positive, so a negative
    one means these points do not describe a dome and nothing is invented.

    Points outside the measured range are INFERRED — callers should draw them
    differently.
    """
    curve = np.asarray(curve)
    if len(curve) < min_points:
        raise ValueError(f"need >= {min_points} points to fit a dome, got {len(curve)}")

    a, b, c = np.polyfit(curve[:, 0].astype(float), curve[:, 1].astype(float), 2)
    if a <= 0:
        raise ValueError("traced points do not describe a dome (fit opens downward)")

    x0, x1 = int(x0), int(x1)
    if max_extend is not None:
        lo, hi = int(curve[:, 0].min()), int(curve[:, 0].max())
        x0, x1 = max(x0, lo - int(max_extend)), min(x1, hi + int(max_extend))

    xs = np.arange(x0, x1 + 1)
    fitted = np.stack([xs, np.polyval([a, b, c], xs)], axis=1).astype(int)
    return _clip_below(fitted, curve[:, 1].max())


def _clip_below(fitted, limit):
    """Run around the peak that stays above limit — the deepest measured point.

    A parabola's arms keep descending; without this they sink past the
    costophrenic angle into the abdomen.
    """
    ok = fitted[:, 1] <= limit
    xs, _ = _run_around(fitted[:, 0], fitted[:, 1], int(np.argmin(fitted[:, 1])),
                        lambda i: ok[i + 1], lambda i: ok[i - 1])
    keep = (fitted[:, 0] >= xs[0]) & (fitted[:, 0] <= xs[-1])
    return fitted[keep]


# --- locating the apex -------------------------------------------------------

def dome_apex(curve, tol=2):
    """Highest point of a curve, averaged across the plateau it sits on.

    A shallow dome sits at one integer height for tens of columns, so a plain
    argmin returns whichever end of that plateau comes first and jitters. Only
    the contiguous run through the peak is averaged: a curve can reach the same
    height in two places, and averaging both would put the apex in the dip
    between them, off the curve entirely.
    """
    curve = np.asarray(curve)
    ys = curve[:, 1]
    top = ys.min()
    tied = ys <= top + tol
    xs, _ = _run_around(curve[:, 0], ys, int(np.argmin(ys)),
                        lambda i: tied[i + 1], lambda i: tied[i - 1])
    return np.array([int(round(xs.mean())), int(top)])


def apex_source(measured, fitted, tolerance=APEX_TOLERANCE):
    """Which curve the apex should come from.

    The fitted dome only while its vertex stays within tolerance of the measured
    span — beyond that the vertex is wherever curvature implies, hundreds of
    columns from any evidence. Callers need this as well as the apex, so a
    height rule can be drawn against the same curve the apex came from.
    """
    measured = np.asarray(measured)
    if fitted is None:
        return measured
    lo, hi = measured[:, 0].min(), measured[:, 0].max()
    margin = tolerance * (hi - lo)
    return fitted if lo - margin <= dome_apex(fitted)[0] <= hi + margin else measured


def apex_of(measured, fitted, tolerance=APEX_TOLERANCE):
    """Dome apex, from the fit where it is trustworthy and the measurement else."""
    return dome_apex(apex_source(measured, fitted, tolerance))


def center_point(curve):
    """Point on curve at its horizontal center -- not necessarily its peak.

    Some ground truth (DDR's DM-MODE_truth) defines "the diaphragm point" this
    way instead of as a true apex: whatever height the curve happens to be at
    its horizontal midpoint. Kept alongside dome_apex so a predicted curve can
    be reduced to a point the same way truth was, for a same-definition
    comparison against dome_apex's own (different) definition.
    """
    curve = np.asarray(curve, dtype=float)
    order = np.argsort(curve[:, 0])
    xs, ys = curve[order, 0], curve[order, 1]
    center_x = (xs.min() + xs.max()) / 2
    return np.array([center_x, np.interp(center_x, xs, ys)])


def center_of(measured, fitted, tolerance=APEX_TOLERANCE):
    """Curve's center point, from the fit where it is trustworthy and the measurement else.

    Same curve-selection as apex_of (apex_source) -- only the point-reduction
    rule differs (center_point vs. dome_apex).
    """
    return center_point(apex_source(measured, fitted, tolerance))


def apex_is_lower_bound(measured, apex, tol_frac=0.05):
    """True when the apex sits at an end of the measured curve.

    The dome then hides behind the mediastinum and the curve is a bare flank, so
    its highest point is wherever measurement stopped — a floor on dome height,
    not a measurement of it.
    """
    measured = np.asarray(measured)
    lo, hi = measured[:, 0].min(), measured[:, 0].max()
    return bool(min(abs(apex[0] - lo), abs(apex[0] - hi)) < tol_frac * (hi - lo))


# --- the whole calculation ---------------------------------------------------

def diaphragm_of(mask, max_rise_frac=RISE_FRACTION, max_step=CLIFF_STEP):
    """Lung mask -> Diaphragm(measured, fitted). fitted is None if no dome fits."""
    measured = trace_diaphragm(mask, max_step=max_step, max_rise_frac=max_rise_frac)

    try:  # bound by the cardiophrenic / costophrenic corners
        corners = landmarks_of(largest_component(mask))
        xs, ys = trim_to_span(measured[:, 0], measured[:, 1],
                              corners["lower_left"][0], corners["lower_right"][0])
        measured = np.stack([xs, ys], axis=1)
    except ValueError:
        pass

    cols = np.where(np.asarray(mask).any(axis=0))[0]
    span = measured[:, 0].max() - measured[:, 0].min()
    try:
        return Diaphragm(measured, fit_dome(measured, cols.min(), cols.max(),
                                            max_extend=DOME_EXTEND * span))
    except ValueError:
        return Diaphragm(measured, None)


def measure_lung(contour, max_rise_frac=RISE_FRACTION, max_step=CLIFF_STEP):
    """One lung contour -> LungPoints(top, lower_left, lower_right, dome).

    contour: (N, 2) or OpenCV's (N, 1, 2), in image coordinates.

    The contour is rasterised onto a canvas just large enough to hold it. Since
    that canvas starts at the origin, every point returned stays in the original
    image's coordinates, and no image is needed.

    Masks from a U-Net prediction should pass max_step=cliff_step_for(height):
    the default suits hand-drawn masks, whose edges are 1px smooth. Callers who
    also need the diaphragm curve itself (to draw it, or to check whether dome
    is a lower bound) should call diaphragm_of on the same contour instead.
    """
    pts = np.asarray(contour).reshape(-1, 2)
    if pts.min() < 0:
        raise ValueError("contour has negative coordinates")
    mask = np.zeros((int(pts[:, 1].max()) + 1, int(pts[:, 0].max()) + 1), np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)

    lower_left, lower_right = bottom_corners(pts)
    dome = apex_of(*diaphragm_of(mask, max_rise_frac, max_step))

    return LungPoints(top=top_point(pts), lower_left=lower_left,
                      lower_right=lower_right, dome=dome)
