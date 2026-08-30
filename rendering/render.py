# Drawing: turn measured geometry into pixels. No analysis happens here.
#
# Two conventions carry meaning and should not be changed casually:
#   solid vs dashed  — measured vs inferred (fitted behind the heart)
#   filled vs hollow — a dome height vs a lower bound on one

import cv2
import numpy as np

from geometry.diaphragm import apex_is_lower_bound, apex_of, apex_source
from geometry.landmarks import contour_of

VIEW_MAX_DIM = 1024
LEFT_COLOR = (80, 200, 255)    # amber (BGR), image-left lung
RIGHT_COLOR = (255, 180, 80)   # blue, image-right lung
OUTLINE_COLOR = (0, 255, 0)
POINT_COLORS = {"top": (255, 255, 0), "lower_left": (0, 165, 255),
                "lower_right": (255, 0, 255)}
MASK_ALPHA = 0.30
RULE_FRACTION = 0.11           # height rule reaches this much of canvas width each side
THUMBS_PER_SHEET = 24
SHEET_COLS = 6
DARK = (0, 0, 0)


# --- coordinate helpers ------------------------------------------------------

def display_scale(shape, max_dim=VIEW_MAX_DIM):
    """Factor bringing the longest side down to max_dim. Never upscales."""
    return min(1.0, max_dim / max(shape[:2]))


def scale_points(points, factor):
    return (np.asarray(points) * factor).astype(int)


def dash_segments(width, dash=18, gap=10):
    """[(x0, x1), ...] spanning width, clipped at the end."""
    return [(x, min(x + dash, width)) for x in range(0, width, dash + gap)]


def rule_extent(width, x, frac=RULE_FRACTION, bounds=None):
    """(x0, x1) for a height rule either side of x, clipped to canvas and curve.

    Short rather than full-width: the rule marks one point, and stretching it
    across the image implies a precision that point does not have. bounds keeps
    it on the curve — a fifth of apexes sit at a curve end, where a centred rule
    would otherwise overhang into empty chest.
    """
    half = max(1, int(round(frac * width)))
    x0, x1 = max(0, x - half), min(width, x + half)
    if bounds is not None:
        lo, hi = int(bounds[0]), int(bounds[1])
        x0, x1 = max(x0, lo), min(x1, hi)
        if x1 <= x0:
            x0, x1 = max(0, x - half), min(width, x + half)
    return x0, x1


# --- primitives --------------------------------------------------------------

def tint(canvas, mask, color, alpha=MASK_ALPHA):
    """Blend a flat colour wherever mask is set."""
    out = canvas.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def draw_height_rule(canvas, apex, color, label, align="left", bounds=None,
                     lower_bound=False):
    """Dashed horizontal rule at a dome apex, for comparing the two sides.

    Dark halo under every mark: a chest film runs near-black to near-white and
    the rule crosses both. Hollow marker plus a '>=' label when the height is a
    lower bound rather than a measurement.
    """
    x, y = int(apex[0]), int(apex[1])
    w = canvas.shape[1]

    x0, x1 = rule_extent(w, x, bounds=bounds)
    for a, b in dash_segments(x1 - x0):
        cv2.line(canvas, (x0 + a, y), (x0 + b, y), DARK, 4, cv2.LINE_AA)
        cv2.line(canvas, (x0 + a, y), (x0 + b, y), color, 2, cv2.LINE_AA)

    cv2.circle(canvas, (x, y), 7, DARK, -1)
    cv2.circle(canvas, (x, y), 5, color, 2 if lower_bound else -1)
    if lower_bound:
        label = ">= " + label

    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    (tw, th), _ = cv2.getTextSize(label, font, size, 1)
    tx = min(max(x0 - tw - 10 if align == "left" else x1 + 10, 6), w - tw - 6)
    ty = max(th + 6, y + th // 2)
    cv2.rectangle(canvas, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), DARK, -1)
    cv2.putText(canvas, label, (tx, ty), font, size, color, 1, cv2.LINE_AA)
    return canvas


def draw_diaphragm(canvas, curve, color, scale=1.0):
    """Measured part solid, fitted part dashed. Takes a Diaphragm."""
    if curve.fitted is not None:
        pts = scale_points(curve.fitted, scale)
        for a, b in dash_segments(len(pts), dash=14, gap=9):
            cv2.polylines(canvas, [pts[a:b]], False, color, 1, cv2.LINE_AA)
    cv2.polylines(canvas, [scale_points(curve.measured, scale)], False,
                  color, 2, cv2.LINE_AA)
    return canvas


def draw_landmarks(canvas, mask, lung, side, scale):
    """Green lung outline plus the three corner markers from a LungPoints."""
    try:
        pts = contour_of(mask)
    except ValueError:
        return canvas
    cv2.polylines(canvas, [scale_points(pts, scale).reshape(-1, 1, 2)], True,
                  OUTLINE_COLOR, 1, cv2.LINE_AA)
    if lung is None:
        return canvas
    for name in ("top", "lower_left", "lower_right"):
        p = tuple(scale_points(getattr(lung, name), scale))
        cv2.circle(canvas, p, 5, POINT_COLORS[name], -1)
        cv2.circle(canvas, p, 5, DARK, 1)
        cv2.putText(canvas, f"{side}-{name}", (p[0] + 8, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, POINT_COLORS[name], 1, cv2.LINE_AA)
    return canvas


# --- composed views ----------------------------------------------------------

def draw_overlay(cxr, left, right, left_mask=None, right_mask=None,
                 show_masks=False, show_apex=True, show_landmarks=False,
                 max_dim=VIEW_MAX_DIM, left_lung=None, right_lung=None):
    """Downscaled X-ray with both hemidiaphragms, optional lung tint and
    landmarks, and a height rule at each apex. left/right are Diaphragm or
    None; left_lung/right_lung are LungPoints or None, only used when
    show_landmarks is set."""
    f = display_scale(cxr.shape, max_dim)
    small = cv2.resize(cxr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    if show_masks:
        for mask, color in ((left_mask, LEFT_COLOR), (right_mask, RIGHT_COLOR)):
            if mask is not None:
                shrunk = cv2.resize(mask.astype(np.uint8),
                                    (canvas.shape[1], canvas.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                canvas = tint(canvas, shrunk, color)

    if show_landmarks:
        for mask, lung, side in ((left_mask, left_lung, "R"),
                                 (right_mask, right_lung, "L")):
            if mask is not None:
                draw_landmarks(canvas, mask, lung, side, f)

    for curve, color in ((left, LEFT_COLOR), (right, RIGHT_COLOR)):
        if curve is not None:
            draw_diaphragm(canvas, curve, color, f)

    if show_apex:
        for curve, color, label, align in ((left, LEFT_COLOR, "img-left", "left"),
                                           (right, RIGHT_COLOR, "img-right", "right")):
            if curve is None:
                continue
            apex = apex_of(*curve)
            own = apex_source(*curve)
            draw_height_rule(canvas, scale_points(apex, f), color, label, align,
                             bounds=scale_points([own[:, 0].min(), own[:, 0].max()], f),
                             lower_bound=apex_is_lower_bound(curve.measured, apex))
    return canvas


def contact_sheet(tiles, cols=SHEET_COLS):
    """Grid of captioned thumbnails, so a whole run can be scanned at once."""
    th = max(t.shape[0] for _, t in tiles)
    tw = max(t.shape[1] for _, t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * (th + 22), cols * tw, 3), np.uint8)
    for i, (name, tile) in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + 22), c * tw
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        cv2.putText(sheet, name, (x + 4, y + th + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return sheet
