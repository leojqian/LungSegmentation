# The CLI's output points: one frame's measured geometry reduced to 8 (x, y)
# points — per side, the lung apex, the diaphragm point, and the two bottom
# corners — plus the naive baseline those corners imply.
#
# Pure data shaping — no file I/O, no drawing. "Diaphragm point" is
# geometry.diaphragm.center_of, the DDR dataset's center-x convention that
# submission/evaluate_ddr.py scores against. The naive baseline is the plain
# midpoint of the two bottom corners: no tracing, trimming, or dome fitting,
# so comparing it to the diaphragm point shows what that machinery buys.

from collections import namedtuple

import numpy as np

from lungmap.geometry.diaphragm import center_of

SIDES = ("R", "L")   # R = image-left lung (patient's right on a PA film)

#: any field is None when that side couldn't be measured on this frame
SidePoints = namedtuple("SidePoints", "apex diaphragm lower_left lower_right")

CSV_COLUMNS = ["frame"] + [f"{side}_{name}_{axis}"
                           for side in SIDES for name in SidePoints._fields for axis in "xy"]


def _xy(point):
    return None if point is None else (float(point[0]), float(point[1]))


def side_points(lung, curve):
    """LungPoints or None, Diaphragm or None -> SidePoints.

    Each source fails independently (measure_masks maps either to None), so a
    lung with no traceable diaphragm still reports its apex and corners.
    """
    return SidePoints(
        apex=_xy(lung.top) if lung is not None else None,
        diaphragm=_xy(center_of(*curve)) if curve is not None else None,
        lower_left=_xy(lung.lower_left) if lung is not None else None,
        lower_right=_xy(lung.lower_right) if lung is not None else None,
    )


def naive_midpoint(points):
    """Midpoint of the two bottom corners, or None if either is missing."""
    if points is None or points.lower_left is None or points.lower_right is None:
        return None
    (x0, y0), (x1, y1) = points.lower_left, points.lower_right
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def csv_row(frame, by_side):
    """{"R": SidePoints or None, "L": ...} -> one row matching CSV_COLUMNS.

    Coordinates to one decimal; a missing side or point is blank, not 0 or
    NaN, so a spreadsheet or pandas reads it as missing.
    """
    row = [frame]
    for side in SIDES:
        points = by_side.get(side)
        for name in SidePoints._fields:
            point = getattr(points, name) if points is not None else None
            row += ["", ""] if point is None else [f"{point[0]:.1f}", f"{point[1]:.1f}"]
    return row


def compare(frames):
    """[{"R": SidePoints, "L": ...}, ...] -> {side: {"n", "mean_px", "median_px"}}.

    Distance from the naive midpoint to the diaphragm point, over every frame
    where both exist for that side.
    """
    stats = {}
    for side in SIDES:
        dists = []
        for by_side in frames:
            points = by_side.get(side)
            naive = naive_midpoint(points)
            if naive is None or points.diaphragm is None:
                continue
            dists.append(float(np.hypot(points.diaphragm[0] - naive[0],
                                        points.diaphragm[1] - naive[1])))
        stats[side] = {"n": len(dists),
                       "mean_px": float(np.mean(dists)) if dists else None,
                       "median_px": float(np.median(dists)) if dists else None}
    return stats
