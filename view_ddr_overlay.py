# Interactive viewer: one DDR sample case with predicted vs. ground-truth
# lung mask and diaphragm apex points overlaid, scrollable across every
# annotated frame. Reuses Step6ValidateDDR's own inference/scoring code so
# this shows exactly what gets scored there.
#
#   python view_ddr_overlay.py [CASE] [--from-truth]
#
# CASE defaults to the first case with all three files present (images_pres +
# DM-MODE_truth + LungArea_truth). --from-truth feeds LungArea_truth straight
# into apex_of() instead of running the U-Net, isolating the geometry
# algorithm's own error — same flag as Step6.
#
# Legend: yellow outline = truth mask, green outline = predicted mask.
# Three diaphragm points per side, one per point-reduction method, each its
# own fixed color (a small R/L label disambiguates side since color no longer
# does):
#   yellow hollow diamond  = DM-MODE_truth's own point (center-x of the
#                             hand-drawn line -- this dataset's convention)
#   green  filled circle   = dome_apex() on the predicted/traced curve --
#                             this codebase's own apex definition
#   magenta filled diamond = center_point() on that SAME traced curve -- our
#                             segmentation, DDR's convention, so it's a fair
#                             same-definition comparison against the yellow point
import argparse

import cv2
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

from formats.dicom_io import (load_frames, parse_dm_mode_truth, parse_lung_area_truth,
                             to_model_input, window_params)
from geometry.diaphragm import center_of
from Step6ValidateDDR import (IMAGES_PRES_DIR, analyze_ddr_frame,
                              analyze_ddr_frame_from_truth, find_cases, load_model,
                              select_sample)

TRUTH_COLOR = (0, 255, 255)    # yellow -- DM-MODE_truth's own center-x point
APEX_COLOR = (0, 255, 0)       # green  -- dome_apex() on our traced curve
CENTER_COLOR = (255, 0, 255)   # magenta -- center_point() on our traced curve


def draw_combined(frame_img, pred_mask, truth_mask, lungs, curves, truth_points, max_dim=1200):
    """One frame's image + mask outlines + 3 diaphragm points/side -> BGR canvas."""
    canvas = cv2.cvtColor((np.clip(frame_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if truth_mask is not None:
        contours, _ = cv2.findContours(truth_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)  # yellow
    if pred_mask is not None:
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)  # green

    for side in set(truth_points or {}) | set(lungs or {}):
        label_xy = None

        truth_pt = (truth_points or {}).get(side)
        if truth_pt is not None:
            pt = tuple(int(v) for v in truth_pt)
            cv2.drawMarker(canvas, pt, TRUTH_COLOR, cv2.MARKER_DIAMOND, 16, 2)
            label_xy = (pt[0] + 12, pt[1] - 12)

        lung = (lungs or {}).get(side)
        if lung is not None:
            pt = tuple(int(v) for v in lung.dome)
            cv2.circle(canvas, pt, 6, APEX_COLOR, -1)
            label_xy = label_xy or (pt[0] + 12, pt[1] - 12)

        curve = (curves or {}).get(side)
        if curve is not None:
            cx, cy = center_of(curve.measured, curve.fitted)
            pt = (int(cx), int(cy))
            cv2.drawMarker(canvas, pt, CENTER_COLOR, cv2.MARKER_DIAMOND, 16, 2)
            label_xy = label_xy or (pt[0] + 12, pt[1] - 12)

        if label_xy is not None:
            cv2.putText(canvas, side, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2, cv2.LINE_AA)

    f = min(1.0, max_dim / max(canvas.shape[:2]))
    return cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", nargs="?", help="case id, e.g. KaU001_PA_deep (default: first available)")
    ap.add_argument("--from-truth", action="store_true",
                    help="feed LungArea_truth into apex_of() instead of running the U-Net")
    ap.add_argument("--window", action="store_true",
                    help="normalize display/model input using the DICOM's WindowCenter/Width "
                         "tags instead of min/max (default, matches Step6's baseline)")
    args = ap.parse_args()

    cases = find_cases()
    if not cases:
        raise SystemExit(f"no cases found under {IMAGES_PRES_DIR}")
    case_id, dcm_path, xml_path, raw_path = (
        select_sample(cases, args.case)[0] if args.case else cases[0])

    frames, spacing_mm, photometric = load_frames(dcm_path)
    lung_area_truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
    dm_mode_truth = parse_dm_mode_truth(xml_path)
    wc, ww = window_params(dcm_path) if args.window else (None, None)

    model = None
    if not args.from_truth:
        print("loading best_model.h5 ...")
        model = load_model()

    frame_idxs = sorted(set(lung_area_truth) | set(dm_mode_truth))
    if not frame_idxs:
        raise SystemExit(f"{case_id}: no annotated frames")

    cache = {}

    def compute(i):
        if i not in cache:
            raw_frame = frames[i]
            img = to_model_input(raw_frame, photometric, wc, ww)
            truth_mask = lung_area_truth.get(i)
            truth_points = dm_mode_truth.get(i)

            if args.from_truth:
                lungs, curves = {}, {}
                if truth_mask is not None:
                    _, lungs, curves = analyze_ddr_frame_from_truth(truth_mask)
                pred_mask = None
            else:
                pred_mask, lungs, curves = analyze_ddr_frame(raw_frame, photometric, model,
                                                              window_center=wc, window_width=ww)

            cache[i] = (img, pred_mask, truth_mask, lungs, curves, truth_points)
        return cache[i]

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.12)
    ax.axis("off")
    state = {"im": None}

    def render(slider_pos):
        frame_idx = frame_idxs[int(slider_pos)]
        img, pred_mask, truth_mask, lungs, curves, truth_points = compute(frame_idx)
        rgb = cv2.cvtColor(draw_combined(img, pred_mask, truth_mask, lungs, curves, truth_points),
                           cv2.COLOR_BGR2RGB)
        if state["im"] is None:
            state["im"] = ax.imshow(rgb)
        else:
            state["im"].set_data(rgb)

        kinds = [k for k, v in (("mask", truth_mask), ("points", truth_points)) if v is not None]
        ax.set_title(f"{case_id}  frame {frame_idx}/{frames.shape[0] - 1}  [{'+'.join(kinds)}]\n"
                    "yellow diamond=DM-MODE_truth point  green circle=our apex  "
                    "magenta diamond=our center-x point (DDR's convention on our curve)")
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.04])
    slider = Slider(slider_ax, "Annotated frame", 0, len(frame_idxs) - 1, valinit=0, valstep=1)
    slider.on_changed(render)

    def step(delta):
        slider.set_val((int(slider.val) + delta) % len(frame_idxs))

    def on_key(event):
        if event.key == "right":
            step(1)
        elif event.key == "left":
            step(-1)

    def on_scroll(event):
        step(1 if event.button == "up" else -1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    render(0)
    plt.show()


if __name__ == "__main__":
    main()
