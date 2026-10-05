# ARCHIVED -- not part of the lungmap package, and not used to build the shipped
# model (that's research/ddr). Kept as the record behind the numbers in
# docs/design/2026-08-24-ddr-domain-adaptation.md. Run from the repo root.
#
# Validate the Montgomery-trained U-Net + diaphragm geometry against the DDR
# dataset (SampleDDR_August2026): real ground truth for both lung area and
# diaphragm position, which this project never had before.
#
#   python research/archive/ddr_validate.py                 all cases -> outputs/ddr/
#   python research/archive/ddr_validate.py --sample NAME    one case, e.g. KaU001_PA_deep
#   python research/archive/ddr_validate.py --limit 3        first N cases
#   python research/archive/ddr_validate.py --no-overlays    skip the spot-check PNGs
#   python research/archive/ddr_validate.py --from-truth     score apex_of() alone: feed it
#                                                LungArea_truth instead of a
#                                                U-Net mask, isolating the
#                                                geometry algorithm's own error
#                                                from segmentation error ->
#                                                outputs/ddr_from_truth/
#
# Only the ~6 ground-truthed frames per case are run through the model, not the
# full ~230-frame cine — LungArea_truth and DM-MODE_truth annotate different,
# sparse frames (verified: 2 and 4 per case respectively, across all 20 cases).
#
# I/O and scoring only: DICOM/truth parsing lives in lungmap/formats/dicom_io.py, lung
# geometry in lungmap/geometry/diaphragm.py and lungmap/geometry/landmarks.py, unchanged.

import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np
from tqdm import tqdm

from lungmap.formats.dicom_io import (load_frames, parse_dm_mode_truth, parse_dm_mode_truth_lines,
                                     parse_lung_area_truth, to_model_input, window_params)
from lungmap.geometry.diaphragm import dome_apex, split_lungs
from lungmap.segmentation.pipeline import (CLIFF_FACTOR, MODEL_SIZE, OPEN_KERNEL, cliff_step_for,
                                           measure_masks, segment)

DDR_DIR = "SampleDDR_August2026"
IMAGES_PRES_DIR = os.path.join(DDR_DIR, "images_pres")
LUNGAREA_DIR = os.path.join(DDR_DIR, "LungArea_truth")
DMMODE_DIR = os.path.join(DDR_DIR, "DM-MODE_truth")
OUT_DIR = "outputs/ddr"
MODEL_FILE = "best_model.h5"

SIDE_COLOR = {"R": (80, 200, 255), "L": (255, 180, 80)}  # BGR, matches render.py
FIELDNAMES = ["case", "frame", "kind", "side", "dice", "iou", "pred_x", "pred_y",
             "truth_x", "truth_y", "error_px", "error_mm", "note"]


# --- case discovery ------------------------------------------------------------

def find_cases(images_dir=IMAGES_PRES_DIR, suffix="LogPlusRegius"):
    """(case_id, dcm_path, xml_path, raw_path) for every case with all three files.

    images_dir/suffix default to the processed images_pres set; Step7's image-
    source comparison points this at the raw `images` dir (suffix="") instead.
    """
    cases = []
    pattern = re.compile(rf"(.+){re.escape(suffix)}\.dcm$" if suffix else r"(.+)\.dcm$")
    for dcm_path in sorted(glob.glob(os.path.join(images_dir, "*.dcm"))):
        name = os.path.basename(dcm_path)
        m = pattern.match(name)
        if not m:
            print(f"  skip {name}: unexpected filename")
            continue
        case_id = m.group(1)
        xml_path = os.path.join(DMMODE_DIR, f"{case_id}_truth.xml")
        raw_path = os.path.join(LUNGAREA_DIR, f"{case_id}_truth.raw")
        if os.path.exists(xml_path) and os.path.exists(raw_path):
            cases.append((case_id, dcm_path, xml_path, raw_path))
        else:
            print(f"  skip {case_id}: missing truth file(s)")
    return cases


def select_sample(cases, name):
    hit = [c for c in cases if c[0] == name]
    if not hit:
        raise SystemExit(f"no case named {name!r} under {IMAGES_PRES_DIR}")
    return hit


def load_model(path=MODEL_FILE):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


# --- inference -------------------------------------------------------------

def analyze_ddr_frame(raw_frame, photometric, model, open_kernel=OPEN_KERNEL,
                      window_center=None, window_width=None):
    """Raw DDR frame -> (binary mask, lungs, curves).

    Composes pipeline.py's building blocks rather than the top-level
    pipeline.analyze_frame: Dice scoring only needs the merged mask, so it must
    stay available even on the (rare) frame where split_lungs can't find two
    lung components — lungs/curves are {} in that case, not a lost mask.
    """
    image = to_model_input(raw_frame, photometric, window_center, window_width)
    mask = segment(image, model, open_kernel)
    return _measure_from_mask(mask)


def analyze_ddr_frame_from_truth(truth_mask):
    """LungArea_truth mask -> (mask, lungs, curves), no U-Net involved.

    Same split_lungs/measure_masks geometry as analyze_ddr_frame, but fed the
    ground-truth mask directly — for isolating apex_of()'s own error from
    U-Net segmentation error (--from-truth).
    """
    return _measure_from_mask(truth_mask)


def _measure_from_mask(mask):
    max_step = cliff_step_for(mask.shape[0], model_size=MODEL_SIZE, factor=CLIFF_FACTOR)
    try:
        masks = dict(zip(["R", "L"], split_lungs(mask)))
    except ValueError:
        return mask, {}, {}
    lungs, curves = measure_masks(masks, max_step=max_step)
    return mask, lungs, curves


def dice_iou(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    total = pred.sum() + truth.sum()
    dice = 2 * intersection / total if total else 1.0
    iou = intersection / union if union else 1.0
    return dice, iou


# --- per-case scoring --------------------------------------------------------

def score_lung_area(case_id, frame_idx, binary_mask, truth_mask):
    dice, iou = dice_iou(binary_mask, truth_mask)
    return {"case": case_id, "frame": frame_idx, "kind": "lung_area",
            "dice": dice, "iou": iou}


def score_diaphragm_points(case_id, frame_idx, lungs, truth_sides, spacing_mm):
    """lungs: {"R"/"L": LungPoints or None or missing}, from analyze_ddr_frame."""
    rows = []
    for side, truth_point in truth_sides.items():
        row = {"case": case_id, "frame": frame_idx, "kind": "diaphragm_point", "side": side}
        lung = lungs.get(side)
        if lung is None:
            row["note"] = "landmarks not found"
        else:
            pred_point = lung.dome
            error_px = float(np.hypot(pred_point[0] - truth_point[0],
                                      pred_point[1] - truth_point[1]))
            row.update(pred_x=pred_point[0], pred_y=pred_point[1],
                      truth_x=truth_point[0], truth_y=truth_point[1],
                      error_px=error_px, error_mm=error_px * spacing_mm)
        rows.append(row)
    return rows


def true_apex_of(polyline_points, tol=2):
    """Hand-drawn DM-MODE_truth polyline -> its actual peak, (x, y).

    DM-MODE_truth's own point (parse_dm_mode_truth) is the dataset's "center
    x-coordinate" convention, not the line's highest point — a different
    definition than apex_of() uses on a traced mask. This resamples the raw
    polyline onto integer x-columns and calls dome_apex() on it directly, so
    the ground-truth point is defined identically to the predicted one.
    """
    pts = np.asarray(sorted(polyline_points))
    xs = np.arange(int(np.ceil(pts[:, 0].min())), int(np.floor(pts[:, 0].max())) + 1)
    ys = np.interp(xs, pts[:, 0], pts[:, 1])
    x, y = dome_apex(np.stack([xs, ys], axis=1), tol)
    return (float(x), float(y))


# --- overlays ----------------------------------------------------------------

def draw_overlay(frame, kind, binary_mask=None, truth_mask=None, points=None):
    """Downscaled spot-check image: predicted vs. truth, for one frame.

    kind: "lung_area" draws mask contours only, "diaphragm_point" draws points
    only, "combined" draws both — masks (yellow=truth, green=predicted) with
    the dome points (open=truth, filled=predicted) on top, one spot-check
    image per frame instead of two.
    """
    canvas = cv2.cvtColor((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if kind in ("lung_area", "combined") and truth_mask is not None and binary_mask is not None:
        for mask, color in ((truth_mask, (0, 255, 255)), (binary_mask, (0, 255, 0))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, contours, -1, color, 2)
    if kind in ("diaphragm_point", "combined") and points is not None:
        for side, (pred, truth) in points.items():
            color = SIDE_COLOR[side]
            if pred is not None:
                cv2.circle(canvas, pred, 6, color, -1)
            if truth is not None:
                cv2.circle(canvas, truth, 8, color, 2)
            if pred is not None and truth is not None:
                cv2.line(canvas, pred, truth, color, 1, cv2.LINE_AA)

    f = min(1.0, 1024 / max(canvas.shape[:2]))
    return cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


# --- driver --------------------------------------------------------------------

def process_case(case_id, dcm_path, xml_path, raw_path, model, write_overlays,
                 open_kernel=OPEN_KERNEL, window=False, from_truth=False, out_dir=OUT_DIR):
    frames, spacing_mm, photometric = load_frames(dcm_path)
    lung_area_truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
    dm_mode_truth = parse_dm_mode_truth(xml_path)
    dm_mode_lines = parse_dm_mode_truth_lines(xml_path)
    wc, ww = window_params(dcm_path) if window else (None, None)

    if write_overlays:
        os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    rows = []
    for frame_idx in sorted(set(lung_area_truth) | set(dm_mode_truth)):
        raw_frame = frames[frame_idx]

        if from_truth:
            # Lung-area Dice is meaningless here (pred mask == truth mask by
            # construction) — only diaphragm points get scored, and only on
            # frames that actually have a ground-truth mask to feed apex_of().
            lungs = {}
            if frame_idx in lung_area_truth:
                _, lungs, _ = analyze_ddr_frame_from_truth(lung_area_truth[frame_idx])
        else:
            binary, lungs, _ = analyze_ddr_frame(raw_frame, photometric, model, open_kernel, wc, ww)

            if frame_idx in lung_area_truth:
                truth_mask = lung_area_truth[frame_idx]
                rows.append(score_lung_area(case_id, frame_idx, binary, truth_mask))
                if write_overlays:
                    overlay = draw_overlay(to_model_input(raw_frame, photometric, wc, ww),
                                           "lung_area", binary_mask=binary, truth_mask=truth_mask)
                    cv2.imwrite(os.path.join(out_dir, "overlays",
                                             f"{case_id}_{frame_idx:03d}_lungarea.png"), overlay)

                # apex_of() run on the *true* mask defines the reference point here —
                # same algorithm both sides, so this isn't confounded by DM-MODE_truth's
                # different point definition, and it scores every LungArea_truth frame
                # (~2/case) rather than only the ~1 frame that also has DM-MODE_truth.
                _, truth_lungs, _ = analyze_ddr_frame_from_truth(truth_mask)
                truth_dome_sides = {side: lp.dome for side, lp in truth_lungs.items() if lp is not None}
                if truth_dome_sides:
                    mask_rows = score_diaphragm_points(case_id, frame_idx, lungs,
                                                       truth_dome_sides, spacing_mm)
                    for r in mask_rows:
                        r["kind"] = "diaphragm_point_mask_truth"
                    rows.extend(mask_rows)
                    if write_overlays:
                        points = {r["side"]: (
                            (int(r["pred_x"]), int(r["pred_y"])) if "pred_x" in r else None,
                            tuple(int(v) for v in truth_dome_sides[r["side"]]),
                        ) for r in mask_rows}
                        image = to_model_input(raw_frame, photometric, wc, ww)
                        overlay = draw_overlay(image, "diaphragm_point", points=points)
                        cv2.imwrite(os.path.join(out_dir, "overlays",
                                                 f"{case_id}_{frame_idx:03d}_diaphragm_masktruth.png"),
                                    overlay)
                        # masks + points together: predicted mask/point vs. ground truth
                        # mask/point, one spot-check image per frame.
                        combined = draw_overlay(image, "combined", binary_mask=binary,
                                               truth_mask=truth_mask, points=points)
                        cv2.imwrite(os.path.join(out_dir, "overlays",
                                                 f"{case_id}_{frame_idx:03d}_combined.png"), combined)

        if frame_idx in dm_mode_truth:
            truth_sides = dm_mode_truth[frame_idx]
            if from_truth and frame_idx not in lung_area_truth:
                new_rows = [{"case": case_id, "frame": frame_idx, "kind": "diaphragm_point",
                            "side": side, "note": "no ground-truth mask for this frame"}
                           for side in truth_sides]
            else:
                new_rows = score_diaphragm_points(case_id, frame_idx, lungs, truth_sides, spacing_mm)
            rows.extend(new_rows)
            if write_overlays:
                points = {r["side"]: (
                    (int(r["pred_x"]), int(r["pred_y"])) if "pred_x" in r else None,
                    tuple(int(v) for v in truth_sides[r["side"]]),
                ) for r in new_rows}
                overlay = draw_overlay(to_model_input(raw_frame, photometric, wc, ww),
                                       "diaphragm_point", points=points)
                cv2.imwrite(os.path.join(out_dir, "overlays",
                                         f"{case_id}_{frame_idx:03d}_diaphragm.png"), overlay)

        if frame_idx in dm_mode_lines:
            # true_apex_of() re-derives the peak from the same hand-drawn line
            # DM-MODE_truth ships, instead of the dataset's "center x" point —
            # apples-to-apples against apex_of()'s own definition. lungs here is
            # whichever mode's already in scope: ground-truth-mask-derived under
            # --from-truth (isolates the geometry algorithm, but only scores
            # frames that also have a LungArea_truth mask), or U-Net-predicted
            # otherwise (every dm_mode_truth frame, full pipeline error).
            truth_apex_sides = {side: true_apex_of(pts) for side, pts in dm_mode_lines[frame_idx].items()}
            if from_truth and frame_idx not in lung_area_truth:
                apex_rows = [{"case": case_id, "frame": frame_idx, "kind": "diaphragm_apex_truth",
                             "side": side, "note": "no ground-truth mask for this frame"}
                            for side in truth_apex_sides]
            else:
                apex_rows = score_diaphragm_points(case_id, frame_idx, lungs, truth_apex_sides, spacing_mm)
                for r in apex_rows:
                    r["kind"] = "diaphragm_apex_truth"
            rows.extend(apex_rows)
    return rows


def _report(rows, out_dir=OUT_DIR):
    dice = [r["dice"] for r in rows if r.get("kind") == "lung_area" and "dice" in r]
    failures = [r for r in rows if r.get("note")]

    print(f"\nwrote {len(rows)} scored rows to {out_dir}/results.csv")
    if dice:
        d = np.array(dice)
        print(f"\nlung area Dice: mean {d.mean():.3f}, median {np.median(d):.3f}, "
              f"min {d.min():.3f} ({len(dice)} frames)")

    for kind, label in (("diaphragm_point", "diaphragm point error vs. DM-MODE_truth (hand-clicked, center-x convention)"),
                        ("diaphragm_point_mask_truth",
                         "diaphragm point error vs. apex_of(LungArea_truth) (mask-derived)"),
                        ("diaphragm_apex_truth",
                         "diaphragm APEX error vs. true_apex_of(DM-MODE_truth line) (same definition as apex_of())")):
        errors_px = [r["error_px"] for r in rows if r.get("kind") == kind and "error_px" in r]
        errors_mm = [r["error_mm"] for r in rows if r.get("kind") == kind and "error_mm" in r]
        if errors_px:
            px, mm = np.array(errors_px), np.array(errors_mm)
            print(f"\n{label}:")
            print(f"  mean {px.mean():.1f}px / {mm.mean():.1f}mm, "
                  f"median {np.median(px):.1f}px / {np.median(mm):.1f}mm "
                  f"({len(errors_px)} points)")

    if failures:
        print(f"\n{len(failures)} points/frames could not be scored (see 'note' column)")
    print(f"\n  open {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", metavar="NAME", help="one case, e.g. KaU001_PA_deep")
    ap.add_argument("--limit", type=int, help="only process the first N cases")
    ap.add_argument("--no-overlays", action="store_true", help="skip spot-check PNGs")
    ap.add_argument("--from-truth", action="store_true",
                    help="score apex_of() alone: feed it LungArea_truth instead of a "
                         "U-Net mask (no model loaded, no Dice scoring) -> outputs/ddr_from_truth/")
    ap.add_argument("--model", default=MODEL_FILE,
                    help=f"checkpoint to score (default {MODEL_FILE})")
    args = ap.parse_args()

    cases = find_cases()
    if not cases:
        raise SystemExit(f"no cases found under {IMAGES_PRES_DIR}")
    if args.sample:
        cases = select_sample(cases, args.sample)
    elif args.limit:
        cases = cases[:args.limit]

    model = None
    if not args.from_truth:
        print(f"loading {args.model} ...")
        model = load_model(args.model)

    out_dir = "outputs/ddr_from_truth" if args.from_truth else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for case_id, dcm_path, xml_path, raw_path in tqdm(cases, desc="validating DDR"):
        all_rows.extend(process_case(case_id, dcm_path, xml_path, raw_path, model,
                                     write_overlays=not args.no_overlays,
                                     from_truth=args.from_truth, out_dir=out_dir))

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    _report(all_rows, out_dir)


if __name__ == "__main__":
    main()
