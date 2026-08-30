# Measures how well a model finds (a) the lung field and (b) the diaphragm
# apex point on the DDR dataset, and prints/saves a side-by-side comparison
# of the baseline (Montgomery-only) model against the DDR fine-tuned one.
#
# Two metrics, chosen to each be the single clearest number for their task:
#
#   Lung field  -> Dice coefficient of the predicted mask against
#                  LungArea_truth (the hand-segmented ground truth), one
#                  score per annotated frame.
#
#   Diaphragm point -> Euclidean distance between the predicted apex (found
#                  from the model's mask using this project's own dome-apex
#                  geometry) and the TRUE peak of the hand-drawn DM-MODE_truth
#                  line. This is deliberately NOT compared against the
#                  dataset's own "center-x" convention point, which is a
#                  different definition of "the diaphragm point" than the
#                  apex geometry uses -- mixing the two would blend real
#                  model error together with a definition mismatch.
#
# Orientation: all image/mask/point loading goes through formats/dicom_io.py,
# which applies the DICOM's Field of View Horizontal Flip and the matching
# DM-MODE_truth point correction -- see that module for the full explanation.
# There is no DICOM- or XML-reading code in this file.
#
# Run:
#   python evaluate_ddr.py
#       compares best_model.h5 (baseline) vs best_model_ddr_finetuned_phase3.h5
#   python evaluate_ddr.py --model PATH --label NAME [--model PATH --label NAME ...]
#       compares an arbitrary list of checkpoints instead
#
# Writes outputs/ddr_eval/<label>.json (per-case detail, for the Apex Review
# artifact) and prints the summary table.

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

# Run from anywhere -- see finetune_ddr.py's own comment for why this is here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune_ddr import (MASK_OPEN_KERNEL, MODEL_SIZE, USE_DICOM_WINDOW_TAGS,
                          find_cases, segment)
from formats.dicom_io import (load_frames, parse_dm_mode_truth_lines, parse_lung_area_truth,
                             to_model_input, window_params)
from geometry.diaphragm import cliff_step_for, dome_apex, split_lungs
from segmentation.pipeline import CLIFF_FACTOR, RISE_FRACTION, measure_masks

OUT_DIR = "outputs/ddr_eval"


def true_apex_of(polyline_points, tol=2):
    """Hand-drawn DM-MODE_truth polyline -> its actual peak, (x, y).

    Resamples onto integer x-columns and applies the same dome_apex()
    definition the model's own predicted apex is measured with, so the two
    are comparable -- apples to apples, not two different definitions of
    "the diaphragm point."
    """
    pts = np.asarray(sorted(polyline_points))
    xs = np.arange(int(np.ceil(pts[:, 0].min())), int(np.floor(pts[:, 0].max())) + 1)
    ys = np.interp(xs, pts[:, 0], pts[:, 1])
    x, y = dome_apex(np.stack([xs, ys], axis=1), tol)
    return (float(x), float(y))


def predicted_lungs(mask):
    """Predicted binary mask -> {"R": LungPoints or None, "L": ...}."""
    try:
        masks = dict(zip(["R", "L"], split_lungs(mask)))
    except ValueError:
        return {}
    max_step = cliff_step_for(mask.shape[0], model_size=MODEL_SIZE, factor=CLIFF_FACTOR)
    lungs, _ = measure_masks(masks, max_rise_frac=RISE_FRACTION, max_step=max_step)
    return lungs


def evaluate_case(case_id, dcm_path, xml_path, raw_path, model):
    """One case -> (list of per-frame Dice, list of per-frame {"R"/"L": error_mm})."""
    frames, spacing_mm, photometric = load_frames(dcm_path)
    lung_truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
    dm_lines = parse_dm_mode_truth_lines(xml_path)
    wc, ww = window_params(dcm_path) if USE_DICOM_WINDOW_TAGS else (None, None)

    dice_rows, apex_rows = [], []
    for frame_idx in sorted(set(lung_truth) | set(dm_lines)):
        img = to_model_input(frames[frame_idx], photometric, wc, ww)
        pred_mask = segment(img, model, open_kernel=MASK_OPEN_KERNEL)

        if frame_idx in lung_truth:
            truth_mask = lung_truth[frame_idx]
            inter = np.logical_and(pred_mask.astype(bool), truth_mask.astype(bool)).sum()
            total = pred_mask.astype(bool).sum() + truth_mask.astype(bool).sum()
            dice_rows.append({"case": case_id, "frame": frame_idx,
                             "dice": 2 * inter / total if total else 1.0})

        if frame_idx in dm_lines:
            lungs = predicted_lungs(pred_mask)
            errors = {}
            for side, poly_pts in dm_lines[frame_idx].items():
                lung = lungs.get(side)
                if lung is None:
                    continue
                truth_x, truth_y = true_apex_of(poly_pts)
                pred_x, pred_y = lung.dome
                error_px = float(np.hypot(pred_x - truth_x, pred_y - truth_y))
                errors[side] = error_px * spacing_mm
            if errors:
                apex_rows.append({"case": case_id, "frame": frame_idx, **errors})

    return dice_rows, apex_rows


def evaluate_model(model_path, cases):
    print(f"loading {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    all_dice, all_apex = [], []
    for case_id, dcm_path, xml_path, raw_path in cases:
        dice_rows, apex_rows = evaluate_case(case_id, dcm_path, xml_path, raw_path, model)
        all_dice.extend(dice_rows)
        all_apex.extend(apex_rows)
    return all_dice, all_apex


def summarize(dice_rows, apex_rows):
    dice_vals = np.array([r["dice"] for r in dice_rows])
    by_side = {"R": [], "L": []}
    for r in apex_rows:
        for side in ("R", "L"):
            if side in r:
                by_side[side].append(r[side])
    all_apex = by_side["R"] + by_side["L"]
    return {
        "dice_mean": float(dice_vals.mean()), "dice_median": float(np.median(dice_vals)),
        "n_dice": len(dice_vals),
        "apex_mm_mean": float(np.mean(all_apex)) if all_apex else None,
        "apex_mm_median": float(np.median(all_apex)) if all_apex else None,
        "apex_mm_mean_R": float(np.mean(by_side["R"])) if by_side["R"] else None,
        "apex_mm_mean_L": float(np.mean(by_side["L"])) if by_side["L"] else None,
        "n_apex": len(all_apex),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[],
                    help="checkpoint to evaluate; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="display name for the matching --model; repeatable")
    args = ap.parse_args()

    if args.model:
        if len(args.label) != len(args.model):
            raise SystemExit("pass one --label per --model")
        models = list(zip(args.model, args.label))
    else:
        models = [("best_model.h5", "baseline"),
                 ("best_model_ddr_finetuned_phase3.h5", "finetuned")]

    cases = find_cases()
    print(f"{len(cases)} DDR cases\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = {}
    for model_path, label in models:
        dice_rows, apex_rows = evaluate_model(model_path, cases)
        summary = summarize(dice_rows, apex_rows)
        summaries[label] = summary
        with open(os.path.join(OUT_DIR, f"{label}.json"), "w") as f:
            json.dump({"model": model_path, "summary": summary,
                      "dice_rows": dice_rows, "apex_rows": apex_rows}, f)
        print(f"  {label:12s} Dice mean {summary['dice_mean']:.3f} median {summary['dice_median']:.3f} "
             f"(n={summary['n_dice']})")
        if summary["apex_mm_mean"] is not None:
            print(f"  {'':12s} apex error mean {summary['apex_mm_mean']:.1f}mm "
                 f"median {summary['apex_mm_median']:.1f}mm "
                 f"(R={summary['apex_mm_mean_R']:.1f}mm L={summary['apex_mm_mean_L']:.1f}mm, "
                 f"n={summary['n_apex']})")
        print()

    print("=== comparison ===")
    labels = list(summaries)
    print(f"{'metric':22s}" + "".join(f"{l:>14s}" for l in labels))
    for key, fmt in (("dice_mean", "{:.3f}"), ("dice_median", "{:.3f}"),
                     ("apex_mm_mean", "{:.1f}"), ("apex_mm_median", "{:.1f}")):
        vals = [summaries[l][key] for l in labels]
        cells = "".join(f"{fmt.format(v) if v is not None else 'n/a':>14s}" for v in vals)
        print(f"{key:22s}{cells}")

    print(f"\nwrote {OUT_DIR}/<label>.json for each model")


if __name__ == "__main__":
    main()
