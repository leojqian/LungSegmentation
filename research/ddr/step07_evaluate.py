# Step 7: score models against the DDR ground truth.
#   Lung field:      Dice of the predicted mask vs LungArea_truth, per annotated frame.
#   Diaphragm point: distance (mm) between the predicted point and DM-MODE_truth's,
#                    both taken at the curve's horizontal center (the dataset's
#                    definition; lungmap.geometry.diaphragm.center_of), not its peak.
#
# Run from the repo root:
#   uv run --extra research python research/ddr/step07_evaluate.py
#       baseline (best_model.h5) vs fine-tuned (best_model_ddr_finetuned_phase3.h5)
#   ... step07_evaluate.py --model PATH --label NAME [--model PATH --label NAME ...]
# Writes outputs/ddr_eval/<label>.json and prints a comparison table.

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from lungmap.formats.dicom_io import (load_frames, parse_dm_mode_truth, parse_lung_area_truth,
                                     to_model_input, window_params)
from lungmap.geometry.diaphragm import center_of
from lungmap.segmentation.pipeline import measure_combined, segment
from step06_finetune import MASK_OPEN_KERNEL, USE_DICOM_WINDOW_TAGS, find_cases

OUT_DIR = "outputs/ddr_eval"


def evaluate_case(case_id, dcm_path, xml_path, raw_path, model):
    """One case -> (list of per-frame Dice, list of per-frame {"R"/"L": error_mm})."""
    frames, spacing_mm, photometric = load_frames(dcm_path)
    lung_truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
    dm_points = parse_dm_mode_truth(xml_path)
    wc, ww = window_params(dcm_path) if USE_DICOM_WINDOW_TAGS else (None, None)

    dice_rows, apex_rows = [], []
    for frame_idx in sorted(set(lung_truth) | set(dm_points)):
        img = to_model_input(frames[frame_idx], photometric, wc, ww)
        pred_mask = segment(img, model, open_kernel=MASK_OPEN_KERNEL)

        if frame_idx in lung_truth:
            truth_mask = lung_truth[frame_idx]
            inter = np.logical_and(pred_mask.astype(bool), truth_mask.astype(bool)).sum()
            total = pred_mask.astype(bool).sum() + truth_mask.astype(bool).sum()
            dice_rows.append({"case": case_id, "frame": frame_idx,
                             "dice": 2 * inter / total if total else 1.0})

        if frame_idx in dm_points:
            _, _, curves = measure_combined(pred_mask)
            errors = {}
            for side, (truth_x, truth_y) in dm_points[frame_idx].items():
                curve = curves.get(side)
                if curve is None:
                    continue
                pred_x, pred_y = center_of(*curve)
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
        "apex_mm_std": float(np.std(all_apex)) if all_apex else None,
        "apex_mm_mean_R": float(np.mean(by_side["R"])) if by_side["R"] else None,
        "apex_mm_mean_L": float(np.mean(by_side["L"])) if by_side["L"] else None,
        "apex_mm_std_R": float(np.std(by_side["R"])) if by_side["R"] else None,
        "apex_mm_std_L": float(np.std(by_side["L"])) if by_side["L"] else None,
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
                 f"median {summary['apex_mm_median']:.1f}mm std {summary['apex_mm_std']:.1f}mm "
                 f"(R={summary['apex_mm_mean_R']:.1f}mm L={summary['apex_mm_mean_L']:.1f}mm, "
                 f"n={summary['n_apex']})")
        print()

    print("=== comparison ===")
    labels = list(summaries)
    print(f"{'metric':22s}" + "".join(f"{l:>14s}" for l in labels))
    for key, fmt in (("dice_mean", "{:.3f}"), ("dice_median", "{:.3f}"),
                     ("apex_mm_mean", "{:.1f}"), ("apex_mm_median", "{:.1f}"),
                     ("apex_mm_std", "{:.1f}")):
        vals = [summaries[l][key] for l in labels]
        cells = "".join(f"{fmt.format(v) if v is not None else 'n/a':>14s}" for v in vals)
        print(f"{key:22s}{cells}")

    print(f"\nwrote {OUT_DIR}/<label>.json for each model")


if __name__ == "__main__":
    main()
