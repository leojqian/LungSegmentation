# Phase 1 preprocessing calibration for the DDR domain-adaptation spec
# (docs/superpowers/specs/2026-08-24-ddr-domain-adaptation-design.md).
#
# Coordinate ascent, not full factorial: 2 windowing candidates -> 4 kernel-size
# candidates -> 2 image-source candidates, 8 evaluations total, each across all
# 20 DDR cases. No model weights change here — this only picks preprocessing
# constants, the same way CLIFF_STEP/RISE_FRACTION were tuned in
# docs/superpowers/specs/2026-08-15-diaphragm-mapping-design.md.
#
# Reuses Step6ValidateDDR.py's scoring wholesale rather than reimplementing it.

import argparse
import json
import os

import cv2
import numpy as np

import Step6ValidateDDR as step6

OUT_DIR = "outputs/ddr_calibration"
KERNEL_CANDIDATES = [5, 9, 15, 21]
RAW_IMAGES_DIR = os.path.join(step6.DDR_DIR, "images")


# --- one variant ---------------------------------------------------------------

def run_variant(cases, model, open_kernel=step6.OPEN_KERNEL, window=False):
    """-> (per_case_dice, per_case_error_mm), each {case_id: mean value}.

    Per-case means, not pooled means: this is what paired_win_count needs, and
    it keeps one case's frame count from silently outweighing another's.
    """
    per_case_dice, per_case_error_mm = {}, {}
    for case_id, dcm_path, xml_path, raw_path in cases:
        rows = step6.process_case(case_id, dcm_path, xml_path, raw_path, model,
                                  write_overlays=False, open_kernel=open_kernel, window=window)
        dice = [r["dice"] for r in rows if r.get("kind") == "lung_area" and "dice" in r]
        error = [r["error_mm"] for r in rows if "error_mm" in r]
        if dice:
            per_case_dice[case_id] = float(np.mean(dice))
        if error:
            per_case_error_mm[case_id] = float(np.mean(error))
    return per_case_dice, per_case_error_mm


def paired_win_count(a, b):
    """-> (a_wins, b_wins, ties) on Dice (higher is better), cases in both."""
    common = set(a) & set(b)
    a_wins = sum(1 for c in common if a[c] > b[c])
    b_wins = sum(1 for c in common if b[c] > a[c])
    return a_wins, b_wins, len(common) - a_wins - b_wins


def summarize(per_case_dice, per_case_error_mm):
    d = np.array(list(per_case_dice.values()))
    e = np.array(list(per_case_error_mm.values()))
    return {
        "dice_mean": float(d.mean()) if len(d) else float("nan"),
        "dice_median": float(np.median(d)) if len(d) else float("nan"),
        "n_cases_dice": len(d),
        "err_mm_mean": float(e.mean()) if len(e) else float("nan"),
        "err_mm_median": float(np.median(e)) if len(e) else float("nan"),
        "n_cases_err": len(e),
    }


def report_row(name, per_case_dice, baseline_dice=None):
    s = summarize(per_case_dice, {})
    line = (f"  {name:28s} Dice mean {s['dice_mean']:.3f} median {s['dice_median']:.3f} "
           f"(n={s['n_cases_dice']})")
    if baseline_dice is not None and baseline_dice is not per_case_dice:
        w, l, t = paired_win_count(per_case_dice, baseline_dice)
        line += f"   vs baseline: {w}W/{l}L/{t}T"
    print(line)


def choose_better(name_a, res_a, name_b, res_b):
    """Pick by per-case win count on Dice, mean Dice as tiebreaker."""
    a_dice, b_dice = res_a[0], res_b[0]
    w_a, w_b, _ = paired_win_count(a_dice, b_dice)
    if w_a != w_b:
        winner = (name_a, res_a) if w_a > w_b else (name_b, res_b)
    else:
        mean_a, mean_b = summarize(a_dice, {})["dice_mean"], summarize(b_dice, {})["dice_mean"]
        winner = (name_a, res_a) if mean_a >= mean_b else (name_b, res_b)
    print(f"  -> winner: {winner[0]}")
    return winner


# --- fixed spot-check cases ------------------------------------------------

def spot_check_cases(per_case_dice, n_worst=2):
    """The 2 worst-Dice cases + 1 median case, fixed across every variant."""
    ranked = sorted(per_case_dice.items(), key=lambda kv: kv[1])
    worst = [c for c, _ in ranked[:n_worst]]
    median = ranked[len(ranked) // 2][0]
    return worst + [median]


def write_spot_check_overlays(tag, cases_by_id, model, open_kernel, window):
    out = os.path.join(OUT_DIR, "overlays", tag)
    os.makedirs(out, exist_ok=True)
    for case_id, dcm_path, xml_path, raw_path in cases_by_id:
        rows = step6.process_case(case_id, dcm_path, xml_path, raw_path, model,
                                  write_overlays=False, open_kernel=open_kernel, window=window)
        frames, spacing_mm, photometric = step6.load_frames(dcm_path)
        lung_area_truth = step6.parse_lung_area_truth(raw_path, shape=frames.shape[1:])
        wc, ww = step6.window_params(dcm_path) if window else (None, None)
        for frame_idx, truth_mask in lung_area_truth.items():
            binary, _, _ = step6.analyze_ddr_frame(frames[frame_idx], photometric, model,
                                                    open_kernel, wc, ww)
            overlay = step6.draw_overlay(step6.to_model_input(frames[frame_idx], photometric, wc, ww),
                                         "lung_area", binary_mask=binary, truth_mask=truth_mask)
            cv2.imwrite(os.path.join(out, f"{case_id}_{frame_idx:03d}.png"), overlay)


# --- driver --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="only use the first N cases (smoke-testing)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading {step6.MODEL_FILE} ...")
    model = step6.load_model()

    cases = step6.find_cases()
    if args.limit:
        cases = cases[:args.limit]
    findings = {}

    print("\n=== baseline (current defaults: min/max, kernel=5, images_pres) ===")
    baseline = run_variant(cases, model)
    report_row("baseline", baseline[0])
    findings["baseline"] = summarize(*baseline)
    spot_cases = spot_check_cases(baseline[0])
    print(f"  fixed spot-check cases: {spot_cases}")

    print("\n=== stage 1: windowing ===")
    window_variants = {
        "min/max (current)": run_variant(cases, model, window=False),
        "WindowCenter/Width tags": run_variant(cases, model, window=True),
    }
    for name, res in window_variants.items():
        report_row(name, res[0], baseline[0])
        findings[name] = summarize(*res)
    win_name, win_res = choose_better("min/max (current)", window_variants["min/max (current)"],
                                      "WindowCenter/Width tags", window_variants["WindowCenter/Width tags"])
    best_window = win_name == "WindowCenter/Width tags"

    print("\n=== stage 2: mask post-processing kernel size ===")
    kernel_variants = {}
    for k in KERNEL_CANDIDATES:
        name = f"kernel={k}"
        # kernel=5 with the winning window setting was already computed in stage 1.
        kernel_variants[name] = win_res if k == 5 else run_variant(
            cases, model, open_kernel=k, window=best_window)
        report_row(name, kernel_variants[name][0], baseline[0])
        findings[name] = summarize(*kernel_variants[name])
    best_kernel_name, best_kernel_res = list(kernel_variants.items())[0]
    for name, res in list(kernel_variants.items())[1:]:
        best_kernel_name, best_kernel_res = choose_better(best_kernel_name, best_kernel_res, name, res)
    best_kernel = int(best_kernel_name.split("=")[1])

    print("\n=== stage 3: image source ===")
    case_ids = {c[0] for c in cases}
    raw_cases = [c for c in step6.find_cases(RAW_IMAGES_DIR, suffix="") if c[0] in case_ids]
    source_variants = {
        "images_pres (current)": run_variant(cases, model, open_kernel=best_kernel, window=best_window),
        "images (raw)": run_variant(raw_cases, model, open_kernel=best_kernel, window=best_window),
    }
    for name, res in source_variants.items():
        report_row(name, res[0], baseline[0])
        findings[name] = summarize(*res)
    best_source_name, best_source_res = choose_better(
        "images_pres (current)", source_variants["images_pres (current)"],
        "images (raw)", source_variants["images (raw)"])
    best_source_raw = best_source_name == "images (raw)"

    print("\n=== winning combination ===")
    print(f"  windowing: {'WindowCenter/Width tags' if best_window else 'min/max (current)'}")
    print(f"  kernel: {best_kernel}")
    print(f"  image source: {'images (raw)' if best_source_raw else 'images_pres (current)'}")
    report_row("winning combination", best_source_res[0], baseline[0])

    winning_cases = raw_cases if best_source_raw else cases
    spot_case_records = [c for c in winning_cases if c[0] in spot_cases]
    write_spot_check_overlays("winning", spot_case_records, model, best_kernel, best_window)
    write_spot_check_overlays("baseline", [c for c in cases if c[0] in spot_cases],
                              model, step6.OPEN_KERNEL, False)

    findings["winner"] = {
        "window": best_window, "kernel": best_kernel, "source_raw": best_source_raw,
        "spot_check_cases": spot_cases,
    }
    with open(os.path.join(OUT_DIR, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nwrote {OUT_DIR}/findings.json and {OUT_DIR}/overlays/{{baseline,winning}}/")


if __name__ == "__main__":
    main()
