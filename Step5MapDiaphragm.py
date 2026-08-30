# Map the diaphragm across the Montgomery County chest X-ray set.
#
#   python Step5MapDiaphragm.py                 overlays + contact sheets + curves.npz
#   python Step5MapDiaphragm.py --view          step through a window, writes nothing
#   python Step5MapDiaphragm.py --sample NAME   one image in a window
#   python Step5MapDiaphragm.py --predicted     segment with the U-Net, not ManualMask
#
# I/O and orchestration only: geometry lives in geometry/diaphragm.py, drawing in
# rendering/render.py.

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from geometry.diaphragm import RISE_FRACTION, apex_is_lower_bound, apex_of
from rendering.render import THUMBS_PER_SHEET, contact_sheet, draw_overlay
from segmentation.pipeline import analyze as analyze_image
from segmentation.pipeline import measure_masks

CXR_DIR = "MontgomerySet/CXR_png"
LEFT_DIR = "MontgomerySet/ManualMask/leftMask"
RIGHT_DIR = "MontgomerySet/ManualMask/rightMask"
OUT_DIR = "outputs"
MODEL_FILE = "best_model.h5"


# --- inputs ------------------------------------------------------------------

def stem(path):
    """'.../MCUCXR_0001_0.png' -> 'MCUCXR_0001_0'"""
    return os.path.splitext(os.path.basename(path))[0]


def find_pairs():
    """(cxr, left_mask, right_mask) triples that all exist on disk."""
    pairs = []
    for cxr in sorted(glob.glob(os.path.join(CXR_DIR, "*.png"))):
        name = os.path.basename(cxr)
        left, right = os.path.join(LEFT_DIR, name), os.path.join(RIGHT_DIR, name)
        if os.path.exists(left) and os.path.exists(right):
            pairs.append((cxr, left, right))
        else:
            print(f"  skip {name}: missing mask")
    return pairs


def select_sample(pairs, name):
    """Narrow the run to one image, by bare name or filename."""
    hit = [p for p in pairs if stem(p[0]) == stem(name)]
    if not hit:
        raise SystemExit(f"no image named {stem(name)!r} under {CXR_DIR}")
    return hit


def load_model(path=MODEL_FILE):
    """TensorFlow is imported here, not at module scope: the manual-mask path is
    the common one and should not pay a multi-second import for it."""
    import tensorflow as tf
    return tf.keras.models.load_model(path)


# --- per-image work ----------------------------------------------------------

def analyse(cxr_path, left_path, right_path, rise=RISE_FRACTION, model=None):
    """-> (cxr, left, right, left_mask, right_mask, left_lung, right_lung).

    left/right are Diaphragm or None; left_lung/right_lung are LungPoints or
    None. left/left_mask/left_lung are the image-left lung — this codebase's
    "R" side (see pipeline.py) — right the image-right ("L").
    """
    cxr = cv2.imread(cxr_path, cv2.IMREAD_GRAYSCALE)

    if model is not None:
        try:
            result = analyze_image(cxr_path, model, max_rise_frac=rise)
        except ValueError as e:
            print(f"  {stem(cxr_path)}: {e}")
            return cxr, None, None, None, None, None, None
        masks, lungs, curves = result.masks, result.lungs, result.curves
    else:
        masks = {"R": cv2.imread(left_path, cv2.IMREAD_GRAYSCALE) > 127,
                "L": cv2.imread(right_path, cv2.IMREAD_GRAYSCALE) > 127}
        lungs, curves = measure_masks(masks, rise, max_step=8)

    for side, label in (("R", "left"), ("L", "right")):
        if lungs[side] is None:
            print(f"  {stem(cxr_path)} {label} landmarks: not found")
        if curves[side] is None:
            print(f"  {stem(cxr_path)} {label}: diaphragm not found")

    return (cxr, curves["R"], curves["L"], masks["R"], masks["L"],
           lungs["R"], lungs["L"])


# --- runs --------------------------------------------------------------------

def run_batch(pairs, rise=RISE_FRACTION, show_masks=False, model=None,
              show_landmarks=False):
    os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)
    saved, tiles, heights, floors = {}, [], [], []

    for cxr_path, left_path, right_path in tqdm(pairs, desc="mapping diaphragm"):
        name = stem(cxr_path)
        cxr, left, right, lmask, rmask, llung, rlung = analyse(
            cxr_path, left_path, right_path, rise, model)
        saved[f"{name}_shape"] = np.array(cxr.shape)
        apexes = {}
        for side, curve, lung in (("left", left, llung), ("right", right, rlung)):
            if curve is None:
                continue
            saved[f"{name}_{side}"] = curve.measured
            if curve.fitted is not None:
                saved[f"{name}_{side}_dome"] = curve.fitted
            apexes[side] = apex_of(*curve)
            saved[f"{name}_{side}_apex"] = apexes[side]
            if lung is not None:
                saved[f"{name}_{side}_top"] = np.array(lung.top)
                saved[f"{name}_{side}_lower_left"] = np.array(lung.lower_left)
                saved[f"{name}_{side}_lower_right"] = np.array(lung.lower_right)
            if apex_is_lower_bound(curve.measured, apexes[side]):
                floors.append(f"{name}_{side}")
        if len(apexes) == 2:
            heights.append(apexes["right"][1] - apexes["left"][1])

        overlay = draw_overlay(cxr, left, right, lmask, rmask, show_masks,
                               show_landmarks=show_landmarks,
                               left_lung=llung, right_lung=rlung)
        cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"{name}.png"), overlay)
        tiles.append((name, cv2.resize(overlay, (256, 256),
                                       interpolation=cv2.INTER_AREA)))

    np.savez_compressed(os.path.join(OUT_DIR, "curves.npz"), **saved)
    sheets = 0
    for i in range(0, len(tiles), THUMBS_PER_SHEET):
        sheets += 1
        cv2.imwrite(os.path.join(OUT_DIR, f"contact_sheet_{sheets:02d}.png"),
                    contact_sheet(tiles[i:i + THUMBS_PER_SHEET]))

    _report(len(pairs), sheets, heights, floors)


def _report(n_images, sheets, heights, floors):
    print(f"\nwrote {n_images} overlays to {OUT_DIR}/overlays/")
    print(f"wrote {sheets} contact sheets and {OUT_DIR}/curves.npz")
    if floors:
        print(f"\n{len(floors)} hemidiaphragms have the apex at a curve end "
              f"(dome hidden behind the mediastinum).")
        print("  Those heights are LOWER BOUNDS — hollow marker, '>=' label.")
    if heights:
        d = np.array(heights)
        higher = "image-left" if np.median(d) > 0 else "image-right"
        n = (d > 0).sum() if higher == "image-left" else (d < 0).sum()
        print("\nhemidiaphragm elevation difference (full-res px):")
        print(f"  median {np.median(d):+.0f} ({higher} sits higher), "
              f"range {d.min():+.0f} to {d.max():+.0f}")
        print(f"  {higher} is higher in {n}/{len(d)} images")
    print(f"\n  open {OUT_DIR}/")


def run_viewer(pairs, rise=RISE_FRACTION, show_masks=False, model=None,
               show_landmarks=False):
    print("space/right = next   b/left = back   esc = quit")
    i = 0
    while 0 <= i < len(pairs):
        cxr_path, left_path, right_path = pairs[i]
        cxr, left, right, lmask, rmask, llung, rlung = analyse(
            cxr_path, left_path, right_path, rise, model)
        frame = draw_overlay(cxr, left, right, lmask, rmask, show_masks,
                             show_landmarks=show_landmarks,
                             left_lung=llung, right_lung=rlung)
        cv2.imshow("diaphragm", frame)
        cv2.setWindowTitle("diaphragm", f"[{i+1}/{len(pairs)}] {stem(cxr_path)}")
        # waitKey answers the KEYBOARD only — the window's red X does nothing
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        i = max(0, i + (-1 if key in (ord("b"), 2) else 1))
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS needs one more tick to dismiss the window


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", action="store_true",
                    help="step through results in a window instead of writing files")
    ap.add_argument("--sample", metavar="NAME",
                    help="show a single image, e.g. MCUCXR_0035_0 (implies --view)")
    ap.add_argument("--limit", type=int, help="only process the first N images")
    ap.add_argument("--predicted", action="store_true",
                    help=f"segment with {MODEL_FILE} instead of the ManualMask PNGs")
    ap.add_argument("--landmarks", action="store_true",
                    help="also draw the lung outline and the three corner markers")
    ap.add_argument("--masks", action="store_true",
                    help="tint the lung regions under the curves")
    ap.add_argument("--rise", type=float, default=RISE_FRACTION, metavar="F",
                    help="how far the curve may climb above the costophrenic angle, "
                         f"as a fraction of lung height (default: {RISE_FRACTION})")
    args = ap.parse_args()

    pairs = find_pairs()
    if not pairs:
        raise SystemExit(f"no image/mask pairs found under {CXR_DIR}")
    if args.sample:
        pairs = select_sample(pairs, args.sample)
    elif args.limit:
        pairs = pairs[:args.limit]

    model = None
    if args.predicted:
        print(f"loading {MODEL_FILE} ...")
        model = load_model()

    run = run_viewer if (args.view or args.sample) else run_batch
    run(pairs, args.rise, args.masks, model, args.landmarks)


if __name__ == "__main__":
    main()
