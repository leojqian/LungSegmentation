# Browse every ground-truthed frame across the whole DDR dataset
# (SampleDDR_August2026): LungArea_truth mask outline + DM-MODE_truth
# diaphragm points overlaid on the DICOM image. --predict adds the model's
# own predicted mask alongside the truth, for sanity-checking predictions
# after fine-tuning (Step8FineTuneDDR.py); without it, this is a pure
# dataset viewer -- no model involved.
#
#   python view_ddr_truth.py                  every case, all annotated frames
#   python view_ddr_truth.py --case NAME       one case only, e.g. KaU001_PA_deep
#   python view_ddr_truth.py --source-raw      use the unprocessed `images` DICOMs
#                                               instead of `images_pres` (default,
#                                               matches the winning calibration in
#                                               outputs/ddr_calibration/findings.json)
#   python view_ddr_truth.py --grid            every annotated frame at once, as
#                                               one contact sheet, instead of
#                                               stepping through them one-by-one
#   python view_ddr_truth.py --predict         also overlay the model's predicted
#                                               mask (green) on every frame; truth
#                                               (yellow) still overlays wherever a
#                                               frame has it
#   python view_ddr_truth.py --predict --model best_model.h5
#                                               predict with a different checkpoint
#                                               (default: the fine-tuned one)
#
# Annotated frames are sparse (LungArea_truth and DM-MODE_truth mark different
# frames per case, ~2 and ~4 respectively) so indexing reads every case's
# truth files up front; only the current case's DICOM pixel data is cached.
#
# Legend: yellow outline = LungArea truth mask; green outline = predicted mask
# (--predict only); hollow circle = DM-MODE truth diaphragm point, colored by
# side (R/L) -- same palette as Step6's overlays.

import argparse
import os

import cv2
import numpy as np
import pydicom
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from tqdm import tqdm

from formats.dicom_io import load_frames, parse_dm_mode_truth, parse_lung_area_truth, to_model_input
from rendering.render import contact_sheet
from segmentation.pipeline import segment
import Step7CalibrateDDR as step7
from Step6ValidateDDR import IMAGES_PRES_DIR, SIDE_COLOR, _measure_from_mask, find_cases, load_model

DEFAULT_PREDICT_MODEL = "best_model_ddr_finetuned_phase3.h5"
PRED_CACHE_DIR = "outputs/.cache/ddr_pred_masks"


def _model_tag(model_path):
    """Identifies a checkpoint file by path+size+mtime -- cheap, no content hash,
    and changes automatically if the file is retrained/overwritten."""
    st = os.stat(model_path)
    return f"{os.path.basename(model_path)}_{st.st_size}_{int(st.st_mtime)}"


def make_model_loader(model_path):
    """-> zero-arg callable returning the loaded model, loaded (and TF imported)
    only on first actual use -- so a fully cache-hit run never pays for either."""
    state = {}

    def get():
        if "model" not in state:
            print(f"loading {model_path} ...")
            state["model"] = load_model(model_path)
        return state["model"]

    return get


def predict_cached(img, model_loader, model_path, case_id, frame_idx):
    """segment(), memoized to PRED_CACHE_DIR by (checkpoint identity, case, frame).

    Model inference + the full-resolution upscale/morphology in segment() is the
    dominant cost of --predict (~0.2s/frame, most of a --grid run's time) --
    caching skips all of it on repeat runs against the same checkpoint.
    """
    os.makedirs(PRED_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(PRED_CACHE_DIR, f"{case_id}_{frame_idx}_{_model_tag(model_path)}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    mask = segment(img, model_loader())
    np.save(cache_path, mask)
    return mask


def pred_apex_points(pred_mask):
    """Predicted mask -> {side: (x, y)}, the same apex_of()/dome geometry Step6 scores with."""
    if pred_mask is None:
        return {}
    _, lungs, _ = _measure_from_mask(pred_mask)
    return {side: lp.dome for side, lp in lungs.items() if lp is not None}


def _put_label(canvas, text, org, color, scale=0.7):
    """cv2.putText with a black outline so it stays legible over any x-ray brightness."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame_img, pred_mask, truth_mask, truth_points, pred_points=None,
                 spacing_mm=None, max_dim=1200, label_scale=0.7):
    """One frame's image + pred/truth mask outlines + pred/truth diaphragm points -> BGR canvas.

    Point convention matches view_ddr_overlay.py: hollow circle = truth, filled
    circle = predicted, connecting line = the error, colored by side (R/L).
    When both points exist and spacing_mm is given, the error distance is
    labeled directly on the line in px and mm.
    """
    canvas = cv2.cvtColor((np.clip(frame_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if truth_mask is not None:
        contours, _ = cv2.findContours(truth_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)  # yellow
    if pred_mask is not None:
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)  # green

    truth_points, pred_points = truth_points or {}, pred_points or {}
    for side in set(truth_points) | set(pred_points):
        color = SIDE_COLOR[side]
        truth_xy = tuple(int(v) for v in truth_points[side]) if side in truth_points else None
        pred_xy = tuple(int(v) for v in pred_points[side]) if side in pred_points else None
        if truth_xy is not None:
            cv2.circle(canvas, truth_xy, 8, color, 2)
        if pred_xy is not None:
            cv2.circle(canvas, pred_xy, 6, color, -1)
        if truth_xy is not None and pred_xy is not None:
            cv2.line(canvas, pred_xy, truth_xy, color, 1, cv2.LINE_AA)
            error_px = float(np.hypot(pred_xy[0] - truth_xy[0], pred_xy[1] - truth_xy[1]))
            label = f"{side} {error_px:.0f}px"
            if spacing_mm:
                label += f" / {error_px * spacing_mm:.1f}mm"
            # anchored off the farther-out point with a fixed offset, not the
            # midpoint -- when error is small the two markers sit almost on top
            # of each other and a midpoint label collides with both
            anchor = truth_xy if truth_xy[1] >= pred_xy[1] else pred_xy
            _put_label(canvas, label, (anchor[0] + 14, anchor[1] + 28), color, scale=label_scale)

    f = min(1.0, max_dim / max(canvas.shape[:2]))
    return cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def enable_scroll_zoom(fig, ax, base_scale=1.2):
    """Scroll wheel zooms in/out of ax, centered on the cursor (not the axes center)."""
    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None:
            return
        scale = 1 / base_scale if event.button == "up" else base_scale
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        relx = (xlim[1] - event.xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - event.ydata) / (ylim[1] - ylim[0])
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        ax.set_xlim(event.xdata - new_w * (1 - relx), event.xdata + new_w * relx)
        ax.set_ylim(event.ydata - new_h * (1 - rely), event.ydata + new_h * rely)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)


def build_index(cases):
    """cases -> [(case_id, dcm_path, frame_idx, truth_mask_or_None, truth_points_or_None), ...]

    Reads each case's truth files (not its DICOM pixel data) to find annotated
    frames, so this is the slow, one-time part of startup -- LungArea_truth
    files are ~250MB each.
    """
    entries = []
    for case_id, dcm_path, xml_path, raw_path in tqdm(cases, desc="indexing ground truth"):
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        lung_truth = parse_lung_area_truth(raw_path, shape=(ds.Rows, ds.Columns))
        dm_truth = parse_dm_mode_truth(xml_path)
        frame_idxs = sorted(set(lung_truth) | set(dm_truth))
        for i in frame_idxs:
            entries.append((case_id, dcm_path, i, lung_truth.get(i), dm_truth.get(i)))
    return entries


def build_thumbnails(entries, get_frames, thumb_dim, model_loader=None, model_path=None):
    """entries -> [(caption, tile), ...] for contact_sheet, one tile per annotated frame."""
    tiles = []
    for case_id, dcm_path, frame_idx, truth_mask, truth_points in tqdm(entries, desc="rendering thumbnails"):
        frames, photometric, _ = get_frames(case_id, dcm_path)
        img = to_model_input(frames[frame_idx], photometric)
        pred_mask = (predict_cached(img, model_loader, model_path, case_id, frame_idx)
                    if model_loader is not None else None)
        # no distance labels here -- too small/cluttered at grid thumbnail scale,
        # the single-frame slideshow view is where those matter (see render() below)
        tile = draw_overlay(img, pred_mask, truth_mask, truth_points,
                            pred_points=pred_apex_points(pred_mask), max_dim=thumb_dim)
        caption = f"{case_id.replace('_PA_deep', '')} f{frame_idx}"
        tiles.append((caption, tile))
    return tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help="restrict to one case id, e.g. KaU001_PA_deep")
    ap.add_argument("--source-raw", action="store_true",
                    help="browse the unprocessed `images` DICOMs instead of `images_pres`")
    ap.add_argument("--grid", action="store_true",
                    help="show every annotated frame at once as one contact sheet, "
                         "instead of stepping through them one-by-one")
    ap.add_argument("--masks-only", action="store_true",
                    help="only include frames with a LungArea_truth mask, dropping "
                         "the DM-MODE_truth point-only frames")
    ap.add_argument("--cols", type=int, default=12, help="grid columns for --grid (default 12)")
    ap.add_argument("--thumb-dim", type=int, default=180,
                    help="thumbnail size in px for --grid (default 180)")
    ap.add_argument("--out", default="outputs/ddr_truth_grid.png",
                    help="where to save the --grid contact sheet")
    ap.add_argument("--predict", action="store_true",
                    help="overlay the model's predicted mask (green) on every frame, "
                         "not just the ones with LungArea_truth")
    ap.add_argument("--model", default=DEFAULT_PREDICT_MODEL,
                    help=f"checkpoint to predict with, only used with --predict "
                         f"(default {DEFAULT_PREDICT_MODEL})")
    args = ap.parse_args()

    images_dir = step7.RAW_IMAGES_DIR if args.source_raw else IMAGES_PRES_DIR
    suffix = "" if args.source_raw else "LogPlusRegius"
    cases = find_cases(images_dir, suffix=suffix)
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
        if not cases:
            raise SystemExit(f"no case named {args.case!r} under {images_dir}")
    if not cases:
        raise SystemExit(f"no cases found under {images_dir}")

    entries = build_index(cases)
    if args.masks_only:
        entries = [e for e in entries if e[3] is not None]
    if not entries:
        raise SystemExit("no annotated frames found")
    n_cases = len(set(e[0] for e in entries))
    print(f"{len(entries)} annotated frames across {n_cases} case(s)")

    model_loader = make_model_loader(args.model) if args.predict else None

    cache = {"case_id": None, "frames": None, "photometric": None, "spacing_mm": None}

    def get_frames(case_id, dcm_path):
        if cache["case_id"] != case_id:
            frames, spacing_mm, photometric = load_frames(dcm_path)
            cache.update(case_id=case_id, frames=frames, photometric=photometric, spacing_mm=spacing_mm)
        return cache["frames"], cache["photometric"], cache["spacing_mm"]

    legend = "yellow=LungArea truth mask  hollow circle=DM-MODE truth point"
    if args.predict:
        legend += "\ngreen=predicted mask  filled circle=predicted point  line=error (labeled in px/mm)"

    if args.grid:
        tiles = build_thumbnails(entries, get_frames, thumb_dim=args.thumb_dim,
                                 model_loader=model_loader, model_path=args.model)
        sheet = contact_sheet(tiles, cols=args.cols)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}px, {len(tiles)} frames)")

        fig, ax = plt.subplots(figsize=(16, 16 * sheet.shape[0] / sheet.shape[1]))
        ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{len(tiles)} annotated frames across {n_cases} case(s)  ({legend})\n"
                    "scroll to zoom (cursor-centered), toolbar's home button resets")
        plt.tight_layout()
        enable_scroll_zoom(fig, ax)
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.12)

    def render(slider_pos):
        case_id, dcm_path, frame_idx, truth_mask, truth_points = entries[int(slider_pos)]
        frames, photometric, spacing_mm = get_frames(case_id, dcm_path)
        img = to_model_input(frames[frame_idx], photometric)
        pred_mask = (predict_cached(img, model_loader, args.model, case_id, frame_idx)
                    if model_loader is not None else None)
        rgb = cv2.cvtColor(draw_overlay(img, pred_mask, truth_mask, truth_points,
                                        pred_points=pred_apex_points(pred_mask),
                                        spacing_mm=spacing_mm, label_scale=0.9),
                           cv2.COLOR_BGR2RGB)

        ax.clear()  # case DICOMs vary in resolution, so re-imshow rather than set_data
        ax.imshow(rgb)
        ax.axis("off")

        kinds = [k for k, v in (("mask", truth_mask), ("points", truth_points)) if v is not None]
        ax.set_title(f"{case_id}  frame {frame_idx}  [{int(slider_pos) + 1}/{len(entries)}]  "
                    f"[{'+'.join(kinds)}]\n{legend}")
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.04])
    slider = Slider(slider_ax, "Annotated frame", 0, len(entries) - 1, valinit=0, valstep=1)
    slider.on_changed(render)

    def step(delta):
        slider.set_val((int(slider.val) + delta) % len(entries))

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
