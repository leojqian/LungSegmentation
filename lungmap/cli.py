# The `lungmap` command (also `python -m lungmap`; the double-click launchers
# run it too). DICOM files or folders in; per file, a points CSV, an overlay
# DICOM and an MP4 (made by process.py).
#
# Argument handling and terminal output only. Output is ASCII-only, for legacy
# Windows consoles.

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from lungmap import __version__
from lungmap import model_file
from lungmap.model_file import LEGACY_NAME, MODEL_ENV
from lungmap.process import process_dicom

OUT_DIR_NAME = "lungmap_output"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="lungmap",
        description="Segment the lungs on every frame of a chest X-ray DICOM and write, per "
                    "input: <name>_points.csv (8 points per frame: apex, diaphragm point and "
                    "both bottom corners, for each lung), <name>_overlay.dcm and "
                    "<name>_overlay.mp4 (the same points and lung masks drawn on each frame).")
    ap.add_argument("inputs", nargs="+", type=Path, metavar="DICOM",
                    help="DICOM files, folders of them, or wildcards like *.dcm")
    ap.add_argument("-o", "--out-dir", type=Path, default=None,
                    help=f"output folder (default: a '{OUT_DIR_NAME}' folder next to each input)")
    ap.add_argument("--model", type=Path, default=None,
                    help=f"U-Net checkpoint (.h5). Default: ${MODEL_ENV} if set, else "
                         f"./{LEGACY_NAME} if present, else the copy downloaded on first run")
    ap.add_argument("--open", action="store_true",
                    help="open the output folder(s) when finished")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def is_dicom(path):
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except (InvalidDicomError, OSError):
        return False


def collect_inputs(ap, args):
    """Paths as typed -> the DICOM files to process, in order, without repeats.

    Folders contribute every DICOM directly inside them, and wildcards are
    expanded here because Windows shells pass `*.dcm` through literally. Files
    found that way are skipped quietly if they aren't DICOM (or are one of our
    own overlays); a file named explicitly must be DICOM.
    """
    def candidates(paths):
        return [p for p in paths
                if p.is_file() and not p.name.endswith("_overlay.dcm") and is_dicom(p)]

    found = []
    for arg in args:
        if arg.is_dir():
            found += candidates(sorted(arg.iterdir()))
        elif arg.is_file():
            if not is_dicom(arg):
                ap.error(f"not a readable DICOM file: {arg}")
            found.append(arg)
        elif any(c in str(arg) for c in "*?["):
            matches = sorted(Path(p) for p in glob.glob(str(arg)))
            if not matches:
                ap.error(f"nothing matches {arg}")
            found += candidates(matches)
        else:
            ap.error(f"no such file or folder: {arg}")
    if not found:
        ap.error("no DICOM files found in: " + ", ".join(str(a) for a in args))
    return list(dict.fromkeys(found))


def open_folder(path):
    """Show a folder in Explorer / Finder. Best effort: skipped where there's no desktop."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        print(f"(could not open {path} automatically)")


def _print_summary(dcm_path, result):
    print(f"\n{dcm_path.name}: {result.n_frames} frames"
          f" ({result.incomplete} with a point missing)")
    print(f"  points : {result.csv_path}")
    print(f"  overlay: {result.dicom_path}")
    print(f"  video  : {result.mp4_path}")
    mm = result.pixel_spacing_mm
    print("  naive corner midpoint vs diaphragm point"
          + (f" (pixel spacing {mm:.3f} mm):" if mm else ":"))
    for side, s in result.comparison.items():
        if not s["n"]:
            print(f"    {side}: no frames with both points")
            continue
        mean, median = (f"{px:.1f} px" + (f" ({px * mm:.1f} mm)" if mm else "")
                        for px in (s["mean_px"], s["median_px"]))
        print(f"    {side}: mean {mean}, median {median}, n={s['n']}")


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    inputs = collect_inputs(ap, args.inputs)

    model_path = model_file.find_model(args.model)
    if model_path is not None and not model_path.is_file():
        ap.error(f"model not found: {model_path} (check --model or ${MODEL_ENV})")

    failed, out_dirs = [], []
    try:
        if model_path is None:
            try:
                model_path = model_file.fetch_model()
            except model_file.ModelError as e:
                print(f"lungmap: error: {e}", file=sys.stderr)
                return 1
        print(f"loading model {model_path} ...")
        model = model_file.load_model(model_path)

        for i, path in enumerate(inputs, 1):
            out_dir = args.out_dir or path.parent / OUT_DIR_NAME
            if len(inputs) > 1:
                print(f"\n[{i}/{len(inputs)}] {path}")
            try:
                result = process_dicom(path, model, out_dir)
            except Exception as e:   # one bad file in a folder must not stop the rest
                failed.append(path)
                print(f"lungmap: error: {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            _print_summary(path, result)
            if out_dir not in out_dirs:
                out_dirs.append(out_dir)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130

    if args.open:
        for out_dir in out_dirs:
            open_folder(out_dir)
    if failed:
        print(f"\n{len(failed)} of {len(inputs)} file(s) failed: "
              + ", ".join(p.name for p in failed))
        return 1
    return 0
