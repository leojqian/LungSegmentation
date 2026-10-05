# One DICOM through the whole pipeline: every frame is segmented, measured,
# reduced to the 8 output points, and drawn — streamed straight into the three
# output files so no full-size colour stack is ever held in memory.
#
# Orchestration only. Every step is an existing building block: loading and
# writing in formats/, segmentation and measuring in segmentation/pipeline.py,
# the points in points.py, drawing in rendering/render.py.

import csv
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm

from lungmap.formats.dicom_io import (cine_fps, encode_overlay_frame, load_frames,
                                      to_model_input, write_overlay_dicom)
from lungmap.formats.video import Mp4Writer
from lungmap.points import CSV_COLUMNS, SIDES, compare, csv_row, side_points
from lungmap.rendering.render import draw_frame_points
from lungmap.segmentation.pipeline import measure_combined, segment

# The fine-tuned model's evaluated setup (submission/evaluate_ddr.py): opening
# kernel 5 (finetune_ddr.MASK_OPEN_KERNEL, not pipeline.OPEN_KERNEL) and
# per-frame min/max normalization rather than the DICOM window tags.
OPEN_KERNEL = 5

#: comparison is points.compare's {side: {"n", "mean_px", "median_px"}};
#: incomplete counts frames where some point on some side couldn't be measured
RunResult = namedtuple("RunResult", "csv_path dicom_path mp4_path n_frames "
                                    "pixel_spacing_mm comparison incomplete")


def output_paths(dcm_path, out_dir):
    """-> (csv, overlay dicom, mp4) paths, named after the input file's stem."""
    stem, out_dir = Path(dcm_path).stem, Path(out_dir)
    return (out_dir / f"{stem}_points.csv", out_dir / f"{stem}_overlay.dcm",
            out_dir / f"{stem}_overlay.mp4")


def process_dicom(dcm_path, model, out_dir, progress=True):
    """DICOM path + loaded U-Net -> the three output files in out_dir, and a RunResult.

    Frames are analysed independently: one where a lung can't be found or
    traced gets blank CSV cells (and whatever could be drawn) instead of
    stopping the run.
    """
    dcm_path = Path(dcm_path)
    csv_path, dicom_path, mp4_path = output_paths(dcm_path, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frames, spacing_mm, photometric = load_frames(str(dcm_path))
    height, width = frames.shape[1:]

    per_frame, jpegs, incomplete = [], [], 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f, \
            Mp4Writer(mp4_path, cine_fps(str(dcm_path)), (width, height)) as video:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for i, raw in enumerate(tqdm(frames, desc=dcm_path.name, unit="frame",
                                     disable=not progress)):
            image = to_model_input(raw, photometric)
            masks, lungs, curves = measure_combined(segment(image, model, OPEN_KERNEL))
            points = {side: side_points(lungs.get(side), curves.get(side)) for side in SIDES}

            writer.writerow(csv_row(i, points))
            canvas = draw_frame_points(image, masks, curves, points, frame_idx=i)
            video.write(canvas)
            jpegs.append(encode_overlay_frame(canvas))

            per_frame.append(points)
            incomplete += any(p is None for sp in points.values() for p in sp)

    write_overlay_dicom(dicom_path, jpegs, (height, width), str(dcm_path))
    return RunResult(csv_path, dicom_path, mp4_path, len(frames), spacing_mm,
                     compare(per_frame), incomplete)
