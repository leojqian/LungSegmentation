# DICOM I/O for the DDR dataset: reading multi-frame cine files and the two
# ground-truth formats that ship with them (SampleDDR_August2026).
#
# Pure functions — no model, no CLI. Mirrors landmarks.py's separation of
# pure data-handling from orchestration.

import defusedxml.ElementTree as ET

import numpy as np
import pydicom

# Identity, not swapped: tested empirically against all 4 combinations of
# (mirror x / don't) x (swap R<->L / don't), aggregated across all 119
# annotated frames, scored against the fine-tuned model's own predicted apex
# (itself derived from the now-correctly-oriented image): untouched labels +
# untouched coordinates won decisively (25.0mm mean error) over every
# alternative, including mirror+swap (43.5mm) -- so DM-MODE_truth's XML was
# NOT annotated against the same raw/unflipped orientation LungArea_truth
# was; its own export pipeline evidently already applied the correct
# orientation. An earlier version of this dict swapped Right<->Left based on
# a single frame's worth of evidence that turned out to have unusual
# (non-dome-shaped) polylines -- see the fuller 119-frame test before
# trusting a one-frame read on this again.
LABEL_TO_SIDE = {"Right": "R", "Left": "L"}


def load_frames(path):
    """Multi-frame DICOM -> (frames uint16 (F, H, W), pixel_spacing_mm, photometric).

    pixel_spacing_mm assumes square pixels, true of every file in this set.

    Applies (0018,7034) Field of View Horizontal Flip when the acquisition
    device set it to "YES" (true of every file in this dataset) -- confirmed
    by direct visual check: the cardiac silhouette sits on the wrong side of
    the unflipped pixel data relative to true anatomy (PA convention: patient's
    right is on image-left, so the heart, mostly patient-left, belongs on
    image-right -- it isn't, until this flip is applied).

    The two ground-truth formats needed different treatment once this flip
    was added: LungArea_truth's raw file already matches the corrected image
    as-is (parse_lung_area_truth() applies no transform of its own).
    DM-MODE_truth's XML also needed no transform, once tested properly
    (parse_dm_mode_truth_lines()) -- its export pipeline evidently already
    applied the same correction independently, unlike LungArea_truth's.
    """
    ds = pydicom.dcmread(path)
    spacing = ds.PixelSpacing if "PixelSpacing" in ds else ds.ImagerPixelSpacing
    pixels = ds.pixel_array
    if pixels.ndim == 2:  # pydicom squeezes the frame axis when NumberOfFrames == 1
        pixels = pixels[None]
    flip_tag = ds.get((0x0018, 0x7034))
    if flip_tag is not None and flip_tag.value == "YES":
        pixels = np.ascontiguousarray(pixels[:, :, ::-1])
    return pixels, float(spacing[0]), ds.PhotometricInterpretation


def window_params(path):
    """DICOM -> (window_center, window_width), or (None, None) if the file has
    no window tags. Reads header only (stop_before_pixels), so it's cheap to
    call once per case without re-decoding pixel data."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    def scalar(value):
        if value is None:
            return None
        return float(value[0]) if isinstance(value, pydicom.multival.MultiValue) else float(value)

    return scalar(ds.get("WindowCenter")), scalar(ds.get("WindowWidth"))


def to_model_input(frame, photometric, window_center=None, window_width=None):
    """One raw frame -> normalized float32 HxW in [0, 1].

    KONICA MINOLTA's RF images are MONOCHROME1 (0 displays as white), the
    opposite of the PNGs the model was trained on, so those get inverted after
    normalizing. window_center/width are DICOM-standard bounds; frame min/max
    is the fallback when a file has no window tags.
    """
    frame = frame.astype(np.float32)
    if window_center is not None and window_width is not None:
        lo, hi = window_center - window_width / 2, window_center + window_width / 2
    else:
        lo, hi = frame.min(), frame.max()
    norm = np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)
    return 1 - norm if photometric == "MONOCHROME1" else norm


# --- ground truth --------------------------------------------------------------

def parse_lung_area_truth(raw_path, shape):
    """LungArea_truth .raw -> {frame_idx: mask}, annotated (nonzero) frames only.

    Unannotated frames are all-zero by convention (per the dataset's README) and
    are dropped here rather than returned as empty masks that would score as a
    perfect true-negative.

    No flip here: the .raw file is annotated against the DICOM's raw,
    unflipped pixel data -- load_frames() now applies the acquisition device's
    own Field of View Horizontal Flip, so this mask aligns with that corrected
    image without changes of its own. (An earlier version of this function
    flipped the mask instead, which papered over load_frames() not applying
    that flip yet -- confirmed wrong by direct visual check: the cardiac
    silhouette sat on the wrong side of the then-unflipped image.)
    """
    h, w = shape
    frames = np.fromfile(raw_path, dtype=np.uint8).reshape(-1, h, w)
    return {i: frame for i, frame in enumerate(frames) if frame.any()}


def _diaphragm_point(points):
    """Polyline points -> the (x, y) at the polyline's horizontal center.

    Matches the dataset README: "The diaphragm point is defined as the center
    x-coordinate of both the right and left diaphragm."
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    center_x = (pts[:, 0].min() + pts[:, 0].max()) / 2
    return (center_x, float(np.interp(center_x, pts[:, 0], pts[:, 1])))


def parse_dm_mode_truth_lines(xml_path):
    """DM-MODE_truth .xml -> {frame_idx: {"R": [(x, y), ...], "L": [(x, y), ...]}}.

    Raw annotated polyline vertices, one list per side, before collapsing to a
    single point — parse_dm_mode_truth's _diaphragm_point call is one way to
    reduce a line to a point (the dataset's own "center x" convention); a
    caller wanting the line's actual peak (matching diaphragm.dome_apex's
    definition) needs these vertices, not that reduced point.

    XML "Right"/"Left" are anatomical sides, which is this codebase's own R/L
    convention (R = image-left, the patient's right lung on a PA film) — no
    transform needed, see LABEL_TO_SIDE's comment for how that was verified.
    Frames without a polyline are unannotated and omitted.
    """
    root = ET.parse(xml_path).getroot()
    out = {}
    for image in root.iter("image"):
        polylines = image.findall("polyline")
        if not polylines:
            continue
        sides = {}
        for poly in polylines:
            side = LABEL_TO_SIDE.get(poly.get("label"))
            if side is None:
                continue
            sides[side] = [tuple(map(float, p.split(","))) for p in poly.get("points").split(";")]
        if sides:
            out[int(image.get("id"))] = sides
    return out


def parse_dm_mode_truth(xml_path):
    """DM-MODE_truth .xml -> {frame_idx: {"R": (x, y), "L": (x, y)}}.

    Each point is parse_dm_mode_truth_lines's polyline reduced to the dataset's
    "center x-coordinate" convention (see _diaphragm_point) — not the line's peak.
    """
    return {frame: {side: _diaphragm_point(pts) for side, pts in sides.items()}
           for frame, sides in parse_dm_mode_truth_lines(xml_path).items()}
