# DICOM I/O for the DDR dataset: reading multi-frame cine files and the two
# ground-truth formats that ship with them (SampleDDR_August2026), and writing
# the CLI's annotated overlay back out as DICOM.
#
# Pure functions — no model, no CLI. Mirrors landmarks.py's separation of
# pure data-handling from orchestration.

import defusedxml.ElementTree as ET

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.tag import Tag
from pydicom.uid import (JPEGBaseline8Bit, MultiFrameTrueColorSecondaryCaptureImageStorage,
                         generate_uid)

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

DEFAULT_FPS = 15.0          # cine_fps fallback; DDR's own FrameTime of 66ms is ~15fps
OVERLAY_JPEG_QUALITY = 90

# Copied from the source so a viewer files the overlay under the same patient
# and study. The Type 2 ones (PatientName ... AccessionNumber) must be present
# in a Secondary Capture even if empty, so they're written blank when missing.
_COPIED_IF_PRESENT = ("SpecificCharacterSet", "StudyDescription")
_COPIED_TYPE2 = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyID",
                 "AccessionNumber")


def load_frames(path):
    """Multi-frame DICOM -> (frames uint16 (F, H, W), pixel_spacing_mm, photometric).

    pixel_spacing_mm assumes square pixels, true of every file in this set. It
    is None for a file with neither PixelSpacing nor ImagerPixelSpacing.

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
    spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    pixels = ds.pixel_array
    if pixels.ndim == 2:  # pydicom squeezes the frame axis when NumberOfFrames == 1
        pixels = pixels[None]
    flip_tag = ds.get((0x0018, 0x7034))
    if flip_tag is not None and flip_tag.value == "YES":
        pixels = np.ascontiguousarray(pixels[:, :, ::-1])
    return pixels, float(spacing[0]) if spacing else None, ds.PhotometricInterpretation


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


def cine_fps(path):
    """DICOM -> playback frame rate. Header only, like window_params.

    FrameTime (ms per frame) first — DDR files carry 66ms, ~15fps — then
    CineRate, then DEFAULT_FPS for files that carry neither.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    frame_time = ds.get("FrameTime")
    if frame_time and float(frame_time) > 0:
        return 1000.0 / float(frame_time)
    cine_rate = ds.get("CineRate")
    if cine_rate and float(cine_rate) > 0:
        return float(cine_rate)
    return DEFAULT_FPS


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


# --- overlay output ------------------------------------------------------------

def encode_overlay_frame(bgr, quality=OVERLAY_JPEG_QUALITY):
    """BGR uint8 frame -> JPEG Baseline bytes, as write_overlay_dicom stores them.

    4:2:2 chroma subsampling, because that is what the DICOM photometric
    interpretation YBR_FULL_422 promises a viewer. cv2's default is 4:2:0.
    """
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def write_overlay_dicom(out_path, jpeg_frames, shape, source_path):
    """encode_overlay_frame outputs -> one multi-frame colour DICOM at out_path.

    A Multi-frame True Color Secondary Capture, JPEG Baseline. Uncompressed
    RGB would be ~790MB for a 233-frame DDR case; this is tens of MB, and JPEG
    Baseline is the most widely supported compressed DICOM transfer syntax. shape is (rows, cols) of every frame.

    Patient/study identity, pixel spacing and frame timing come from
    source_path so the overlay sits beside the original in a viewer and plays
    at the same speed. The Field of View Horizontal Flip tag is deliberately
    NOT copied: load_frames already applied that flip to the pixels drawn
    here, and a viewer honouring the tag would flip them back.
    """
    src = pydicom.dcmread(source_path, stop_before_pixels=True)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MultiFrameTrueColorSecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = JPEGBaseline8Bit

    ds = FileDataset(str(out_path), {}, file_meta=meta, preamble=b"\x00" * 128)
    for keyword in _COPIED_IF_PRESENT:
        if keyword in src:
            setattr(ds, keyword, src[keyword].value)
    for keyword in _COPIED_TYPE2:
        setattr(ds, keyword, src[keyword].value if keyword in src else "")

    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = src.get("StudyInstanceUID") or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.SeriesDescription = "lungmap overlay"
    ds.SeriesNumber = ""
    ds.InstanceNumber = 1
    ds.PatientOrientation = ""
    ds.ConversionType = "WSD"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.DerivationDescription = ("lungmap: lung masks, apexes, bottom corners, diaphragm "
                                "points and naive corner midpoints burned in")
    ds.BurnedInAnnotation = "YES"

    spacing = src.get("PixelSpacing") or src.get("ImagerPixelSpacing")
    if spacing is not None:
        ds.PixelSpacing = list(spacing)
    if src.get("FrameTime"):
        ds.FrameTime = src.FrameTime
        ds.FrameIncrementPointer = Tag("FrameTime")

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = int(shape[0]), int(shape[1])
    ds.NumberOfFrames = len(jpeg_frames)
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.LossyImageCompression = "01"
    ds.LossyImageCompressionMethod = "ISO_10918_1"

    ds.PixelData = encapsulate(list(jpeg_frames))
    ds["PixelData"].VR = "OB"
    ds.save_as(str(out_path), enforce_file_format=True)


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
