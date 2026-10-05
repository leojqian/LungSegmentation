# Shared test doubles: a minimal DICOM writer and a fake U-Net. Imported by
# test modules as a sibling (`from fakes import ...`) — pytest puts tests/ on
# sys.path because the directory has no __init__.py.

import cv2
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from lungmap.segmentation.pipeline import MODEL_SIZE


def write_dicom(path, frames, photometric="MONOCHROME2", pixel_spacing=(0.4, 0.4),
                window_center=None, window_width=None, **tags):
    """frames: uint16 array (F, H, W). Minimal tags load_image actually reads,
    plus any extra keyword tags (e.g. FrameTime=66, PatientID="X")."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.NumberOfFrames, ds.Rows, ds.Columns = frames.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = list(pixel_spacing)
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    for keyword, value in tags.items():
        setattr(ds, keyword, value)
    ds.PixelData = frames.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, enforce_file_format=True)


class FakeModel:
    """Predicts two fixed, separated circular blobs at MODEL_SIZE resolution,
    regardless of input — decouples pipeline-wiring tests from real trained
    weights, following this codebase's synthetic-mask testing convention."""

    def predict(self, batch, verbose=0):
        canvas = np.zeros((MODEL_SIZE, MODEL_SIZE), np.float32)
        cv2.circle(canvas, (70, 140), 50, 1.0, -1)
        cv2.circle(canvas, (186, 140), 50, 1.0, -1)
        return canvas[None, ..., None]
