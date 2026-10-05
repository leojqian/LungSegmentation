# Tests for dicom_io.py — reading DDR multi-frame DICOMs and the two ground-truth
# formats that ship alongside them (SampleDDR_August2026).

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from lungmap.formats.dicom_io import (load_frames, parse_dm_mode_truth, parse_lung_area_truth,
                             to_model_input, window_params)


def write_dicom(path, frames, photometric="MONOCHROME1", pixel_spacing=(0.4, 0.4),
                window_center=None, window_width=None, horizontal_flip=None):
    """frames: uint16 array (F, H, W). Minimal tags load_frames actually reads."""
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
    if horizontal_flip is not None:
        ds.FieldOfViewHorizontalFlip = horizontal_flip
    ds.PixelData = frames.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def dicom_path(tmp_path):
    path = str(tmp_path / "case.dcm")
    frames = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    write_dicom(path, frames)
    return path


class TestLoadFrames:
    def test_returns_pixel_array_matching_source(self, dicom_path):
        frames, spacing, photometric = load_frames(dicom_path)
        assert frames.shape == (2, 4, 4)
        assert frames[1, 0, 0] == 16  # second frame's first pixel, per np.arange layout

    def test_single_frame_dicom_still_returns_3d(self, tmp_path):
        # pydicom squeezes the frame axis away when NumberOfFrames == 1.
        path = str(tmp_path / "single.dcm")
        write_dicom(path, np.zeros((1, 4, 4), dtype=np.uint16))
        frames, _, _ = load_frames(path)
        assert frames.shape == (1, 4, 4)

    def test_returns_pixel_spacing_in_mm(self, dicom_path):
        _, spacing, _ = load_frames(dicom_path)
        assert spacing == pytest.approx(0.4)

    def test_pixel_spacing_is_none_when_the_file_has_no_spacing_tag(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        del ds.PixelSpacing
        ds.save_as(dicom_path)
        _, spacing, _ = load_frames(dicom_path)
        assert spacing is None

    def test_returns_photometric_interpretation(self, dicom_path):
        _, _, photometric = load_frames(dicom_path)
        assert photometric == "MONOCHROME1"

    def test_flips_horizontally_when_device_tag_says_yes(self, tmp_path):
        path = str(tmp_path / "flipped.dcm")
        frames = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
        write_dicom(path, frames, horizontal_flip="YES")
        loaded, _, _ = load_frames(path)
        assert np.array_equal(loaded, frames[:, :, ::-1])

    def test_does_not_flip_when_device_tag_says_no(self, tmp_path):
        path = str(tmp_path / "notflipped.dcm")
        frames = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
        write_dicom(path, frames, horizontal_flip="NO")
        loaded, _, _ = load_frames(path)
        assert np.array_equal(loaded, frames)

    def test_does_not_flip_when_device_tag_absent(self, dicom_path):
        # dicom_path's fixture DICOM has no FieldOfViewHorizontalFlip tag at all.
        frames = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
        loaded, _, _ = load_frames(dicom_path)
        assert np.array_equal(loaded, frames)


class TestWindowParams:
    def test_returns_scalar_window_tags(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.zeros((1, 4, 4), dtype=np.uint16),
                   window_center=2048, window_width=4095)
        wc, ww = window_params(path)
        assert wc == pytest.approx(2048)
        assert ww == pytest.approx(4095)

    def test_multivalued_window_tags_take_the_first(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.zeros((1, 4, 4), dtype=np.uint16),
                   window_center=[2048, 1024], window_width=[4095, 2000])
        wc, ww = window_params(path)
        assert wc == pytest.approx(2048)
        assert ww == pytest.approx(4095)

    def test_missing_window_tags_return_none(self, dicom_path):
        wc, ww = window_params(dicom_path)
        assert wc is None and ww is None


class TestToModelInput:
    def test_monochrome2_normalizes_without_inverting(self):
        frame = np.array([[0, 100], [50, 100]], dtype=np.uint16)
        out = to_model_input(frame, "MONOCHROME2")
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(1.0)

    def test_monochrome1_inverts(self):
        frame = np.array([[0, 100], [50, 100]], dtype=np.uint16)
        out = to_model_input(frame, "MONOCHROME1")
        assert out[0, 0] == pytest.approx(1.0)
        assert out[0, 1] == pytest.approx(0.0)

    def test_window_center_width_overrides_min_max(self):
        frame = np.array([[2048, 4096]], dtype=np.uint16)
        out = to_model_input(frame, "MONOCHROME2", window_center=2048, window_width=4096)
        assert out[0, 0] == pytest.approx(0.5)

    def test_output_is_clipped_to_unit_range(self):
        frame = np.array([[0, 5000]], dtype=np.uint16)
        out = to_model_input(frame, "MONOCHROME2", window_center=2048, window_width=100)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestParseLungAreaTruth:
    def test_keeps_only_nonzero_frames(self, tmp_path):
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        frames[1] = 1
        path = tmp_path / "truth.raw"
        frames.tofile(path)

        truth = parse_lung_area_truth(str(path), shape=(4, 4))

        assert set(truth) == {1}
        assert np.array_equal(truth[1], frames[1])

    def test_empty_file_returns_no_frames(self, tmp_path):
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        path = tmp_path / "truth.raw"
        frames.tofile(path)

        assert parse_lung_area_truth(str(path), shape=(4, 4)) == {}


DM_MODE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="0000.png" width="100" height="100">
  </image>
  <image id="5" name="0005.png" width="100" height="100">
    <polyline label="Right" source="manual" occluded="0"
              points="0.00,10.00;10.00,20.00;20.00,30.00" z_order="0">
    </polyline>
    <polyline label="Left" source="manual" occluded="0"
              points="50.00,0.00;60.00,10.00" z_order="0">
    </polyline>
  </image>
</annotations>
"""


class TestParseDmModeTruth:
    def test_skips_frames_without_polylines(self, tmp_path):
        path = tmp_path / "truth.xml"
        path.write_text(DM_MODE_XML)
        assert set(parse_dm_mode_truth(str(path))) == {5}

    def test_point_is_polyline_value_at_horizontal_center(self, tmp_path):
        path = tmp_path / "truth.xml"
        path.write_text(DM_MODE_XML)
        # No transform (see LABEL_TO_SIDE's comment -- tested empirically against
        # all 4 combinations of mirror/swap, identity won). Right spans x=0..20,
        # center x=10, which is an exact vertex -> y=20.
        point = parse_dm_mode_truth(str(path))[5]["R"]
        assert point == pytest.approx((10.0, 20.0))

    def test_maps_left_and_right_labels(self, tmp_path):
        path = tmp_path / "truth.xml"
        path.write_text(DM_MODE_XML)
        # Left spans x=50..60, center x=55, interpolated y=5.
        point = parse_dm_mode_truth(str(path))[5]["L"]
        assert point == pytest.approx((55.0, 5.0))
