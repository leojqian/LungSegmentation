# Tests for the CLI's file writers: the overlay DICOM and cine frame rate in
# formats/dicom_io.py, and the MP4 writer in formats/video.py.

import cv2
import numpy as np
import pydicom
import pytest
from pydicom.encaps import generate_frames
from pydicom.uid import JPEGBaseline8Bit, MultiFrameTrueColorSecondaryCaptureImageStorage

from fakes import write_dicom
from lungmap.formats.dicom_io import (DEFAULT_FPS, cine_fps, encode_overlay_frame,
                                      write_overlay_dicom)
from lungmap.formats.video import Mp4Writer


def color_frames(n=3, shape=(64, 80)):
    """Distinct flat-colour BGR frames, so a decode can be matched to its source."""
    return [np.full((*shape, 3), (40 * i, 100, 255 - 40 * i), np.uint8) for i in range(n)]


@pytest.fixture
def source(tmp_path):
    path = str(tmp_path / "source.dcm")
    write_dicom(path, np.zeros((3, 64, 80), np.uint16), pixel_spacing=(0.4, 0.4),
                FrameTime=66, PatientID="P001", StudyInstanceUID="1.2.3.4",
                FieldOfViewHorizontalFlip="YES")
    return path


@pytest.fixture
def overlay(tmp_path, source):
    frames = color_frames()
    out = tmp_path / "overlay.dcm"
    write_overlay_dicom(out, [encode_overlay_frame(f) for f in frames], frames[0].shape[:2], source)
    return frames, pydicom.dcmread(out)


class TestWriteOverlayDicom:
    def test_is_a_multiframe_true_color_secondary_capture(self, overlay):
        _, ds = overlay
        assert ds.SOPClassUID == MultiFrameTrueColorSecondaryCaptureImageStorage
        assert ds.file_meta.TransferSyntaxUID == JPEGBaseline8Bit
        assert ds.NumberOfFrames == 3
        assert (ds.Rows, ds.Columns) == (64, 80)
        assert ds.SamplesPerPixel == 3

    def test_every_frame_decodes_back_to_its_source(self, overlay):
        frames, ds = overlay
        decoded = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                   for b in generate_frames(ds.PixelData, number_of_frames=ds.NumberOfFrames)]
        for got, want in zip(decoded, frames):
            assert np.abs(got.astype(int) - want.astype(int)).mean() < 4

    def test_carries_over_patient_study_spacing_and_frame_time(self, overlay, source):
        _, ds = overlay
        assert ds.PatientID == "P001"
        assert ds.StudyInstanceUID == "1.2.3.4"
        assert [float(v) for v in ds.PixelSpacing] == [0.4, 0.4]
        assert float(ds.FrameTime) == 66

    def test_is_a_new_series_not_a_copy_of_the_source(self, overlay, source):
        _, ds = overlay
        src = pydicom.dcmread(source, stop_before_pixels=True)
        assert ds.SOPInstanceUID != src.file_meta.MediaStorageSOPInstanceUID
        assert ds.SeriesInstanceUID != src.get("SeriesInstanceUID")

    def test_drops_the_horizontal_flip_tag_since_pixels_are_already_flipped(self, overlay):
        _, ds = overlay
        assert (0x0018, 0x7034) not in ds

    def test_flags_burned_in_annotation_and_lossy_compression(self, overlay):
        _, ds = overlay
        assert ds.BurnedInAnnotation == "YES"
        assert ds.LossyImageCompression == "01"


class TestCineFps:
    def test_from_frame_time_in_ms(self, tmp_path):
        path = str(tmp_path / "a.dcm")
        write_dicom(path, np.zeros((1, 8, 8), np.uint16), FrameTime=50)
        assert cine_fps(path) == pytest.approx(20.0)

    def test_falls_back_to_cine_rate(self, tmp_path):
        path = str(tmp_path / "a.dcm")
        write_dicom(path, np.zeros((1, 8, 8), np.uint16), CineRate=12)
        assert cine_fps(path) == pytest.approx(12.0)

    def test_falls_back_to_default(self, tmp_path):
        path = str(tmp_path / "a.dcm")
        write_dicom(path, np.zeros((1, 8, 8), np.uint16))
        assert cine_fps(path) == DEFAULT_FPS


class TestMp4Writer:
    def test_writes_every_frame(self, tmp_path):
        path = tmp_path / "out.mp4"
        with Mp4Writer(path, fps=15, size=(80, 64)) as video:
            for frame in color_frames():
                video.write(frame)
        cap = cv2.VideoCapture(str(path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        cap.release()

    def test_odd_sizes_are_padded_to_even(self, tmp_path):
        path = tmp_path / "odd.mp4"
        with Mp4Writer(path, fps=15, size=(81, 63)) as video:
            for frame in color_frames(shape=(63, 81)):
                video.write(frame)
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        assert ok and frame.shape[:2] == (64, 82)

    def test_unwritable_path_raises(self, tmp_path):
        with pytest.raises(OSError):
            Mp4Writer(tmp_path / "missing_dir" / "out.mp4", fps=15, size=(80, 64))
