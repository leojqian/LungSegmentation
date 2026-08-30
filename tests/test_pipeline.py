# Tests for pipeline.py — the one entry point that turns any supported image
# (PNG/JPG or multi-frame DICOM) into segmented lung masks, diaphragm curves,
# and landmark points.

import cv2
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from segmentation.pipeline import MODEL_SIZE, analyze, analyze_frame, load_image, measure_masks, segment


def write_dicom(path, frames, photometric="MONOCHROME2", pixel_spacing=(0.4, 0.4),
                window_center=None, window_width=None):
    """frames: uint16 array (F, H, W). Minimal tags load_image actually reads."""
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
    ds.PixelData = frames.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, enforce_file_format=True)


def frame_with_marker(shape, marker_col, value=4000):
    """All-zero except a bright vertical stripe at marker_col — lets a test tell
    which frame it got without relying on a flat frame's degenerate normalization."""
    f = np.zeros(shape, dtype=np.uint16)
    f[:, marker_col:marker_col + 5] = value
    return f


class TestLoadImage:
    def test_reads_png_as_normalized_float(self, tmp_path):
        path = str(tmp_path / "img.png")
        cv2.imwrite(path, np.full((20, 30), 128, np.uint8))

        image, spacing = load_image(path)

        assert image.dtype == np.float32
        assert image.shape == (20, 30)
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert spacing is None

    def test_png_frame_argument_rejected(self, tmp_path):
        path = str(tmp_path / "img.png")
        cv2.imwrite(path, np.zeros((10, 10), np.uint8))
        with pytest.raises(ValueError):
            load_image(path, frame=0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_image(str(tmp_path / "missing.png"))

    def test_dicom_defaults_to_frame_zero(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        frames = np.stack([frame_with_marker((20, 30), 0), frame_with_marker((20, 30), 20)])
        write_dicom(path, frames)

        image, _ = load_image(path)

        assert image[:, :5].mean() > image[:, 20:25].mean()

    def test_dicom_selects_requested_frame(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        frames = np.stack([frame_with_marker((20, 30), 0), frame_with_marker((20, 30), 20)])
        write_dicom(path, frames)

        image, _ = load_image(path, frame=1)

        assert image[:, 20:25].mean() > image[:, :5].mean()

    def test_dicom_returns_pixel_spacing(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.zeros((1, 20, 30), dtype=np.uint16), pixel_spacing=(0.5, 0.5))

        _, spacing = load_image(path)

        assert spacing == pytest.approx(0.5)

    def test_dicom_respects_monochrome1_inversion(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.stack([frame_with_marker((20, 30), 0)]), photometric="MONOCHROME1")

        image, _ = load_image(path)

        # MONOCHROME1: the bright raw stripe (high value) should invert to LOW.
        assert image[:, :5].mean() < image[:, 20:25].mean()

    def test_window_flag_uses_dicom_window_tags(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        frames = np.full((1, 20, 30), 2048, dtype=np.uint16)
        write_dicom(path, frames, window_center=2048, window_width=4096)

        image, _ = load_image(path, window=True)

        # A flat frame at the window's exact center should normalize to 0.5,
        # not 0 (which per-frame min/max would give for a uniform frame).
        assert image.mean() == pytest.approx(0.5, abs=0.01)

    def test_window_flag_ignored_without_tags(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.stack([frame_with_marker((20, 30), 0)]))
        # No window tags on this file — should fall back to per-frame min/max
        # rather than crashing.
        image, _ = load_image(path, window=True)
        assert image.max() == pytest.approx(1.0)


class FakeModel:
    """Predicts two fixed, separated circular blobs at MODEL_SIZE resolution,
    regardless of input — decouples pipeline-wiring tests from real trained
    weights, following this codebase's synthetic-mask testing convention."""

    def predict(self, batch, verbose=0):
        canvas = np.zeros((MODEL_SIZE, MODEL_SIZE), np.float32)
        cv2.circle(canvas, (70, 140), 50, 1.0, -1)
        cv2.circle(canvas, (186, 140), 50, 1.0, -1)
        return canvas[None, ..., None]


class FakeModelWithIsland:
    """Like FakeModel's two lung blobs, plus a small disconnected third blob —
    reproduces the spurious false-positive "islands" seen in real DDR
    predictions (soft-tissue/skin-fold activations). Radius 15 is deliberately
    bigger than OPEN_KERNEL (21px across) so it survives the existing
    morphological open and actually exercises component filtering, unlike a
    smaller island the opening would already erase on its own."""

    def predict(self, batch, verbose=0):
        canvas = np.zeros((MODEL_SIZE, MODEL_SIZE), np.float32)
        cv2.circle(canvas, (70, 140), 50, 1.0, -1)
        cv2.circle(canvas, (186, 140), 50, 1.0, -1)
        cv2.circle(canvas, (128, 20), 15, 1.0, -1)
        return canvas[None, ..., None]


class TestSegment:
    def test_drops_small_islands_keeping_only_the_two_largest_components(self):
        image = np.full((400, 400), 0.5, np.float32)
        mask = segment(image, FakeModelWithIsland())
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        assert n_labels - 1 == 2

    def test_two_lung_mask_is_unaffected(self):
        image = np.full((400, 400), 0.5, np.float32)
        with_island = segment(image, FakeModelWithIsland())
        without_island = segment(image, FakeModel())
        # the island fake model's two main blobs are identical to FakeModel's;
        # dropping the island should leave that shared mask untouched.
        assert np.array_equal(with_island, without_island)


class TestAnalyze:
    def png_path(self, tmp_path, size=400):
        path = str(tmp_path / "img.png")
        cv2.imwrite(path, np.full((size, size), 128, np.uint8))
        return path

    def test_returns_both_sides_for_masks_lungs_and_curves(self, tmp_path):
        result = analyze(self.png_path(tmp_path), FakeModel())
        assert set(result.masks) == {"R", "L"}
        assert set(result.lungs) == {"R", "L"}
        assert set(result.curves) == {"R", "L"}

    def test_lung_points_have_all_four_fields(self, tmp_path):
        result = analyze(self.png_path(tmp_path), FakeModel())
        for side in ("R", "L"):
            assert result.lungs[side]._fields == ("top", "lower_left", "lower_right", "dome")

    def test_r_side_is_the_image_left_blob(self, tmp_path):
        result = analyze(self.png_path(tmp_path), FakeModel())
        assert result.lungs["R"].top[0] < result.lungs["L"].top[0]

    def test_pixel_spacing_none_for_png(self, tmp_path):
        result = analyze(self.png_path(tmp_path), FakeModel())
        assert result.pixel_spacing_mm is None

    def test_pixel_spacing_set_for_dicom(self, tmp_path):
        path = str(tmp_path / "case.dcm")
        write_dicom(path, np.full((1, 400, 400), 2000, dtype=np.uint16), pixel_spacing=(0.4, 0.4))
        result = analyze(path, FakeModel())
        assert result.pixel_spacing_mm == pytest.approx(0.4)

    def test_mask_post_processing_kernel_is_configurable(self, tmp_path):
        # Should not raise with a non-default kernel — exercises the parameter
        # Phase 1 calibration needs to sweep.
        result = analyze(self.png_path(tmp_path), FakeModel(), open_kernel=15)
        assert result.mask.any()


class TestAnalyzeFrame:
    """analyze() = load_image() + analyze_frame(); this is the building block
    for callers with frames already in memory (Step6's multi-frame DICOM loop)."""

    def test_matches_analyze_on_the_same_image(self, tmp_path):
        result = analyze(TestAnalyze().png_path(tmp_path), FakeModel())
        image = np.full((400, 400), 128, np.uint8).astype(np.float32) / 255.0
        direct = analyze_frame(image, FakeModel())
        assert direct.lungs["R"].top == result.lungs["R"].top

    def test_passes_through_given_pixel_spacing(self):
        image = np.full((400, 400), 0.5, np.float32)
        result = analyze_frame(image, FakeModel(), pixel_spacing_mm=0.4)
        assert result.pixel_spacing_mm == pytest.approx(0.4)


def lung_mask(shape=(400, 400), cx=100, cy=200):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (cx, cy), (70, 150), 0, 0, 360, 1, -1)
    return m


class TestMeasureMasks:
    """The no-model tail of analyze_frame — for callers with masks already in
    hand (e.g. Step5's manual-mask path, which never runs the U-Net)."""

    def test_returns_lungs_and_curves_keyed_by_side(self):
        masks = {"R": lung_mask(cx=100), "L": lung_mask(cx=300)}
        lungs, curves = measure_masks(masks)
        assert set(lungs) == {"R", "L"}
        assert set(curves) == {"R", "L"}

    def test_lung_points_match_a_direct_measure_lung_call(self):
        masks = {"R": lung_mask(cx=100)}
        lungs, _ = measure_masks(masks)
        from geometry.diaphragm import measure_lung
        from geometry.landmarks import contour_of
        expected = measure_lung(contour_of(masks["R"]))
        assert lungs["R"].top == expected.top
        assert lungs["R"].lower_left == expected.lower_left
        assert lungs["R"].lower_right == expected.lower_right
        assert tuple(lungs["R"].dome) == tuple(expected.dome)

    def test_accepts_an_explicit_max_step(self):
        masks = {"R": lung_mask(cx=100)}
        lungs, _ = measure_masks(masks, max_step=8)
        assert lungs["R"].top is not None

    def test_one_sides_failure_does_not_lose_the_other(self):
        masks = {"R": lung_mask(cx=100), "L": np.zeros((400, 400), np.uint8)}
        lungs, curves = measure_masks(masks)
        assert lungs["R"] is not None and curves["R"] is not None
        assert lungs["L"] is None and curves["L"] is None
