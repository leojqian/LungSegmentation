# Tests for measure_lung — the single call that turns one lung contour into
# the four points we derive from it: top, lower_left, lower_right, dome.

import cv2
import numpy as np
import pytest

from geometry.diaphragm import apex_of, diaphragm_of, measure_lung


def lung_mask(h=600, w=500, cx=250, cy=280, rx=150, ry=220):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m


def contour_of_mask(m):
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def sample():
    return measure_lung(contour_of_mask(lung_mask()))


class TestReturnedFields:
    def test_returns_exactly_the_four_points(self):
        assert sample()._fields == ("top", "lower_left", "lower_right", "dome")

    def test_every_field_is_an_xy_point(self):
        r = sample()
        for point in r:
            assert len(point) == 2


class TestGeometry:
    def test_top_is_above_both_bottom_corners(self):
        r = sample()
        assert r.top[1] < r.lower_left[1]
        assert r.top[1] < r.lower_right[1]

    def test_lower_left_is_left_of_lower_right(self):
        r = sample()
        assert r.lower_left[0] < r.lower_right[0]

    def test_dome_sits_between_the_bottom_corners(self):
        r = sample()
        assert r.lower_left[0] <= r.dome[0] <= r.lower_right[0]

    def test_dome_matches_the_diaphragm_pipeline_computed_independently(self):
        # measure_lung's dome should agree with calling diaphragm_of + apex_of
        # directly on the same contour, rasterised the same way
        contour = contour_of_mask(lung_mask())
        pts = contour.reshape(-1, 2)
        mask = np.zeros((pts[:, 1].max() + 1, pts[:, 0].max() + 1), np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
        expected = apex_of(*diaphragm_of(mask))

        assert measure_lung(contour).dome.tolist() == expected.tolist()


class TestInputHandling:
    def test_accepts_opencv_shaped_contours(self):
        raw = contour_of_mask(lung_mask())            # (N, 1, 2)
        flat = raw.reshape(-1, 2)                     # (N, 2)
        assert measure_lung(raw).dome.tolist() == measure_lung(flat).dome.tolist()

    def test_points_stay_in_the_original_image_coordinates(self):
        # the canvas is contour-sized, but nothing is re-based to it
        c = contour_of_mask(lung_mask())
        pts = c.reshape(-1, 2)
        r = measure_lung(c)
        assert pts[:, 0].min() <= r.dome[0] <= pts[:, 0].max()
        assert pts[:, 1].min() <= r.top[1] <= pts[:, 1].max()

    def test_degenerate_contour_raises(self):
        with pytest.raises(ValueError):
            measure_lung(np.array([[10, 10], [11, 10], [11, 11]]))

    def test_negative_coordinates_raise(self):
        with pytest.raises(ValueError, match="negative"):
            measure_lung(np.array([[-5, 10], [20, 10], [20, 40], [-5, 40]]))
