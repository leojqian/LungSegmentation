# Tests for landmarks.py — the corner-finding logic moved out of Step 4.
# It was untested there; these pin the behaviour before it gets shared.

import cv2
import numpy as np
import pytest

from geometry.landmarks import (average_point, bottom_corners, contour_of, diagonal_extreme,
                       landmarks_of, top_point)


def lung_like(h=400, w=300, top=40, bottom=360):
    """Rounded blob standing in for a lung: flat-ish top, curved bottom."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, (top + bottom) // 2), (w // 3, (bottom - top) // 2),
                0, 0, 360, 1, -1)
    return m


class TestAveragePoint:
    def test_averages_tied_coordinates(self):
        assert average_point(np.array([[10, 5], [20, 5]])) == (15, 5)

    def test_rounds_to_integers(self):
        # round() is banker's rounding, so 10.5 -> 10, not 11
        assert average_point(np.array([[10, 5], [11, 5]])) == (10, 5)


class TestTopPoint:
    def test_finds_the_topmost_point(self):
        pts = np.array([[10, 100], [50, 20], [90, 100]])
        assert top_point(pts) == (50, 20)

    def test_averages_a_flat_top(self):
        pts = np.array([[10, 20], [30, 20], [50, 20], [40, 90]])
        assert top_point(pts) == (30, 20)

    def test_tolerance_groups_near_ties(self):
        # y=20 and y=21 are within tol, so both are averaged: x=20, y=20.5 -> 20
        pts = np.array([[10, 20], [30, 21], [90, 100]])
        assert top_point(pts, tol=2) == (20, 20)


class TestDiagonalExtreme:
    def test_finds_the_down_right_corner(self):
        pts = np.array([[0, 0], [100, 100], [0, 100], [100, 0]])
        assert diagonal_extreme(pts, want_right=True) == (100, 100)

    def test_finds_the_down_left_corner(self):
        pts = np.array([[0, 0], [100, 100], [0, 100], [100, 0]])
        assert diagonal_extreme(pts, want_right=False) == (0, 100)


class TestBottomCorners:
    def test_returns_two_separated_corners(self):
        pts = contour_of(lung_like())
        left, right = bottom_corners(pts)
        assert left[0] < right[0]

    def test_corners_sit_near_the_bottom(self):
        pts = contour_of(lung_like())
        left, right = bottom_corners(pts)
        lowest = pts[:, 1].max()
        assert lowest - left[1] < 120 and lowest - right[1] < 120

    def test_collapsed_diagonals_fall_back_to_halves(self):
        # narrow deep spike: both diagonals converge on the same tip
        m = np.zeros((300, 200), np.uint8)
        cv2.fillPoly(m, [np.array([[80, 20], [120, 20], [105, 280], [95, 280]])], 1)
        left, right = bottom_corners(contour_of(m))
        assert left != right


class TestContourOf:
    def test_takes_the_largest_blob(self):
        m = lung_like()
        m[5:12, 5:12] = 1  # speck
        pts = contour_of(m)
        assert pts[:, 0].min() > 12

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError):
            contour_of(np.zeros((50, 50), np.uint8))


class TestLandmarksOf:
    def test_returns_all_three_named_points(self):
        assert set(landmarks_of(lung_like())) == {"top", "lower_left", "lower_right"}

    def test_top_is_above_both_corners(self):
        lm = landmarks_of(lung_like())
        assert lm["top"][1] < lm["lower_left"][1]
        assert lm["top"][1] < lm["lower_right"][1]
