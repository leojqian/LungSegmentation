# Tests for points.py — reducing one frame's measured geometry to the 8 output
# points (apex, diaphragm, two bottom corners, per side), their CSV form, and
# the naive bottom-corner-midpoint baseline the CLI compares against.

import numpy as np
import pytest

from lungmap.geometry.diaphragm import Diaphragm, LungPoints, center_of
from lungmap.points import (CSV_COLUMNS, SidePoints, compare, csv_row, naive_midpoint,
                            side_points)


def lung(top=(50, 10), lower_left=(20, 200), lower_right=(90, 210), dome=(55, 180)):
    return LungPoints(top=top, lower_left=lower_left, lower_right=lower_right, dome=dome)


def curve():
    xs = np.arange(20, 91)
    measured = np.stack([xs, 180 + ((xs - 55) ** 2) // 60], axis=1)
    return Diaphragm(measured, None)


class TestSidePoints:
    def test_maps_apex_and_corners_from_lung_points(self):
        sp = side_points(lung(), curve())
        assert sp.apex == (50, 10)
        assert sp.lower_left == (20, 200)
        assert sp.lower_right == (90, 210)

    def test_diaphragm_is_the_curves_center_point(self):
        c = curve()
        sp = side_points(lung(), c)
        assert sp.diaphragm == pytest.approx(tuple(center_of(*c)))

    def test_missing_curve_blanks_only_the_diaphragm(self):
        sp = side_points(lung(), None)
        assert sp.diaphragm is None
        assert sp.apex == (50, 10)

    def test_missing_lung_blanks_apex_and_corners(self):
        sp = side_points(None, curve())
        assert sp.apex is None and sp.lower_left is None and sp.lower_right is None
        assert sp.diaphragm is not None

    def test_fields_are_plain_floats_not_numpy(self):
        sp = side_points(lung(), curve())
        for point in sp:
            assert all(type(v) is float for v in point)


class TestNaiveMidpoint:
    def test_is_the_midpoint_of_the_two_bottom_corners(self):
        sp = SidePoints(apex=None, diaphragm=None, lower_left=(20, 200), lower_right=(90, 210))
        assert naive_midpoint(sp) == (55.0, 205.0)

    def test_none_when_a_corner_is_missing(self):
        sp = SidePoints(apex=None, diaphragm=None, lower_left=None, lower_right=(90, 210))
        assert naive_midpoint(sp) is None

    def test_none_for_a_missing_side(self):
        assert naive_midpoint(None) is None


class TestCsv:
    def test_frame_plus_eight_xy_points(self):
        assert len(CSV_COLUMNS) == 1 + 8 * 2
        assert CSV_COLUMNS[0] == "frame"

    def test_column_order_is_side_then_point_then_axis(self):
        assert CSV_COLUMNS[1:9] == ["R_apex_x", "R_apex_y", "R_diaphragm_x", "R_diaphragm_y",
                                    "R_lower_left_x", "R_lower_left_y",
                                    "R_lower_right_x", "R_lower_right_y"]
        assert CSV_COLUMNS[9] == "L_apex_x"

    def test_row_matches_columns_and_rounds_to_one_decimal(self):
        sp = SidePoints(apex=(1, 2), diaphragm=(3.14159, 4.0), lower_left=(5, 6), lower_right=(7, 8))
        row = csv_row(7, {"R": sp, "L": sp})
        assert len(row) == len(CSV_COLUMNS)
        assert row[0] == 7
        assert row[1:9] == ["1.0", "2.0", "3.1", "4.0", "5.0", "6.0", "7.0", "8.0"]

    def test_missing_side_or_field_is_blank(self):
        sp = SidePoints(apex=(1, 2), diaphragm=None, lower_left=(5, 6), lower_right=(7, 8))
        row = csv_row(0, {"R": sp, "L": None})
        assert row[3:5] == ["", ""]
        assert row[9:] == [""] * 8


class TestCompare:
    def test_mean_and_median_distance_per_side(self):
        a = SidePoints(apex=None, diaphragm=(55, 205), lower_left=(20, 200), lower_right=(90, 210))
        b = SidePoints(apex=None, diaphragm=(55, 195), lower_left=(20, 200), lower_right=(90, 210))
        c = SidePoints(apex=None, diaphragm=(85, 205), lower_left=(20, 200), lower_right=(90, 210))
        stats = compare([{"R": a, "L": None}, {"R": b, "L": None}, {"R": c, "L": None}])
        assert stats["R"]["n"] == 3
        assert stats["R"]["mean_px"] == pytest.approx((0 + 10 + 30) / 3)
        assert stats["R"]["median_px"] == pytest.approx(10)

    def test_side_with_no_comparable_frames_reports_zero_and_none(self):
        stats = compare([{"R": None, "L": None}])
        assert stats["L"] == {"n": 0, "mean_px": None, "median_px": None}
