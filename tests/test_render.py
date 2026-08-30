# Tests for the arithmetic behind the drawing. A wrong scale factor puts curves
# in the wrong place while everything still "runs", so the maths is tested even
# though the pixels are not.

import numpy as np
import pytest

from geometry.diaphragm import cliff_step_for
from rendering.render import dash_segments, display_scale, rule_extent, scale_points, tint
from Step5MapDiaphragm import select_sample, stem


class TestDisplayScale:
    def test_shrinks_long_side_to_target(self):
        assert display_scale((4020, 4892), 1024) == pytest.approx(1024 / 4892)

    def test_uses_height_when_portrait(self):
        assert display_scale((4892, 4020), 1024) == pytest.approx(1024 / 4892)

    def test_never_upscales_small_images(self):
        assert display_scale((200, 300), 1024) == 1.0


class TestScalePoints:
    def test_halves_coordinates(self):
        pts = np.array([[100, 200], [300, 400]])
        assert np.array_equal(scale_points(pts, 0.5), np.array([[50, 100], [150, 200]]))

    def test_identity_scale_is_lossless(self):
        pts = np.array([[1, 2], [3, 4]])
        assert np.array_equal(scale_points(pts, 1.0), pts)

    def test_returns_integers(self):
        out = scale_points(np.array([[10, 10]]), 0.333)
        assert np.issubdtype(out.dtype, np.integer)


class TestTint:
    """Translucent lung-region fill, so the curve can be checked against the
    mask edge it was derived from."""

    def canvas(self):
        return np.zeros((10, 10, 3), np.uint8)

    def mask(self):
        m = np.zeros((10, 10), bool)
        m[:5] = True  # top half
        return m

    def test_leaves_pixels_outside_the_mask_untouched(self):
        out = tint(self.canvas(), self.mask(), (255, 255, 255), alpha=0.5)
        assert (out[5:] == 0).all()

    def test_moves_masked_pixels_toward_the_colour(self):
        out = tint(self.canvas(), self.mask(), (200, 100, 50), alpha=0.5)
        assert np.allclose(out[0, 0], (100, 50, 25), atol=1)

    def test_alpha_zero_changes_nothing(self):
        out = tint(self.canvas(), self.mask(), (255, 255, 255), alpha=0.0)
        assert (out == 0).all()

    def test_alpha_one_replaces_with_the_colour(self):
        out = tint(self.canvas(), self.mask(), (10, 20, 30), alpha=1.0)
        assert (out[:5] == (10, 20, 30)).all()

    def test_does_not_mutate_the_input(self):
        c = self.canvas()
        tint(c, self.mask(), (255, 255, 255), alpha=1.0)
        assert (c == 0).all()

    def test_stays_uint8(self):
        out = tint(self.canvas(), self.mask(), (255, 255, 255), alpha=0.5)
        assert out.dtype == np.uint8


class TestStem:
    def test_strips_directory_and_extension(self):
        assert stem("MontgomerySet/CXR_png/MCUCXR_0001_0.png") == "MCUCXR_0001_0"


class TestCliffStepFor:
    """max_step is an absolute pixel threshold, so it has to scale with the
    effective resolution of whatever produced the mask — not the image size."""

    def test_scales_with_upscale_quantum(self):
        # 2935px tall from a 256px prediction -> ~11.5px per predicted pixel
        assert cliff_step_for(2935) == 34

    def test_full_resolution_needs_a_bigger_threshold(self):
        assert cliff_step_for(4892) > cliff_step_for(2935)

    def test_hand_drawn_resolution_returns_the_default(self):
        # a mask at its own native resolution has no upscale quantum
        assert cliff_step_for(2935, model_size=2935) == 3

    def test_never_returns_zero(self):
        assert cliff_step_for(100, model_size=4892) >= 1


class TestRuleExtent:
    """The height rule is a short segment either side of the apex, not a
    full-width rule — it marks one point, and should look like it."""

    def test_centres_on_the_apex(self):
        assert rule_extent(1000, x=500, frac=0.1) == (400, 600)

    def test_clips_at_the_left_edge(self):
        assert rule_extent(1000, x=30, frac=0.1) == (0, 130)

    def test_clips_at_the_right_edge(self):
        assert rule_extent(1000, x=970, frac=0.1) == (870, 1000)

    def test_span_scales_with_canvas_width(self):
        x0, x1 = rule_extent(2000, x=1000, frac=0.1)
        assert x1 - x0 == 400

    def test_always_returns_a_visible_segment(self):
        x0, x1 = rule_extent(100, x=0, frac=0.01)
        assert x1 > x0

    def test_bounds_clip_the_rule_to_the_curve(self):
        # apex at the curve's left end: the rule must not run off into space
        assert rule_extent(1000, x=400, frac=0.1, bounds=(400, 900)) == (400, 500)

    def test_bounds_clip_at_the_right_end_too(self):
        assert rule_extent(1000, x=900, frac=0.1, bounds=(400, 900)) == (800, 900)

    def test_bounds_wider_than_the_rule_change_nothing(self):
        assert rule_extent(1000, x=500, frac=0.1, bounds=(0, 1000)) == (400, 600)

    def test_bounds_never_collapse_the_segment(self):
        x0, x1 = rule_extent(1000, x=400, frac=0.1, bounds=(400, 400))
        assert x1 > x0


class TestDashSegments:
    """Dashes distinguish the horizontal height reference from the anatomical
    curve, which is solid."""

    def test_starts_at_zero(self):
        assert dash_segments(100, dash=10, gap=5)[0] == (0, 10)

    def test_leaves_gaps_between_dashes(self):
        segs = dash_segments(100, dash=10, gap=5)
        assert segs[1] == (15, 25)

    def test_never_exceeds_the_width(self):
        assert all(b <= 40 for _, b in dash_segments(40, dash=10, gap=5))

    def test_covers_most_of_the_width(self):
        segs = dash_segments(200, dash=10, gap=5)
        assert segs[-1][1] >= 190

    def test_width_shorter_than_one_dash_gives_one_clipped_segment(self):
        assert dash_segments(6, dash=10, gap=5) == [(0, 6)]


class TestSelectSample:
    PAIRS = [
        ("cxr/MCUCXR_0001_0.png", "l/MCUCXR_0001_0.png", "r/MCUCXR_0001_0.png"),
        ("cxr/MCUCXR_0035_0.png", "l/MCUCXR_0035_0.png", "r/MCUCXR_0035_0.png"),
    ]

    def test_finds_by_bare_name(self):
        assert select_sample(self.PAIRS, "MCUCXR_0035_0") == [self.PAIRS[1]]

    def test_finds_by_filename_with_extension(self):
        assert select_sample(self.PAIRS, "MCUCXR_0035_0.png") == [self.PAIRS[1]]

    def test_unknown_name_raises_with_the_name_in_the_message(self):
        with pytest.raises(SystemExit, match="MCUCXR_9999_9"):
            select_sample(self.PAIRS, "MCUCXR_9999_9")
