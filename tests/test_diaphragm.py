# Tests for diaphragm.py — synthetic masks with analytically known answers,
# so correctness doesn't depend on eyeballing X-rays.

import glob

import cv2
import numpy as np
import pytest

from geometry.diaphragm import (
    bottom_profile,
    trace_diaphragm,
    largest_component,
    apex_source,
    apex_of,
    dome_apex,
    apex_is_lower_bound,
    fit_dome,
    smooth,
    split_lungs,
    trim_cliffs,
    trim_rise,
    trim_terminal_upturn,
    trim_to_span,
)


def half_disc(h=400, w=600, cx=300, cy=150, r=120):
    """Filled upper half-disc: flat top, curved bottom. Stands in for a lung
    whose inferior border is a known circle."""
    ys, xs = np.mgrid[0:h, 0:w]
    return (((xs - cx) ** 2 + (ys - cy) ** 2 <= r**2) & (ys >= cy)).astype(np.uint8)


class TestLargestComponent:
    def test_keeps_only_the_bigger_blob(self):
        m = np.zeros((100, 100), np.uint8)
        m[10:20, 10:20] = 1  # small: 100 px
        m[40:80, 40:80] = 1  # large: 1600 px
        out = largest_component(m)
        assert out[40:80, 40:80].all()
        assert not out[10:20, 10:20].any()

    def test_single_blob_survives_unchanged(self):
        m = np.zeros((50, 50), np.uint8)
        m[10:30, 10:30] = 1
        assert np.array_equal(largest_component(m), m.astype(bool))

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError):
            largest_component(np.zeros((20, 20), np.uint8))


class TestBottomProfile:
    def test_finds_lowest_pixel_per_column(self):
        m = np.zeros((10, 4), np.uint8)
        m[2:5, 0] = 1  # lowest row 4
        m[2:8, 1] = 1  # lowest row 7
        m[3:4, 3] = 1  # lowest row 3
        xs, ys = bottom_profile(m)
        assert list(xs) == [0, 1, 3]  # column 2 empty, omitted
        assert list(ys) == [4, 7, 3]

    def test_ignores_empty_columns(self):
        m = np.zeros((10, 5), np.uint8)
        m[5, 2] = 1
        xs, ys = bottom_profile(m)
        assert list(xs) == [2] and list(ys) == [5]


class TestSmooth:
    def test_removes_single_pixel_spike(self):
        ys = np.array([10, 10, 10, 99, 10, 10, 10])
        assert 99 not in smooth(ys, 3)

    def test_preserves_length(self):
        ys = np.arange(50)
        assert len(smooth(ys, 5)) == 50

    def test_even_kernel_is_forced_odd(self):
        smooth(np.arange(20), 4)  # must not raise


class TestTrimCliffs:
    """Near the midline the lung field sits above the heart, so the per-column
    bottom jumps up to the cardiac border. That climb is not diaphragm."""

    def test_drops_the_vertical_cardiac_climb(self):
        xs = np.arange(120)
        ys = np.concatenate([np.full(100, 300), np.full(20, 50)])  # 250px cliff
        kx, ky = trim_cliffs(xs, ys, max_step=8)
        assert kx.max() == 99
        assert set(ky) == {300}

    def test_climb_on_the_other_side_is_also_dropped(self):
        xs = np.arange(120)
        ys = np.concatenate([np.full(20, 50), np.full(100, 300)])
        kx, _ = trim_cliffs(xs, ys, max_step=8)
        assert kx.min() == 20

    def test_gentle_slopes_are_untouched(self):
        xs = np.arange(100)
        ys = 300 - np.arange(100) * 2  # 2px per column, well under threshold
        kx, ky = trim_cliffs(xs, ys, max_step=8)
        assert len(kx) == 100

    def test_keeps_the_run_containing_the_deepest_point(self):
        xs = np.arange(60)
        # short shallow run, cliff, then the long run holding the deepest point
        ys = np.concatenate([np.full(10, 100), np.full(50, 400)])
        kx, ky = trim_cliffs(xs, ys, max_step=8)
        assert kx.min() == 10 and len(kx) == 50


class TestTrimRise:
    """On the left the lung's lower border joins the cardiac border as a
    gradual diagonal, not a cliff — same slope range as the dome, so only the
    total climb above the costophrenic angle distinguishes them."""

    def test_cuts_where_climb_exceeds_budget(self):
        xs = np.arange(100)
        ys = 500 - np.arange(100) * 10  # deepest (y=500) at x=0, climbs from there
        kx, ky = trim_rise(xs, ys, max_rise=100)
        assert ky.min() >= 400  # never climbs more than 100 above y=500
        assert kx.min() == 0 and kx.max() == 10

    def test_shallow_dome_is_untouched(self):
        xs = np.arange(101)
        ys = 500 - (50 - np.abs(np.arange(101) - 50))  # 50px dome
        assert len(trim_rise(xs, ys, max_rise=200)[0]) == 101

    def test_trims_both_directions_from_deepest_point(self):
        xs = np.arange(201)
        ys = 500 - np.abs(np.arange(201) - 100) * 0  # flat
        ys = 500 - np.minimum(np.abs(np.arange(201) - 100) * 5, 400)
        kx, ky = trim_rise(xs, ys, max_rise=100)
        assert kx.min() > 0 and kx.max() < 200

    def test_deepest_point_always_survives(self):
        xs = np.arange(50)
        ys = 500 - np.arange(50) * 100  # climbs violently from x=0
        kx, _ = trim_rise(xs, ys, max_rise=10)
        assert 0 in kx  # the anchor is never trimmed away


class TestExtractDiaphragm:
    def test_traces_known_circle_within_one_pixel(self):
        cx, cy, r = 300, 150, 120
        curve = trace_diaphragm(half_disc(cx=cx, cy=cy, r=r), kernel=1)
        # analytic bottom of the half-disc at each x
        x, y = curve[:, 0], curve[:, 1]
        expected = cy + np.sqrt(np.maximum(r**2 - (x - cx) ** 2, 0))
        assert np.abs(y - expected).max() <= 1.0

    def test_apex_is_at_the_circle_centre_column(self):
        curve = trace_diaphragm(half_disc(cx=300, cy=150, r=120), kernel=1)
        deepest = curve[np.argmax(curve[:, 1])]
        assert abs(int(deepest[0]) - 300) <= 1

    def test_x_is_strictly_increasing(self):
        curve = trace_diaphragm(half_disc())
        assert (np.diff(curve[:, 0]) > 0).all()

    def test_returns_n_by_2_int_array(self):
        curve = trace_diaphragm(half_disc())
        assert curve.ndim == 2 and curve.shape[1] == 2
        assert np.issubdtype(curve.dtype, np.integer)

    def test_ignores_stray_second_blob(self):
        m = half_disc()
        m[380:395, 10:25] = 1  # stray blob far below-left
        curve = trace_diaphragm(m, kernel=1)
        assert curve[:, 0].min() >= 180  # disc starts at cx-r = 180, not x=10

    def test_grayscale_255_mask_is_binarised(self):
        m = half_disc() * 255
        assert len(trace_diaphragm(m)) > 0

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError):
            trace_diaphragm(np.zeros((100, 100), np.uint8))


class TestSplitLungs:
    """Step 1 trains on left+right merged into one mask, so a U-Net prediction
    holds both lungs in a single image and must be split before tracing."""

    def two_lungs(self):
        m = np.zeros((200, 200), np.uint8)
        m[40:160, 20:80] = 1  # image-left blob
        m[40:160, 120:180] = 1  # image-right blob
        return m

    def test_returns_the_two_blobs_in_image_order(self):
        a, b = split_lungs(self.two_lungs())
        assert a[:, 20:80].any() and not a[:, 120:180].any()
        assert b[:, 120:180].any() and not b[:, 20:80].any()

    def test_discards_specks_beyond_the_two_largest(self):
        m = self.two_lungs()
        m[5:10, 5:10] = 1  # speck
        a, b = split_lungs(m)
        assert not (a[5:10, 5:10].any() or b[5:10, 5:10].any())

    def test_single_lung_raises(self):
        m = np.zeros((100, 100), np.uint8)
        m[10:90, 10:40] = 1
        with pytest.raises(ValueError):
            split_lungs(m)

    def test_accepts_0_255_masks(self):
        a, b = split_lungs(self.two_lungs() * 255)
        assert a.any() and b.any()

    def test_split_halves_trace_independently(self):
        for half in split_lungs(self.two_lungs()):
            assert len(trace_diaphragm(half)) > 0


class TestDomeApex:
    def test_finds_the_highest_point_of_the_curve(self):
        curve = np.array([[0, 500], [1, 400], [2, 300], [3, 420], [4, 510]])
        assert tuple(dome_apex(curve)) == (2, 300)

    def test_only_the_plateau_containing_the_peak_is_averaged(self):
        # two separate runs at the same height: averaging both would place the
        # apex in the dip between them, where the curve is 100px lower
        curve = np.array([[0, 300], [1, 300],
                          [2, 400], [3, 400], [4, 400],
                          [5, 300], [6, 300]])
        apex = dome_apex(curve)
        assert apex[0] in (0, 1)          # the first run, not the midpoint
        j = int(np.argmin(np.abs(curve[:, 0] - apex[0])))
        assert curve[j, 1] == apex[1]     # and it lands on the curve

    def test_tied_columns_average_to_the_plateau_centre(self):
        curve = np.array([[0, 300], [1, 300], [2, 300], [3, 400]])
        assert tuple(dome_apex(curve)) == (1, 300)

    def test_flat_plateau_does_not_bias_to_an_edge(self):
        xs = np.arange(100, 201)
        ys = np.full(101, 250)
        ys[0] = ys[-1] = 400  # flat dome with steep ends
        assert dome_apex(np.stack([xs, ys], axis=1))[0] == 150

    def test_apex_of_a_dome_shaped_lung(self):
        # A real diaphragm domes UPWARD: highest in the middle, dropping to the
        # costophrenic angles at both ends. (half_disc bulges the other way, so
        # it is the wrong fixture for an apex test.)
        h, w, cx, span, depth, base = 400, 600, 300, 200, 40, 300
        ys, xs = np.mgrid[0:h, 0:w]
        floor = base - depth * (1 - ((xs - cx) / span) ** 2)
        m = ((ys <= floor) & (np.abs(xs - cx) <= span)).astype(np.uint8)

        apex = dome_apex(trace_diaphragm(m, kernel=1))
        assert abs(apex[0] - cx) <= 3  # apex sits at the dome's centre column
        assert abs(apex[1] - (base - depth)) <= 3  # and at its peak height


class TestTrimTerminalUpturn:
    """Past the costophrenic angle the lung border turns back up the lateral
    chest wall. That hook is not diaphragm, and being the highest point of the
    curve it captures the apex if left in."""

    def test_trims_a_short_hook_at_the_right_end(self):
        xs = np.arange(200)
        ys = np.concatenate([np.arange(180) + 100,      # diaphragm descending
                             275 - np.arange(20) * 4])  # hook back up the wall
        kx, _ = trim_terminal_upturn(xs, ys)
        assert kx.max() == 179

    def test_trims_a_short_hook_at_the_left_end(self):
        xs = np.arange(200)
        # the wall descends INTO the costophrenic angle, so y rises with x
        ys = np.concatenate([120 + np.arange(20) * 4,
                             200 - np.arange(180)])
        kx, _ = trim_terminal_upturn(xs, ys)
        assert kx.min() == 20

    def test_leaves_a_long_dome_flank_alone(self):
        # a real dome flank rises for most of the curve — not a hook
        xs = np.arange(200)
        ys = 400 - np.arange(200)
        kx, _ = trim_terminal_upturn(xs, ys)
        assert len(kx) == 200

    def test_symmetric_dome_is_untouched(self):
        xs = np.arange(201)
        ys = 300 + np.abs(np.arange(201) - 100)  # apex in the middle
        kx, _ = trim_terminal_upturn(xs, ys)
        assert len(kx) == 201

    def test_trims_hooks_at_both_ends(self):
        xs = np.arange(220)
        ys = np.concatenate([120 + np.arange(20) * 4,   # left wall descending in
                             200 - np.arange(90),       # up to the dome
                             111 + np.arange(90),       # down to the right angle
                             196 - np.arange(20) * 4])  # right wall going back up
        kx, _ = trim_terminal_upturn(xs, ys)
        assert kx.min() == 20 and kx.max() == 199

    def test_never_returns_an_empty_curve(self):
        xs = np.arange(50)
        ys = 500 - np.arange(50) * 3
        kx, _ = trim_terminal_upturn(xs, ys)
        assert len(kx) > 0


class TestTrimToSpan:
    """The diaphragm runs between the cardiophrenic and costophrenic angles.
    Those corners are found independently by landmarks.bottomCorners, so they
    bound the curve without inventing another slope threshold."""

    def curve(self, x0=0, x1=200):
        xs = np.arange(x0, x1)
        return xs, 500 + xs // 2

    def test_clips_to_the_span(self):
        xs, ys = self.curve()
        kx, _ = trim_to_span(xs, ys, 50, 150)
        assert kx.min() == 50 and kx.max() == 150

    def test_span_covering_everything_changes_nothing(self):
        xs, ys = self.curve()
        kx, _ = trim_to_span(xs, ys, -100, 1000)
        assert len(kx) == len(xs)

    def test_accepts_reversed_bounds(self):
        xs, ys = self.curve()
        kx, _ = trim_to_span(xs, ys, 150, 50)
        assert kx.min() == 50 and kx.max() == 150

    def test_keeps_the_original_when_too_little_would_survive(self):
        xs, ys = self.curve()
        kx, _ = trim_to_span(xs, ys, 100, 102, min_points=10)
        assert len(kx) == len(xs)  # refuses to gut the curve

    def test_y_values_stay_paired_with_their_x(self):
        xs, ys = self.curve()
        kx, ky = trim_to_span(xs, ys, 50, 150)
        assert (ky == 500 + kx // 2).all()


class TestFitDome:
    """The medial half of a hemidiaphragm hides behind the heart, so it has no
    lung above it to trace. Fitting a dome lets that part be inferred."""

    def parabola(self, a=0.004, vertex=(400, 900), x0=200, x1=600):
        xs = np.arange(x0, x1)
        ys = a * (xs - vertex[0]) ** 2 + vertex[1]
        return np.stack([xs, ys.astype(int)], axis=1)

    def test_recovers_the_curve_it_was_fitted_to(self):
        curve = self.parabola()  # xs 200..599
        fitted = fit_dome(curve, 200, 599)
        assert np.array_equal(fitted[:, 0], curve[:, 0])
        assert np.abs(fitted[:, 1] - curve[:, 1]).max() <= 2

    def test_extrapolates_beyond_the_traced_range(self):
        # traced only on the left flank; dome must continue to the right
        fitted = fit_dome(self.parabola(x0=200, x1=400), 200, 600)
        assert fitted[:, 0].max() == 600

    def test_extrapolated_apex_lands_on_the_true_vertex(self):
        fitted = fit_dome(self.parabola(x0=200, x1=400), 200, 600)
        assert abs(dome_apex(fitted)[0] - 400) <= 10

    def test_straight_line_is_not_a_dome(self):
        xs = np.arange(200, 600)
        line = np.stack([xs, (1000 - xs).astype(int)], axis=1)
        with pytest.raises(ValueError):
            fit_dome(line, 200, 600)

    def test_upside_down_arc_is_not_a_dome(self):
        curve = self.parabola(a=-0.004)  # opens the wrong way
        with pytest.raises(ValueError):
            fit_dome(curve, 200, 600)

    def test_too_few_points_to_fit(self):
        with pytest.raises(ValueError):
            fit_dome(np.array([[1, 2], [3, 4]]), 0, 10)

    def test_returns_integer_points(self):
        fitted = fit_dome(self.parabola(), 200, 600)
        assert np.issubdtype(fitted.dtype, np.integer)

    def test_extension_is_capped_to_max_extend(self):
        # traced 200..399 with its deepest point at x=200. Asked to reach
        # 0..900, capped to 50px either side -> 150..449. Extending LEFT is then
        # refused by depth clipping (it would sink past the deepest measured
        # point), so only the right-hand cap is visible in the result.
        fitted = fit_dome(self.parabola(x0=200, x1=400), 0, 900, max_extend=50)
        assert fitted[:, 0].max() == 449  # capped, not the requested 900
        assert fitted[:, 0].min() == 200  # depth-clipped, not the capped 150

    def test_cap_never_widens_a_narrow_request(self):
        fitted = fit_dome(self.parabola(x0=200, x1=400), 250, 300, max_extend=500)
        assert fitted[:, 0].min() == 250 and fitted[:, 0].max() == 300

    def test_uncapped_still_bounded_by_depth(self):
        # no max_extend, so the only limit is the depth clip: the fit may run
        # out to where it reaches the deepest measured y (1060 at x=200), which
        # for this symmetric parabola is x=600 on the far side.
        fitted = fit_dome(self.parabola(x0=200, x1=400), 100, 700)
        assert fitted[:, 0].min() == 200 and fitted[:, 0].max() == 600

    def test_never_descends_below_the_deepest_measured_point(self):
        # traced on one flank only; extending far would plunge past the
        # costophrenic angle into the abdomen
        traced = self.parabola(x0=200, x1=400)
        fitted = fit_dome(traced, 0, 900)
        assert fitted[:, 1].max() <= traced[:, 1].max()

    def test_clipping_keeps_the_apex(self):
        traced = self.parabola(x0=200, x1=400)
        fitted = fit_dome(traced, 0, 900)
        assert abs(dome_apex(fitted)[0] - 400) <= 10

    def test_clipping_returns_a_contiguous_run(self):
        fitted = fit_dome(self.parabola(x0=200, x1=400), 0, 900)
        assert (np.diff(fitted[:, 0]) == 1).all()


class TestBestApex:
    """A parabola fitted to one flank puts its vertex wherever the curvature
    implies, which can be hundreds of columns past the last measurement. Such an
    apex floats off the drawn curve and is unverifiable, so it is not used."""

    def traced(self, x0=100, x1=300):
        xs = np.arange(x0, x1)
        ys = 500 - (xs - x0) // 2  # rising steadily to the right
        return np.stack([xs, ys], axis=1)

    def test_uses_the_fit_when_its_vertex_is_inside_the_traced_range(self):
        t = self.traced()
        inferred = np.stack([np.arange(100, 300), 400 + (np.arange(200) - 100) ** 2 // 50],
                            axis=1)
        assert apex_of(t, inferred)[0] == dome_apex(inferred)[0]

    def test_falls_back_when_the_vertex_is_far_outside(self):
        t = self.traced()                      # traced 100..299
        far = np.stack([np.arange(100, 700),   # vertex way out at x=600
                        300 + (np.arange(600) - 500) ** 2 // 100], axis=1)
        assert np.array_equal(apex_of(t, far), dome_apex(t))

    def test_small_overshoot_is_tolerated(self):
        t = self.traced()  # span 200 -> 15% margin = 30 columns
        near = np.stack([np.arange(100, 340), 400 + (np.arange(240) - 215) ** 2 // 50],
                        axis=1)
        assert apex_of(t, near)[0] == dome_apex(near)[0]

    def test_no_fit_uses_the_traced_curve(self):
        t = self.traced()
        assert np.array_equal(apex_of(t, None), dome_apex(t))

    def test_result_is_always_on_one_of_the_two_curves(self):
        t = self.traced()
        far = np.stack([np.arange(100, 700),
                        300 + (np.arange(600) - 500) ** 2 // 100], axis=1)
        a = apex_of(t, far)
        assert a[0] in set(t[:, 0]) or a[0] in set(far[:, 0])


class TestApexSource:
    """Whatever curve the apex was taken from is the curve the height rule must
    be drawn against — otherwise the marker can land outside its own rule."""

    def traced(self, x0=100, x1=300):
        xs = np.arange(x0, x1)
        return np.stack([xs, 500 - (xs - x0) // 2], axis=1)

    def test_returns_the_inferred_curve_when_its_vertex_is_accepted(self):
        t = self.traced()
        i = np.stack([np.arange(100, 300), 400 + (np.arange(200) - 100) ** 2 // 50], axis=1)
        assert apex_source(t, i) is i

    def test_returns_the_traced_curve_when_the_vertex_is_rejected(self):
        t = self.traced()
        far = np.stack([np.arange(100, 700),
                        300 + (np.arange(600) - 500) ** 2 // 100], axis=1)
        assert apex_source(t, far) is t

    def test_returns_the_traced_curve_when_there_is_no_fit(self):
        t = self.traced()
        assert apex_source(t, None) is t

    def test_apex_of_agrees_with_the_curve_it_names(self):
        t = self.traced()
        i = np.stack([np.arange(100, 340), 400 + (np.arange(240) - 215) ** 2 // 50], axis=1)
        assert np.array_equal(apex_of(t, i), dome_apex(apex_source(t, i)))

    def test_the_apex_always_lies_within_its_own_curve(self):
        t = self.traced()
        for inferred in (None,
                         np.stack([np.arange(100, 700),
                                   300 + (np.arange(600) - 500) ** 2 // 100], axis=1)):
            c = apex_source(t, inferred)
            a = apex_of(t, inferred)
            assert c[:, 0].min() <= a[0] <= c[:, 0].max()


class TestApexIsLowerBound:
    """When the dome hides behind the mediastinum the traced curve is a bare
    flank with no peak, so its highest point is an endpoint — a floor on dome
    height, not a measurement of it. Those must not be reported as equals."""

    def test_apex_at_the_medial_end_is_a_lower_bound(self):
        xs = np.arange(100, 300)
        ys = 500 + np.arange(200)  # descends the whole way; peak at x=100
        t = np.stack([xs, ys], axis=1)
        assert apex_is_lower_bound(t, dome_apex(t))

    def test_apex_at_the_lateral_end_is_a_lower_bound(self):
        xs = np.arange(100, 300)
        ys = 700 - np.arange(200)  # peak at the far end
        t = np.stack([xs, ys], axis=1)
        assert apex_is_lower_bound(t, dome_apex(t))

    def test_interior_peak_is_a_real_measurement(self):
        xs = np.arange(100, 301)
        ys = 500 + np.abs(np.arange(201) - 100)  # peak in the middle
        t = np.stack([xs, ys], axis=1)
        assert not apex_is_lower_bound(t, dome_apex(t))

    def test_just_inside_the_tolerance_still_counts_as_an_end(self):
        xs = np.arange(0, 200)          # span 199, 5% = ~10 columns
        ys = np.concatenate([[600], 500 + np.arange(199)])
        t = np.stack([xs, ys], axis=1)
        assert apex_is_lower_bound(t, np.array([5, 505]))

    def test_apex_well_inside_is_not_flagged(self):
        xs = np.arange(0, 200)
        t = np.stack([xs, 500 + np.abs(xs - 100)], axis=1)
        assert not apex_is_lower_bound(t, np.array([100, 500]))


class TestRealMasks:
    """Regression + smoke tests against the actual dataset."""

    @pytest.mark.parametrize(
        "path",
        [
            "MontgomerySet/ManualMask/leftMask/MCUCXR_0399_1.png",
            "MontgomerySet/ManualMask/rightMask/MCUCXR_0052_0.png",
        ],
    )
    def test_multi_component_masks_use_only_largest(self, path):
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            pytest.skip(f"{path} not present")
        binary = raw > 127
        kept = largest_component(binary)
        discarded = binary & ~kept
        assert discarded.any(), f"{path} was expected to have a stray blob"

        curve = trace_diaphragm(raw)
        kept_cols = np.where(kept.any(axis=0))[0]
        stray_cols = set(np.where(discarded.any(axis=0))[0])

        # curve stays inside the kept blob and never visits the stray one
        assert curve[:, 0].min() >= kept_cols.min()
        assert curve[:, 0].max() <= kept_cols.max()
        assert not stray_cols & set(curve[:, 0])

    def test_all_masks_produce_sane_curves(self):
        paths = sorted(glob.glob("MontgomerySet/ManualMask/*Mask/*.png"))
        if not paths:
            pytest.skip("dataset not present")
        assert len(paths) == 276
        for p in paths:
            curve = trace_diaphragm(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
            assert len(curve) > 0, p
            assert (np.diff(curve[:, 0]) > 0).all(), p
