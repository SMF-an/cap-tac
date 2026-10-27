"""Tests for the tactile key: one reading, reduced to bands.

The cases are the ones this repo has actually measured, so a change to a cut
point that would have called a measured situation wrong fails here:
an empty close reads under 0.01 px, a real grasp 1.5 px or more; healthy
pads report coherence 0.42-0.68 while a pad whose marker field was dragged
away by the jaw reports 0.09-0.10; the far plane is 34 mm, the undeformed
gel face 28.3, the camera floor 24.
"""

from __future__ import annotations

import pytest

from capx.tactile_memory.signature import (
    BANDS,
    DIMENSIONS,
    describe,
    key_from_readings,
    matches,
)


def reading(**overrides) -> dict:
    """A resting far-plane reading, with the fields a case cares about set."""
    base = {
        "left_contact": False, "right_contact": False,
        "left_depth_mm": 34.0, "right_depth_mm": 34.0,
        "left_marker_displacement_px": 0.0, "right_marker_displacement_px": 0.0,
        "left_marker_coherence": 0.0, "right_marker_coherence": 0.0,
        "left_marker_shear_dx": 0.0, "left_marker_shear_dy": 0.0,
        "right_marker_shear_dx": 0.0, "right_marker_shear_dy": 0.0,
    }
    base.update(overrides)
    return base


def test_every_dimension_is_always_present():
    key = key_from_readings({})
    assert tuple(key) == DIMENSIONS
    for dimension, band in key.items():
        assert band in BANDS[dimension], (dimension, band)


def test_an_open_gripper_keys_to_nothing_touching_anything():
    key = key_from_readings(reading())
    assert key["contact"] == "none"
    assert key["depth"] == "far"
    assert key["disp"] == "missing"
    assert key["coh"] == "missing"
    assert key["shear"] == "none"
    assert key["aniso"] == "unknown"
    assert key["relief"] == "unknown"


def test_a_seated_grasp():
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=27.9, right_depth_mm=28.0,
        left_marker_displacement_px=2.4, right_marker_displacement_px=2.2,
        left_marker_coherence=0.52, right_marker_coherence=0.61,
        left_marker_shear_dx=-0.35, left_marker_shear_dy=0.04,
        right_marker_shear_dx=0.33, right_marker_shear_dy=-0.03,
    ))
    assert key["contact"] == "both"
    assert key["depth"] == "face"       # 27.9 is below the face band's top
    assert key["asym"] == "even"        # the two pads see the same object
    assert key["disp"] == "moving"      # 2.4 px is a real grasp, not an empty close
    assert key["coh"] == "mid"
    assert key["shear"] == "small"
    assert key["shear_axis"] == "x"     # the mirrored pair both moved along x


def test_an_empty_close_is_still_not_a_grasp():
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=28.8, right_depth_mm=28.9,
        left_marker_displacement_px=0.004, right_marker_displacement_px=0.003,
        left_marker_coherence=0.02, right_marker_coherence=0.01,
    ))
    assert key["depth"] == "face"       # the depth alone cannot tell
    assert key["disp"] == "still"       # ...and the displacement can
    assert key["coh"] == "lost"


def test_one_dead_pad_shows_up_as_the_gap_not_as_its_own_reading():
    # The pad whose marker field was dragged with the jaw: 22.8 px of
    # displacement and coherence 0.09 against a healthy 1.6 px / 0.62.
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=28.0, right_depth_mm=28.1,
        left_marker_displacement_px=22.8, right_marker_displacement_px=1.6,
        left_marker_coherence=0.09, right_marker_coherence=0.62,
    ))
    assert key["disp"] == "heavy"       # the dragged pad dominates the max
    assert key["disp_gap"] == "split"   # ...and the disagreement is what says so
    assert key["coh"] == "mid"          # the healthy pad is the coherent one
    assert key["coh_gap"] == "split"


def test_a_zero_reading_is_missing_not_a_measurement():
    # A pad that did not report: 0.0 is out of range for every field but the
    # shear vector, so it must not band as "the firmest grip there is".
    key = key_from_readings(reading(
        left_contact=True, right_contact=False,
        left_depth_mm=0.0, right_depth_mm=27.0,
        left_marker_displacement_px=0.0, right_marker_displacement_px=2.0,
        left_marker_coherence=0.0, right_marker_coherence=0.5,
    ))
    assert key["contact"] == "left"
    assert key["depth"] == "pressed"    # from the pad that did report
    assert key["asym"] == "unknown"     # one pad silent is not "even"
    assert key["disp"] == "moving"
    assert key["disp_gap"] == "unknown"
    assert key["coh"] == "mid"


def test_the_two_pads_disagreeing_on_depth_is_one_sided_loading():
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=26.0, right_depth_mm=28.4,
        left_marker_displacement_px=3.0, right_marker_displacement_px=2.8,
        left_marker_coherence=0.5, right_marker_coherence=0.5,
    ))
    assert key["asym"] == "left"


@pytest.mark.parametrize("depth,band", [
    (34.0, "far"), (33.0, "near"), (29.0, "face"),
    (27.0, "pressed"), (25.0, "pressed"), (24.0, "floor"), (21.0, "floor"),
])
def test_depth_bands(depth, band):
    key = key_from_readings(reading(left_contact=True, right_contact=True,
                                    left_depth_mm=depth, right_depth_mm=depth))
    assert key["depth"] == band


def test_relief_is_the_only_channel_that_sees_the_surface():
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=27.0, right_depth_mm=27.0,
        left_marker_displacement_px=3.0, right_marker_displacement_px=3.0,
        left_marker_coherence=0.5, right_marker_coherence=0.5,
        left_height_roughness_mm=0.081, right_height_roughness_mm=0.023,
    ))
    assert key["relief"] == "rough"


def test_anisotropy_needs_a_number_to_be_banded():
    quiet = key_from_readings(reading(left_marker_anisotropy=0.0,
                                      right_marker_anisotropy=None))
    assert quiet["aniso"] == "unknown"
    banded = key_from_readings(reading(left_marker_anisotropy=1.05,
                                       right_marker_anisotropy=1.30))
    assert banded["aniso"] == "banded"


def test_shear_axis_follows_the_larger_pad():
    key = key_from_readings(reading(
        left_marker_shear_dx=0.10, left_marker_shear_dy=0.90,
        right_marker_shear_dx=0.05, right_marker_shear_dy=0.10,
    ))
    assert key["shear"] == "medium"
    assert key["shear_axis"] == "y"


def test_an_ambiguous_shear_vector_names_no_axis():
    key = key_from_readings(reading(left_marker_shear_dx=0.6,
                                    left_marker_shear_dy=0.6))
    assert key["shear"] == "medium"
    assert key["shear_axis"] == "none"


def test_matches_consults_only_the_dimensions_the_card_names():
    key = key_from_readings(reading(
        left_contact=True, right_contact=True,
        left_depth_mm=27.9, right_depth_mm=27.9,
        left_marker_displacement_px=2.4, right_marker_displacement_px=2.4,
        left_marker_coherence=0.5, right_marker_coherence=0.5,
    ))
    assert matches(key, {"depth": "face"})
    assert matches(key, {"depth": "face", "disp": "moving"})
    assert not matches(key, {"depth": "pressed"})
    assert not matches(key, {"depth": "face", "disp": "still"})
    assert matches(key, {})  # a card with no condition is always true


def test_describe_reads_as_the_condition_it_is():
    assert describe({"depth": "face", "coh_gap": "split"}) == "depth=face, coh_gap=split"
