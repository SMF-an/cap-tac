"""The tactile key: one reading, reduced to the situations a skill can name.

A skill is not written against a number.  "coherence 0.42" is a fact about
one instant of one trial; "the gel is pressed and its markers are moving
together" is a situation, and a situation is what a skill can be retrieved
by.  This module is that reduction: it takes the dict ``get_tactile_data``
returns and produces a small, fixed set of ``dimension -> band`` pairs.

Two properties matter more than resolution:

  * zero input.  The agent never supplies the key.  Every dimension is read
    off a response the server already sent, so a card cannot be matched to
    a reading that was never taken, and a model that misremembers a number
    cannot steer retrieval with it.
  * bands, not values.  A band survives a different seed, a different gel
    and a different object; an exact number does not.  The cut points below
    sit in gaps the repo has measured (an empty close reads under 0.01 px
    of marker motion while a real grasp reads 1.5 px or more; healthy pads
    report coherence 0.42-0.68, a pad whose marker field was dragged away
    by the jaw reports 0.09-0.10), so a card can name a band and still be
    right on the next seed.

The cost is resolution: two situations inside one band are the same
situation here.  That is deliberate.  A card that needs finer resolution
than the sensor's own repeatability is a card that will be wrong about half
the time, and a wrong card is worse than no card -- it arrives with the
authority of something that was once measured.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# ---------------------------------------------------------------------------
# Dimensions and their bands
# ---------------------------------------------------------------------------
#
# ``0.0`` is the API's fallback for "this pad did not report", and it is out
# of range for every field except the shear vector -- where zero is also the
# honest reading for "no shear".  So for depth, displacement, coherence and
# relief a 0.0 becomes the ``missing`` band, while shear keeps ``none`` and
# cannot tell the two apart.  That asymmetry is the reason ``contact`` is its
# own dimension: it is the only field that says whether a pad is talking at
# all.

#: Every dimension, in the order they are rendered.
DIMENSIONS: tuple[str, ...] = (
    "contact",     # which pads are inside the sensor's range
    "depth",       # the nearer of the two pads, banded
    "asym",        # whether the two pads agree on depth
    "disp",        # the pad whose markers moved most
    "disp_gap",    # how far apart the two pads' displacements are
    "coh",         # the more coherent pad
    "coh_gap",     # how far apart the two pads' coherence is
    "shear",       # magnitude of the larger mean marker displacement vector
    "shear_axis",  # which component of that vector dominates
    "aniso",       # marker image row/column gradient ratio
    "relief",      # fine relief of the gel's indentation map
    # The three below need a previous reading; without one they are
    # "unknown", which is what a first reading deserves.
    "depth_trend",  # is the surface still coming in, or has it stopped
    "disp_trend",   # are the markers still being dragged
    "shear_trend",  # is the load still sliding
)

#: The legal values of each dimension.
BANDS: dict[str, tuple[str, ...]] = {
    "contact": ("none", "left", "right", "both"),
    "depth": ("missing", "far", "near", "face", "pressed", "floor"),
    "asym": ("unknown", "even", "left", "right"),
    "disp": ("missing", "still", "trace", "light", "moving", "heavy"),
    "disp_gap": ("unknown", "even", "split"),
    "coh": ("missing", "lost", "low", "mid", "high"),
    "coh_gap": ("unknown", "even", "split"),
    "shear": ("none", "small", "medium", "large"),
    "shear_axis": ("none", "x", "y"),
    "aniso": ("unknown", "iso", "banded"),
    "relief": ("unknown", "smooth", "rough"),
    "depth_trend": ("unknown", "falling", "static", "rising"),
    "disp_trend": ("unknown", "static", "growing"),
    "shear_trend": ("unknown", "static", "growing"),
}


def is_absence(dimension: str, band: str) -> bool:
    """Whether a band reports an absent reading rather than a measured one.

    ``depth=missing`` and every ``*_trend=unknown`` say that the sensor, or
    the previous reading a trend needs, was not there.  That is a fact about
    what could be measured -- a record's validity -- not a condition an
    action can be gated on: nothing follows from "this measurement does not
    exist".  ``contact=none``, ``shear=none`` and ``shear_axis=none`` are not
    absences; zero shear and no pad in range are readings in their own right.

    Args:
        dimension: A name from :data:`DIMENSIONS`.
        band: One of that dimension's legal values.

    Returns:
        True when the band names the absence of a reading.
    """
    names = BANDS.get(dimension)
    if not names or dimension in ("contact", "shear", "shear_axis"):
        return False
    return band == names[0]

# Cut points, each with where the number came from.
_DEPTH_FAR = 33.5        # 34.0 is the far plane: nothing in the sensor's range
_DEPTH_FACE = 29.6       # the undeformed gel face sits at ~28.3; above this, no touch
_DEPTH_FLOOR = 24.4      # 24.0 is the camera floor; the reading saturates there
_DEPTH_PRESSED = 27.6    # 26.5 mm is a full grasp depth on a 60 mm ball
_ASYM_MM = 0.5           # pad-to-pad depth difference that means one-sided loading

_DISP_TRACE = 0.05       # an empty close reads under 0.01 px
_DISP_LIGHT = 0.5
_DISP_MOVING = 1.5       # a real grasp moves the markers 1.5 px or more
_DISP_HEAVY = 6.0

_COH_LOST = 0.15         # a dead pad (field dragged by the jaw) reads 0.09-0.10
_COH_LOW = 0.40          # healthy pads read 0.42-0.68
_COH_HIGH = 0.70         # a pad dragged a long way through the gel reads 0.9+
_GAP = 0.20              # pad-to-pad disagreement that means the two are not describing one object

_SHEAR_SMALL = 0.10
_SHEAR_MEDIUM = 0.50
_SHEAR_LARGE = 1.50
_AXIS_RATIO = 2.0        # one component this many times the other names the axis

_ANISO_BANDED = 1.15     # 1.0 is an evenly deforming gel
_RELIEF_ROUGH = 0.05     # mm; smooth surfaces read 0.014-0.038, dotted ones 0.09-0.25

# Trends compare one reading with the one before it.  The steps are set at
# the sensor's own noise: anything below is not a movement, it is a re-render.
_DEPTH_STEP = 0.20       # mm; repeated readings of a still surface agree this closely
_SHEAR_STEP = 0.10       # px
_DISP_STEP = 0.05        # px; an empty close reads under 0.01


def _readings_value(readings: Mapping[str, Any], key: str) -> float | None:
    """One numeric field, or None when the pad did not report it."""
    value = readings.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _reported(readings: Mapping[str, Any], key: str) -> float | None:
    """One field, with the API's 0.0-means-missing convention applied."""
    number = _readings_value(readings, key)
    return None if number is None or number == 0.0 else number


def _band(value: float | None, edges: tuple[float, ...],
          names: tuple[str, ...]) -> str:
    """Place a number in its band; ``names[0]`` is the missing band."""
    if value is None:
        return names[0]
    for edge, name in zip(edges, names[1:]):
        if value < edge:
            return name
    return names[-1]


def _sides(readings: Mapping[str, Any], field: str,
           missing_zero: bool = True) -> dict[str, float | None]:
    read = _reported if missing_zero else _readings_value
    return {side: read(readings, f"{side}_{field}") for side in ("left", "right")}


def _gap(values: Mapping[str, float | None]) -> str:
    """Even or split, for a pair of per-pad readings."""
    if values["left"] is None or values["right"] is None:
        return "unknown"
    return "split" if abs(values["left"] - values["right"]) >= _GAP else "even"


def _trend(current: float | None, previous: float | None, step: float) -> str:
    """Rising, falling or static, with ``unknown`` when there is nothing to pair."""
    if current is None or previous is None:
        return "unknown"
    change = current - previous
    return "rising" if change >= step else "falling" if change <= -step else "static"


def _growing(current: float | None, previous: float | None, step: float) -> str:
    """Growing or static -- for fields that only ever accumulate."""
    if current is None or previous is None:
        return "unknown"
    return "growing" if current - previous >= step else "static"


def key_from_readings(readings: Mapping[str, Any],
                      previous: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Reduce one tactile reading to its ``dimension -> band`` key.

    Args:
        readings: The mapping ``get_tactile_data`` returned, or any motion
            primitive's ``tactile`` sub-dict.  Missing fields are allowed;
            absent or 0.0 numbers become the ``missing`` band rather than a
            suspicious zero.
        previous: The reading taken before this one, if it is known.  Only
            the three ``*_trend`` dimensions consult it; without it they
            read ``unknown``, which is the honest answer for a first look.
            Displacement and shear accumulate from the reset, so their
            trend is a rate -- "the gel is still being dragged" -- not a
            direction.

    Returns:
        dict with one entry per name in DIMENSIONS.  The values are bands
        from BANDS, never numbers.
    """
    key: dict[str, str] = {}

    left_in = bool(readings.get("left_contact"))
    right_in = bool(readings.get("right_contact"))
    key["contact"] = ("both" if left_in and right_in
                      else "left" if left_in else "right" if right_in else "none")

    depth = _sides(readings, "depth_mm")
    present = [value for value in depth.values() if value is not None]
    nearest = min(present) if present else None
    key["depth"] = _band(nearest,
                         (_DEPTH_FLOOR, _DEPTH_PRESSED, _DEPTH_FACE, _DEPTH_FAR),
                         ("missing", "floor", "pressed", "face", "near", "far"))

    # The two pads see two sides of one object, so their depths should agree
    # to within the object's own compliance.  A pad out of range on either
    # side makes the comparison meaningless rather than "even".
    if depth["left"] is None or depth["right"] is None:
        key["asym"] = "unknown"
    else:
        gap = depth["left"] - depth["right"]
        key["asym"] = ("even" if abs(gap) < _ASYM_MM
                       else "left" if gap < 0 else "right")

    disp = _sides(readings, "marker_displacement_px")
    present = [value for value in disp.values() if value is not None]
    furthest = max(present) if present else None
    key["disp"] = _band(furthest,
                        (_DISP_TRACE, _DISP_LIGHT, _DISP_MOVING, _DISP_HEAVY),
                        ("missing", "still", "trace", "light", "moving", "heavy"))
    key["disp_gap"] = _gap(disp)

    coh = _sides(readings, "marker_coherence")
    present = [value for value in coh.values() if value is not None]
    key["coh"] = _band(max(present) if present else None,
                       (_COH_LOST, _COH_LOW, _COH_HIGH),
                       ("missing", "lost", "low", "mid", "high"))
    key["coh_gap"] = _gap(coh)

    # Signed, mirrored between the pads, and 0.0 is legal here -- so a pad
    # that did not report is indistinguishable from one that did not shear.
    shear: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        dx = _readings_value(readings, f"{side}_marker_shear_dx")
        dy = _readings_value(readings, f"{side}_marker_shear_dy")
        shear[side] = (0.0 if dx is None else dx, 0.0 if dy is None else dy)
    magnitude = max((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in shear.values())
    key["shear"] = ("none" if magnitude < _SHEAR_SMALL
                    else "small" if magnitude < _SHEAR_MEDIUM
                    else "medium" if magnitude < _SHEAR_LARGE else "large")
    vector = max(shear.values(), key=lambda pair: pair[0] ** 2 + pair[1] ** 2)
    dx, dy = abs(vector[0]), abs(vector[1])
    key["shear_axis"] = ("none" if max(dx, dy) < _SHEAR_SMALL
                         else "x" if dx >= dy * _AXIS_RATIO
                         else "y" if dy >= dx * _AXIS_RATIO else "none")

    aniso = [value for value in
             (_reported(readings, f"{side}_marker_anisotropy")
              for side in ("left", "right")) if value is not None]
    key["aniso"] = _band(max(aniso) if aniso else None, (_ANISO_BANDED,),
                         ("unknown", "iso", "banded"))

    relief = [value for value in
              (_reported(readings, f"{side}_height_roughness_mm")
               for side in ("left", "right")) if value is not None]
    key["relief"] = _band(max(relief) if relief else None, (_RELIEF_ROUGH,),
                          ("unknown", "smooth", "rough"))

    if previous is None:
        key["depth_trend"] = key["disp_trend"] = key["shear_trend"] = "unknown"
    else:
        was = [value for value in _sides(previous, "depth_mm").values()
               if value is not None]
        key["depth_trend"] = _trend(nearest, min(was) if was else None, _DEPTH_STEP)

        was = [value for value in _sides(previous, "marker_displacement_px").values()
               if value is not None]
        key["disp_trend"] = _growing(furthest, max(was) if was else None, _DISP_STEP)

        before: list[float] = []
        for side in ("left", "right"):
            x = _readings_value(previous, f"{side}_marker_shear_dx")
            y = _readings_value(previous, f"{side}_marker_shear_dy")
            if x is not None and y is not None:
                before.append((x ** 2 + y ** 2) ** 0.5)
        key["shear_trend"] = _growing(magnitude, max(before) if before else None,
                                      _SHEAR_STEP)
    return key


def matches(key: Mapping[str, str], wanted: Mapping[str, str]) -> bool:
    """Whether a key satisfies a card's (partial) condition.

    Args:
        key: A full key from :func:`key_from_readings`.
        wanted: The card's condition: any subset of DIMENSIONS, each with one
            band it requires.  Dimensions it does not name are not consulted.

    Returns:
        True when every named dimension carries the required band.
    """
    return all(key.get(dimension) == band for dimension, band in wanted.items())


def describe(wanted: Mapping[str, str]) -> str:
    """A card's condition as prose, for the line the LLM reads."""
    return ", ".join(f"{dimension}={band}" for dimension, band in wanted.items())
