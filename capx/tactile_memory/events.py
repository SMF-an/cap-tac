"""Contact events: the unit a tactile memory stores.

A trace row is a sample, not a memory.  Marker displacement and shear are
cumulative from the reset baseline, so their absolute values only compare
inside one episode; what survives an episode is the change across a bounded
contact.  This module cuts a trace into those bounded contacts.

The boundary is the contact flag, not the action.  One grasp often spans
several primitives -- close, lift, move -- and cutting at every call would
turn one contact into a dozen rows, each too short to say what the gel did.
A segment opens when a pad first reports range contact and closes when both
pads leave it.  A segment that is already open at the first row is labelled
as such: a trace can begin after the contact was made (an initial hold), and
recording that as an onset would be a measurement the trace never made.

Each event stores the change, not the absolute reading: depth and
displacement deltas, the shear delta vector and its direction, coherence at
both ends, and how many readings it spanned.  The absolute values stay in
the trace; the event is the reduction a memory row carries.

Keyframes are row indices, not copies.  The image representation can fetch
the frames those rows produced and the formalized one can re-read the
fields, without the event growing a second payload.  The three frames are
onset, deepest, and release, which is exactly the comparison the retention
experiment varies.

Offline by construction.  Segmenting a finished trace may use rows after a
candidate boundary; selecting memory online must not.  A caller that runs
this on a live stream has to pass the rows observed so far, and the event
it gets is the event as it looked at that moment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from capx.tactile_memory import signature

#: Calls that read the sensor without moving anything.  Their rows attach to
#: the event that is already open and never open one by themselves.
READ_CALLS = frozenset({"/tactile"})

#: Boundaries an event can begin with.  ``trace_start`` means the first row
#: was already in contact; ``first_report`` means the row before it did not
#: report contact state at all, so the transition is unobserved.  Only
#: ``contact_onset`` is a transition this trace saw happen.
STARTS = ("trace_start", "contact_onset", "first_report")

#: Boundaries an event can end with.  ``trace_end`` means contact was still
#: reported when the rows ran out -- not a release.
ENDS = ("contact_loss", "trace_end")

#: Fields where the API's 0.0 means "this pad did not report", not a reading.
#: Shear is deliberately absent: a zero shear vector is an honest reading.
_ZERO_IS_MISSING = (
    "depth_mm",
    "marker_displacement_px",
    "marker_coherence",
    "height_roughness_mm",
)


def _side(readings: Mapping[str, Any], side: str, field: str) -> float | None:
    """One pad's field.  ``field`` is the bare name (``depth_mm``)."""
    raw = readings.get(f"{side}_{field}")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if field in _ZERO_IS_MISSING and number == 0.0:
        return None
    return number


def _shear(readings: Mapping[str, Any], side: str) -> tuple[float, float] | None:
    """One pad's shear vector, or None when either component is absent."""
    dx = readings.get(f"{side}_marker_shear_dx")
    dy = readings.get(f"{side}_marker_shear_dy")
    if dx is None or dy is None or isinstance(dx, bool) or isinstance(dy, bool):
        return None
    try:
        return float(dx), float(dy)
    except (TypeError, ValueError):
        return None


def _delta(new: float | None, old: float | None) -> float | None:
    """A change, or None when either end is missing; a real 0.0 is kept."""
    if new is None or old is None:
        return None
    return new - old


def _vector_delta(new: tuple[float, float] | None,
                  old: tuple[float, float] | None
                  ) -> tuple[float, float] | None:
    """A shear change, or None when either end is missing."""
    if new is None or old is None:
        return None
    return new[0] - old[0], new[1] - old[1]


def _contact_state(readings: Mapping[str, Any]) -> str | None:
    """Which pads report range contact, or None when the row cannot say.

    A row that reports only one pad and says it is out of range is not
    "no contact": the other pad is unknown, and a memory built on that
    reading would remember a loss of contact that was never measured.
    """
    left = readings.get("left_contact")
    right = readings.get("right_contact")
    if left is None and right is None:
        return None
    left_in, right_in = bool(left), bool(right)
    if left_in and right_in:
        return "both"
    if left_in:
        return "left"
    if right_in:
        return "right"
    if left is not None and right is not None:
        return "none"
    return None


@dataclass(frozen=True)
class PadDelta:
    """What one pad reported across an event, as changes where they exist.

    The absolute ``*_start`` and ``*_end`` values are kept for depth because
    depth is a physical distance and reads the same in the next episode.
    Displacement and shear are kept at both ends only so their delta is
    auditable; do not compare those absolutes across a reset.
    """

    depth_start_mm: float | None = None
    depth_end_mm: float | None = None
    depth_delta_mm: float | None = None
    depth_min_mm: float | None = None
    disp_start_px: float | None = None
    disp_end_px: float | None = None
    disp_delta_px: float | None = None
    shear_start_px: tuple[float, float] | None = None
    shear_end_px: tuple[float, float] | None = None
    shear_delta_px: tuple[float, float] | None = None
    shear_direction_deg: float | None = None
    coherence_start: float | None = None
    coherence_end: float | None = None
    relief_start_mm: float | None = None
    relief_end_mm: float | None = None


@dataclass(frozen=True)
class ContactEvent:
    """One bounded contact, reduced to what outlives the episode.

    Attributes:
        run: The trace's run id, when it carried one.
        trial: The trial number, when it carried one.
        task: The task name the caller attributed the trace to, or "".
        call: The call that produced the event's first in-contact reading --
            the action that triggered it.
        args: That action's arguments: the acquisition conditions of the
            public interface (commanded displacement, grip width, ...).
        calls: The action calls inside the event, in order, with pure reads
            left out.  One event can have several; the trigger is ``call``.
        start: Row index of the first in-contact reading (inclusive).
        end: Row index of the last in-contact reading (inclusive).
        boundary_start: One of :data:`STARTS`.
        boundary_end: One of :data:`ENDS`.
        samples: How many traces rows the event covers.
        t_start: Timestamp of the first row, or None.
        t_end: Timestamp of the last row, or None.
        left: The left pad's deltas.
        right: The right pad's deltas.
        key_start: The band key at the first in-contact reading.
        key_end: The band key at the last in-contact reading.
        onset: Row index of the onset keyframe (== ``start``).
        peak: Row index of the deepest in-contact reading, or None when no
            row reported a depth.
        release: Row index of the release keyframe (== ``end``; a release
            only if ``boundary_end`` is ``contact_loss``).
        label: A task-specific label supplied by the caller (``seated``,
            ``jammed``, ...).  Never inferred here: the generic boundaries
            are the sensor's, the task labels are the task's.
    """

    run: str | None = None
    trial: int | None = None
    task: str = ""
    call: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    calls: tuple[str, ...] = ()
    start: int = 0
    end: int = 0
    boundary_start: str = "contact_onset"
    boundary_end: str = "trace_end"
    samples: int = 0
    t_start: float | None = None
    t_end: float | None = None
    left: PadDelta = field(default_factory=PadDelta)
    right: PadDelta = field(default_factory=PadDelta)
    key_start: Mapping[str, str] = field(default_factory=dict)
    key_end: Mapping[str, str] = field(default_factory=dict)
    onset: int = 0
    peak: int | None = None
    release: int = 0
    label: str = ""

    def to_record(self) -> dict[str, Any]:
        """The event as a JSON-serializable memory row.

        Nested by group so a reader can tell a span from a payload from a
        keyframe.  The row keeps no raw readings: the trace is the only
        place those live, and a memory that copied them would grow without
        bound while saying nothing new.
        """
        return {
            "run": self.run,
            "trial": self.trial,
            "task": self.task,
            "label": self.label,
            "action": {
                "call": self.call,
                "args": dict(self.args),
                "calls": list(self.calls),
            },
            "span": {
                "start": self.start,
                "end": self.end,
                "samples": self.samples,
                "t_start": self.t_start,
                "t_end": self.t_end,
            },
            "boundary": {"start": self.boundary_start, "end": self.boundary_end},
            "left": _pad_record(self.left),
            "right": _pad_record(self.right),
            "key_start": dict(self.key_start),
            "key_end": dict(self.key_end),
            "keyframes": {
                "onset": self.onset,
                "peak": self.peak,
                "release": self.release,
            },
        }


def _pad_record(pad: PadDelta) -> dict[str, Any]:
    return {
        "depth_mm": {
            "start": pad.depth_start_mm,
            "end": pad.depth_end_mm,
            "delta": pad.depth_delta_mm,
            "min": pad.depth_min_mm,
        },
        "disp_px": {
            "start": pad.disp_start_px,
            "end": pad.disp_end_px,
            "delta": pad.disp_delta_px,
        },
        "shear_px": {
            "start": pad.shear_start_px,
            "end": pad.shear_end_px,
            "delta": pad.shear_delta_px,
            "direction_deg": pad.shear_direction_deg,
        },
        "coherence": {
            "start": pad.coherence_start,
            "end": pad.coherence_end,
        },
        "relief_mm": {
            "start": pad.relief_start_mm,
            "end": pad.relief_end_mm,
        },
    }


def _pad_delta(rows: Sequence[Mapping[str, Any]], side: str) -> PadDelta:
    """Fold one pad's readings across the event into start/end/delta."""
    readings = [_rows_readings(row) for row in rows]
    depth_pairs = [(index, _side(reading, side, "depth_mm"))
                   for index, reading in enumerate(readings)]
    depths = [(index, value) for index, value in depth_pairs if value is not None]
    disp = [_side(reading, side, "marker_displacement_px") for reading in readings]
    shear = [_shear(reading, side) for reading in readings]
    coh = [_side(reading, side, "marker_coherence") for reading in readings]
    relief = [_side(reading, side, "height_roughness_mm") for reading in readings]

    def first(values):
        return next((value for value in values if value is not None), None)

    def last(values):
        return next((value for value in reversed(values) if value is not None), None)

    depth_start = depths[0][1] if depths else None
    depth_end = depths[-1][1] if depths else None
    disp_start, disp_end = first(disp), last(disp)
    shear_start, shear_end = first(shear), last(shear)
    shear_delta = _vector_delta(shear_end, shear_start)
    direction = None
    if shear_delta is not None and math.hypot(*shear_delta) > 0.0:
        direction = math.degrees(math.atan2(shear_delta[1], shear_delta[0]))

    return PadDelta(
        depth_start_mm=depth_start,
        depth_end_mm=depth_end,
        depth_delta_mm=_delta(depth_end, depth_start),
        depth_min_mm=min(value for _index, value in depths) if depths else None,
        disp_start_px=disp_start,
        disp_end_px=disp_end,
        disp_delta_px=_delta(disp_end, disp_start),
        shear_start_px=shear_start,
        shear_end_px=shear_end,
        shear_delta_px=shear_delta,
        shear_direction_deg=direction,
        coherence_start=first(coh),
        coherence_end=last(coh),
        relief_start_mm=first(relief),
        relief_end_mm=last(relief),
    )


def _peak(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Row of the deepest contact inside the event, first one wins."""
    best, best_depth = None, None
    for index, row in enumerate(rows):
        readings = _rows_readings(row)
        present = [value for value in (_side(readings, "left", "depth_mm"),
                                       _side(readings, "right", "depth_mm"))
                   if value is not None]
        if not present:
            continue
        depth = min(present)
        if best_depth is None or depth < best_depth:
            best, best_depth = index, depth
    return best


def _rows_readings(row: Mapping[str, Any]) -> Mapping[str, Any]:
    readings = row.get("readings")
    return readings if isinstance(readings, Mapping) else {}


def _build(rows: Sequence[Mapping[str, Any]], start: int, end: int,
           boundary_start: str, boundary_end: str, task: str,
           label: str) -> ContactEvent:
    span = rows[start:end + 1]
    first = span[0]
    calls = tuple(dict.fromkeys(
        str(row.get("call")) for row in span
        if row.get("call") and row.get("call") not in READ_CALLS
    ))
    # A pure read can be the first row that reports the contact (the last
    # action moved the pad in, and the read is the first look at it).  That
    # does not make the read the trigger; the trigger is the first action in
    # the event, and when the event holds none the trigger is unobserved.
    trigger = next((row for row in span
                    if row.get("call") and row.get("call") not in READ_CALLS),
                   None)
    call = str(trigger.get("call")) if trigger else ""
    args = (trigger.get("args") if trigger is not None
            and isinstance(trigger.get("args"), Mapping) else {})
    times = [float(row["t"]) for row in span
             if isinstance(row.get("t"), (int, float))
             and not isinstance(row.get("t"), bool)]

    def key_at(index: int) -> dict[str, str]:
        previous = (_rows_readings(rows[index - 1]) if index > 0 else None)
        return signature.key_from_readings(_rows_readings(rows[index]),
                                           previous=previous)

    peak = _peak(span)
    return ContactEvent(
        run=first.get("run"),
        trial=first.get("trial"),
        task=task,
        call=call,
        args=dict(args),
        calls=calls,
        start=start,
        end=end,
        boundary_start=boundary_start,
        boundary_end=boundary_end,
        samples=sum(1 for row in span if _rows_readings(row)),
        t_start=times[0] if times else None,
        t_end=times[-1] if times else None,
        left=_pad_delta(span, "left"),
        right=_pad_delta(span, "right"),
        key_start=key_at(start),
        key_end=key_at(end),
        onset=start,
        peak=None if peak is None else start + peak,
        release=end,
        label=label,
    )


def segment_contact_events(rows: Sequence[Mapping[str, Any]], *,
                           task: str = "", label: str = "") -> list[ContactEvent]:
    """Cut one trace into bounded contacts.

    Args:
        rows: The trace rows, in order, each a mapping with ``call``,
            ``args`` and ``readings`` as :mod:`capx.integrations.tactile.trace`
            writes them.  Rows that carry no readings are allowed; they
            neither open nor close an event.
        task: The task name to attach to every event.  The trace does not
            carry it; the caller does (the miner reads it from the path).
        label: A task-specific label to attach to every event.  Generic
            boundaries are the sensor's; this is the task's.

    Returns:
        The events in trace order.  Empty when no row ever reported contact.
        An event that was open at the end of the rows is closed with
        ``boundary_end == "trace_end"``; that is not a release.
    """
    events: list[ContactEvent] = []
    open_at: int | None = None
    last_seen: int | None = None
    start_reason = ""

    for index, row in enumerate(rows):
        state = _contact_state(_rows_readings(row))
        if open_at is None:
            if state in (None, "none"):
                continue
            open_at = index
            last_seen = index
            if index == 0:
                start_reason = "trace_start"
            elif _contact_state(_rows_readings(rows[index - 1])) == "none":
                start_reason = "contact_onset"
            else:
                start_reason = "first_report"
            continue
        if state == "none":
            # Close on the last row that actually reported contact, not on
            # the row before the loss: rows in between may be unreadable,
            # and attributing an unreadable row to the contact would make
            # the event longer than the evidence for it.
            events.append(_build(rows, open_at, last_seen, start_reason,
                                 "contact_loss", task, label))
            open_at = None
            last_seen = None
        elif state is not None:
            last_seen = index

    if open_at is not None:
        events.append(_build(rows, open_at, last_seen, start_reason,
                             "trace_end", task, label))
    return events
