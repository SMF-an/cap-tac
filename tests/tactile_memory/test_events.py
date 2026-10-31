"""Tests for contact events: the unit a tactile memory stores.

Two things must hold for the layer to be worth having.  The boundary has to
follow the contact flag rather than the action, because one grasp spans
several primitives and cutting at every call would turn one contact into a
dozen events too short to say what the gel did.  And the event has to store
*changes*, because marker displacement and shear are cumulative from the
reset baseline: their absolutes mean nothing in the next episode, while the
difference across a bounded contact means the same thing every time.

The rest pins the attribution edge cases -- an unreadable row is not a
release, a contact that was already open is not an onset, and the trigger is
the action that moved the pad in, not the read that first noticed it.
"""

from __future__ import annotations

import json

from capx.tactile_memory.events import (
    ENDS,
    STARTS,
    ContactEvent,
    PadDelta,
    segment_contact_events,
)


def reading(contact: bool | None = True, depth: float | None = 27.9,
            disp: float | None = 2.0, coh: float | None = 0.5,
            shear: tuple[float, float] | None = (0.0, 0.0),
            relief: float | None = 0.02) -> dict:
    """One trace row's ``readings`` mapping, with both pads alike."""
    out = {}
    for side in ("left", "right"):
        if contact is not None:
            out[f"{side}_contact"] = contact
        if depth is not None:
            out[f"{side}_depth_mm"] = depth
        if disp is not None:
            out[f"{side}_marker_displacement_px"] = disp
        if coh is not None:
            out[f"{side}_marker_coherence"] = coh
        if shear is not None:
            out[f"{side}_marker_shear_dx"] = shear[0]
            out[f"{side}_marker_shear_dy"] = shear[1]
        if relief is not None:
            out[f"{side}_height_roughness_mm"] = relief
    return out


def row(call: str = "/move", args: dict | None = None,
        readings: dict | None = None, t: float = 0.0) -> dict:
    """One trace row, as ``trace.record`` writes it."""
    return {"t": t, "run": "run-1", "trial": 3, "call": call,
            "args": args or {}, "readings": reading() if readings is None
            else readings}


def closed_event() -> list[dict]:
    return [
        row("/move", readings=reading(contact=False)),
        row("/gripper", {"position": 0.6}, reading()),
        row("/tactile", readings=reading()),
        row("/move", readings=reading()),
        row("/move", readings=reading(contact=False)),
    ]


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_a_trace_that_never_touched_the_gel_has_no_events():
    rows = [row(readings=reading(contact=False)) for _ in range(4)]
    assert segment_contact_events(rows) == []


def test_one_contact_is_one_event_however_many_calls_it_spans():
    events = segment_contact_events(closed_event())
    assert len(events) == 1
    event = events[0]
    assert (event.start, event.end) == (1, 3)
    assert event.boundary_start == "contact_onset"
    assert event.boundary_end == "contact_loss"
    # The pure read in the middle neither opens nor closes anything.
    assert event.samples == 3


def test_an_event_that_was_already_open_at_the_first_row_says_so():
    rows = [row(readings=reading()), row(readings=reading(contact=False))]
    event = segment_contact_events(rows)[0]
    assert event.boundary_start == "trace_start"
    assert event.start == 0


def test_a_contact_first_seen_after_an_unreadable_row_is_not_an_onset():
    # The row before it did not report contact state, so the transition was
    # never observed.  Calling it an onset would invent a measurement.
    rows = [{"t": 0.0, "call": "/move", "args": {}, "readings": {}},
            row(readings=reading()),
            row(readings=reading(contact=False))]
    event = segment_contact_events(rows)[0]
    assert event.boundary_start == "first_report"
    assert event.start == 1


def test_a_contact_still_on_at_the_end_is_not_a_release():
    events = segment_contact_events([row(readings=reading())])
    assert len(events) == 1
    assert events[0].boundary_end == "trace_end"


def test_unreadable_rows_before_the_loss_are_not_attributed_to_the_event():
    # The pads were last seen in contact at row 1.  Rows 2 and 3 say nothing,
    # so the event ends at the last row that actually reported contact.
    rows = [row(readings=reading(contact=False)),
            row(readings=reading()),
            {"t": 0.0, "call": "/move", "args": {}, "readings": {}},
            row(readings=reading(contact=False))]
    event = segment_contact_events(rows)[0]
    assert (event.start, event.end) == (1, 1)


def test_one_pad_reporting_out_of_range_does_not_end_the_contact():
    # The other pad is unknown, not out of contact.  A memory built on this
    # row would remember a release that was never measured.
    rows = [row(readings=reading()),
            row(readings={"left_contact": False}),
            row(readings=reading()),
            row(readings=reading(contact=False))]
    events = segment_contact_events(rows)
    assert len(events) == 1
    assert (events[0].start, events[0].end) == (0, 2)


def test_two_contacts_are_two_events():
    rows = [row(readings=reading()),
            row(readings=reading(contact=False)),
            row(readings=reading()),
            row(readings=reading(contact=False))]
    events = segment_contact_events(rows)
    assert [(event.start, event.end) for event in events] == [(0, 0), (2, 2)]


def test_every_boundary_is_from_the_declared_vocabulary():
    rows = [row(readings=reading()), row(readings=reading(contact=False))]
    event = segment_contact_events(rows)[0]
    assert event.boundary_start in STARTS
    assert event.boundary_end in ENDS


# ---------------------------------------------------------------------------
# The payload: changes, not cumulative absolutes
# ---------------------------------------------------------------------------


def test_displacement_and_shear_are_stored_as_changes():
    # Both are cumulative from the reset baseline, so the absolutes compare
    # only inside one episode; the change across the contact compares across
    # episodes, which is the whole point of storing it.
    rows = [
        row("/move", readings=reading(contact=False, disp=6.0,
                                      shear=(1.0, 0.0))),
        row("/gripper", readings=reading(disp=6.5, shear=(1.0, 0.0))),
        row("/move", readings=reading(disp=7.0, shear=(1.0, 2.0))),
        row("/move", readings=reading(contact=False)),
    ]
    event = segment_contact_events(rows)[0]
    assert event.left.disp_start_px == 6.5
    assert event.left.disp_end_px == 7.0
    assert event.left.disp_delta_px == 0.5
    assert event.left.shear_delta_px == (0.0, 2.0)
    assert event.left.shear_direction_deg == 90.0
    assert event.right.shear_direction_deg == 90.0


def test_depth_keeps_its_absolute_and_its_minimum():
    # Depth is a physical distance and reads the same in the next episode,
    # so the absolute is kept; the minimum is where the pad was pressed
    # hardest, which the two endpoints do not show.
    rows = [
        row(readings=reading(depth=27.0)),
        row(readings=reading(depth=25.0)),
        row(readings=reading(depth=26.0)),
        row(readings=reading(contact=False)),
    ]
    pad = segment_contact_events(rows)[0].left
    assert pad.depth_start_mm == 27.0
    assert pad.depth_end_mm == 26.0
    assert pad.depth_delta_mm == -1.0
    assert pad.depth_min_mm == 25.0


def test_coherence_keeps_both_ends_so_a_lost_track_is_visible():
    rows = [row(readings=reading(coh=0.62)),
            row(readings=reading(coh=0.09)),
            row(readings=reading(contact=False))]
    pad = segment_contact_events(rows)[0].left
    assert (pad.coherence_start, pad.coherence_end) == (0.62, 0.09)


def test_a_zero_depth_is_missing_but_a_zero_shear_is_a_reading():
    # The API writes 0.0 for "this pad did not report" in the scalar fields;
    # a zero shear vector is a real measurement of no slide.
    event = segment_contact_events([
        row(readings=reading(depth=0.0, shear=(0.0, 0.0))),
        row(readings=reading(contact=False))])[0]
    assert event.left.depth_start_mm is None
    assert event.left.shear_start_px == (0.0, 0.0)
    assert event.left.shear_direction_deg is None


# ---------------------------------------------------------------------------
# The action and the keyframes
# ---------------------------------------------------------------------------


def test_the_trigger_is_the_action_not_the_read_that_first_saw_it():
    rows = [
        row("/move", readings=reading(contact=False)),
        row("/tactile", readings=reading()),
        row("/gripper", {"position": 0.6}, reading()),
        row("/tactile", readings=reading()),
        row("/move", readings=reading(contact=False)),
    ]
    event = segment_contact_events(rows)[0]
    assert event.call == "/gripper"
    assert event.args == {"position": 0.6}
    assert event.calls == ("/gripper",)


def test_an_event_with_no_action_has_no_trigger():
    rows = [row("/tactile", readings=reading()),
            row("/tactile", readings=reading(contact=False))]
    assert segment_contact_events(rows)[0].call == ""


def test_keyframes_index_the_trace_not_the_event():
    rows = [row(readings=reading(contact=False)),
            row(readings=reading(depth=27.0)),
            row(readings=reading(depth=25.0)),
            row(readings=reading(depth=26.0)),
            row(readings=reading(contact=False))]
    event = segment_contact_events(rows)[0]
    assert (event.onset, event.peak, event.release) == (1, 2, 3)


def test_the_run_and_trial_travel_with_the_event():
    event = segment_contact_events(closed_event())[0]
    assert (event.run, event.trial) == ("run-1", 3)


def test_the_caller_supplies_the_task_and_label():
    event = segment_contact_events(closed_event(), task="blind_pick_balls",
                                   label="seated")[0]
    assert (event.task, event.label) == ("blind_pick_balls", "seated")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_record_is_json_and_carries_no_raw_readings():
    event = segment_contact_events(closed_event())[0]
    record = event.to_record()
    json.dumps(record)  # would raise on a tuple key or a numpy scalar
    assert "readings" not in json.dumps(record)
    assert record["action"]["call"] == "/gripper"
    assert record["span"]["samples"] == 3
    assert record["left"]["disp_px"]["delta"] == event.left.disp_delta_px


def test_an_event_whose_rows_reported_no_depth_has_no_peak():
    rows = [row(readings=reading(depth=None)),
            row(readings=reading(depth=None, contact=False))]
    event = segment_contact_events(rows)[0]
    assert event.peak is None
    assert event.left.depth_start_mm is None


def test_the_segmenter_accepts_a_finished_trace_and_says_it_is_offline():
    # The docstring's contract: segmenting a finished trace may look ahead,
    # but the caller decides when "now" is by passing only the rows so far.
    live = closed_event()
    assert len(segment_contact_events(live[:4])) == 1
    assert segment_contact_events(live[:4])[0].boundary_end == "trace_end"


def test_the_event_types_are_frozen():
    assert ContactEvent.__dataclass_params__.frozen
    assert PadDelta.__dataclass_params__.frozen
