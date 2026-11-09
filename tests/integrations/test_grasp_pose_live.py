"""``sample_grasp_pose`` must answer from the object, not from the hand.

gpt-6-astra trial 26 (2026-11-07) asked for ``candidate_left``, which lay at
y=-0.347, and the server answered y=+0.239 — the hand's own location over the
reference, because the EE-relative offset is cached on the first call for an
object and the hand was nowhere near that object.  The program drove nowhere
and closed on air six times.

A task that exposes ``get_live_grasp_pose`` answers from the object's own
pose, and the cache is never entered.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import capx.serving.launch_univtac_server as server

CAN_POSITIONS = {
    "reference_object": [0.68, 0.238, 0.021],
    "candidate_left": [0.60, -0.347, 0.021],
    "candidate_right": [0.60, -0.177, 0.021],
}
# Where the hand sat when the two candidates were sampled in trial 26.
HAND_AT_REFERENCE = [0.616, 0.239, 0.208]
# What the task's own can rule answers for candidate_left.
LIVE_ANSWER = ([0.535, -0.347, 0.144], [0.0, 1.0, 0.0, 0.0])


class _Pose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


class _RobotManager:
    def __init__(self, p=HAND_AT_REFERENCE):
        self._pose = _Pose(p)

    def get_ee_pose(self):
        return self._pose


@pytest.fixture
def scene(monkeypatch):
    """A scene with the three cans, and an empty offset cache."""

    def fake_object_pose(name, return_bbox_extent=False):
        return {"found": name in CAN_POSITIONS, "position": CAN_POSITIONS.get(name)}

    monkeypatch.setattr(server, "_sim_get_object_pose", fake_object_pose)
    monkeypatch.setattr(server, "_GRASP_OFFSET", {})
    yield monkeypatch
    server._GRASP_OFFSET.clear()


def _task(**hooks):
    task = types.SimpleNamespace(_robot_manager=_RobotManager())
    for name, hook in hooks.items():
        setattr(task, name, hook)
    return task


def test_a_can_the_hand_is_not_above_gets_its_own_pose(scene):
    scene.setattr(
        server,
        "_TASK",
        _task(get_live_grasp_pose=lambda name: LIVE_ANSWER),
    )

    got = server._sim_get_grasp_pose("candidate_left")

    assert got["position"] == pytest.approx(LIVE_ANSWER[0])
    assert got["quaternion_wxyz"] == pytest.approx(LIVE_ANSWER[1])
    # The trap is not entered at all, so nothing is cached from the hand.
    assert server._GRASP_OFFSET == {}


def test_the_live_rule_tracks_the_object_not_the_first_call(scene):
    moved = {"xy": [0.60, -0.347]}

    def live(name):
        position = [0.535, moved["xy"][1], 0.144]
        return position, LIVE_ANSWER[1]

    scene.setattr(server, "_TASK", _task(get_live_grasp_pose=live))
    first = server._sim_get_grasp_pose("candidate_left")

    moved["xy"] = [0.60, -0.300]      # an earlier probe shoved the can
    second = server._sim_get_grasp_pose("candidate_left")

    assert first["position"][1] == pytest.approx(-0.347)
    assert second["position"][1] == pytest.approx(-0.300)


def test_without_the_rule_the_cache_still_returns_the_hand(scene):
    """The behaviour the rule exists to replace: this is the trial-26 pose."""
    scene.setattr(server, "_TASK", _task())

    got = server._sim_get_grasp_pose("candidate_left")

    assert got["position"] == pytest.approx(HAND_AT_REFERENCE)


def test_a_rule_that_cannot_answer_falls_back_to_the_cache(scene, caplog):
    def boom(name):
        raise RuntimeError("no live pose for that name")

    scene.setattr(server, "_TASK", _task(get_live_grasp_pose=boom))

    with caplog.at_level("WARNING"):
        got = server._sim_get_grasp_pose("candidate_left")

    assert got["position"] == pytest.approx(HAND_AT_REFERENCE)
    assert "live-pose rule failed" in caplog.text


def test_a_rule_that_does_not_know_the_name_falls_back(scene):
    scene.setattr(server, "_TASK", _task(get_live_grasp_pose=lambda name: None))

    got = server._sim_get_grasp_pose("candidate_left")

    assert got["position"] == pytest.approx(HAND_AT_REFERENCE)


def test_an_unknown_object_is_still_reported_missing(scene):
    scene.setattr(server, "_TASK", _task(get_live_grasp_pose=lambda name: LIVE_ANSWER))

    got = server._sim_get_grasp_pose("no_such_can")

    assert got["position"] is None
