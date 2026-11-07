"""Tactile API for the Shadow-Hand piano task.

Same transport and the same motion primitives as :class:`UniVTACTactileApi`;
the difference is which tools this task can actually honour.

The index finger carries one GelSight Mini (the UIPC backend in
``env/tasks/_index_tactile.py``).  Its readings are served as the **left**
pad: ``get_tactile_data`` and every motion reply carry the height-map depth in
millimetres, the marker displacement/shear/coherence and the marker image
under ``left_*``.  The ``right_*`` fields are missing -- the hand has no second
pad -- so 0.0 and False there mean "no reading", not "a zero reading".

Withdrawn (four):

* ``detect_slip`` -- the parent's detector is built from a left/right pad
  pair.  A single gel has markers, but the pair arithmetic behind that tool
  has no second side to compare against.
* ``get_object_pose`` / ``sample_grasp_pose`` -- the keys are articulation
  bodies with no pose query, and there are no objects to grasp.
* ``rotate_gripper`` -- the wrist's orientation is fixed by the arm-hand mount
  and is not one of the task's action dimensions.

Added (two):

* ``get_ee_pose`` -- promoted from the parent's private diagnostic.  The palm
  position is the frame every commanded pose lives in, and there is no other
  way to read where the hand is.
* ``get_key_position`` -- the palm targets for a key.  The 88-key geometry is
  fixed and public; asking the task for it keeps the model from reproducing
  the white/black-key arithmetic and its own mistakes.

No progress or score reading is added.  The episode is a note sequence and
the model plays it in order, the way it plans any other task.  What the gel
*does* deliver is the force itself: a calibrated model turns the pad's image
into a newton estimate (``left_force_N``, with ``left_force_quality`` saying
whether the reading is inside the model's envelope), and
``close_gripper(press_force_N=...)`` servos the press on that estimate.  The
estimate is a measurement of the pad's contact, not the task's verdict -- a
note still has to be held in its band and released, and a missed note is the
model's risk.

``goto_pose`` and ``move_delta`` keep the parent's signatures but their
docstrings are reissued below: on this task the end effector is the **palm**
and the commanded quaternion is ignored.  ``close_gripper`` keeps the parent's
gel-depth servo (``force_threshold``) -- it closes the index finger until the
gel reads that depth -- and ``open_gripper`` lifts the finger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.integrations.tactile.univtac_api import (
    UniVTACTactileApi,
    _trim_motion_tactile,
)

# Tools this task cannot honour; see the module docstring.  get_tactile_data
# is NOT here: the index gel is real and its readings are served as the left
# pad.
_WITHDRAWN = (
    "detect_slip",
    "get_object_pose",
    "sample_grasp_pose",
    "rotate_gripper",
)


class UniVTACPianoApi(UniVTACTactileApi):
    """Robot control API for a Franka arm carrying a five-finger Shadow Hand.

    Only the index finger moves; the rest of the hand is welded and the wrist
    orientation is fixed.  The task is scored on the measured contact force of
    the piano key under that finger.
    """

    def functions(self) -> dict[str, Any]:
        tools = super().functions()
        for name in _WITHDRAWN:
            tools.pop(name, None)
        return {
            "get_ee_pose": self.get_ee_pose,
            "get_key_position": self.get_key_position,
            **tools,
        }

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> dict:
        """Move the palm to a target position.

        The end effector is the palm, and the wrist's orientation is fixed by
        the arm-hand mount: ``quaternion_wxyz`` is accepted for the shared
        signature and ignored, so pass the one ``get_ee_pose`` returned.

        Args:
            position: (3,) palm XYZ in metres, world frame.
            quaternion_wxyz: (4,) WXYZ; ignored on this task.
            z_approach: (float) If positive, first move to position + this
                height, then descend. Default 0.0.

        Returns:
            dict with ``step`` (the episode clock, 120 ticks per simulated
            second) and the index gel's scalar readings (depth, marker
            displacement/shear/coherence) as the ``left_*`` fields -- see
            ``get_tactile_data``.  The ``right_*`` fields are missing because
            the hand has one pad.
        """
        return super().goto_pose(position, quaternion_wxyz, z_approach)

    def move_delta(
        self,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
        drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
        coord: str = "world",
    ) -> dict:
        """Move the palm by a small delta.

        Limits per call: max 0.10 m translation per axis; a delta below 2 mm on
        every axis raises instead of moving.  Rotation is not available on this
        task -- the wrist is fixed -- so any nonzero ``drx`` / ``dry`` /
        ``drz`` raises.

        Args:
            dx: (float) X translation in metres.
            dy: (float) Y translation in metres.
            dz: (float) Z translation in metres.
            drx: (float) Must be 0.0.
            dry: (float) Must be 0.0.
            drz: (float) Must be 0.0.
            coord: ("world" or "ee") Reference frame for the translation.

        Returns:
            dict with ``step`` (120 ticks per simulated second).
        """
        if any(abs(float(v)) > 1e-9 for v in (drx, dry, drz)):
            raise RuntimeError(
                "The wrist orientation is fixed on this task; move_delta "
                "cannot rotate it. Pass drx=dry=drz=0 and move the palm."
            )
        return super().move_delta(dx, dy, dz, 0.0, 0.0, 0.0, coord)

    def open_gripper(self) -> dict:
        """Straighten the index finger (opening 1.0).

        Returns:
            dict with ``step`` (120 ticks per simulated second).
        """
        data = self._post("/gripper", {"action": "open"})
        print("Index finger straightened")
        return self._motion_result(data)

    def close_gripper(self, force_threshold: float | None = None,
                      position: float | None = None,
                      adaptive: bool | None = None,
                      press_force_N: float | None = None) -> dict:
        """Press a key with the index finger, optionally to a force target.

        The opening is the finger's travel: 1.0 leaves it straight, 0.0 drives
        it fully down.

        With ``press_force_N`` the call is the whole press and the other two
        arguments are ignored.  The hand must already be at the key's press
        pose with the finger open -- ``get_key_position(midi)["press"]`` -- and
        the call then rises 2 mm, closes the index with the calibrated ramp,
        takes a no-contact depth frame, and lowers the palm in 0.1 mm steps,
        re-reading the gel at each one, until the gel's force estimate reaches
        the target.  It holds the band for 0.2 s and returns.

        The estimate is calibrated over 0.14-0.43 N and saturates above that:
        the pad's tilt bottoms out, so a target above 0.43 N cannot be
        distinguished from 0.5 N by the reading itself.  Such a target is
        served differently -- the press stops when the pad's tilt stops
        growing while the palm keeps descending, which is the heavy band --
        and the reply's message says so.  A press that finds the key too far
        below (or missing) fails with the estimate it reached.

        Args:
            force_threshold: (float, optional) Gel-depth target in mm for the
                older depth servo.  Not used together with ``press_force_N``.
            position: (float, optional) Opening from 0.0 (fully pressed) to
                1.0 (straight), driving the finger travel directly.
            adaptive: (bool) Whether the depth servo runs.  Omitted, it is
                True unless a position was given.
            press_force_N: (float, optional) Target contact force in newtons,
                e.g. 0.35 for a light note and 0.50 for a heavy one.  The
                task's tactile skill carries the band for each dynamic.

        Returns:
            dict with ``step`` (120 ticks per simulated second),
            ``left_force_N`` (the estimate at the stop, newtons) and
            ``left_force_quality``.
        """
        if adaptive is None:
            adaptive = position is None
        payload: dict = {"action": "close", "use_adaptive_grasp": bool(adaptive)}
        if force_threshold is not None:
            payload["force_threshold"] = float(force_threshold)
        if position is not None:
            payload["position"] = float(position)
        if press_force_N is not None:
            payload["press_force_N"] = float(press_force_N)
        data = self._post("/gripper", payload)
        if press_force_N is not None:
            print(f"Pressing to {press_force_N:.2f} N on the gel estimate")
        elif force_threshold is not None and adaptive:
            print(f"Index finger closing to gel depth {force_threshold:.2f} mm")
        else:
            print("Index finger closed")
        return self._motion_result(data)

    def get_tactile_data(self) -> dict:
        """Read the index finger's GelSight Mini without moving.

        The hand carries one gel, on the index; its readings are the ``left_*``
        fields.  ``depth_mm`` is the height map's deepest point in millimetres
        -- about 28.3 with the pad unloaded, falling as the key presses into
        the gel.  The force estimate is reported separately, because depth
        alone does not track the force (it falls about 0.13 mm over the whole
        working range and turns around above 0.45 N):

        * ``left_force_N`` -- the contact force the pad's image implies,
          newtons, from a model fitted on measured presses.  It is None until
          a press has captured its no-contact frame: a depth image on its own
          has no force in it.  A press does that automatically, so read it
          after ``close_gripper(press_force_N=...)``.
        * ``left_force_quality`` -- ``ok`` when the reading is inside the
          model's calibrated envelope, ``out_of_range`` when it has left it
          (a different press pose, a lost contact, a stale reference) and the
          number is extrapolating, ``no_reference`` before the first press.

        The marker fields are in the marker image's pixels.  The ``right_*``
        fields are missing and always 0.0 / False; they are not a second pad.

        Args:
            None.

        Returns:
            dict with ``step`` (120 ticks per simulated second),
            ``left_contact``, ``left_depth_mm``, ``left_force_N``,
            ``left_force_quality``, ``left_marker_displacement_px``,
            ``left_marker_coherence``, ``left_marker_shear_dx`` /
            ``left_marker_shear_dy``, ``left_marker_anisotropy``, and the
            ``right_*`` fields reported as missing.
        """
        return super().get_tactile_data()

    def get_ee_pose(self) -> dict:
        """Return the palm pose in the world frame.

        Args:
            None.

        Returns:
            dict with ``position``: [x, y, z] metres and ``quaternion_wxyz``:
            [w, x, y, z].  The quaternion is the fixed wrist orientation to
            pass back to ``goto_pose``.
        """
        return self._get("/ee_pose")

    def get_key_position(self, midi: int) -> dict:
        """Return the palm targets for one piano key.

        The keyboard geometry is fixed, so these are the same numbers on every
        episode; they are served from the task's own layout rather than
        reconstructed from a formula.

        Args:
            midi: (int) MIDI note number, 21 (A0) to 108 (C8).

        Returns:
            dict with ``midi`` (int), ``black`` (bool), ``hover`` ([x, y, z],
            the palm 40 mm above the key), ``press`` ([x, y, z], the palm
            height at which the fingertip meets the key) and
            ``quaternion_wxyz`` (the fixed wrist orientation).
        """
        return self._get("/piano_key_position", {"midi": int(midi)})

    @staticmethod
    def _motion_result(data: dict) -> dict:
        """The clock plus the index gel's scalars.

        The gel's readings ride on every motion reply, the way they do on
        every other task: a press returns the force estimate it stopped at,
        and a move returns what the pad reads where it arrived.
        """
        out = {"step": int(data.get("step", 0))}
        out.update(_trim_motion_tactile(data.get("tactile", {})))
        return out
