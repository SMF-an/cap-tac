"""Tactile text API for the Shadow-Hand piano task.

Same transport and the same motion primitives as :class:`UniVTACTactileApi`;
the difference is which tools this task can actually honour.

The index finger carries one GelSight Mini (the UIPC backend in
``env/tasks/_index_tactile.py``).  Its readings are served as the **left** pad,
and this branch reads them only through ``describe_tactile()``: the gel image
goes to the Octopi-1.5 visual-tactile language model, which returns a text
description of the surface and its compliance.  There is no scalar tactile
layer here -- no depth in millimetres, no marker field, no force estimate.  The
hand has one pad, so ``side="left"`` is the reading, and "both" would ask for a
right pad that does not exist.

The task's score is a key force, and this branch deliberately does not carry
the calibrated image-to-force model that the scalar arm used.  The description
is the only tactile reading a policy gets; whether it separates a 0.35 N press
from a 0.50 N one is the question this arm measures, not a mapping the server
supplies.

Withdrawn (three):

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

No progress or score reading is added.  The episode is a note sequence and the
model plays it in order, the way it plans any other task.

``goto_pose`` and ``move_delta`` keep the parent's signatures but their
docstrings are reissued below: on this task the end effector is the **palm**
and the commanded quaternion is ignored.  ``close_gripper`` is positional --
the finger's travel from straight to fully pressed -- and ``open_gripper``
lifts the finger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.integrations.tactile.univtac_api import UniVTACTactileApi

# Tools this task cannot honour; see the module docstring.  describe_tactile is
# NOT here: the index gel is real and its description is the task's tactile
# reading.
_WITHDRAWN = (
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
        signature and ignored, so pass the one ``get_ee_pose`` returned.  A
        target a fraction of a millimetre below the current palm is a valid
        move; the press is built from such steps.

        Args:
            position: (3,) palm XYZ in metres, world frame.
            quaternion_wxyz: (4,) WXYZ; ignored on this task.
            z_approach: (float) If positive, first move to position + this
                height, then descend. Default 0.0.

        Returns:
            dict with ``step`` (the episode clock, 120 ticks per simulated
            second).
        """
        return super().goto_pose(position, quaternion_wxyz, z_approach)

    def move_delta(
        self,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
        drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
        coord: str = "world",
    ) -> dict:
        """Move the palm by a delta.

        Limits per call: max 0.10 m translation per axis; a delta below 2 mm on
        every axis raises instead of moving, so this call cannot step a press
        finer than 2 mm -- use ``goto_pose`` with an absolute height for that.
        Rotation is not available on this task -- the wrist is fixed -- so any
        nonzero ``drx`` / ``dry`` / ``drz`` raises.

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
        return super().open_gripper()

    def close_gripper(self, position: float) -> dict:
        """Bring the index finger down by a commanded opening.

        The opening is the finger's travel: 1.0 leaves it straight, 0.0 drives
        it fully down to the press angle.  The close is ramped -- the finger
        walks to the command in small steps rather than snapping there -- and a
        key under it stops the finger where it stands.  This task has no
        gel-depth or force target on this branch: the press itself is built
        from small palm descents, with ``describe_tactile`` after each one.

        Args:
            position: (float) Finger opening, 0.0 (fully pressed) to 1.0
                (straight).

        Returns:
            dict with ``step`` (120 ticks per simulated second).
        """
        return super().close_gripper(position)

    def describe_tactile(
        self,
        side: str = "left",
        prompt: str = (
            "Describe the object in the following tactile video(s)."
        ),
    ) -> dict:
        """Describe what the index fingertip gel feels, in words.

        The hand carries one gel, on the index, and it is served as the left
        pad; ``side="left"`` is the reading.  The gel image goes to the
        Octopi-1.5 visual-tactile model, which describes the surface and its
        compliance -- hard or soft, smooth or textured, and how firmly the pad
        is loaded.  It does not return a force in newtons.

        On this task a reading belongs to each step of the press, not only to
        a decision point: read the unloaded pad, then read again after each
        0.1 mm descent and compare.  The description is the only tactile
        channel the policy has.

        Args:
            side: ("left") Which fingertip to read.  The hand has no right
                pad, so ask for the left one.
            prompt: (str) Instruction sent to Octopi.  For a forced choice,
                ask the question and name the alternatives -- a question with
                two candidate answers is easier to act on than a free
                description.

        Returns:
            dict with:
            step (int -- physics ticks since reset, same clock as motion
                primitives);
            description (str -- Octopi's text description of the surface;
                empty string when the Octopi service is unavailable);
            octopi_seconds (float -- wall-clock time inside the Octopi service,
                for latency accounting; 0.0 on fallback).
        """
        return super().describe_tactile(side=side, prompt=prompt)

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
        return self._get("/piano_key_position", params={"midi": int(midi)})
