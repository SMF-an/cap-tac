"""Adapter serving ViTaForge's shadow-hand piano task.

``ViTaForge/env/tasks/arm_shadow_piano.py`` is not a ViTaForge ``BaseTask``.
It is a standalone Gymnasium environment: it owns its PhysX world, its own
differential-IK controller and its own scorer, and it has none of the manager
objects (``_robot_manager`` / ``_tactile_manager`` / ``_actor_manager``) that
every endpoint in ``launch_univtac_server.py`` reads.  Its constructor is
``Task(cfg, render_mode=...)`` and its ``TaskCfg`` has no ``step_lim``,
``robot``, ``use_adaptive_grasp`` or ``adaptive_grasp_depth_threshold``.

This module supplies exactly that missing surface, in the terms the endpoints
already speak.  Five things are worth writing down.

**The task has no gripper.**  A five-finger Shadow Hand is mounted on the
Franka flange and only the index finger moves; every other hand joint is
welded.  The finger is exposed through the gripper endpoints as a single
opening: ``percent`` 1.0 is the finger straight out, 0.0 is it fully flexed.
The opening the endpoints read back is the **commanded** fraction, because
there is no jaw to measure; the measured finger angle is not part of the
delivered API.  That keeps ``close_gripper`` / ``open_gripper`` meaning "put
the finger down" / "lift it" without teaching the endpoints a fourth
primitive.

**The task's action is not a pose.**  Its four numbers are
``(wrist_x_offset, wrist_y, wrist_z, index_flexion)``: the first is an offset
from the palm's reset x, the second and third are coordinates whose zero is
``palm_y = -0.09175`` and ``palm_z = 0.03``, and the fourth is the index joint
angle.  Poses cross this boundary as **palm world poses**, matching every other
task served here; both directions are converted in ``_action_from_world`` and
``palm_position``, so an oracle written against ``goto_pose`` never has to know
the task's internal frame.  The wrist orientation is fixed by the arm-hand
mount and is not part of the action, so a commanded quaternion is ignored --
the pose returned by ``get_ee_pose`` is the one to pass back.

**The index finger carries one GelSight Mini.**  The GelSight/UIPC backend
(``env/tasks/_index_tactile.py``) owns the gel-key contact *and* the key
mechanics for the white keys it can reach, so the task is served with
``tactile=True`` and ``tactile_sensor='gsmini'``.  The hand has a single pad,
so its readings are republished as cap-x's **left** pad: ``depth`` is the
sensor's height map in millimetres (~28.3 at the gel face, falling as the key
presses in), ``marker`` is the [2, M, 2] marker motion and ``rgb_marker`` the
marker image.  The ``right_*`` fields are missing (0.0 / False) because there
is no second pad -- a caller must not read them as a zero reading.  The key's
measured contact force, which is what the dynamics are scored on, is **not**
delivered as a tool: the model presses to the gel depth the task's calibrated
skill gives, and the pad is its only press reading.  On this backend that
force is the UIPC gel-key reaction, i.e. the same contact the pad reports, so
the gel is the physical channel either way.

**The episode clock is republished at 120 Hz.**  The piano runs at 240 Hz
(``physics_dt = 1/240``), while every API docstring, every other task and the
strap's hold arithmetic are built on 120 ticks per simulated second.  The
score itself reads the task's own 240 Hz clock directly; only the number the
API shows is halved, so a caller's ``step`` differences mean the same thing
here as on every other task.

**The camera is the task's own recording camera.**  ``single_finger_piano``
only spawns one when ``record_video`` is set, and its default pose is the
world origin; the builder turns it on and this adapter aims it at the
keyboard, matching the upstream runner's arm view.  It is a recording/observer
view -- the policy observation carries no image -- and ``/observation`` serves
it for the visual-differencing channel.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_F

from capx.serving.piano_spec import load_calibration

logger = logging.getLogger(__name__)

# The task's own action-space coordinate offsets (arm_shadow_piano._physics_step):
# desired palm = initial palm + (x, y + 0.09175, z - 0.03).
ACTION_Y_OFFSET_M = 0.09175
ACTION_Z_OFFSET_M = 0.03

# 120 Hz is the clock every API docstring is written in; the task steps physics
# at 240 Hz (single_finger_piano.TaskCfg.physics_dt).
API_CLOCK_HZ = 120.0

# Reached tolerance and dwell for a commanded palm position.  The task's own
# demonstrator accepts 0.5 mm laterally and 0.5 mm vertically before pressing,
# so the arm can hold this; 0.4 mm leaves room for a sub-millimetre correction
# (a 1 mm z step still moves) without calling a settled hand lost.
REACHED_M = 0.0004
STABLE_STEPS = 2
# A commanded move that physics blocks (a key under the finger) or the arm
# cannot hold must end the call, not spend the whole budget in place.
STALL_WINDOW = 8
STALL_TOL_M = 0.0001
# A move that stops this close to its goal, having covered a real part of the
# distance asked for, counts as executed and says where the hand stopped.
SETTLE_M = 0.003
STILL_FRACTION = 0.10
# One control step moves the palm a few millimetres; 300 is more than any
# single commanded move in this task needs.
MOVE_STEP_LIMIT = 300

# Finger settling.  The index actuator is a position servo; a command is only
# reported as landed when the joint is within this of it, or when the effort
# limit stops it there (a key at its hard stop).  The hold is what makes a
# press or a release spend enough simulated time to score.
FLEX_TOL_RAD = 0.01
FLEX_SETTLE_LIMIT = 20
FLEX_STALL_RAD = 0.0005
SETTLE_STEPS = 2
HOLD_STEPS = 6  # 0.2 s at 30 control steps/s -- more than the 0.12 s a note needs
# The finger is ramped, not stepped: the calibrated force-vs-depth curve is
# measured with an opening step of this size settled by this many control
# steps per command, and a step command spikes the force irreproducibly.
GRIPPER_RAMP_STEP = 0.05
GRIPPER_RAMP_SETTLE = 2
# Gel-depth servo: each control step closes the finger by this much opening
# (0.02 of full travel ~ 0.01 rad at the tactile profile's 0.50 press angle)
# and re-reads the pad.  The sweep is bounded because a target the key can
# never load stops at the closed end, not after a fixed budget.
DEPTH_SERVO_STEP = 0.01
DEPTH_SERVO_LIMIT = 120

# The GelSight Mini far plane, the same value every ViTaForge gsmini task
# declares.  The index gel reports its height map in millimetres against this
# scale (~28.3 with the pad unloaded); the right pad does not exist.
TACTILE_FAR_PLANE_MM = 34.0
ADAPTIVE_DEPTH_MM = 27.5

EPISODE_OVER_MESSAGE = (
    "The episode is over: the task has already ended, the simulation no "
    "longer advances and the hand did not move. Every later motion or "
    "gripper call returns this same error."
)

# Diagnostic record, one row per control step.  The dynamics are scored on a
# force this server does not otherwise log, so a run that misses a note
# cannot be explained after the fact without it.  Nothing reads it back.
TRACE_FILE = "physical_trace.jsonl"


def _quat_angle(q_a, q_b) -> float:
    """Absolute angle between two wxyz quaternions, radians.

    Kept for the orientation check a future piano variant with a movable
    wrist would need; the served arm variant's orientation is fixed and
    ``goto`` ignores the commanded quaternion.
    """
    a = np.asarray(q_a, dtype=np.float64).reshape(4)
    b = np.asarray(q_b, dtype=np.float64).reshape(4)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


class _RobotManagerShim:
    """The RobotManager surface the endpoints read, backed by the piano task."""

    def __init__(self, adapter: "PianoTaskAdapter") -> None:
        self._adapter = adapter
        self.device = torch.device(adapter.task.cfg.device)

    def get_ee_pose(self):
        return self._adapter.ee_pose()

    def get_gripper_center_pose(self):
        # There is no separate tool centre; the palm is the commanded frame.
        return self._adapter.ee_pose()

    def gripper_center_to_ee(self, centre):
        return centre

    def get_gripper_qpos(self) -> float:
        return self._adapter.gripper_qpos()

    def gripper_percent2qpos(self, percent: float) -> float:
        return self._adapter.gripper_percent2qpos(percent)

    def is_gripper_opening(self, target_qpos: float) -> bool:
        return float(target_qpos) > self.get_gripper_qpos()


class _TactileManagerShim:
    """``get_min_depth()`` over the index gel, in millimetres.

    The hand has one pad, not two; the value is repeated so a caller taking
    ``.min()`` gets the only reading there is instead of a fabricated right
    pad.  ``/tactile`` itself reports the right side as missing.
    """

    def __init__(self, adapter: "PianoTaskAdapter") -> None:
        self._adapter = adapter

    def get_min_depth(self) -> torch.Tensor:
        return self._adapter.gel_min_depth()


class _ActorManagerShim:
    """No actors: the keys are articulation bodies with no pose query."""

    actors: dict = {}


class PianoTaskAdapter:
    """BaseTask-shaped view of ``env.tasks.arm_shadow_piano.Task``."""

    def __init__(self, task, seed: int | None = None) -> None:
        self.task = task
        self._seed = seed
        self.plan_success = True
        self.last_motion_error = ""
        self.last_motion_note = ""
        self._last_frame: torch.Tensor | None = None
        self._gel_cache: tuple[int, dict] | None = None
        self._last_gel_depth: float | None = None
        # Frozen gel-image force model (piano_spec.CALIBRATION_FILE) and the
        # state a press needs to apply it: the no-contact depth frame taken at
        # the approach pose, its gradient, the recent area readings behind the
        # rate feature, and the last estimate for /tactile to report.
        self._force_calibration: dict | None = None
        self._reference_depth: np.ndarray | None = None
        self._reference_gradient = 0.0
        self._area05_history: list[float] = []
        self._estimate_step: int | None = None
        self._last_force_estimate: float | None = None
        self._last_force_quality = "no_reference"
        self._last_features: np.ndarray | None = None
        self._robot_manager = _RobotManagerShim(self)
        self._tactile_manager = _TactileManagerShim(self)
        self._actor_manager = _ActorManagerShim()
        # The observer camera is not the policy's; the non-privileged
        # perception path degrades to Nones when this is absent.
        self._camera_manager = None

        cfg = task.cfg
        # ``Task.__init__`` copies TaskCfg with dataclasses.replace, so a
        # field added before construction is dropped.  Set these on the live
        # config instead, as the ball adapter does.
        cfg.step_lim = int(round(cfg.episode_seconds / (cfg.physics_dt * cfg.action_repeat)))
        cfg.robot = _RobotNamespace(tactile_far_plane=TACTILE_FAR_PLANE_MM)
        cfg.use_adaptive_grasp = False
        cfg.adaptive_grasp_depth_threshold = ADAPTIVE_DEPTH_MM

        self._refresh_geometry()
        self._clock_scale = API_CLOCK_HZ * float(cfg.physics_dt)
        self._flexion = 0.0
        self._arm_goal = self._initial_palm.copy()

        self._trace_stream = None
        self._open_trace()

        logger.info(
            "Piano task served: %s (%s), episode=%.0fs, control %.0f Hz",
            " ".join(str(v) for v in task.sequence.midi),
            " ".join(str(v) for v in task.sequence.dynamics),
            cfg.episode_seconds,
            1.0 / (cfg.physics_dt * cfg.action_repeat),
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _refresh_geometry(self) -> None:
        """Cache the palm frame and the finger's full press angle."""
        task = self.task
        self._initial_palm = (
            task._initial_palm[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        self._palm_quat = (
            task._palm_quat[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        self._press_angle = float(task.cfg.press_joint_angle)
        self._palm = int(task._palm)
        self._index = int(task._index)

    def palm_position(self) -> np.ndarray:
        """Current palm position in the world frame, metres."""
        return (
            self.task.hand.data.body_pos_w[0, self._palm]
            .detach().cpu().numpy().astype(np.float64).reshape(3)
        )

    def palm_quaternion(self) -> np.ndarray:
        """Current palm quaternion, wxyz."""
        return (
            self.task.hand.data.body_quat_w[0, self._palm]
            .detach().cpu().numpy().astype(np.float64).reshape(4)
        )

    def _action_from_world(self, position, flexion: float) -> np.ndarray:
        """The task's four-number action that commands the palm to ``position``."""
        p = np.asarray(position, dtype=np.float64).reshape(3)
        d = p - self._initial_palm
        return np.array(
            [d[0], d[1] - ACTION_Y_OFFSET_M, d[2] + ACTION_Z_OFFSET_M, float(flexion)],
            dtype=np.float64,
        )

    def _world_from_action(self, x: float, y: float, z: float) -> np.ndarray:
        """The palm world position an action coordinate names."""
        return self._initial_palm + np.array(
            [x, y + ACTION_Y_OFFSET_M, z - ACTION_Z_OFFSET_M], dtype=np.float64
        )

    def _clip_action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clip a four-number action to the task's own box."""
        low = np.asarray(self.task.action_space.low, dtype=np.float64)
        high = np.asarray(self.task.action_space.high, dtype=np.float64)
        clipped = np.clip(action, low, high)
        return clipped, bool(np.linalg.norm(clipped - action) > 1e-6)

    def _clip_world_goal(self, position) -> np.ndarray:
        """The reachable palm position closest to ``position``, in the world frame."""
        action = self._action_from_world(position, self._flexion)
        clipped, _ = self._clip_action(action)
        return self._world_from_action(clipped[0], clipped[1], clipped[2])

    # ------------------------------------------------------------------
    # Physical trace (diagnostics only)
    # ------------------------------------------------------------------

    def _open_trace(self) -> None:
        leaf = "seed_unknown" if self._seed is None else f"seed_{int(self._seed)}"
        path = (Path(tempfile.gettempdir()) / "univtac_server"
                / "arm_shadow_piano" / leaf / TRACE_FILE)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            self._trace_stream = path.open("a", buffering=1)
        except OSError:
            logger.exception("Piano adapter: cannot write %s; tracing disabled", path)
            self._trace_stream = None
        else:
            logger.info("Piano physical trace: %s", path)

    def _close_trace(self) -> None:
        stream, self._trace_stream = self._trace_stream, None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    def _trace(self, event: str | None = None, **extra: object) -> None:
        """Append one row holding the channel the scorer reads.

        Never raises: a diagnostics failure must not change what the task does.
        """
        if self._trace_stream is None:
            return
        try:
            angles, forces = self.task._key_state()
            score = self.task.score
            touched = int(np.argmax(forces)) if len(forces) else 0
            row = {
                "step": int(self.task.step_count),
                "notes_completed": int(score.count),
                "target_midi": int(self.task.sequence.midi[min(score.count, score.repetitions - 1)]),
                "force_N": round(float(forces[touched]), 4),
                "gel_depth_mm": None if self._last_gel_depth is None
                else round(float(self._last_gel_depth), 4),
                # Only when a frame was rendered at this step: an estimate
                # from an older frame is about a pose the hand has left.
                "gel_force_N": (
                    None if (self._last_force_estimate is None
                             or self._gel_cache is None
                             or self._gel_cache[0] != int(self.task.step_count))
                    else round(float(self._last_force_estimate), 4)),
                "gel_force_quality": (
                    self._last_force_quality
                    if (self._gel_cache is not None
                        and self._gel_cache[0] == int(self.task.step_count))
                    else "stale"),
                "touched_key": int(touched + 21),
                "key_angle_rad": round(float(angles[touched]), 5),
                "finger_flexion_rad": round(float(self.task.hand.data.joint_pos[0, self._index]), 5),
                "palm": [round(float(v), 5) for v in self.palm_position()],
                "failure": str(score.failure or ""),
            }
            row.update(extra)
            if event is not None:
                row["event"] = event
            self._trace_stream.write(json.dumps(row) + "\n")
        except Exception:  # noqa: BLE001 - see the docstring
            logger.exception("Piano adapter: physical trace failed; disabled")
            self._close_trace()

    # ------------------------------------------------------------------
    # Interface the endpoints read
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        return self.task.cfg

    @property
    def step_count(self) -> int:
        """Physics ticks on the API's 120 Hz clock (the task runs at 240 Hz)."""
        return int(self.task.step_count * self._clock_scale)

    @property
    def choice(self):
        """What /reset logs as the episode's ground truth line."""
        return " ".join(
            f"{midi}:{dynamic}"
            for midi, dynamic in zip(self.task.sequence.midi, self.task.sequence.dynamics)
        )

    @property
    def failure(self) -> str:
        return str(self.task.score.failure or "")

    def ee_pose(self):
        """The palm pose, in the task's world frame."""
        from envs.utils.transforms import Pose

        return Pose(self.palm_position(), self.palm_quaternion())

    def gripper_qpos(self) -> float:
        """Commanded finger opening, 1.0 straight to 0.0 fully pressed.

        The endpoints compare this against ``gripper_percent2qpos`` to verify
        that a close/open did what it asked.  This task has no jaw to measure,
        so the opening reported is the command; the measured finger angle is
        not delivered.
        """
        return 1.0 - self._flexion / max(self._press_angle, 1e-9)

    def gripper_percent2qpos(self, percent: float) -> float:
        """The opening fraction a command names, on the same 0-1 scale."""
        return float(percent)

    def key_position(self, midi: int) -> dict:
        """Palm targets for one key: hover, and the height that meets it.

        The coordinates are the task's own demonstrator's: a white key is
        reached with the wrist at its reset x and the fingertip at action z 0;
        a black key is 70 mm forward and 15.5 mm higher.  ``y`` is the key's
        own centre less the 33 mm that puts the fingertip over its front.
        """
        midi = int(midi)
        if not 21 <= midi <= 108:
            raise ValueError(f"midi {midi} is outside the 88-key piano (21-108)")
        key = self.task.layout[midi - 21]
        x = 0.07 if key.black else 0.0
        y = float(key.position[1]) - 0.033
        press_z = 0.0155 if key.black else 0.0
        return {
            "midi": midi,
            "black": bool(key.black),
            "hover": self._world_from_action(x, y, 0.04).tolist(),
            "press": self._world_from_action(x, y, press_z).tolist(),
            "quaternion_wxyz": self._palm_quat.tolist(),
        }

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _refuse_if_over(self) -> bool:
        if not self.task._done:
            return False
        self.last_motion_error = EPISODE_OVER_MESSAGE
        return True

    def _sync_gelpad(self) -> None:
        """Follow the hand with the UIPC gelpad, once per control step.

        The upstream runner calls ``tactile.observation()`` after each control
        step, whose ``VisualTactileSensor.update`` first moves the gelpad to
        the finger's current pose and only then renders.  Our settle loops can
        command several control steps inside one API call, so without this the
        pad would lag the finger by the whole call and the gel-key force would
        be read off a stale contact.  The optics are not rendered here; that
        happens on demand in ``_gel_observations``.
        """
        tactile = getattr(self.task, "tactile", None)
        gelpad = getattr(getattr(tactile, "sensor", None), "gelpad", None)
        if gelpad is not None:
            gelpad.update(dt=self.task.cfg.physics_dt * self.task.cfg.action_repeat)

    def _command(self, goal_world, flexion: float) -> None:
        """One control step: hold the palm at ``goal_world`` and the finger at ``flexion``."""
        action = self._action_from_world(goal_world, flexion)
        action, clipped = self._clip_action(action)
        goal = self._world_from_action(action[0], action[1], action[2])
        self._arm_goal = goal
        if clipped:
            logger.debug("Piano adapter: action clipped to the task box")
        self._sync_gelpad()
        self.task.step(action)
        self._trace()

    def move(self, actions, tag: str = "move", is_save: bool = True,
             delay: bool = True, **_: object) -> bool:
        """Execute a list of ``envs.utils.atom.Action`` objects.

        Mirrors ``_base_task.move``'s contract for the endpoints: return False
        and latch ``plan_success`` when an action cannot be executed.  The
        server restores ``plan_success`` itself after a failure.
        """
        self.last_motion_error = ""
        self.last_motion_note = ""
        if not self.plan_success:
            return False
        for action in actions:
            kind = getattr(action, "action", None)
            if kind in ("move", "all"):
                pre = float(action.args.get("pre_dis") or 0.0)
                pose = action.target_pose
                if not self.goto(pose.p, pose.q, pre_dis=pre):
                    logger.warning("Piano adapter: motion failed (tag=%s)", tag)
                    self.plan_success = False
                    return False
            if kind in ("gripper", "all"):
                depth = action.args.get("gripper_depth_threshold")
                force = action.args.get("gripper_press_force_N")
                if not self.set_gripper(float(action.target_gripper_pos),
                                        depth, force):
                    logger.warning("Piano adapter: gripper action failed (tag=%s)", tag)
                    self.plan_success = False
                    return False
        return True

    def goto(self, p_ee, q_ee=None, pre_dis: float = 0.0,
             limit: int = MOVE_STEP_LIMIT) -> bool:
        """Move the palm to ``p_ee``, optionally via a world +z rise.

        ``q_ee`` is accepted for the shared call signature and ignored: the
        wrist's orientation is fixed by the arm-hand mount and is not one of
        the task's action dimensions.
        """
        p_ee = np.asarray(p_ee, dtype=np.float64).reshape(3)
        if not np.isfinite(p_ee).all():
            self.last_motion_error = "The commanded position is not finite."
            return False
        if pre_dis > 1e-6:
            if not self._goto_segment(p_ee + np.array([0.0, 0.0, pre_dis]), limit):
                return False
        return self._goto_segment(p_ee, limit)

    def _goto_segment(self, target, limit: int) -> bool:
        self.last_motion_note = ""
        goal = self._clip_world_goal(target)
        if float(np.linalg.norm(goal - np.asarray(target, float))) > 1e-6:
            logger.warning(
                "Piano adapter: palm goal %s clipped to the task workspace %s",
                np.round(np.asarray(target, float), 4).tolist(),
                np.round(goal, 4).tolist(),
            )

        start = self.palm_position()
        requested = float(np.linalg.norm(goal - start))
        stable = 0
        history: list[float] = []
        for _ in range(limit):
            if self._refuse_if_over():
                return False
            self._command(goal, self._flexion)
            now = self.palm_position()
            error = float(np.linalg.norm(now - goal))
            if error <= REACHED_M:
                stable += 1
                if stable >= STABLE_STEPS:
                    return True
                continue
            stable = 0
            history.append(error)
            if len(history) > STALL_WINDOW:
                recent = history[-STALL_WINDOW:]
                if max(recent) - min(recent) < STALL_TOL_M:
                    progress = float(np.linalg.norm(now - start))
                    covered = progress / requested if requested > 1e-9 else 1.0
                    if error <= SETTLE_M and covered >= STILL_FRACTION:
                        logger.info(
                            "Piano adapter: motion settled %.1f mm from the goal",
                            error * 1000.0)
                        self.last_motion_note = (
                            f"OK: the hand stopped {error * 1000.0:.1f} mm short "
                            "of the commanded point and would not go closer.")
                        return True
                    self.last_motion_error = (
                        f"The hand travelled {progress * 1000.0:.0f} mm and "
                        f"stopped {error * 1000.0:.1f} mm short of the commanded "
                        "point, and did not move again while the goal was held.")
                    return False
        return False

    def set_gripper(self, percent: float,
                    depth_threshold: float | None = None,
                    press_force_N: float | None = None) -> bool:
        """Drive the index finger to ``percent`` opening (1 straight, 0 pressed).

        With ``press_force_N`` the whole call is the force press instead (see
        :meth:`press_to_force`): it rises to the approach pose, closes the
        finger with the calibrated ramp, captures the no-contact depth frame
        and descends the palm until the gel's force estimate reaches the
        target -- or, above the gel's saturation point, until the pad's tilt
        plateaus.  ``percent`` is then ignored.

        With ``depth_threshold`` the close is the gel-depth servo instead: the
        finger keeps closing in small steps until the index gel's height map
        reads at most that depth, then holds.  The pad is the only contact
        sensor on this hand and the score is a key force; the task's tactile
        skill maps each dynamic to a depth band, so this is the press
        primitive.  The step is small and the settle stops on the first
        reading inside the band, so the force rises from below rather than
        crossing it.

        A changed position command ramps the opening in small steps: the
        calibrated force-vs-depth curve is measured with this ramp, and a step
        command lands the fingertip on the key with an impact whose force is
        not reproducible.  The ramp stops early when the joint stalls -- a key
        at its hard stop holds the finger short of the command.  The same
        command re-issued is a hold, and spends ``HOLD_STEPS`` control steps,
        which is how a press or a release accumulates the simulated time the
        score wants.
        """
        self.last_motion_error = ""
        self.last_motion_note = ""
        if press_force_N is not None:
            return self.press_to_force(float(press_force_N))
        if depth_threshold is not None:
            return self._servo_to_depth(float(depth_threshold))

        percent = float(np.clip(percent, 0.0, 1.0))
        target = self._press_angle * (1.0 - percent)
        same = abs(target - self._flexion) <= 1e-6

        if not same:
            opening = self._opening()
            direction = 1.0 if percent > opening else -1.0
            previous = float(self.task.hand.data.joint_pos[0, self._index])
            stalled = 0
            while (percent - opening) * direction > 1e-9:
                opening += direction * min(GRIPPER_RAMP_STEP,
                                           abs(percent - opening))
                opening = float(np.clip(opening, 0.0, 1.0))
                self._flexion = self._press_angle * (1.0 - opening)
                for _ in range(GRIPPER_RAMP_SETTLE):
                    if self._refuse_if_over():
                        return False
                    self._command(self._arm_goal, self._flexion)
                actual = float(self.task.hand.data.joint_pos[0, self._index])
                if abs(actual - previous) < FLEX_STALL_RAD:
                    stalled += 1
                else:
                    stalled = 0
                previous = actual
                if stalled >= 2:
                    self.last_motion_note = (
                        f"the finger stopped at {actual:.3f} rad of a "
                        f"{self._flexion:.3f} rad command")
                    break

        # A re-issued command is a hold; so is opening fully, which is the
        # release every note needs.
        hold = SETTLE_STEPS if not same else HOLD_STEPS
        if percent >= 0.999:
            hold = max(hold, HOLD_STEPS)
        for _ in range(hold):
            if self._refuse_if_over():
                return False
            self._command(self._arm_goal, self._flexion)
        return True

    def _opening(self) -> float:
        """Current commanded finger opening, 1 straight to 0 fully pressed."""
        return 1.0 - self._flexion / max(self._press_angle, 1e-9)

    def _servo_to_depth(self, threshold: float) -> bool:
        """Close the finger until the index gel reads at most ``threshold`` mm.

        The score is a key force, and that force is what the pad feels: the
        further the key presses into the gel, the lower the height map reads.
        The task's skill carries the two depth bands, so the model presses to a
        depth rather than to a force.  Every step is one small opening
        increment followed by a fresh gel reading, and the servo stops on the
        first reading inside the target -- from below, so the key force is not
        driven past the band by a coarse jump.
        """
        opening = self._opening()
        if self.gel_depth_mm() is None:
            self.last_motion_error = (
                "The index gel is not reporting a depth, so a gel-depth close "
                "cannot be servoed. Use a position close instead.")
            return False

        reached = False
        for _ in range(DEPTH_SERVO_LIMIT):
            # Settle the finger at the current command before reading: the
            # action filter and the contact are still moving one control step
            # after a command, and a depth read off a half-moved finger would
            # make the servo take another step and overshoot the band.
            for _ in range(SETTLE_STEPS):
                if self._refuse_if_over():
                    return False
                self._command(self._arm_goal, self._flexion)
            depth = self.gel_depth_mm()
            if depth is not None and depth <= threshold:
                reached = True
                break
            if opening <= 0.0:
                break
            opening = max(0.0, opening - DEPTH_SERVO_STEP)
            self._flexion = self._press_angle * (1.0 - opening)

        if not reached:
            depth = self.gel_depth_mm()
            self.last_motion_error = (
                f"The finger closed to opening {opening:.3f} and the index gel "
                f"still reads {depth:.2f} mm; the close target was "
                f"{threshold:.2f} mm. The key is not being pressed hard enough."
                if depth is not None else
                "The gel reading went missing during the close.")
            return False

        for _ in range(HOLD_STEPS):
            if self._refuse_if_over():
                return False
            self._command(self._arm_goal, self._flexion)
        return True

    # ------------------------------------------------------------------
    # Observation / rendering
    # ------------------------------------------------------------------

    def _gel_observations(self) -> dict:
        """The index gel's tensor outputs, refreshed once per physics tick.

        ``IndexTactile.observation()`` is what advances the optics; its return
        value carries the marker image only.  ``VisualTactileSensor``'s own
        ``get_observations`` then reads the same updated buffers for the height
        map (``depth``, mm) and the [2, M, 2] marker motion.  The cache is
        keyed on the task's physics step so one API call -- a motion reply, its
        before/after frames and ``/tactile`` -- renders the gel once, not once
        per reader.
        """
        tactile = getattr(self.task, "tactile", None)
        if tactile is None or getattr(tactile, "sensor", None) is None:
            return {}
        step = int(self.task.step_count)
        if self._gel_cache is not None and self._gel_cache[0] == step:
            return self._gel_cache[1]
        tactile.observation()
        raw = tactile.sensor.get_observations(["depth", "marker", "rgb_marker"])
        self._gel_cache = (step, raw)
        depth = raw.get("depth")
        self._last_gel_depth = None if depth is None else float(depth.min())
        return raw

    def gel_depth_mm(self) -> float | None:
        """Deepest point of the index gel's height map, millimetres."""
        raw = self._gel_observations()
        depth = raw.get("depth")
        return None if depth is None else float(depth.min())

    # ------------------------------------------------------------------
    # Gel force estimate (the frozen model in the calibration file)
    # ------------------------------------------------------------------

    def _calibration(self) -> dict:
        if self._force_calibration is None:
            self._force_calibration = load_calibration()
        return self._force_calibration

    def _depth_image(self) -> np.ndarray | None:
        """The index gel's height map in millimetres, ``(rows, columns)``."""
        raw = self._gel_observations()
        depth = raw.get("depth")
        if depth is None:
            return None
        return np.squeeze(depth.detach().cpu().numpy().astype(np.float64))

    @staticmethod
    def _gradient(image: np.ndarray, mask: np.ndarray) -> float:
        """Depth change per image column over ``mask``, mm/px.

        The contact moment tilts the whole gelpad; that tilt is the force
        channel, and it is a plane's x-slope.  Fitting it on the pixels both
        frames see keeps the silhouette's moving edge from dragging the plane.
        """
        rows, cols = np.nonzero(mask)
        if rows.size < 100:
            return float("nan")
        design = np.stack(
            [np.ones(rows.size), cols, rows], axis=1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(design, image[mask], rcond=None)
        return float(coef[1])

    def _features(self, image: np.ndarray) -> np.ndarray | None:
        """The model's six features for one frame, relative to the reference."""
        reference = self._reference_depth
        if reference is None or reference.shape != image.shape:
            return None
        limit = float(self._calibration()["valid_depth_mm"])
        mask = (reference < limit) & (image < limit)
        if int(mask.sum()) < 100:
            return None
        d = np.abs(reference - image)
        values = d[mask]
        return np.array([
            values.mean(),
            float((values > 0.01).mean()),
            float((values > 0.05).mean()),
            float((values > 0.20).mean()),
            0.0,
            self._gradient(image, mask) - self._reference_gradient,
        ])

    def _estimate(self, image: np.ndarray) -> tuple[float, str, float] | None:
        """``(force_N, quality, gradient)`` for one frame, or None without one."""
        feature = self._features(image)
        if feature is None:
            return None
        step = int(self.task.step_count)
        if step != self._estimate_step:
            self._area05_history.append(float(feature[2]))
            del self._area05_history[:-8]
            self._estimate_step = step
        if len(self._area05_history) >= 3:
            feature[4] = feature[2] - self._area05_history[-3]
        self._last_features = feature.copy()
        model = self._calibration()["model"]
        median = np.asarray(model["median"], dtype=np.float64)
        iqr = np.asarray(model["iqr"], dtype=np.float64)
        weight = np.asarray(model["weight"], dtype=np.float64)
        z = (feature - median) / iqr
        force = float(np.concatenate([[1.0], z, z * z]) @ weight)
        low = np.asarray(model["range_min"], dtype=np.float64)
        high = np.asarray(model["range_max"], dtype=np.float64)
        margin = 0.05 * (high - low) + 1e-6
        inside = np.all((feature >= low - margin) & (feature <= high + margin))
        self._last_force_estimate = force
        self._last_force_quality = "ok" if inside else "out_of_range"
        return force, self._last_force_quality, float(feature[5])

    def gel_force_estimate(self) -> tuple[float, str] | None:
        """The key force the gel reads, newtons, with a quality tag.

        None until a press has captured its no-contact reference.  ``ok`` when
        the reading sits inside the training envelope; ``out_of_range`` when
        it has left it -- a different press pose, a lost contact or a stale
        reference leaves the model extrapolating, and the number is then not
        to be trusted.  The estimate is what the press servo closes the loop
        on, so a caller can also read it between steps.
        """
        image = self._depth_image()
        if image is None:
            return None
        estimate = self._estimate(image)
        if estimate is None:
            return None
        force, quality, _ = estimate
        return force, quality

    def press_to_force(self, target_N: float) -> bool:
        """Press a key until the gel's force estimate reaches ``target_N``.

        The recipe is the one the model was fitted with, and it matters: the
        palm rises to the approach pose (``press.raise_mm`` above where the
        caller left it), the index closes to the press angle with the ramped
        close, ten control steps settle the gel, the no-contact depth frame is
        captured, and only then does the palm descend in ``press.step_mm``
        steps, re-reading the gel at each one.  A press that starts anywhere
        else -- the palm already down, the finger already closed against the
        key -- has no reference and must not be served: call it with the palm
        at the key's press pose and the finger open.

        Above ``saturation_N`` the estimate stops separating forces (the pad's
        tilt has bottomed out), so the target is served by the plateau rule
        instead: keep descending while the gradient still grows, stop when it
        stops.  The band that lands in is the calibration's heavy band, and
        the calibration file carries the fraction that defines "stops".
        """
        calibration = self._calibration()
        press = calibration["press"]
        self.last_motion_error = ""
        self.last_motion_note = ""

        # 1. Approach pose: the reference is only valid at the calibrated
        #    height, so the rise is relative and exact.
        approach = self.palm_position() + np.array(
            [0.0, 0.0, float(press["raise_mm"]) / 1000.0])
        if not self._goto_segment(approach, MOVE_STEP_LIMIT):
            return False

        # 2. Close the index onto the key with the calibrated ramp.
        if not self.set_gripper(0.0):
            return False
        for _ in range(int(press["settle_at_pose_steps"])):
            if self._refuse_if_over():
                return False
            self._command(self._arm_goal, self._press_angle)

        # 3. The no-contact frame, and the sample at the approach pose that
        #    the training sweep's first row is.
        reference = self._depth_image()
        if reference is None:
            self.last_motion_error = (
                "The index gel reported no depth at the approach pose, so no "
                "force estimate can be made. Use a position close instead.")
            return False
        limit = float(calibration["valid_depth_mm"])
        self._reference_depth = reference
        self._reference_gradient = self._gradient(reference, reference < limit)
        self._area05_history = []
        self._estimate_step = int(self.task.step_count)
        first = self._estimate(reference)
        if first is None:
            self.last_motion_error = (
                "The gel's contact area is too small to read at the approach "
                "pose; the key is not in front of the pad.")
            return False
        estimate, quality, gradient = first

        # 4. Descend until the target is reached (or the tilt bottoms out).
        plateau = bool(float(target_N) > float(calibration["saturation_N"]))
        plateau_cfg = calibration["plateau"]
        fraction = float(plateau_cfg["fraction"])
        min_samples = int(plateau_cfg["min_samples"])
        window = int(plateau_cfg["window"])
        gradients: list[float] = []
        z = float(self._arm_goal[2])
        approach_z = z
        reached = False
        estimate_N = estimate
        for _ in range(int(press["max_steps"])):
            if self._refuse_if_over():
                return False
            z -= float(press["step_mm"]) / 1000.0
            goal = self._arm_goal.copy()
            goal[2] = z
            for _ in range(int(press["settle_steps"])):
                if self._refuse_if_over():
                    return False
                self._command(goal, self._press_angle)
            image = self._depth_image()
            estimate = None if image is None else self._estimate(image)
            if estimate is None:
                break
            estimate_N, quality, gradient = estimate
            # Debug only: the estimate against the step it was taken at.  The
            # measured force is never read here -- the whole point is that the
            # press closes on the pad's reading alone.
            logger.debug("press step z=%.5f estimate=%.5f features=%s", z,
                         estimate_N,
                         np.round(self._last_features, 8).tolist()
                         if self._last_features is not None else None)
            gradients.append(gradient)
            if not plateau and estimate_N >= float(target_N):
                reached = True
                break
            if plateau and len(gradients) >= max(min_samples, window + 2):
                recent = np.abs(np.diff(gradients[-window:]))
                initial = np.median(np.abs(np.diff(gradients[:min_samples])))
                if np.all(recent < fraction * initial):
                    reached = True
                    break

        if not reached:
            self.last_motion_error = (
                f"The press reached {estimate_N:.3f} N on the gel estimate "
                f"(target {target_N:.2f} N) after {press['max_steps']} "
                f"{press['step_mm']} mm steps and neither the estimate nor "
                "the pad's tilt would move further; the key was not reached."
                if estimate_N is not None else
                "The gel reading went missing during the press.")
            return False

        self._last_force_estimate = estimate_N
        self._last_force_quality = quality
        self.last_motion_note = (
            f"OK: the gel reads {estimate_N:.3f} N against a {target_N:.2f} N "
            f"target, stopped {(approach_z - z) * 1000.0:.1f} mm below the "
            "approach pose."
            + ("  The estimate saturates above "
               f"{calibration['saturation_N']:.2f} N, so the press stopped "
               "when the pad's tilt plateaued." if plateau else ""))

        # 5. Hold: the score wants the band held, not touched.
        for _ in range(HOLD_STEPS):
            if self._refuse_if_over():
                return False
            self._command(self._arm_goal, self._press_angle)
        return True

    def gel_min_depth(self) -> torch.Tensor:
        """Deepest point of the index gel's height map, millimetres.

        The hand has one pad; the value is repeated so a caller that takes the
        minimum of the pair gets the only reading there is.
        """
        value = self.gel_depth_mm()
        if value is None:
            value = TACTILE_FAR_PLANE_MM
        return torch.tensor(
            [value, value], dtype=torch.float32,
            device=self.task.cfg.device,
        )

    def _get_observations(self) -> dict:
        """The shape every endpoint parses: the index gel as the left pad."""
        tactile = {}
        raw = self._gel_observations()
        entry = {key: raw[key] for key in ("depth", "marker", "rgb_marker")
                 if key in raw}
        if entry:
            tactile["left_tactile"] = entry
        return {
            "tactile": tactile,
            "observation": {"head": {"rgb": self._last_frame}},
        }

    def _update_render(self) -> None:
        """Refresh the observer camera and cache the frame."""
        if getattr(self.task, "camera", None) is None:
            self._last_frame = None
            return
        # The task's own hook syncs the UIPC gel's visual mesh before a render;
        # without it the observer camera can show the pad a step stale.
        before = getattr(self.task, "_before_render", None)
        if before is not None:
            before()
        self._last_frame = torch.as_tensor(self.task.render(), dtype=torch.uint8)

    def get_frame_shot(self, obs: dict) -> torch.Tensor:
        """Observer view plus the index gel: 320x480 scene + 160x160 pad."""
        canvas = torch.zeros((320, 640, 3), dtype=torch.uint8)
        if self._last_frame is not None:
            canvas[:, :480] = _resize_hwc(self._last_frame, 320, 480)
        gel = obs.get("tactile", {}).get("left_tactile", {}).get("rgb_marker")
        if gel is not None:
            canvas[:160, 480:] = _resize_hwc(gel, 160, 160)
        return canvas

    def _aim_observer_camera(self) -> None:
        """Put the recording camera where the upstream runner puts it."""
        camera = getattr(self.task, "camera", None)
        if camera is None:
            return
        keys_y = [
            float(self.task.layout[int(m) - 21].position[1])
            for m in self.task.sequence.midi
        ]
        target = [0.12, float(np.mean(keys_y)) if keys_y else 0.0, 0.095]
        eye = [-0.58, 0.85, 0.66]
        device = self.task.cfg.device
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=device),
            torch.tensor([target], device=device),
        )

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.plan_success = True
        self.last_motion_error = ""
        self.last_motion_note = ""
        observation, info = self.task.reset(seed=seed, options=options)
        self._refresh_geometry()
        self._gel_cache = None
        # The no-contact frame belongs to one press: a new episode starts
        # without one, and the estimate says so rather than reusing a frame
        # taken at another key.
        self._reference_depth = None
        self._reference_gradient = 0.0
        self._area05_history = []
        self._estimate_step = None
        self._last_force_estimate = None
        self._last_force_quality = "no_reference"
        self._flexion = float(self.task.hand.data.joint_pos[0, self._index])
        self._arm_goal = self._initial_palm.copy()
        self._aim_observer_camera()
        self._trace("reset", seed=seed, sequence=list(self.task.sequence.midi),
                    dynamics=list(self.task.sequence.dynamics))
        return observation, info

    def check_success(self) -> bool:
        return bool(self.task.score.success)

    def check_early_stop(self) -> bool:
        return False

    def check_success_detail(self) -> dict:
        return self.task._info()

    def close(self) -> None:
        self._close_trace()
        self.task.close()


class _RobotNamespace:
    """Stand-in for ``cfg.robot``, which this TaskCfg does not have."""

    __slots__ = ("tactile_far_plane",)

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _resize_hwc(image, height: int, width: int) -> torch.Tensor:
    """Bilinear resize of an HWC image tensor to (height, width), uint8."""
    if image is None:
        return torch.zeros((height, width, 3), dtype=torch.uint8)
    x = torch.as_tensor(image).detach()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, 3)
    x = x[..., :3].to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    x = torch_F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
