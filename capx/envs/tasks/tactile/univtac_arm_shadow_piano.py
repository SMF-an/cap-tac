import copy

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.serving.piano_spec import (
    force_target,
    goal_sentence,
    load_calibration,
    sample_sequence,
)

PROMPT = """
You are controlling a Franka arm carrying a five-finger Shadow Hand. Only the
index finger moves; every other hand joint is welded, and the wrist's
orientation is fixed. The API is described below.

Goal: {goal}

get_key_position(midi) returns the two palm targets for a key -- `hover`, 40 mm
above it, and `press`, the height at which the fingertip meets it -- and
goto_pose moves between them; pass back the quaternion it returns.

To play a note: goto_pose to `press`, then close_gripper(press_force_N=...)
with the force that note's dynamic asks for.  That one call rises to the
approach pose, closes the index onto the key, and lowers the palm until the
index gel's force estimate reaches the target; it holds the band for 0.2 s and
returns the estimate it stopped at.  The task's tactile skill carries the
target for each dynamic and what the estimate does at the top of its range.

The score is the key's measured force: a note counts only if the right key is
held inside its band for 0.12 s and released before the next one, and a force
above the band for more than a moment ends the run.  Never ask for more than
the skill's target.

Time advances only while a motion or a press is being commanded.

A wrong key ends the episode, and a note that is never released does not count.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_TEMPLATE = """
import math

# =============================================================================
# arm_shadow_piano v3 -- play each note with the gel's calibrated force servo.
#
# The hand reports no force, but the index gel's image carries it: the contact
# moment tilts the whole gelpad, and a model fitted on measured presses turns
# five image statistics plus that tilt into newtons.  A press is therefore
# servoed on the estimate itself -- close_gripper(press_force_N=...) -- rather
# than on a palm depth.  The estimate separates forces up to ~0.43 N and
# saturates there; a heavy target is served by the pad's tilt plateau instead,
# which the servo does on its own.
#
# Nothing here reads progress: the program plays the notes in order.
# =============================================================================

NOTES = __NOTES__            # [(midi, "light"/"heavy", target force N), ...]

start = get_ee_pose()
Q = start["quaternion_wxyz"]


def depth_mm():
    return float(get_tactile_data()["left_depth_mm"])


def press(midi, dynamic, target_N):
    key = get_key_position(midi)
    hover, at_key = key["hover"], key["press"]
    here = get_ee_pose()["position"]
    lift = max(float(here[2]), float(hover[2]) + 0.01)
    goto_pose([here[0], here[1], lift], Q)
    goto_pose(hover, Q)
    goto_pose(at_key, Q)

    # One call: approach, close, and the palm descent servoed on the gel's
    # force estimate.  It holds the band for 0.2 s, which is what the score
    # wants, and returns the estimate it stopped at.
    result = close_gripper(press_force_N=target_N)
    print("note %d (%s): gel reads %.3f N (target %.2f)"
          % (midi, dynamic, float(result.get("left_force_N", -1.0)), target_N))

    # Unload before opening.  Opening the finger while the palm still holds
    # the key drags the pad across it and the contact force spikes for a step
    # or two -- on a heavy note that spike reached 0.58 N against a 0.57 N
    # over-force line.
    goto_pose([at_key[0], at_key[1], at_key[2] + 0.004], Q)
    try:
        open_gripper()
        goto_pose(hover, Q)
    except RuntimeError:
        # The last note's release ends the episode; nothing after it can be
        # commanded.  That is the finish, not a missed note.
        pass
    print("  released: gel depth %.2f mm" % depth_mm())


for midi, dynamic, target_N in NOTES:
    press(int(midi), str(dynamic), float(target_N))

print("episode finished; the harness scores it")
"""


def render_oracle(sequence, calibration) -> str:
    """Fill the seed's notes and each dynamic's force target into the template."""
    notes = [
        (int(midi), str(dynamic), force_target(calibration, str(dynamic)))
        for midi, dynamic in zip(sequence.midi, sequence.dynamics)
    ]
    return ORACLE_TEMPLATE.replace("__NOTES__", repr(notes))


class UniVTACArmShadowPianoCodeEnv(CodeExecutionEnvBase):
    """The shadow-hand piano task with the note sequence its trial was given.

    The sequence is a property of the seed, so it cannot be a class constant:
    the goal the model reads and the oracle's note list are re-rendered in
    ``reset`` from the trial id, which is the same seed the server drew the
    task's notes from (capx/serving/piano_spec.py).
    """

    prompt = PROMPT.format(goal=goal_sentence(sample_sequence(0)))
    # No progress or score tool is delivered.  The model plays the sequence in
    # order and verifies each press through the gel, as it plans any other
    # task.  The force-to-palm-depth mapping is this task's skill.
    tactile_skills = ("piano-gel-force-band",)

    def __init__(self, cfg) -> None:
        self._sequence = sample_sequence(0)
        super().__init__(cfg)

    @property
    def oracle_code(self) -> str:
        return render_oracle(self._sequence, load_calibration())

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sequence = sample_sequence(0 if seed is None else seed)
        self._task_prompt = PROMPT.format(goal=goal_sentence(self._sequence))
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",
             "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]
        obs["full_prompt"] = copy.deepcopy(self._full_prompt)
        info["task_prompt"] = self._task_prompt
        return obs, info
