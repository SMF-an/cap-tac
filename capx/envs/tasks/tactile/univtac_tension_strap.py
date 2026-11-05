import copy

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.serving.tension_strap_spec import (
    goal_sentence,
    hold_moves,
    sample_sequence,
)

PROMPT = """
You are controlling a Franka Emika robot with GelSight Mini tactile sensors
on both fingertips. The API is described below.

Goal: {goal}. A hold
counts only if it stays within half a newton of its target. The strap is the
only thing in the scene, and it is called "elastic_strap".

The simulation advances only while the hand is being commanded; waiting passes
no time at all, so a hold is made of commands.

The arm's own travel is not a measure of the load. How far this strap
stretches for a given force is a property of the strap, and it is different
every episode, so a plan made of distances cannot be this task. What does not
change is that the load passes through the fingers: the pads carry it, and the
gel is therefore the only thing here that can say how hard the strap is being
pulled.

The strap is elastic, so the load is set by where the hand is and by nothing
else - the pull raises it and backing off lowers it by as much. There is no
second lever and nothing to press against, and nothing is scored on arriving:
the score is the two holds.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import math

# --- Where the tab is ---------------------------------------------------------
# The strap is added at base [0.55, 0, 0.005] and the tab -- the block at its
# top, the only part the pads can close on -- is centred 0.174 above that base
# (envs/tension_strap.py:561,578).  The wrist sits GRIPPER_OFFSET along the
# gripper centre's own +z (the tool points down), so the flange that puts the
# pads on the tab's centre is the centre plus that offset.
STRAP = "elastic_strap"
STRAP_BASE = (0.55, 0.0, 0.005)
TAB_LOCAL_Z = 0.174
GRIPPER_OFFSET_M = 0.131
APPROACH_CLEAR_M = 0.05

# --- The grip -----------------------------------------------------------------
# The task frees the strap from its initialization fixture only once BOTH pads
# have acquired the tab (envs/tension_strap.py:604-609); until then the strap
# stays pinned to its rest shape however hard it is pulled, so a pull measures
# the fixture rather than the strap.  The close is asked for the task's own
# depth, but the depth is NOT the test of whether the tab was taken: the
# release is read off the tactile manager's own minimum while the fingers are
# still moving, and the reading that settles afterwards is shallower -- both
# closes that worked returned 27.79 and 27.06 mm, on the wrong side of the
# task's own 27.7 mm line.  What separates a grasp from an empty close is the
# marker field: an empty close moves the markers a few thousandths of a pixel
# and a real grasp moves them several.
GRASP_DEPTH_MM = 27.6
# A depth close lands somewhere new on every run -- 27.31/26.99, 27.39/27.10,
# 27.70/27.38, 27.80/27.32 for one command -- and the calibration belongs to
# where it lands.  A close asked for an OPENING lands the pads at 25.70/25.25
# every time instead, on four straps running, so the opening is what this
# oracle asks for.
CLOSE_OPENING = 0.22
CONTACT_PX = 1.0

# --- The two stages -----------------------------------------------------------
# Drawn from the seed this server serves, and rendered in below; see
# capx/serving/tension_strap_spec.py for the draw and why it is that narrow.
TARGETS_N = __TARGETS__

# --- The gel as the force sensor ----------------------------------------------
# What carries the load is the tab slipping down between the pads, and what
# measures it is the gel's marker field.  The two pads are mounted mirrored,
# so the same physical slip appears with OPPOSITE sign in the two marker
# frames: as the strap is pulled, left_marker_shear_dx climbs and
# right_marker_shear_dx falls by the same amount, while anything that moves
# both the same way (a change of squeeze, a drift of the whole hand) cancels
# in the difference.
#
# Measured on four seeds by raising the hand in 2 mm steps from the release
# point: the difference is a straight line in tension at 0.93 N per px on all
# four (0.928, 0.932, 0.936, 0.947 -- 2% apart), residual 0.1 N, a fifth of
# the half-newton band this task is scored on.  The depth readings do not move
# at all (the tab is dragged along the pads, not pushed into them).
GEL_N_PER_PX = 0.93

# What is NOT constant across seeds is where zero sits.  That offset is the
# gel's own gain -- pixels of slip per millimetre the hand rises -- times the
# slack in the strap, and the gain follows the modulus the seed draws: 2.09
# px/mm on a 0.35 MPa strap against 1.62 on a 0.26 one, which is a pixel of
# take-up and most of a newton.  The slack itself does not move: 2.27, 2.28,
# 2.32 and 2.31 mm on those four straps.  So the slack is written down here in
# millimetres and the gain is measured from two rises, never carried in.
SLACK_M = 0.0023
PROBE_M = 0.002

# --- How the hand moves -------------------------------------------------------
# A 2 mm rise is about 4 N on this strap, far too coarse to land inside
# +-0.5 N, so each stage is a coarse run that measures the strap's own
# stiffness, one secant jump, and a short correction loop.  The stiffness is a
# property of the seed -- E is drawn from U(0.25, 0.40) MPa -- which is why it
# is measured here instead of written down: seed 0 came out at 2.12 N/mm and
# the next seed needs whatever it needs.
COARSE_M = 0.002
COARSE_STOP_MARGIN_N = 2.5
MAX_CORRECTIONS = 6
CORRECTION_CAP_M = 0.0005
SETTLE_N = 0.15

# --- The hold -----------------------------------------------------------------
# A hold is not a sleep.  Nothing advances the simulation but a motion call, so
# holding means asking for the pose the hand already holds, over and over, and
# the hold is counted in those calls.  One such call advanced the episode by
# 41 physics ticks -- a 31-step plan plus the 10 the server always delays --
# against a 120 Hz clock, so a 3 s stage is 360 ticks and lands inside ten
# calls.  Fourteen is that with room, and overshooting is harmless (the stage
# is already counted) while undershooting loses the run.
#
# The hold does NOT close a loop, and that is deliberate.  Standing still, the
# strap keeps its load: measured over eight of these calls, the tension fell
# 0.05 N in 330 ticks, so 3 s costs about a tenth of a newton and a stage that
# starts on target stays on target.  The gel, however, does not stand still --
# the tab keeps creeping between the pads, and the same eight calls moved the
# marker difference by 0.47 px, half a newton of reading for no change in load.
# A loop that believed that would pull the hand down and turn a good hold into
# a real release.  The creep is handled where it belongs: the reading is
# re-anchored after each hold, at a point where the load is known.
HOLD_MOVES = __HOLD_MOVES__


def pads(reply):
    \"\"\"(left, right) gel depth in mm from a motion reply, or None.\"\"\"
    left, right = reply.get("left_depth_mm"), reply.get("right_depth_mm")
    if left is None or right is None:
        return None
    return float(left), float(right)


def taken(reply):
    \"\"\"Whether both pads are actually on the tab, by the markers, not the depth.\"\"\"
    left = reply.get("left_marker_displacement_px")
    right = reply.get("right_marker_displacement_px")
    if left is None or right is None:
        return False
    return float(left) > CONTACT_PX and float(right) > CONTACT_PX


def shear(reply):
    \"\"\"The difference of the two pads' shear vectors, in marker pixels.\"\"\"
    keys = ("left_marker_shear_dx", "left_marker_shear_dy",
            "right_marker_shear_dx", "right_marker_shear_dy")
    values = [reply.get(k) for k in keys]
    if any(value is None for value in values):
        return None
    return (float(values[0]) - float(values[2]),
            float(values[1]) - float(values[3]))


def slip_px(reply):
    \"\"\"How far the tab has slipped since the close, in marker pixels.\"\"\"
    vector = shear(reply)
    zero = state.get("zero")
    if vector is None or zero is None:
        return None
    return math.hypot(vector[0] - zero[0], vector[1] - zero[1])


def tension_n(reply):
    \"\"\"The strap's load as the gel measures it, in newtons.\"\"\"
    slip = slip_px(reply)
    return None if slip is None else GEL_N_PER_PX * slip + state["offset"]


# --- 1. Where to grasp --------------------------------------------------------
# The task's own grasp pose: the server builds it from the strap's base plus the
# tab's height, which is the same point the expert takes, and it is the pose the
# calibration was measured at.  Building it by hand instead -- the tab's height
# above the base, plus the flange offset -- misses it by four or five
# millimetres, and on an opening close that is the difference between pads at
# 25.70/25.25 mm and pads at 25.60/24.94, where the gel reads a different gain
# and, on 2026-11-05, a negative slope: the gauge said 3 N at a pose carrying
# more than eleven, and the pull went on until the strap tore.
grasp, q0 = sample_grasp_pose(STRAP)
grasp = (float(grasp[0]), float(grasp[1]), float(grasp[2]))
xy = (grasp[0], grasp[1])
grasp_z = grasp[2]
print("grasp pose %s" % (grasp,))

# --- 2. Take the tab ----------------------------------------------------------
goto_pose((xy[0], xy[1], grasp_z), q0, z_approach=APPROACH_CLEAR_M)
try:
    reply = close_gripper(position=CLOSE_OPENING)
except Exception as error:
    # A close that stopped short still leaves the fingers where the tab
    # stopped them, and the reading that comes back with the error describes
    # that state.  Losing the trial over the exception would throw away the
    # only information it carries.
    print("close raised: %s -- reading the stopped fingers instead" % error)
    reply = get_tactile_data()
held = taken(reply)
print("grip depths %s, tab taken: %s" % (pads(reply), held))

# One dict rather than four globals: every helper below writes the hand's
# height, the stiffness it has measured and the last gel reply, and a plain
# module-level assignment inside a function would shadow instead of update it.
state = {"z": grasp_z, "slope": None, "reply": reply, "offset": 0.0,
         "zero": shear(reply)}


def go(z):
    \"\"\"Move the hand to height z, remember the reply, and return it.\"\"\"
    state["z"] = float(z)
    state["reply"] = goto_pose((xy[0], xy[1], state["z"]), q0)
    return state["reply"]


def raise_to(target):
    \"\"\"Bring the gel reading up to `target` newtons, and report where it landed.

    The coarse steps exist to measure the strap, not to close in: the first is
    a fixed 2 mm, and every later one is sized from the stiffness the previous
    step revealed so that it lands just under the target.  The corrections
    then re-fit that stiffness from their own jumps, so a strap that stiffens
    as it stretches is followed instead of fought.
    \"\"\"
    start = tension_n(state["reply"])
    est = start
    if est is None:
        return None
    for step_index in range(12):
        if est >= target - COARSE_STOP_MARGIN_N:
            break
        # A strap still held by its fixture does not load no matter how far
        # the hand rises, so after three rises that bought nothing the reading
        # is about the fixture and the rest of the episode would only be a
        # longer way of measuring the same nothing.
        if step_index >= 3 and est - start < 0.5:
            print("the strap is not following the hand -- it never left its fixture")
            return None
        if state["slope"] and state["slope"] > 0.0:
            delta = (target - COARSE_STOP_MARGIN_N - est) / state["slope"]
            delta = min(max(delta, 0.0002), 0.005)
        else:
            delta = COARSE_M
        # Never downwards, for the reason the correction loop gives below.
        delta = max(delta, 0.0)
        est_next = tension_n(go(state["z"] + delta))
        if est_next is None:
            return None
        state["slope"] = (est_next - est) / delta
        est = est_next
    for _ in range(MAX_CORRECTIONS):
        error = target - est
        if abs(error) <= SETTLE_N:
            break
        slope = state["slope"] if state["slope"] else 2000.0
        # Only ever upwards.  The gel's field does not retrace its path: a
        # correction that goes back down and up again comes out short -- on
        # 2026-11-05 an episode that was free to swing four millimetres read
        # 20.88 px at a pose where a monotone ramp reads 22.52, and the stage
        # it landed carried 1.6 N more than its gauge said.  So overshooting a
        # stage is not corrected by going down; it is prevented by taking the
        # steps small.
        delta = min(max(error / slope, 0.0), CORRECTION_CAP_M)
        est_next = tension_n(go(state["z"] + delta))
        if est_next is None:
            return None
        if abs(delta) > 1e-6:
            state["slope"] = (est_next - est) / delta
        est = est_next
    return est


def hold(moves):
    \"\"\"Spend `moves` motion calls at the height the hand already holds.\"\"\"
    for _ in range(moves):
        go(state["z"])
    return tension_n(state["reply"])


# --- 3. Where zero sits, from the gel's own gain ------------------------------
# Two rises give pixels of slip per millimetre of hand rise; the strap's slack
# is 2.3 mm of that rise, so the take-up in pixels is the product and the
# offset is what is left when the line is asked for zero there.  It is
# measured here, not carried in, because it follows the strap the seed drew:
# 4.74 px on a soft strap against 3.73 on a firm one, a whole band apart.
if held:
    first = slip_px(go(grasp_z + PROBE_M))
    second = slip_px(go(grasp_z + 2 * PROBE_M))
    if first is None or second is None or second <= first:
        print("the gel did not see the two probing rises; carrying no take-up")
    else:
        gain = (second - first) / PROBE_M
        take_up = SLACK_M * gain
        state["offset"] = -GEL_N_PER_PX * take_up
        print("gel gain %.0f px/m, take-up %.2f px, offset %+.2f N"
              % (gain, take_up, state["offset"]))


if not held:
    print("ABORT: the pads did not reach the tab; a pull here measures nothing")
else:
    for index, target in enumerate(TARGETS_N):
        landed = raise_to(target)
        if landed is None:
            print("gel reading went missing; stopping")
            break
        print("stage %d: %.2f N at z %.4f (slope %.0f N/m)"
              % (index + 1, landed, state["z"], state["slope"] or 0.0))
        hold(HOLD_MOVES[index])
        held_n = tension_n(state["reply"])
        print("stage %d held; reading now %.2f N"
              % (index + 1, -99.0 if held_n is None else held_n))
    print("episode finished; the harness scores it")
"""


ORACLE_TEMPLATE = ORACLE_CODE


def render_oracle(sequence) -> str:
    """The oracle for one draw: its targets and its hold length, not constants."""
    first, first_hold, second, second_hold = sequence
    return (
        ORACLE_TEMPLATE
        .replace("__TARGETS__", "(%r, %r)" % (first, second))
        .replace("__HOLD_MOVES__",
                 "(%d, %d)" % (hold_moves((first_hold,)),
                               hold_moves((second_hold,))))
    )


class UniVTACTensionStrapCodeEnv(CodeExecutionEnvBase):
    """The strap task with the force/time sequence its trial was given.

    The sequence is a property of the seed, so it cannot be a class constant:
    the prompt, the instruction the model reads and the oracle's own targets
    are all re-rendered in ``reset`` from the trial id, which is the same seed
    the server configured the task's acceptance limits from.
    """

    prompt = PROMPT.format(goal=goal_sentence(sample_sequence(0)))

    def __init__(self, cfg) -> None:
        self._sequence = sample_sequence(0)
        super().__init__(cfg)

    @property
    def oracle_code(self) -> str:
        return render_oracle(self._sequence)

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
