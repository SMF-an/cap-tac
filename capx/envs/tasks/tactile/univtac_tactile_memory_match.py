from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with GelSight Mini tactile sensors
on both fingertips. The API is described below.

Task: tactile memory matching.  Three identical orange cans lie flat on the
table: one REFERENCE can at the back, two CANDIDATE cans in front of it
(candidate_left and candidate_right).  A green MATCH SLOT lies beside the row.
Pose names for get_object_pose: reference_object, candidate_left,
candidate_right and match_slot.

The hand starts above the reference can; sample_grasp_pose('reference_object')
returns the grasp it knows.  The candidates are identical cans, so reuse that
measured offset for them.

The cans look the same but differ in three independent ways: 2 surface
textures x 2 weights x 2 hardnesses.  One candidate is the reference's twin,
identical in all three; the other differs in at least one.  Read all three
cans by touch with describe_tactile(), using the same grip and the same
question on each so the descriptions can be compared, then place the
candidate whose feel is closest to the reference on the match slot.  Vision
cannot tell them apart — decide only from the tactile descriptions.

The task is scored on the cans' final places: the twin has to be released and
resting on the match slot, and the other candidate must not be placed.  The
check reads the can's footprint on the slot and whether the hand has let go;
it does not score the can's exact height or its orientation.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import re
import time

import numpy as np

# The match is made from what Octopi says about the same grip on each can.
# Motion is the scalar oracle's geometry: grasp the lying can by its side at
# the offset sample_grasp_pose reports for the reference, lift it clear, and
# set it down by watching its own height.
REF = "reference_object"
CANDS = ["candidate_left", "candidate_right"]
PAD_XY = np.array([0.42, 0.00])
CAN_REST_Z = 0.021          # can centre resting on the table
RELEASE_Z = 0.039           # can centre just above the pad / table rest
CRUISE_Z = 0.28             # travel height: above the cans, fingers open
GRIP_POS = 0.25             # firm position close: same on every can
QUESTION = (
    "Describe this can by touch in a few words: its surface texture, how hard "
    "or soft it feels, and whether it feels light or heavy."
)

# Property words, in opposite pairs.  A text that names neither side of a pair
# leaves that axis unknown, and an unknown axis never decides the match.
PAIRS = [
    ({"rough", "textured", "bumpy", "grooved", "ridged", "grainy", "coarse"},
     {"smooth", "flat", "plain", "polished"}),
    ({"hard", "rigid", "firm", "stiff", "solid"},
     {"soft", "flexible", "compliant", "yielding", "squishy", "springy"}),
    ({"heavy", "dense", "weighty"}, {"light"}),
]
VOCAB = set().union(*(a | b for a, b in PAIRS))


def move_world(dx=0.0, dy=0.0, dz=0.0):
    for key, val in (("dx", dx), ("dy", dy), ("dz", dz)):
        while abs(val) > 0.002:          # 2 mm is the smallest executable move
            step = max(-0.08, min(0.08, val))
            move_delta(**{key: float(step)}, coord="world")
            val -= step


def can_z(name):
    return float(get_object_pose(name)[0][2])


def descend_can(name, target_z):
    for _ in range(10):
        z = can_z(name)
        if z <= target_z + 0.003:
            break
        move_delta(dz=-min(z - target_z, 0.08))


def put_back(name):
    descend_can(name, CAN_REST_Z)
    move_delta(dz=0.005)
    open_gripper()
    move_delta(dz=0.08)


# ---- grasp geometry: the reference's own offset, reused on the twins --------
grasp_pos, grasp_quat = sample_grasp_pose(REF)
ref_pos, _, _ = get_object_pose(REF)
offset = np.asarray(grasp_pos, dtype=float) - np.asarray(ref_pos[:3], dtype=float)
print(f"grasp offset = {np.round(offset, 3)}")


def grasp(name, grip=GRIP_POS, x_shift=0.0):
    pos, _, _ = get_object_pose(name)
    gp = np.asarray(pos[:3], dtype=float) + offset + np.array([x_shift, 0.0, 0.0])
    goto_pose(np.array([gp[0], gp[1], CRUISE_Z]), grasp_quat, z_approach=0.0)
    goto_pose(gp, grasp_quat, z_approach=0.06)
    try:
        if grip is None:
            # The task's own firm close: the transport grip, where the can
            # only has to stay in the fingers through a lateral move.
            close_gripper(position=0.25)
        else:
            close_gripper(position=grip, adaptive=False)
    except Exception as exc:
        # A blocked position close raises, but the fingers stop on the can and
        # the grip is real; the caller's lift check decides whether it held.
        print(f"[grasp {name}] close stopped short: {exc}")


def probe(name):
    for grip, shift in ((GRIP_POS, 0.0), (0.20, 0.015), (0.20, -0.015)):
        grasp(name, grip=grip, x_shift=shift)
        z0 = can_z(name)
        move_delta(dz=0.03)
        time.sleep(0.4)
        z1 = can_z(name)
        if z1 - z0 >= 0.015:
            desc = describe_tactile(side="both", prompt=QUESTION)["description"]
            desc = desc.lower()
            print(f"[probe {name}] {desc[:180]}")
            put_back(name)
            return desc
        print(f"[probe {name}] the can did not rise ({z0:.3f} -> {z1:.3f})")
        open_gripper()
        move_delta(dz=0.06)
    print(f"[probe {name}] no reading")
    return None


def side(text, positive, negative):
    words = set(re.findall(r"[a-z]+", text))
    p = len(words & positive)
    n = len(words & negative)
    return 1 if p > n else (-1 if n > p else 0)


def profile(text):
    return tuple(side(text, pos, neg) for pos, neg in PAIRS)


def similarity(ref_text, cand_text):
    ref_prof = profile(ref_text)
    cand_prof = profile(cand_text)
    score = 0
    for a, b in zip(ref_prof, cand_prof):
        if a and b:
            score += 2 if a == b else -2
    # A small tie-break from the words the two descriptions share, so two
    # texts that name no property word are still compared somehow.
    ref_words = set(re.findall(r"[a-z]+", ref_text))
    cand_words = set(re.findall(r"[a-z]+", cand_text))
    score += 0.01 * len((ref_words & cand_words) & VOCAB)
    return score


texts = {}
for name in [REF] + CANDS:
    texts[name] = probe(name)

print(f"profiles: " + " ".join(f"{n}={profile(texts[n]) if texts[n] else None}"
                              for n in [REF] + CANDS))

scores = {}
for name in CANDS:
    if texts[REF] is None or texts[name] is None:
        scores[name] = -float("inf")
    else:
        scores[name] = similarity(texts[REF], texts[name])
print(f"scores: {scores}")

picked = max(CANDS, key=lambda n: scores[n])
print(f"picked {picked}")

# ---- carry the chosen can to the match slot and release it -----------------
# The measuring close can catch an end cap (the public grasp sits near the
# can's end), which holds a 3 cm lift but lets the can rotate and drop on the
# lateral move.  Transport therefore uses the task's firm close, and walks the
# grasp point along the can axis until a lift proves the grip.
transported = False
for grip, shift in ((None, 0.0), (None, 0.015), (None, -0.015),
                    (None, 0.03), (None, -0.03)):
    grasp(picked, grip=grip, x_shift=shift)
    z0 = can_z(picked)
    move_delta(dz=0.05)
    time.sleep(0.3)
    if can_z(picked) - z0 >= 0.02:
        transported = True
        break
    open_gripper()
    move_delta(dz=0.06)

if transported:
    move_delta(dz=0.10)                     # clear of the table before travel
    can = np.asarray(get_object_pose(picked)[0][:3], dtype=float)
    move_world(dx=float(PAD_XY[0] - can[0]), dy=float(PAD_XY[1] - can[1]))
    descend_can(picked, RELEASE_Z)
    move_delta(dz=0.005)                    # unload before opening
    open_gripper()
    move_delta(dz=0.08)

    # The env marks a placement after 10 stable steps; 2 mm is the smallest
    # executable move and the pair cancels out, so each pass advances two.
    for _ in range(15):
        if check_success():
            break
        try:
            move_delta(dz=-0.002)
            move_delta(dz=0.002)
        except Exception:
            break

print(f"Success: {check_success()}")
"""


class UniVTACTactileMemoryMatchCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE
