from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with GelSight Mini tactile sensors
on both fingertips. The API is described below.

Goal: exactly one red block stands within reach. Decide by touch whether it is
soft or hard, then place it on the matching plate:
  - Soft block -> blue_plate
  - Hard block -> yellow_plate

Read the block with describe_tactile(), which returns a natural-language
description of what the gel feels.  The second block is parked out of reach,
so the decision has to come from this one.

The task is scored on where the block ends up: on the right plate, within 4 cm
horizontally and 1 cm vertically, upright within 15 degrees.  The block does
not have to be released — the score is its resting pose, and a gripper left
closed on it is not excluded.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import time

# One grip is the measurement: the gel's description carries the compliance
# (soft / hard / flexible).  Command the close by position, not by a depth
# target -- this task runs with adaptive grasping off, and a soft block yields
# until the jaws stop.
GRIP_LIGHT = 0.10
GRIP_FIRM = 0.06
SETTLE_S = 0.4

grasp_pos, grasp_quat = sample_grasp_pose("block")
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)

close_gripper(position=GRIP_LIGHT, adaptive=False)
time.sleep(SETTLE_S)
close_gripper(position=GRIP_FIRM, adaptive=False)
time.sleep(SETTLE_S)
move_delta(dz=0.05)

result = describe_tactile(
    side="both",
    prompt="Is this object hard or soft? Answer with one word: hard or soft.",
)
description = result["description"].lower()
print(f"Tactile description: {description}")

if "soft" in description:
    label = "soft"
else:
    label = "hard"
print(f"Classified as: {label}")


def move_by(dx=0.0, dy=0.0, dz=0.0):
    \"\"\"move_delta caps each axis at 0.10 m per call, so chunk long moves.

    It also has a 2 mm FLOOR: a smaller delta is rejected, not executed.  So
    stop once the remainder is under that, and leave the last millimetre or
    two unmoved -- the placement tolerance is 4 cm.
    \"\"\"
    for key, val in (("dx", dx), ("dy", dy), ("dz", dz)):
        while abs(val) > 0.002:
            step = max(-0.05, min(0.05, val))
            move_delta(**{key: step})
            val -= step


# --- Place on the matching plate ---
plate = "blue_plate" if label == "soft" else "yellow_plate"
plate_pos, _, plate_ext = get_object_pose(plate, return_bbox_extent=True)
rest_z = plate_pos[2] + plate_ext[2]          # plate top; origins sit at the bottom face

# Carry the block across with world-frame deltas: it is held rigidly, so moving
# the hand by d moves the block by d.  Deliberately NOT goto_pose: re-commanding
# the wrist to an absolute orientation while the block touches down twists it
# inside the grip.
block_now, _, _ = get_object_pose("block")
move_by(dx=plate_pos[0] - block_now[0],
        dy=plate_pos[1] - block_now[1],
        dz=rest_z + 0.002 - block_now[2])

# Seat it: step the descent down until the block rests on the plate.
for _ in range(12):
    p, _, _ = get_object_pose("block")
    gap = p[2] - rest_z
    if gap <= 0.002:
        break
    move_delta(dz=-max(gap, 0.002))

# Stop here, still holding the block.  Do NOT open the gripper: when the jaws
# come apart they drag the block off the plate or tip it over.  The task is
# scored on where the block ends up, not on whether the hand let go.
check_success()
"""


class UniVTACHardnessClassifyCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE
