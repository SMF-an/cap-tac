from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with GelSight Mini tactile sensors
on both fingertips. The API is described below.

Goal: exactly one block stands within reach. Decide by touch whether it is
light or heavy, then place it on the matching plate:
  - Light block -> blue_plate
  - Heavy block -> yellow_plate

Read the block with describe_tactile(), which returns a natural-language
description of what the gel feels.  The second block is parked far out of
reach, so there is no reference object to compare against.

The task is scored on where the block ends up: on the right plate, within 4 cm
horizontally and 1 cm vertically, upright within 15 degrees.  The block does
not have to be released — the score is its resting pose, and a gripper left
closed on it is not excluded.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import time

# One grip is the measurement: the gel's description carries the load.  A rigid
# block pins the fingers, so the depth servo lands on the same reading whatever
# the block weighs -- the weight has to come off what the pads feel while it
# hangs.  ~28.5 is light enough to let the hanging load shear the pads and
# still firm enough to carry the block.
GRIP_LIGHT = 0.20
GRIP_FIRM = 0.19            # a touch firmer if the light grip drops it
SETTLE_S = 0.4


def grasp_and_lift(position):
    grasp_pos, grasp_quat = sample_grasp_pose("block")
    goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
    close_gripper(position=position)

    # Hang the block clear of the table: one still resting on it loads nothing.
    for _ in range(3):
        move_delta(dz=0.01)
        time.sleep(SETTLE_S)


grasp_and_lift(GRIP_LIGHT)

# A block that never left the table was dropped, not read: a close at 30 mm
# sits above the gel face and the 180 g block slips out.  Check the lift, not
# the reading -- a dropped block describes as light because nothing hangs.
block_pos, _, _ = get_object_pose("block")
if block_pos[2] < 0.02:
    print("block still on the table - retrying with a firmer grip")
    open_gripper()
    time.sleep(SETTLE_S)
    grasp_and_lift(GRIP_FIRM)

result = describe_tactile(
    side="both",
    prompt="Is this block light or heavy? Answer with one word: light or heavy.",
)
description = result["description"].lower()
print(f"Tactile description: {description}")

if "heavy" in description:
    label = "heavy"
else:
    label = "light"
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
plate = "yellow_plate" if label == "heavy" else "blue_plate"
plate_pos, _, plate_ext = get_object_pose(plate, return_bbox_extent=True)
rest_z = plate_pos[2] + plate_ext[2]          # plate top; origins sit at the bottom face

# Carry the block across with world-frame deltas: it is held rigidly, so moving
# the hand by d moves the block by d, with no need to know the hand's pose.
# Deliberately NOT goto_pose: re-commanding the wrist to an absolute
# orientation while the block touches down twists it inside the light grip --
# measured 11-45 deg of tilt that way against 0.2-0.3 deg this way.
# Horizontal first (the block is well clear of the 6 mm plate), then down.
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

# Stop here, still holding the block.  Do NOT open the gripper.  The block is
# only 1.8 g, so the gel holds on to it: when the jaws come apart they drag it
# off the plate or tip it over (8 failures in 11 tries on one seed, and neither
# pressing it down first nor re-gripping more gently helped).  The task is
# scored on where the block ends up, not on whether the hand let go, and
# ViTaForge's own expert places it the same way -- place_actor(is_open=False)
# with no open afterwards.
check_success()
"""


class UniVTACWeightClassifyCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE
