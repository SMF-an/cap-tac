from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with GelSight Mini tactile sensors
on both fingertips. The API is described below.

Goal: Grasp the prism, use touch to decide whether its surface texture is
rough or smooth, and place it on the matching pad:
  - Rough texture → orange_pad
  - Smooth texture → green_pad

Read the texture with describe_tactile(), which returns a natural-language
description of what the gel feels.  The task is scored on where the prism ends
up: within 2 cm horizontally and 1 cm vertically, upright within 15°.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import numpy as np

# --- Grasp at a light commanded grip ---
grasp_pos, grasp_quat = sample_grasp_pose("prism")
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
close_gripper(position=0.20)

# Lift clear of the table before probing
move_delta(dz=0.05)

# Firm up the grip so the texture registers on the gel image
close_gripper(position=0.15)

# --- Texture classification via Octopi language description ---
result = describe_tactile(
    side="both",
    prompt="Is this surface rough or smooth? Answer with one word: rough or smooth.",
)
description = result["description"].lower()
print(f"Tactile description: {description}")

if "rough" in description:
    texture = "rough"
else:
    texture = "plain"
print(f"Classified as: {texture}")

# --- Verify the prism is still held ---
prism_pos, _, prism_ext = get_object_pose("prism", return_bbox_extent=True)
if prism_pos[2] < 0.03:
    open_gripper()
    goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
    close_gripper(position=0.20)
    prism_pos, _, prism_ext = get_object_pose("prism", return_bbox_extent=True)

# --- Place on the matching pad ---
target_pad = "orange_pad" if texture == "rough" else "green_pad"
pad_pos, _, pad_ext = get_object_pose(target_pad, return_bbox_extent=True)

ee_pos = np.array(grasp_pos) + np.array([0.0, 0.0, 0.05])
offset = ee_pos - prism_pos
target_prism = np.array([pad_pos[0], pad_pos[1],
                         pad_pos[2] + pad_ext[2] - 0.001])
goto_pose(target_prism + offset, grasp_quat, z_approach=0.1)

for _ in range(10):
    p, _, _ = get_object_pose("prism")
    gap = p[2] - (pad_pos[2] + pad_ext[2])
    if gap <= 0.002:
        break
    move_delta(dz=-max(gap, 0.002))
open_gripper()
move_delta(dz=0.08)
check_success()
"""


class UniVTACGraspClassifyCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE
