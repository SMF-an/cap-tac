"""The tasks this library can route to, by type.

One list, imported by both the card bank and the core block, so a card
scoped to a task that does not exist fails at load instead of never firing.
The names are the simulator's own ``task_name`` values -- what the server
answers to and what :meth:`BaseTask._task_name` resolves -- not class names
or config file stems.

Scoping is by task, not by type, but the type is what a card is usually
written against ("the classify tasks all read a still gel"), and keeping the
grouping visible is what stops the list from drifting into a pile of names.
"""

from __future__ import annotations

#: task name -> type.
TASK_TYPES: dict[str, str] = {
    "weight_classify": "classify",
    "hardness_classify": "classify",
    "grasp_classify": "classify",
    "tactile_memory_match": "classify",
    "insert_hdmi": "insert",
    "insert_hole": "insert",
    "insert_tube": "insert",
    "blind_pick": "manipulate",
    "blind_pick_balls": "manipulate",
    "lift_bottle": "manipulate",
    "lift_can": "manipulate",
    "pull_out_key": "manipulate",
    "put_bottle_in_shelf": "manipulate",
    "tactile_transfer_rearrange_clean": "manipulate",
    "bulb_tightening": "screw",
    "wipe_vase": "wipe",
}

#: Tasks of each type.
TASKS_BY_TYPE: dict[str, tuple[str, ...]] = {
    kind: tuple(name for name, other in TASK_TYPES.items() if other == kind)
    for kind in dict.fromkeys(TASK_TYPES.values())
}
