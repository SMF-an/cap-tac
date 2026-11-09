---
id: R19
name: grasp-pose-per-can
family: act
summary: Each can's grasp pose comes from sample_grasp_pose — the server rebuilds it from that can's own pose every call, so it is right for all three cans.
task_types: []
tasks: [tactile_memory_match]
channels: []
objects: the three memory-match cans
evidence: >-
  2026-11-07 gpt-6-astra: trial 27 called it and reached [0.615 0.239 0.144]
  and [0.535 -0.346 0.144] — the expert pose on the reference and on
  candidate_left, the two cans it got to. Trials 26 and 28 built the pose
  themselves from get_object_pose plus half the bbox extent plus the wrist
  offset, and closed on air three times each: their
  targets were [0.72 0.258 0.175] and [0.68 0.24 0.172] against a can that
  wants [0.615 0.238 0.144], 65 to 105 mm past it in x.
---

## How

Ask for it by the public name — `sample_grasp_pose("reference_object")`,
`"candidate_left"`, `"candidate_right"`. The task has its own rule, so the
server rebuilds the pose from that can's pose on every call instead of caching
a hand-relative offset: it stays right after a can has been moved, and the
first-call caveat does not apply here.

## Traps

- **Do not build it from `get_object_pose` and the bounding box.** The can's
  pose is not its bbox corner and not its centre: it sits on the can's axis at
  one end cap, and the grasp point is 65 mm along the body from it, 8 mm below
  the axis. A pose assembled from the origin and half the extent misses in all
  three axes, and the miss is silent — the fingers close on air and every pad
  reads the 34 mm far plane ([[depth-grip-scale]]).
- **A missed grasp is not a reading.** Both pads at the far plane after the
  firm close means the can was never between the fingers; go back and re-query
  the pose rather than carrying the empty numbers forward
  ([[memory-match-one-grasp]]).
