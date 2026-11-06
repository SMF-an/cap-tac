---
id: R15
name: strap-hold-creeps
family: protocol
summary: Hold the height and the clock, not the description — a mid-hold description is not a load reading, and the drift recovers.
task_types: []
tasks: [tension_strap]
channels: [text]
objects: the elastic strap's tab, held between the pads
evidence: >-
  The gel image keeps moving under a still hand even when the load does not:
  holding 2.5 s at 11.9 N dropped the marker reading 0.45 px while the tension
  fell 0.07 N, and the points after the hold landed on the hold-free ramp's
  line (19.30, 22.52 and 25.65 px at 12, 14 and 16 mm against 19.54, 22.56 and
  25.67).  The text arm reads that same image through Octopi, so a mid-hold
  description is built from a moving image; the protocol below is what keeps
  the creep out of the next judgement.
---

## How

A hold is the same pose asked for again, not a wait: `describe_tactile` is a
GET, so it renders a gel and spends wall clock without advancing the
simulation.

1. Bring the hand to the height the stage needs and leave it there.
2. Re-issue `goto_pose` at that same position and quaternion.  A no-op motion
   advanced the episode by about 41 ticks when it was measured; stop when the
   `step` difference covers the hold (120 ticks a second, so 3 s is 360).
3. Read the description once after the hold, not during it.

The description may sag while the hand stands still — the tab keeps creeping
between the pads — but the load does not follow it.  Do not correct on a
mid-hold description, and do not re-zero after the hold: the sag recovers on
its own, and treating it as the new baseline bakes it in.

## Traps

- **A description is not a hold move.**  A hold made of `describe_tactile`
  calls never advances the 1.5-3 s the task scores.
- **A sagging description is not a loosening strap.**  If the height has not
  moved, the load has not moved; the text is reading the creep.
- **Do not chase the stage's target with the description.**  The stage is
  counted from the height you set; correcting mid-hold turns a good hold into
  a real release.
