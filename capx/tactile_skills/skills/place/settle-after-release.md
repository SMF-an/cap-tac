---
id: P6
name: settle-after-release
family: place
summary: After opening, keep moving until the object rests — the support, not your commanded height, sets where it lands; a can on the match pad settles at ~0.031, above the pad's own surface.
task_types: [classify, insert, manipulate]
channels: []
objects: any
evidence: >-
  The opening contract is explicit that a placement is counted only once the
  object rests stably, which is a separate condition from the grip having
  released. 2026-11-07 gpt-6-astra trial 26 released its chosen can onto the
  match pad at z=0.033 — a can resting flat on that pad sits at about 0.031 —
  and then raised on its own check that demanded the pad's surface height
  (0.021), ending an episode in which the placement itself had worked.
copies:
  - docs/univtac_api_reference.md §3.4
---

## How

Issue a few small settling moves after the open and stop only once the object
has stopped changing. Two or three short steps is the usual shape; the point is
that the episode does not end on the frame where the fingers left.

## Traps

- **"The grip released" and "the object is placed" are different claims.** The
  contact flags answer the first, and nothing in the gel answers the second — a
  dropped object and a placed one read the same ([[tactile-cannot-locate]]).
- **The support decides the resting height.** Once released, the object stops
  where the support holds it — a can leaning on a pad, a block that settles a
  few millimetres high — not at the height you commanded. A settle check that
  requires the centre to match the target height, or the can's axis to match
  the pad's own surface, rejects placements the task counts; what it must show
  is that the object has stopped on the support.
- A settling object that is still moving when the episode ends is scored as it
  lies, which is usually not where it was aimed.
- If the object does not settle at all, it is going to keep rolling — the
  recovery is to lift, re-grasp and re-place rather than to wait longer
  ([[unload-before-open]]).
