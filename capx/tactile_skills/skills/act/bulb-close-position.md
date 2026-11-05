---
id: A18
name: bulb-close-position
family: act
summary: Close the bulb by position at about 0.40 — a close at its own width only grazes it, and the pads' description is the grip check.
task_types: []
tasks: [bulb_tightening]
channels: [text]
objects: the light bulb
evidence: >-
  The scalar arm closed the bulb on a depth target (force_threshold=27.2).  On
  the text arm the close is positional: 0.5 of the full 0.08 m opening puts the
  pads 40 mm apart — a 40 mm bulb's own width — and a 2026-11-04 close computed
  there (position~0.49) turned the hand through 12 helix iterations with the
  bulb's yaw and depth unchanged; the pads were sliding on the glass.  A second
  text-arm trial that day with position=0.42 fared the same (12/12 iterations,
  the bulb's yaw 0.791 -> 0.792); 0.40 is the next step down.  The glass is
  text-visible: a pad on it names a surface, a pad that has slipped names
  nothing.
---

## How

The close is positional on this task: the opening is a fraction of the pads'
full 0.08 m travel, so the 40 mm bulb just touches at 0.5.  Close to about
0.40 — the pads press into the glass and hold:

```python
close_gripper(position=0.40)
```

Then read the pads' description: on the glass both pads name a surface
(smooth, hard, glass); a pad that has slipped describes nothing or talks about
the image ([[description-needs-contact]]).  That is the grip check — the
bulb's pose cannot tell a light grip, because the bulb stands in its socket
either way.

If a pad goes empty, or describes the surface sliding under it while the block
turns, the grip is gone: open, re-approach in one call
([[one-call-approach]]), close firmer at 0.40 again.

## Traps

- **A close at the bulb's own width is a graze.**  On 2026-11-04 closes at
  `position~0.49` and `position=0.42` each turned the hand through 12 helix
  iterations with the bulb's yaw and depth unchanged; 0.40 is firmer.
- **The depth servo is not this arm's.**  `force_threshold`/adaptive closes are
  the scalar tactile layer's depth channel; this API's close takes position
  only, and a call without one raises.
- The grip can go while the bulb keeps turning: the pads slide on the glass.
  Re-read the description after the blocks, not only after the close.
