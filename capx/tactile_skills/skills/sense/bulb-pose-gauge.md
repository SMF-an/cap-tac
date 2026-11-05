---
id: S27
name: bulb-pose-gauge
family: sense
summary: The pose is not the gauge — depth is the feed you commanded at a matched lead, and the pads say whether it matched.
task_types: [screw]
channels: []
objects: the light_bulb in its socket
evidence: >-
  The task's check_success reads only the bulb's pose: the yaw delta, the
  insertion depth, the thread top's clearance above the socket rim, the
  insertion-depth error against the lead, the centre drift, the tilt and the
  socket's own stability (ViTaForge envs/bulb_tightening.py:1252).  That is the
  verdict's channel, not the agent's — steering a block by the bulb's pose is
  reading the quantity the task is scored on.
---

## How

Keep your own account of what a block did.  The pose is the checker's channel;
on this line the run's decisions - direction, feed, depth, seat - are taken on
the pads, and the pose is used only to find the bulb before the first close.

- **Direction** - a short leg each way; the pads say which one takes the thread
  and which one lets the glass slide ([[description-needs-contact]]).
- **Lead** - per bulb.  Start from a nominal value, read the pads in the same
  block, and correct until the pad holds its patch while the thread turns
  ([[feed-must-match-lead]]).
- **Depth** - the feed you commanded at a matched lead, times the turns you
  turned.  Sum it yourself; there is no reading to take.
- **Seat** - where the pads stop reporting the thread taking the glass while
  the turns still accumulate; stop turning there.  The wrist's own budget is
  the other stop ([[wrist-limit-segmentation]]).

**The first block carries a feed too.**  A nominal lead is enough to start;
with `feed_lead_m` omitted the call runs the planner's path instead of the
helix servo, and a thread refuses that path.

## Traps

- **A block that "felt" like it turned is not evidence either.**  The pads are
  read for what the surface does under them - holding, sliding, gone - and that
  is the whole of the feedback; the pose is not consulted between blocks.
- The lead is per bulb: carrying a number from another seed feeds the wrong
  thread.
- A refused rotate is one block, not the run: wrap it, re-grasp and
  re-approach.  The episode ends only if the failure is left uncaught.
