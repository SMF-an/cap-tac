---
id: A25
name: reach-band-at-ball-height
family: act
summary: The arm cannot put the pads on a ball's centre at the box's near or far edge; there they stop above every ball.
task_types: []
tasks: [blind_pick_balls]
channels: []
objects: the twelve balls in the blind box
evidence: >-
  Measured 2026-10-24 against the live seed-1 server (hand straight down, pads
  80 mm apart) by commanding the flange to 0.16 m: the pad centre reaches
  0.043 m from about x = 0.390 m to x = 0.624 m, and stalls higher outside it
  — 0.0670 m at x = 0.365, 0.0523 at 0.375, 0.0476 at 0.380, and 0.1116 at
  0.655 — with the hand coming back 2-3 cm east.  The same 0.365 command
  stopped at 0.067-0.068 m on all three rows, so the limit is a function of x,
  not of the row.
---

## How

The arm's reach ends at the box's near and far edges before the box's own
walls do.  Measured with the hand pointing down, the pad centre can be put on
a ball's centre (z = 0.050 m) only between about x = 0.39 m and x = 0.62 m.
At the near edge the hand trades depth for horizontal travel — commanded
lower, it comes back higher and further east — and at the far edge it stops
short the same way.

Outside the band the pads stop above every ball, whatever a ball below is
doing.  That stop is the arm, not a contact: it does not mean the ball is not
there.

## Traps

- A refused descent or a hand that comes back short near an edge has not found
  a wall in the box; it has reached the arm's limit
  ([[search-continues-after-miss]]).
- The band is a function of x only — the three rows stop at the same height on
  the near edge.
