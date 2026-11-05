---
id: R15
name: strap-hold-creeps
family: protocol
summary: A held strap's reading drifts by most of a band over the hold while the load stays flat — hold the height and the clock, not the reading.
task_types: []
tasks: [tension_strap]
channels: [shear]
objects: the elastic strap
evidence: Standing still at 11.9 N for 2.5 s, 328 physics ticks, the marker difference fell 0.45 px and the tension 0.07 N — the drift runs either way and it is not the load. What it also does is come back — the points after that hold landed on the same line the hold-free ramp drew, 19.30, 22.52 and 25.65 px at 12, 14 and 16 mm against 19.54, 22.56 and 25.67, and the two fits agree at 0.949 and 0.950 N per px with 0.16 and 0.09 N of residual. So the drift recovers by itself, which is the opposite of correcting for it — a rate subtracted by elapsed time put one stage 0.8 N high, and re-anchoring on the dipped reading bakes the dip in.
---

## How

A hold is made of motion calls, not of waiting: nothing advances the
simulation while the hand is idle, so a hold is the same pose asked for
again and again. About ten such calls is the three seconds a stage needs.

Hold the height, not the reading. The load a held strap carries does not
change on its own, so there is nothing for a hold-time loop to correct.

Do not re-zero after the hold. The drift it leaves behind recovers on its
own; measuring the next stage against the dipped reading instead bakes the
dip in as an offset, and on this band an offset is a pass or a fail.

## Traps

- **The creeping reading looks like a load that is rising.** Correcting on it
  drives the hand down and turns a good hold into a real release, and watching
  it against a band abandons a hold that was fine: on a 2026-10-31 run the
  gauge climbed 0.50 N in 287 ticks while the hand did not move, and the
  program's own +-0.4 N hold guard raised on it.
- **It is a settling, not a clock.** The drift does not accumulate from one
  stage to the next, so subtracting a per-tick rate from every later reading
  makes the next stage land most of a newton high. Re-zero where the load is
  known; do not carry a rate.
- **The creep is in the gel, not in the strap.** The physical tension is
  flat to 0.05 N over the same window, so a second reading does not mean the
  load moved ([[differential-across-states]]).
- **A hold that is cut short is not partially credited.** The stage needs its
  three seconds continuously, so more calls than the arithmetic minimum is
  the safe direction and the extra ones cost nothing.
