---
id: R15
name: strap-hold-creeps
family: protocol
summary: Hold the height and the clock, not the reading — the load stays put while the reading drifts by most of a band and recovers, so never gate the hold on the reading; hold well past the duration.
task_types: []
tasks: [tension_strap]
channels: [shear]
objects: the elastic strap
evidence: Holding 2.5 s at 11.9 N dropped the reading 0.45 px while the tension fell 0.07 N, and the points after the hold landed on the line the hold-free ramp drew — 19.30, 22.52 and 25.65 px at 12, 14 and 16 mm against 19.54, 22.56 and 25.67. The task's own acceptance clears its window on any step whose tension is outside 0.5 N, so its expert asks for 13 calls where 2.5 s needs 8. Two 2026-11-07 seed 3 runs bracket the rule — one held the bare eight calls and scored zero with its first stage's clock incomplete, and the next exited on the gel's own in-band count and never exited — the same standing pose read 12.6 N, then 43, 77 and 91 N as the tab crept while the load had not moved, and the trial ran 2292 s to the harness's alarm with the task already accepted.
---

## How

A hold is a pose re-asked for, not a wait: nothing advances the simulation
while the hand is idle. One such call advanced the episode by about 41 physics
ticks — a 31-step plan plus the 10 the server always delays — against a 120 Hz
clock, so a 3 s stage is 360 ticks and lands inside ten calls, fourteen with
room; overshooting is harmless once the stage is counted, and the spare calls
are what let the hold survive a step that falls outside the band.

Keep the height and count the calls. Do not correct on the reading during the
hold, and do not re-zero after it — the drift comes back on its own, so
re-zeroing turns it into an offset. **Do not make the reading the exit
condition.** The creep runs one way while the hand stands still, so a timer
that needs the reading inside a band keeps resetting itself and the hold never
ends; the load is unchanged through all of it.
