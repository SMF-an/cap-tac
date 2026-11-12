---
id: R15
name: strap-hold-creeps
family: protocol
summary: Hold the height and the clock, not the reading — the load stays put while the reading dips up to ~1.2 N and stays low next move; never gate the hold on it, do not abort, hold well past the duration, anchor the next stage to the pre-hold reading.
task_types: []
tasks: [tension_strap]
channels: [shear]
objects: the elastic strap
evidence: Holding 2.5 s at 11.9 N dropped the reading 0.45 px while the tension fell 0.07 N, and the points after the hold landed on the line the hold-free ramp drew — 19.30, 22.52 and 25.65 px at 12, 14 and 16 mm against 19.54, 22.56 and 25.67. The task's own acceptance clears its window on any step whose tension is outside 0.5 N, so its expert asks for 13 calls where 2.5 s needs 8. Two 2026-11-07 seed 3 runs bracket the rule — one held the bare eight calls and scored zero with its first stage's clock incomplete, and the next exited on the gel's own in-band count and never exited — the same standing pose read 12.6 N, then 43, 77 and 91 N as the tab crept while the load had not moved, and the trial ran 2292 s to the harness's alarm with the task already accepted. The 0.45 px case does not generalise — on 2026-11-11 seed 15 (a dedicated server, read through /force_state) the same path showed the dip at full size and not recovered on the next move — the physical tension held at 7.44-7.47 N through eight hold calls while the gel estimate fell 7.42 to 6.16 N (-1.28 N, 2.5 bands); at +8 mm the estimate was 10.31 N while the physical tension was 11.44 N, inside the 11.0-12.0 band, and the program's next 0.5 mm step took the true load to 12.39 N, outside it. The change in reading from the end of the hold to +8 mm was 4.14 N against a true change of 4.00 N — the difference is accurate, the absolute is not.
---

## How

A hold is a pose re-asked for, not a wait: nothing advances the simulation
while the hand is idle. One such call advanced the episode by about 41 physics
ticks — a 31-step plan plus the 10 the server always delays — against a 120 Hz
clock, so a 3 s stage is 360 ticks and lands inside ten calls, fourteen with
room; overshooting is harmless once the stage is counted, and the spare calls
are what let the hold survive a step that falls outside the band.

Keep the height and count the calls. Do not correct on the reading during the
hold, and do not take the reading at the end of the hold as the new zero: the
dip is the gel's transient, it can still be present on the next move, and an
absolute estimate taken then is low by up to about a newton while the load is
not. Anchor the next stage to the reading taken before the hold (or to the
hold's target); from the hold pose onward, carry the change in reading — the
shared dip cancels — not the absolute. The dip is not a fault: do not abort,
re-grasp or recalibrate on it. **Do not make the reading the exit condition.**
The creep runs one way while the hand stands still, so a timer that needs the
reading inside a band keeps resetting itself and the hold never ends; the load
is unchanged through all of it.
