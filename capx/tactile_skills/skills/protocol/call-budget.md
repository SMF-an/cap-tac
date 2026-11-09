---
id: R18
name: call-budget
family: protocol
summary: A call costs the same ~13 s at 2 mm as at 50 mm, so about 250 calls are the whole episode — count calls, not distance.
task_types: []
tasks: [tactile_memory_match]
channels: []
objects: any
evidence: >-
  2026-11-07 gpt-6-astra: trial 26 made 59 calls in 766 s and trial 27 made 26
  in 456 s — 13 to 18 s a call, the same for a 2 mm sag nudge as for a 50 mm
  carry. Trial 26 spent its last third re-approaching one candidate at the
  same offset, and the same far-plane reading came back each time.
---

## How

Count calls, not millimetres. The allowance is about 250 calls for the whole
episode, at most three grips per can plus one re-grasp. When a can's share is
spent, decide with the readings already in hand — two cans fully read beats
three cans half-read.

Make one call do several calls' work: reach a pose with `z_approach` instead of
rising, travelling and descending by hand ([[one-call-approach]]), and read the
tactile state out of the motion's own return instead of spending a separate
`get_tactile_data` ([[close-lands-on-target]]).

## Traps

- **A retry is not free.** Re-approaching the same can at the same offset costs
  what a fresh probe costs and returns what the fresh probe returned; spend
  the call on a different offset or a different can
  ([[regrasp-after-slip]]).
- **A loose close that met nothing answers nothing.** It costs a call and
  still leaves the can unmeasured, so descend over the can before closing it
  ([[memory-match-one-grasp]]).
