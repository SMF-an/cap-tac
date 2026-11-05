---
id: R14
name: search-continues-after-miss
family: protocol
summary: A missed close or a refused motion costs one candidate, not the episode — catch it, open, and try the next site.
task_types: []
tasks: [blind_pick, blind_pick_balls]
channels: []
objects: the hidden object a blind search is looking for
evidence: >-
  blind_pick, seeds 3-10 (2026-10-19): four runs raised at the first lost
  grasp and ended, leaving their own later columns unvisited, while both
  passing seeds moved on from a failed site.  A gpt-6-astra seed-4 run died on
  an uncaught planner refusal during a routine retract, at site 11 of 35, 3.4
  mm from the object's true centre.  Routed to blind_pick_balls on 2026-10-24:
  its first LLM run ended at the first row on an uncaught goto_pose refusal,
  16 mm short of the commanded point.
---

## How

The episode is a search over candidate sites, and one close tests only the
site it is made at.  A site that fails is one candidate, not the run: open,
rise back to the search height, move to the next site, and close again.

Wrap the motion calls in the same try/except as the close.  **The retract is a
motion call too** — it is the last thing a loop does, so an unguarded refusal
there ends the episode after the rest of the search has already paid.

## Traps

- A close that raises on air, a lift that leaves the object behind, and a
  motion the arm refuses all read as "this site missed".  None of them is a
  verdict on the task.
- On a blind ball sweep every ball the hand stops at is one candidate; a
  refusal there costs that station, not the sweep.
