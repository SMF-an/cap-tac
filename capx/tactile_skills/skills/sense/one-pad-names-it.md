---
id: S26
name: one-pad-names-it
family: sense
summary: One pad naming the material is the reading — a ball is uniform, so a disagreement is text noise, not two materials.
task_types: []
tasks: [blind_pick_balls]
channels: [text]
objects: the balls in the blind box
evidence: >-
  2026-11-04, the first text-arm blind_pick_balls run: a station read
  "left: TEXTURED_BALL | right: SOFT_BALL" on one grip.  The program required
  both pads to name the same material, moved on, and the episode ran out while
  it was still searching.  A ball is one surface on every side; two pads on
  one ball are looking at the same material, so a split is the text, not the
  ball.
---

## How

A ball is one surface on every side, so the two pads on one ball are looking
at the same material.  When their descriptions disagree, one of them is noise:
re-read once at the same grip, and if it is still split, take the pad that
named a material as the reading — a pad that describes the image or says
nothing has not measured the surface.

## Traps

- **Demanding both pads name the same material rejects real balls.**  A ball
  that one pad names and the other mumbles is a reading, not a no-reading; the
  strict pair is the wrong gate ([[lift-verifies-grip]] is the right one).
- A pad that names no material is not a "different material" vote — it is an
  absent reading ([[description-needs-contact]]).
- The relaxation is not a licence to guess: the named material still has to be
  the one the task asks for, and the ball still has to rise with the hand
  before it is carried.
