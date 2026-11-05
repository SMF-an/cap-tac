---
id: A27
name: lift-verifies-grip
family: act
summary: An object that did not rise with the hand was never held — close firmer and lift again.
task_types: [classify, insert, manipulate, wipe]
channels: []
objects: any
evidence: >-
  2026-11-02 weight_classify (text arm), probe on seed 3: three generated
  programs closed at force_threshold=30.0 and the 180 g block stayed at
  z=0.0014 through a 12 cm lift.  At 28.5 the same close carried it (dz=0.031
  with a 3 x 1 cm lift, 0.122 with the same two 6 cm steps); at 30.0 the slow
  lift still left it at z=0.0014.  Octopi answered "left: light. | right:
  light." in every case, including while the block hung.
---

## How

After closing, lift a few centimetres in roughly 1 cm steps, then read the
object's pose with get_object_pose and compare its height with where it
started.  The lift is the test of the grip: an object still at table height
was never in the fingers, whatever the close reported and whatever the tactile
text says afterwards.

When it did not rise, open, re-approach in a single goto_pose call
([[one-call-approach]]) and close firmer — force_threshold is a gel-depth
target and lower values grip harder, while a close that never reaches its
target raises instead of returning.  Lift again and re-check.  Only a lift
that carries the object gives a reading worth classifying.

## Traps

- **A failed lift leaves no trace in the text.**  The pads describe what they
  touch, and a dropped object leaves them touching nothing
  ([[description-needs-contact]]); the description answers the question
  anyway — a 180 g block that never left the table read as "light" on both
  pads.  Never let the description decide whether the object is held.
- The close's own success is not the grip's success.  A light close can press
  the object's edge, stop, and return without trapping it; the command
  succeeded and the object is still on the table.
- The text is prose, not a schema.  It answers with the word somewhere in a
  sentence ("left: LIGHT, ...") and will not echo a requested tag such as
  `CLASS: LIGHT`; match the word it gives, and accept one pad naming it.  A
  parser that demands its own format throws away a correct reading.
- Re-grasping costs a release, which is where a working grip is most easily
  lost ([[preserve-existing-grasp]]); it is still the right move when the lift
  has proved there is no grip at all.
