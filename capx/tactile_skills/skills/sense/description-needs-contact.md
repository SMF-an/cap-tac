---
id: S25
name: description-needs-contact
family: sense
summary: The text covers only what the pads touch; a gel under nothing describes no surface.
task_types: [classify, force]
channels: [text]
objects: any surface read through describe_tactile
evidence: >-
  2026-11-02 grasp_classify seed 1 (gpt-6-astra): a close at
  force_threshold=26.6 with adaptive=True followed by a 10 cm lift returned
  "left: image photo | right: rough"; the oracle's firmer two-stage close
  (27.5 then 20.5) on the same prism returned "rough" from both gels.
---

## How

The gel image is a picture of the gel itself: an object appears in it only
where the object indents the gel.  Ask for the description while the pads are
held against the object, not while they are still closing and not after the
grip has loosened.

## Traps

- **A description of the picture is a failed reading, not a smooth surface.**
  "image photo", ".jpg", or a sentence about a photo means the model is
  describing the rendered image and has found no surface.  Reading that as
  smooth is a guess, not a measurement.

- The two sides can disagree under one grip.  When one gel names the surface
  and the other talks about the image, the side that named it is the reading.
