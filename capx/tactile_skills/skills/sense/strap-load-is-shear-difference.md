---
id: A26
name: strap-load-is-shear-difference
family: sense
summary: The load is the two pads' shear difference, 0.93 N per px — and the pixel take-up is the gel's own gain times 2.3 mm, so measure the gain.
task_types: [force]
channels: [shear]
objects: the elastic strap, gripped by the tab at its top
evidence: Measured on four seeds by raising the hand in 2 mm steps from the release point, against the task's own tension. The slope is the same on all four — 0.928, 0.932, 0.936, 0.947 N per px, 2% apart, residual 0.1 N. The take-up is not — 4.74, 4.53, 4.12, 3.73 px, which is 0.94 N, most of the half-newton band. But the take-up divided by the gel's gain — pixels of slip per millimetre the hand rises, 2.09 / 1.98 / 1.78 / 1.62 — is 2.27, 2.28, 2.32 and 2.31 mm of hand rise — 2.2% apart. Aiming with a fixed 4.28 px take-up puts those seeds at -0.53, -0.28, +0.15 and +0.67 N from their targets; aiming with 2.3 mm times a gain measured in the episode puts them at -0.05, -0.03, -0.04 and +0.12.
---

## How

Read the difference of the two pads' shear **vectors**, measured from the
close:

```python
d = hypot((L.dx - R.dx) - (L.dx - R.dx at the close),
          (L.dy - R.dy) - (L.dy - R.dy at the close))
```

The pads are mounted mirrored, so the tab slipping down between them drags
their marker fields opposite ways. The difference is the load; either pad
alone is half the signal and loses the other half to whatever moved both of
them. Read both axes: the difference vector rotates as the pull changes.

**Close by opening, not by depth.** `close_gripper(position=0.22)` lands the
pads at 25.70 and 25.25 mm every time, on four straps running, where the same
depth command lands somewhere new each attempt: 27.31/26.99, 27.39/27.10,
27.70/27.38, 27.80/27.32. One of those dragged a pad's whole marker field
off its lattice and never let go, and 27.80 leaves the strap pinned to its
fixture. The opening frees it; 0.18 crushes to the sensor floor and
0.10-0.14 do not fit the tab at all.

**Then the take-up, which is not a number of pixels.** The load is 0.93 N per
px, but where zero sits is the gel's gain — px of slip per millimetre the
hand rises — times the strap's slack, and the gain follows the modulus the
seed draws. So measure it: two rises of a known distance give the gain, and
the take-up is 2.3 mm of that rise.

```python
first, second = slip_after(2 * mm), slip_after(4 * mm)
gain = (second - first) / 2 * mm          # px per unit of hand rise
load = 0.93 * (slip_now - 2.3 * mm * gain)
```

The slack itself is the same on every strap measured (2.27-2.32 mm); only
the gain moves. Carrying a fixed take-up instead costs up to 0.7 N and loses
two seeds out of four.

**Take the steps the numbers were measured at.** The same pose does not read
the same when it is reached a different way: a 5 mm jump comes out 0.9 px
short of the pose a 2 mm ramp reaches, and the half-millimetre corrections
that finish a stage cost another 0.6 px -- 0.6 N on a band of 1.0. Climb in
2 mm steps, and let the small steps be as few as possible.

What does *not* bend is magnitude. A 2 mm ramp stays straight from zero to
20.7 N with a 0.09 N residual over seven points, and both pads'
displacements and coherence are flat across it (74.5 to 77.3 px, 0.52 to
0.57 on one). The channel has the range; what it does not have is patience
for being moved around in it.

Measure from the close and keep climbing. The field does not retrace its
path: a descent and a fresh zero re-zeroes against a state that is not the
close — 0.90 px on a 2026-11-01 run, 0.85 N, and its second stage landed at
19.070 N.

What is worth measuring in-episode is the gain, and the strap's stiffness in
millimetres per newton, because those are what the seed draws differently.

## Traps

- **A fixed pixel take-up is the failure this replaced.** Four straps span
  3.73 to 4.74 px — a whole band — while their slopes agree to 2%.
- **A dead pad reads as load.** A close can drag one pad's whole marker
  field along with the jaw; that pad then reports the drag. Its displacement
  is several times the other's — 49 px against 8.7 on a 2026-10-31 run — and
  it never recovers during the episode. Closing again in the same place does
  not mend it (five tries, same 5.6-6.4 ratio). A healthy pair reads within
  about a third of each other.
- **The depth is not the load.** It reads the squeeze between the pads, which
  the pull does not change: the tab is dragged along the gel, not pushed into
  it. Across a 0-19 N ramp both depths held constant to 0.02 mm.
- **Neither pad's own displacement can carry it.** The magnitude saturates on
  one pad while the load keeps rising, and the two pads do not saturate at
  the same time.
