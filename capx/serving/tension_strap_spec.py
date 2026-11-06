"""The force/time sequence each trial's strap task runs, drawn from its seed.

The sequence belongs to the seed.  One server serves one physics seed, the
server configures the task's acceptance from that seed before the task is
built, and the client renders the prompt and the oracle from the same draw
after the reset -- so the numbers the model is given, the numbers the oracle
aims at and the numbers the scorer reads cannot drift apart.

A module rather than a config key because the two processes cannot read each
other's memory: the server is started before the trial and the client only
learns the trial id at reset.  Both import this and call it with the same
seed.

The draw is deliberately narrow, for reasons the task itself does not know:

* Forces stay at or below 20 N.  The gel's absolute accuracy is what decides
  this task, and that error grows with the load -- on the straps measured,
  the same reading is worth about a quarter of a newton per newton of
  disagreement in the strap's own stiffness.
* Holds stay at or below three seconds.  The reading drifts while the hand
  holds still, and the drift accumulates with time; three seconds already
  moves most of the half-newton band.
"""

from __future__ import annotations

import math

import numpy as np

#: First stage's force, in newtons.
FIRST_FORCE_N = (6.0, 12.0)
#: How much more the second stage asks for.
SECOND_FORCE_GAP_N = (3.0, 6.0)
#: Hold duration per stage, in seconds.
HOLD_SECONDS = (1.5, 3.0)
#: The task refuses anything above this, and refuses a tolerance but 0.5 N.
MAXIMUM_FORCE_N = 20.0
#: Salt, so this draw is not the physics draw and not another task's draw.
SEED_SALT = 41879
#: A no-op hold move advanced the episode by 41 ticks when it was measured.
TICKS_PER_HOLD_MOVE = 41
POLICY_HZ = 120


def _half(value: float) -> float:
    """Round to the nearest half, which is the resolution the band can carry."""
    return round(float(value) * 2.0) / 2.0


def sample_sequence(seed: int) -> tuple[float, float, float, float]:
    """The (force, hold, force, hold) this seed runs, in newtons and seconds."""
    rng = np.random.default_rng(int(seed) + SEED_SALT)
    first = _half(rng.uniform(*FIRST_FORCE_N))
    second = min(_half(first + rng.uniform(*SECOND_FORCE_GAP_N)), MAXIMUM_FORCE_N)
    first_hold = _half(rng.uniform(*HOLD_SECONDS))
    second_hold = _half(rng.uniform(*HOLD_SECONDS))
    return first, first_hold, second, second_hold


def as_list(sequence) -> list[float]:
    """The sequence in the shape ViTaForge's ``force_time_sequence`` takes."""
    return [float(value) for value in sequence]


def goal_sentence(sequence) -> str:
    """The task's goal, in the words the prompt gives the model."""
    first, first_hold, second, second_hold = sequence
    return (
        "pull the elastic strap to %g newtons and hold it there for %g seconds, "
        "then pull it to %g newtons and hold that for %g seconds"
        % (first, first_hold, second, second_hold)
    )


def hold_moves(hold_seconds_values) -> int:
    """Motion calls that comfortably exceed each hold, for the oracle.

    The oracle holds by re-issuing the pose it already holds; one such call
    advanced the episode by 41 ticks when it was measured, against a 120 Hz
    clock, so the count is the arithmetic minimum plus a wide margin --
    overshooting a hold is harmless (the stage is already counted) and
    undershooting it loses the run.
    """
    longest = max(hold_seconds_values)
    return int(math.ceil(longest * POLICY_HZ / TICKS_PER_HOLD_MOVE)) + 5
