"""The note sequence each trial's shadow-hand piano task runs, drawn from its seed.

The sequence belongs to the seed, the way the strap's force sequence does.  One
server serves one seed: it is started with ``--seed`` before any trial, and the
client only learns the trial id at reset.  Both sides therefore draw the
sequence here, from the same number, so the notes the model is told to play and
the notes the task's scorer listens for cannot drift apart.

A module rather than config keys because the two processes cannot read each
other's memory.  The server writes the draw into its TaskCfg before the task is
built (``launch_univtac_server._build_piano_task``); the client renders the same
draw into the prompt and the oracle.

The pool is the **white keys of C4-B4**.  The served profile uses the GelSight
Mini / UIPC backend, which owns the key mechanics for the white keys its gel
can reach and refuses a black-key sequence (``shadow_piano._set_sequence``:
black keys are allowed only without tactile sensing or with the DIGIT
renderer, and the DIGIT renderer exposes no gel scalars).  A black-key task
would have to go back to the force-only PhysX profile, so the pool stays white.

Dynamics are drawn per note.  They are not decoration: the task scores the
measured key force against a per-note target, so a light note played heavy is a
failed note, not a stylistic choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Written by ViTaForge/scripts/fit_piano_force_model.py: the frozen
#: gel-image-to-force model, the press recipe that produced it and the
#: leave-one-key-out error.  The adapter, the oracle and the task's skill are
#: built from it; nothing fabricates the numbers.
CALIBRATION_FILE = Path(__file__).with_name("piano_gel_calibration.json")
#: The frozen model's format tag.  A file written by the old palm-offset
#: calibration ("no format") is a hard error: its numbers were measured against
#: a gel mount that never touched the key.
CALIBRATION_FORMAT = "piano-gel-force/2"

#: Six notes is the upstream default for the movable hand (TaskCfg
#: ``piano_note_count = 6``), and a comfortable size for one episode.
NOTE_COUNT = 6
#: C4, D4, E4, F4, G4, A4, B4 -- the white keys the UIPC gel backend owns.
#: ``shadow_piano._set_sequence`` builds exactly this pool when tactile sensing
#: is on and the sensor is the GelSight Mini.
NOTE_POOL = (60, 62, 64, 65, 67, 69, 71)
#: Upstream's per-note hold when the profile does not name one.  The score
#: wants the key held inside its force band for this long before the release
#: counts.
HOLD_SECONDS = 0.12
#: Salt, so this draw is not a physics draw and not another task's draw.
SEED_SALT = 7331

_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class PianoSequence:
    """One seed's notes, their dynamics and their hold times."""

    midi: tuple[int, ...]
    dynamics: tuple[str, ...]
    holds: tuple[float, ...]

    def as_task_kwargs(self) -> dict:
        """The three TaskCfg fields the draw becomes."""
        return {
            "piano_notes": list(self.midi),
            "piano_dynamics": list(self.dynamics),
            "piano_hold_seconds": list(self.holds),
        }


def note_name(midi: int) -> str:
    """Scientific pitch name, e.g. 60 -> C4."""
    return f"{_PITCHES[int(midi) % 12]}{int(midi) // 12 - 1}"


def sample_sequence(seed: int) -> PianoSequence:
    """The notes this seed plays, in order.

    The draw mirrors upstream ``make_sequence``'s two choices -- notes from the
    pool, then one dynamic per note -- so a seed that was tuned against the
    upstream runner draws the same kind of sequence here.  It is not the same
    random stream: the server writes this draw into the task explicitly, so the
    task does not redraw on reset.
    """
    rng = np.random.default_rng(int(seed) + SEED_SALT)
    midi = tuple(int(v) for v in rng.choice(NOTE_POOL, size=NOTE_COUNT))
    dynamics = tuple(str(v) for v in rng.choice(("light", "heavy"), size=NOTE_COUNT))
    holds = (HOLD_SECONDS,) * len(midi)
    return PianoSequence(midi, dynamics, holds)


def goal_sentence(sequence: PianoSequence) -> str:
    """The task's goal, in the words the prompt gives the model."""
    notes = ", ".join(
        f"{note_name(midi)} ({dynamic})"
        for midi, dynamic in zip(sequence.midi, sequence.dynamics)
    )
    return (
        f"play these notes on the piano, in this order: {notes}.  Hold each "
        f"note for {HOLD_SECONDS:g} s of simulated time and release it before "
        "the next one; a repeated pitch needs its own release."
    )


def load_calibration() -> dict:
    """The frozen gel-image force model and the press recipe behind it.

    The score compares a key force to a band, and the hand reports no force.
    The index gel does see it, but not as a depth: the key rotates with the
    finger and the pad's indentation stays under 0.15 mm, while the contact
    moment tilts the whole gelpad by up to 0.4 mm across its face.  What is
    proportional to the force is that tilt, spread over the image -- so the
    model is fitted on five whole-image statistics plus the pad's depth
    gradient, against the force the simulator measured at the same step
    (``ViTaForge/scripts/piano_force_map_probe.py``).

    The numbers are measurements, not constants to guess, so a missing or
    stale file is a hard error: run
    ``ViTaForge/scripts/fit_piano_force_model.py`` and point it at
    ``capx/serving/piano_gel_calibration.json``.
    """
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(
            f"No piano force calibration at {CALIBRATION_FILE}. Run "
            "ViTaForge/scripts/fit_piano_force_model.py against a force map "
            "collected with the tactile Isaac environment."
        )
    data = json.loads(CALIBRATION_FILE.read_text())
    if data.get("format") != CALIBRATION_FORMAT:
        raise ValueError(
            f"{CALIBRATION_FILE}: format {data.get('format')!r}, expected "
            f"{CALIBRATION_FORMAT!r}. The palm-offset calibration was measured "
            "against a gel mount that never touched the key; re-run "
            "ViTaForge/scripts/fit_piano_force_model.py."
        )
    model = data.get("model", {})
    for field in ("median", "iqr", "weight"):
        values = model.get(field)
        if not values:
            raise ValueError(f"{CALIBRATION_FILE}: missing model.{field}")
    if len(data.get("features", ())) != len(model["median"]):
        raise ValueError(
            f"{CALIBRATION_FILE}: features and model.median disagree in length")
    for field in ("valid_depth_mm", "saturation_N", "press", "bands",
                  "plateau"):
        if field not in data:
            raise ValueError(f"{CALIBRATION_FILE}: missing {field}")
    return data


def force_target(calibration: dict, dynamic: str) -> float:
    """The force a press of this dynamic is served to, newtons.

    Light is an ordinary estimate crossing.  Heavy's target sits above the
    gel's saturation point, where the estimate stops separating forces; the
    adapter serves it by stopping when the pad's gradient plateaus instead
    (``saturation_N`` in the calibration is the boundary).
    """
    try:
        return float(calibration["bands"][dynamic]["target_N"])
    except KeyError as exc:
        raise ValueError(
            f"{CALIBRATION_FILE}: no bands.{dynamic}.target_N") from exc
