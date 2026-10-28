#!/usr/bin/env python
"""Mine tactile keys out of recorded trials: which bands belong to which outcome.

The library's cards are supposed to be earned, not invented.  This script is
the earning step.  It reads every tactile reading a trial produced -- from a
live ``CAPX_TACTILE_TRACE`` JSONL when one exists, otherwise from the
``Tactile:`` lines in the trial's ``summary.txt`` -- keys each reading with
:func:`capx.tactile_memory.signature.key_from_readings`, and reports, per
task and per band, how many trials that passed through the band went on to
work and how many did not.

The unit is the TRIAL, not the reading.  A chatty trial takes hundreds of
readings and would otherwise outvote a quiet one; "how many trials passed
through this band" is the question a card actually asks.

It writes a report, never a bank.  A band that separates outcomes is a fact
about the sensor; the line a card carries is a judgement about what to do
with it, and a judgement is the one thing that cannot be mined.  Read the
report, pick the bands that separate, write the card by hand.

The gate this script exists to answer: if the bands do not separate outcomes
on the trials already run, a card bank cannot help, and the work stops here
rather than after a month of prompt tuning.

Example:
    python scripts/build_tactile_bank.py --root outputs

Two things the report cannot tell you, both worth remembering:

  * A band's ``bad share`` is only interesting next to the task's own base
    rate, which is printed with the run.  Every band a healthy trial passes
    through will show a bad share close to that base -- that is the floor,
    not a signal.
  * Historical summaries predate the marker shear and anisotropy fields, so
    on those trials ``shear`` and ``aniso`` are constant and separate
    nothing.  Dimensions that never vary are listed separately instead of
    being ranked with the rest.
  * A summary only records the reads the model chose to make, so two
    consecutive entries there may be a whole grasp apart.  The ``*_trend``
    dimensions still mean "what changed between these two looks", which is
    a longer span than a trace's adjacent rows -- read them as "the trial
    saw the depth fall / stop", not as a rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.tactile_memory.signature import DIMENSIONS, key_from_readings  # noqa: E402
from capx.tactile_memory.tasks import TASK_TYPES  # noqa: E402

TRACE_NAME = "tactile_trace.jsonl"
SUMMARY_NAME = "summary.txt"

#: A trial directory ends with the outcome the framework recorded.
_OUTCOME = re.compile(r"taskcompleted_([01])$")

#: One field of the printed ``Tactile:`` line, as it has been written since
#: the line was introduced: ``L_depth=28.49mm``, ``R_contact=True``.
_PRINTED = re.compile(r"([LR])_(contact|depth|disp|coh|relief)=(\S+)")

_FIELDS = {
    "contact": "{}_contact",
    "depth": "{}_depth_mm",
    "disp": "{}_marker_displacement_px",
    "coh": "{}_marker_coherence",
    "relief": "{}_height_roughness_mm",
}


# ---------------------------------------------------------------------------
# Reading a trial
# ---------------------------------------------------------------------------


def task_of(path: Path) -> str:
    """The task a trial belongs to, from the directories above it.

    Run trees are laid out as ``<model>/univtac_<task>/trial_NN_...``, so the
    task is the nearest ancestor whose name is one the library knows.  A run
    that sits nowhere near such a directory is attributed to no task, and its
    readings can only speak to cross-task bands.
    """
    for parent in path.parents:
        name = parent.name.removeprefix("univtac_")
        if name in TASK_TYPES:
            return name
    return ""


def outcome_of(directory: Path) -> str | None:
    """``success``/``failure`` from the trial directory name, if it says so."""
    match = _OUTCOME.search(directory.name)
    if not match:
        return None
    return "success" if match.group(1) == "1" else "failure"


def readings_from_printed(path: Path) -> list[dict]:
    """Every ``Tactile:`` line in a trial summary, as reading dicts.

    The line has always carried contact, depth, marker displacement and
    coherence, and some tasks add the imprint relief.  The marker shear and
    anisotropy were computed but never printed, so a summary simply has no
    value for them -- which is why those dimensions come out constant here.
    """
    readings = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Tactile:" not in line:
            continue
        reading: dict = {}
        for side, field, raw in _PRINTED.findall(line):
            key = _FIELDS[field].format("left" if side == "L" else "right")
            if field == "contact":
                reading[key] = raw == "True"
            else:
                try:
                    reading[key] = float(raw.removesuffix("mm").removesuffix("px"))
                except ValueError:
                    continue
        if reading:
            readings.append(reading)
    return readings


def read_trace(path: Path) -> list[dict]:
    """Every reading in one trace file, skipping a half-written last line."""
    readings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            readings.append(json.loads(line).get("readings") or {})
        except json.JSONDecodeError:
            continue
    return readings


# ---------------------------------------------------------------------------
# Folding trials into bands
# ---------------------------------------------------------------------------


def bands_of(directory: Path) -> list[str] | None:
    """The ``dimension=band`` pairs one trial passed through, or None.

    A trial contributes each band once, however many readings carried it.
    Readings are replayed in file order so the ``*_trend`` dimensions see
    the reading before each one, exactly as a live client does.
    """
    readings: list[dict] = []
    for trace in sorted(directory.rglob(TRACE_NAME)):
        readings += read_trace(trace)
    if not readings:
        for summary in sorted(directory.rglob(SUMMARY_NAME)):
            readings += readings_from_printed(summary)
    if not readings:
        return None
    seen: set[str] = set()
    previous: dict | None = None
    for reading in readings:
        seen.update(f"{dimension}={band}"
                    for dimension, band
                    in key_from_readings(reading, previous=previous).items())
        previous = reading
    return sorted(seen)


def collect(roots: list[Path]) -> tuple[dict, dict]:
    """Every labelled trial under these roots, folded into band counts.

    Returns:
        ``(outcomes, bands)``.  ``outcomes`` is task -> Counter of trials by
        outcome; ``bands`` is (task, dimension, band) -> the same, counting
        only the trials that passed through that band.
    """
    outcomes: dict[str, Counter] = defaultdict(Counter)
    bands: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for root in roots:
        for directory in sorted(root.rglob("trial_*")):
            if not directory.is_dir():
                continue
            outcome = outcome_of(directory)
            if outcome is None:
                continue
            seen = bands_of(directory)
            if seen is None:
                continue
            task = task_of(directory) or "(unknown)"
            outcomes[task][outcome] += 1
            for entry in seen:
                dimension, _, band = entry.partition("=")
                bands[(task, dimension, band)][outcome] += 1
    return outcomes, bands


def report(outcomes: dict, bands: dict, min_support: int) -> str:
    """The band table, plus the base rates it has to be read against."""
    lines: list[str] = []
    for task in sorted(outcomes):
        base = outcomes[task]
        total = base["success"] + base["failure"]
        if not total:
            continue
        lines.append(f"\n## {task}  ({total} trials, "
                     f"{base['failure']} bad, base bad share "
                     f"{base['failure'] / total:.2f})")
        by_dimension: dict[str, list[tuple]] = defaultdict(list)
        for (owner, dimension, band), counts in bands.items():
            if owner != task:
                continue
            good, bad = counts["success"], counts["failure"]
            if good + bad < min_support:
                continue
            by_dimension[dimension].append((good + bad, band, good, bad))
        if not by_dimension:
            lines.append(f"(no band met --min-support {min_support})")
            continue
        for dimension in sorted(by_dimension):
            rows = by_dimension[dimension]
            # A dimension separates nothing when every band it took covers
            # every trial: the reading is the same story in successes and
            # failures alike.  That is the normal case for a channel the
            # trial never exercised, and for the channels a historical
            # summary cannot carry at all.
            if all(coverage == total for coverage, _b, _g, _d in rows):
                lines.append(f"{dimension}: no band excludes any trial -- "
                             f"separates nothing here")
                continue
            lines.append(f"{dimension}:")
            for _coverage, band, good, bad in sorted(
                    rows, key=lambda row: (-(row[3] / (row[2] + row[3])),
                                           -(row[2] + row[3]))):
                lines.append(f"  {band:<8} | ok {good:>3} | bad {bad:>3} | "
                             f"bad share {bad / (good + bad):.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", default=[],
                        help="a tree to walk for trial_* directories "
                             "(outcome taken from the directory name)")
    parser.add_argument("--out", help="write the report here as well as printing it")
    parser.add_argument("--min-support", type=int, default=3,
                        help="trials a band needs before it is reported")
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("give at least one --root")

    outcomes, bands = collect([Path(root) for root in args.root])
    text = report(outcomes, bands, args.min_support).strip() or "(no labelled trials found)"
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
