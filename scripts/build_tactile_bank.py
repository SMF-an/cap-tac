#!/usr/bin/env python
"""Mine tactile keys out of recorded trials: which bands belong to which outcome.

The library's cards are supposed to be earned, not invented.  This script is
the earning step.  It reads every tactile reading a trial produced -- from a
live ``CAPX_TACTILE_TRACE`` JSONL when one exists, otherwise from the
``Tactile:`` lines in the trial's ``summary.txt`` -- keys the trial's deepest
reading with :func:`capx.tactile_memory.signature.key_from_readings`, and
reports, per task and per band, how many trials that read the band went on to
work and how many did not.

One reading per trial, not every reading.  Keying everything a trial looked
at makes almost every band cover almost every trial -- that is a fact about
a moving arm, not about an outcome -- and the table could never separate
anything.  The deepest reading is the one where the pads had the most to say
and where the grasp either worked or did not.

It writes a report, never a bank.  A band that separates outcomes is a fact
about the sensor; the line a card carries is a judgement about what to do
with it, and a judgement is the one thing that cannot be mined.  Read the
report, pick the bands that separate, write the card by hand.

The gate this script exists to answer: if the bands do not separate outcomes
on the trials already run, a card bank cannot help, and the work stops here
rather than after a month of prompt tuning.

Example:
    python scripts/build_tactile_bank.py --root outputs

Two layouts are read, because a run can leave either:

  * one trace per run, with a ``run`` and a ``trial`` column on every row
    (what the runner writes today -- one process executes all the trials,
    and a re-run appends a second set of the same trial numbers to the same
    file);
  * one trace inside each trial directory (what a process running a single
    trial leaves behind).

Three things the report cannot tell you, all worth remembering:

  * A band's ``bad share`` is only interesting next to the task's own base
    rate, which is printed with the run.  A band holding most of a task's
    trials will show a bad share close to that base -- that is the floor,
    not a signal.  What separates is a band that holds few trials and most
    of the failures.
  * Historical summaries predate the marker shear and anisotropy fields, so
    on those trials ``shear``, ``shear_axis`` and ``aniso`` are constant and
    separate nothing.  A dimension that never varied is listed as such
    instead of being ranked with the rest.
  * A summary only records the reads the model chose to make, so the two
    readings a ``*_trend`` dimension compares may be a whole grasp apart.
    Read them as "the trial saw the depth falling when it was deepest", not
    as a rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.tactile_memory.signature import key_from_readings  # noqa: E402
from capx.tactile_memory.tasks import TASK_TYPES  # noqa: E402

TRACE_NAME = "tactile_trace.jsonl"
SUMMARY_NAME = "summary.txt"

#: A trial directory ends with the outcome the framework recorded.
_OUTCOME = re.compile(r"taskcompleted_([01])$")

#: The trial's number, which is also what the trace's ``trial`` column holds.
_NUMBER = re.compile(r"^trial_(\d+)")

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


def number_of(directory: Path) -> int | None:
    """The trial number a directory is named after, if it is named that way."""
    match = _NUMBER.match(directory.name)
    return int(match.group(1)) if match else None


def labelled_trials(root: Path) -> list[tuple[str, str, Path]]:
    """Every trial directory under ``root`` that states its own outcome.

    Returns:
        ``(task, outcome, directory)`` per directory, in path order.  One
        directory is one trial even when its number repeats: a re-run of a
        seed that ends differently leaves a second directory beside the
        first instead of overwriting it, and the two attempts are two
        trials.  A directory whose name carries no number, or no outcome,
        is not a labelled trial.
    """
    found: list[tuple[str, str, Path]] = []
    for directory in sorted(root.rglob("trial_*")):
        if not directory.is_dir():
            continue
        outcome = outcome_of(directory)
        if outcome is None or number_of(directory) is None:
            continue
        found.append((task_of(directory) or "(unknown)", outcome, directory))
    return found


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


def read_trace(path: Path) -> list[tuple[Any, Any, dict]]:
    """Every row in one trace file, as ``(run, trial, readings)``.

    Either column is None on a row written before it existed.  A half-written
    last line -- a process killed mid-append -- is skipped rather than failing
    the run: the rows before it are still good.
    """
    rows: list[tuple[Any, Any, dict]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append((row.get("run"), row.get("trial"), row.get("readings") or {}))
    return rows


# ---------------------------------------------------------------------------
# Folding trials into bands
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """One field as a number, with the API's 0.0-means-missing convention."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number == 0.0 else number


def deepest(readings: list[dict]) -> int | None:
    """Index of the reading taken at the moment of deepest contact.

    A trial is not one situation.  Keying every reading it took and asking
    which bands it visited makes almost every band cover almost every trial,
    which is a property of the trial -- it moved the arm -- and not a
    property of the outcome, so the table cannot separate anything however
    good the sensor is.  One reading per trial restores a partition: the
    deepest one, where the pads had the most to say and where a grasp either
    worked or did not.

    The first reading to reach the minimum wins, because the question is
    "what did it look like when the pads first got as deep as they got".

    Returns:
        The index, or None when no reading reported a depth at all.
    """
    best, best_depth = None, None
    for index, reading in enumerate(readings):
        present = [value for value in
                   (_number(reading.get(f"{side}_depth_mm"))
                    for side in ("left", "right")) if value is not None]
        if not present:
            continue
        depth = min(present)
        if best_depth is None or depth < best_depth:
            best, best_depth = index, depth
    return best


def band_at(readings: list[dict], index: int) -> list[str]:
    """The ``dimension=band`` pairs of one reading, given its predecessor."""
    previous = readings[index - 1] if index else None
    return [f"{dimension}={band}" for dimension, band
            in key_from_readings(readings[index], previous=previous).items()]


def bands_of(readings: list[dict]) -> list[str]:
    """The key of the trial's deepest reading: one situation per trial."""
    index = deepest(readings)
    return band_at(readings, index) if index is not None else []


def units(root: Path) -> list[tuple[str, str, list[dict]]]:
    """Every labelled trial under ``root``, as ``(task, outcome, readings)``.

    A trace is split by its ``run`` and ``trial`` columns, because a run
    writes one file for every trial it executes and a re-run appends a second
    set of the same trial numbers to that file.  A trace with no trial column
    belongs to the trial directory it sits in.  Trials that produced no trace
    fall back to the printed lines in their summary.

    A trial number naming more than one directory -- a re-run that ended
    differently -- cannot be joined to the run-level rows: the run column
    says which rows are one attempt, not which attempt a directory is.
    Those rows are dropped, and each directory's own printed lines carry
    that trial instead.
    """
    labelled = labelled_trials(root)
    by_number: dict[tuple[Path, int], list[tuple[str, str, Path]]] = defaultdict(list)
    for entry in labelled:
        by_number[(entry[2].parent, number_of(entry[2]))].append(entry)

    found: list[tuple[str, str, list[dict]]] = []
    covered: set[Path] = set()

    for trace_file in sorted(root.rglob(TRACE_NAME)):
        grouped: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
        for run, trial, readings in read_trace(trace_file):
            grouped[(run, trial)].append(readings)
        for (_run, trial), readings in grouped.items():
            if trial is None:
                # No column: the file belongs to whatever trial it sits in.
                outcome = outcome_of(trace_file.parent)
                if outcome is None:
                    continue
                task = task_of(trace_file.parent) or "(unknown)"
                covered.add(trace_file.parent)
            else:
                try:
                    number = int(trial)
                except (TypeError, ValueError):
                    continue
                hits = by_number.get((trace_file.parent, number), [])
                if len(hits) != 1:
                    # No directory to read an outcome off, or more than one
                    # claiming the same number.
                    continue
                task, outcome = hits[0][0], hits[0][1]
                covered.add(hits[0][2])
            found.append((task, outcome, readings))

    for task, outcome, directory in labelled:
        if directory in covered:
            continue
        readings: list[dict] = []
        for summary in sorted(directory.rglob(SUMMARY_NAME)):
            readings += readings_from_printed(summary)
        if readings:
            found.append((task, outcome, readings))
    return found


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
        for task, outcome, readings in units(root):
            outcomes[task][outcome] += 1
            for entry in bands_of(readings):
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
            # One reading per trial means each dimension's bands partition
            # the trials, so a single band covering all of them is a channel
            # that never varied -- the normal case for a field an older
            # server never sent, and for a task that never exercised it.
            if all(coverage == total for coverage, _b, _g, _d in rows):
                lines.append(f"{dimension}: one band covers every trial -- "
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
