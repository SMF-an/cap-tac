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

The table is also read by action, because the same reading can mean
different things under different primitives: a depth reached while closing
the gripper is a grasp, the same depth reached while moving is a collision.
Within each trial, every reading is attributed to the action that produced
it -- a pure read reports what the last action left behind, so it inherits
that action rather than forming a bucket of its own -- and the deepest
reading of each action is keyed separately.  Trials whose only record is a
printed summary carry no action and are grouped as ``(unknown)``.

The report ends with promotion candidates: bands that held at least k
independent trials and whose bad share sits a margin away from the task's
base rate.  Each is printed with the trials that argue against it and any
other task where the same band separated the same way.  This is the
automated half of promotion.  It can say that a band separated outcomes; it
cannot write the sentence a card carries, and it never edits a bank.  A
candidate with counterexamples stays a hypothesis -- narrow its
applicability until they fall outside it, or leave it a hypothesis.  A clean
candidate is one a person may promote to ``observed`` by hand.  Bands that
only say a reading was absent -- ``depth=missing``, ``disp_gap=unknown`` and
the like -- are not candidates: absence is a validity fact, not a condition
an action can be gated on, and the band table above already reports it.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.tactile_memory.events import READ_CALLS  # noqa: E402
from capx.tactile_memory.signature import is_absence, key_from_readings  # noqa: E402
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


def read_rows(path: Path) -> list[dict]:
    """Every parseable row in one trace file, in order.

    A half-written last line -- a process killed mid-append -- is skipped
    rather than failing the run: the rows before it are still good.
    """
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_trace(path: Path) -> list[tuple[Any, Any, dict]]:
    """Every row in one trace file, as ``(run, trial, readings)``.

    Either column is None on a row written before it existed.
    """
    return [(row.get("run"), row.get("trial"), row.get("readings") or {})
            for row in read_rows(path)]


def actions_of(rows: list[dict]) -> list[str | None]:
    """The action each row's reading is attributed to, aligned with ``rows``.

    A pure read does not produce a reading; it reports the state the last
    action left behind.  Attributing the deepest reading of a grasp to
    ``/tactile`` would put the most informative reading in a bucket that
    says nothing about what was done, so a read inherits the most recent
    action.  Rows before any action are unattributed.
    """
    actions: list[str | None] = []
    current: str | None = None
    for row in rows:
        call = row.get("call")
        if call and call not in READ_CALLS:
            current = str(call)
        actions.append(current)
    return actions


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


@dataclass(frozen=True)
class Unit:
    """One labelled trial: its readings and the action behind each.

    ``actions[i]`` is the primitive that produced ``readings[i]``, or None
    when the trial left no trace and only printed lines survive.  ``where``
    is the trial's path relative to the root that was walked, kept so a
    counterexample can be named rather than only counted.  ``identity`` is
    the same directory as a full path: two roots can hold the same relative
    path for two different trials, so anything that decides whether two
    trials are the same must compare this, not ``where``.
    """

    task: str
    outcome: str
    readings: list[dict]
    actions: list[str | None]
    where: str = ""
    identity: str = ""


def units(root: Path) -> list[Unit]:
    """Every labelled trial under ``root``, as :class:`Unit`.

    A trace is split by its ``run`` and ``trial`` columns, because a run
    writes one file for every trial it executes and a re-run appends a second
    set of the same trial numbers to that file.  A trace with no trial column
    belongs to the trial directory it sits in.  Trials that produced no trace
    fall back to the printed lines in their summary, and a printed line has
    no action recorded, so those readings are attributed to None.

    A trial that left no readings at all is still a unit, with an empty list.
    It is a labelled trial that never touched the gel, or never got far enough
    to say so, and dropping it here would quietly flatter every base rate in
    the report: the trials missing from this list are the ones that crashed
    before they could fail properly.

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

    found: list[Unit] = []
    covered: set[Path] = set()

    for trace_file in sorted(root.rglob(TRACE_NAME)):
        grouped: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
        for row in read_rows(trace_file):
            grouped[(row.get("run"), row.get("trial"))].append(row)
        for (_run, trial), rows in grouped.items():
            if trial is None:
                # No column: the file belongs to whatever trial it sits in.
                outcome = outcome_of(trace_file.parent)
                if outcome is None:
                    continue
                task = task_of(trace_file.parent) or "(unknown)"
                directory = trace_file.parent
                covered.add(directory)
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
                directory = hits[0][2]
                covered.add(directory)
            found.append(Unit(
                task, outcome,
                [_readings_of(row) for row in rows],
                actions_of(rows),
                str(directory.relative_to(root)),
                str(directory),
            ))

    for task, outcome, directory in labelled:
        if directory in covered:
            continue
        readings: list[dict] = []
        for summary in sorted(directory.rglob(SUMMARY_NAME)):
            readings += readings_from_printed(summary)
        found.append(Unit(task, outcome, readings, [None] * len(readings),
                          str(directory.relative_to(root)), str(directory)))
    return found


def _readings_of(row: dict) -> dict:
    readings = row.get("readings")
    return readings if isinstance(readings, dict) else {}


#: The bucket a reading falls into when the trial has no recorded action.
#: Printed summaries predate the trace and carry no call column.
UNKNOWN_ACTION = "(unknown)"


@dataclass
class Mined:
    """Everything one mining pass counted, keyed for the report.

    Attributes:
        outcomes: task -> Counter of trials by outcome.
        bands: (task, dimension, band) -> Counter, counting trials whose
            deepest reading fell in that band.
        keyless: task -> Counter of trials whose deepest state carried no
            depth, so no band could be read.
        action_outcomes: (task, action) -> Counter of trials that had at
            least one reading attributed to that action.  This is the base
            rate each action's bands must be read against, not the task's
            overall base rate.
        action_bands: (task, action, dimension, band) -> Counter at the
            deepest reading of that action inside the trial.
        action_keyless: (task, action) -> Counter of trials whose readings
            under that action carried no depth.
        examples: (task, dimension, band) -> the trials that produced that
            band, as ``(relative path, outcome, full path)``.  Kept so a
            counterexample can be named rather than only counted, and so
            two trials can be compared for identity across roots.
    """

    outcomes: dict[str, Counter]
    bands: dict[tuple[str, str, str], Counter]
    keyless: dict[str, Counter]
    action_outcomes: dict[tuple[str, str], Counter]
    action_bands: dict[tuple[str, str, str, str], Counter]
    action_keyless: dict[tuple[str, str], Counter]
    examples: dict[tuple[str, str, str], list[tuple[str, str, str]]]


def collect(roots: list[Path]) -> Mined:
    """Every labelled trial under these roots, folded into band counts.

    The overall table keys one reading per trial -- the deepest -- because a
    trial is one situation.  The action table keys one reading per trial per
    action, for the same reason one level down: within the stretch of a
    trial produced by one primitive, the deepest reading is what that
    primitive did to the gel.
    """
    outcomes: dict[str, Counter] = defaultdict(Counter)
    bands: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    keyless: dict[str, Counter] = defaultdict(Counter)
    action_outcomes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    action_bands: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    action_keyless: dict[tuple[str, str], Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str, str],
                   list[tuple[str, str, str]]] = defaultdict(list)

    for root in roots:
        for unit in units(root):
            outcomes[unit.task][unit.outcome] += 1
            index = deepest(unit.readings)
            if index is None:
                keyless[unit.task][unit.outcome] += 1
            else:
                for entry in band_at(unit.readings, index):
                    dimension, _, band = entry.partition("=")
                    bands[(unit.task, dimension, band)][unit.outcome] += 1
                    examples[(unit.task, dimension, band)].append(
                        (unit.where, unit.outcome, unit.identity))

            buckets: dict[str, list[int]] = defaultdict(list)
            for position, action in enumerate(unit.actions):
                buckets[action or UNKNOWN_ACTION].append(position)
            for action, positions in buckets.items():
                action_outcomes[(unit.task, action)][unit.outcome] += 1
                local = deepest([unit.readings[i] for i in positions])
                if local is None:
                    action_keyless[(unit.task, action)][unit.outcome] += 1
                    continue
                # Key the reading in the trial it came from, so a trend
                # dimension compares it with the row that actually preceded
                # it rather than with the previous row of this action.
                for entry in band_at(unit.readings, positions[local]):
                    dimension, _, band = entry.partition("=")
                    action_bands[(unit.task, action, dimension, band)][unit.outcome] += 1

    return Mined(outcomes, bands, keyless, action_outcomes, action_bands,
                 action_keyless, examples)


def _band_table(entries: dict[tuple[str, str], Counter],
                min_support: int, keyed: int,
                note_constant: bool = True) -> tuple[list[str], bool]:
    """Band rows grouped by dimension, ranked by how bad the band looks.

    ``entries`` maps ``(dimension, band)`` to the trial counter; ``keyed`` is
    how many trials under this table produced a key at all, used to name a
    channel that never varied instead of ranking it.  ``note_constant``
    prints a line for such a channel; the per-action table turns it off
    because the overall table already named the channel and repeating it
    under every action is noise.

    Returns:
        ``(lines, supported)``.  ``supported`` says whether any band met
        ``--min-support``, which lets the caller tell "nothing had enough
        trials" from "everything with enough trials was constant".
    """
    by_dimension: dict[str, list[tuple]] = defaultdict(list)
    supported = False
    for (dimension, band), counts in entries.items():
        good, bad = counts["success"], counts["failure"]
        if good + bad < min_support:
            continue
        supported = True
        by_dimension[dimension].append((good + bad, band, good, bad))
    lines: list[str] = []
    for dimension in sorted(by_dimension):
        rows = by_dimension[dimension]
        # One reading per trial means each dimension's bands partition the
        # trials, so a single band covering all of them is a channel that
        # never varied -- the normal case for a field an older server never
        # sent, and for a task that never exercised it.
        if keyed and all(coverage == keyed for coverage, *_ in rows):
            if note_constant:
                lines.append(f"{dimension}: one band covers every trial that "
                             f"had a key -- separates nothing here")
            continue
        lines.append(f"{dimension}:")
        for _coverage, band, good, bad in sorted(
                rows, key=lambda row: (-(row[3] / (row[2] + row[3])),
                                       -(row[2] + row[3]))):
            lines.append(f"  {band:<8} | ok {good:>3} | bad {bad:>3} | "
                         f"bad share {bad / (good + bad):.2f}")
    return lines, supported


def _action_table(task: str, mined: Mined, min_support: int) -> list[str]:
    """The same band table, per action, against that action's base rate.

    Skipped when every trial recorded no action: the block would repeat the
    overall table under the heading ``(unknown)`` and say nothing new.
    """
    groups = {
        action: counts
        for (owner, action), counts in mined.action_outcomes.items()
        if owner == task
    }
    if not groups or set(groups) == {UNKNOWN_ACTION}:
        return []
    lines = ["by action (deepest reading per trial per action):"]
    for action in sorted(groups):
        counts = groups[action]
        total = counts["success"] + counts["failure"]
        if total < min_support:
            continue
        silent = mined.action_keyless.get((task, action), Counter())
        silent_total = silent["success"] + silent["failure"]
        lines.append(f"  {action}  ({total} trial{'s' if total != 1 else ''}, "
                     f"{counts['failure']} bad, "
                     f"base bad share {counts['failure'] / total:.2f})")
        if silent_total:
            lines.append(f"    {silent_total} of {total} produced no key "
                         f"under this action")
        entries = {
            (dimension, band): counter
            for (owner, owner_action, dimension, band), counter
            in mined.action_bands.items()
            if owner == task and owner_action == action
        }
        block, supported = _band_table(entries, min_support,
                                       total - silent_total,
                                       note_constant=False)
        if block:
            lines.extend("    " + line for line in block)
        elif supported:
            lines.append("    (no dimension varied within this action)")
        else:
            lines.append(f"    (no band met --min-support {min_support})")
    return lines


@dataclass(frozen=True)
class Candidate:
    """One band that separates outcomes enough to be worth a rule.

    Attributes:
        task: The task the counts come from.
        dimension: The reading the band belongs to.
        band: The band.
        n: Independent trials whose deepest reading fell in it.
        bad: How many of those failed.
        base: The task's base bad share.
        lift: ``bad / n - base``.  Positive means the band holds more
            failures than the task average, negative fewer.
        direction: ``"bad"`` or ``"good"`` -- which side of the base.
        counterexamples: Trials in the band that went the other way, as
            relative paths.  For a good band the failures, for a bad band
            the successes.
        covers: Every trial in the band, as full paths so two roots cannot
            collide.  Candidates with the same ``covers`` are one observation
            seen through different dimensions, not independent support.
        elsewhere: Other tasks where the same band separates in the same
            direction with enough trials to count.
    """

    task: str
    dimension: str
    band: str
    n: int
    bad: int
    base: float
    lift: float
    direction: str
    counterexamples: tuple[str, ...]
    covers: tuple[str, ...]
    elsewhere: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether every trial in the band went the way the band predicts."""
        return not self.counterexamples


def promotion_candidates(mined: Mined, *, min_episodes: int = 3,
                         margin: float = 0.20,
                         min_task_support: int = 3,
                         include_absent: bool = False) -> list[Candidate]:
    """Bands whose bad share sits far enough from the task's base rate.

    This is the automated half of promotion.  A candidate is a band with at
    least ``min_episodes`` independent trials whose bad share differs from
    the task's own base by ``margin``.  The human half is the sentence: the
    miner can say the band separates, never what to do about it.

    A separate decision is whether the evidence is clean.  A candidate with
    counterexamples is reported with them and should stay a hypothesis, or
    have its applicability narrowed until they fall outside it.  Candidates
    keyed on exactly the same trials are one observation seen through
    different dimensions, not independent support, and the table says so.
    Nothing here writes to a bank; the suggestion is printed for a person
    to act on.

    Bands that report an absent reading (``is_absence``) are skipped unless
    ``include_absent`` is set.  They belong with a record's validity, not in
    the condition that decides whether a card applies.
    """
    by_band: dict[tuple[str, str], list[tuple[str, Counter]]] = defaultdict(list)
    for (task, dimension, band), counts in mined.bands.items():
        by_band[(dimension, band)].append((task, counts))

    candidates: list[Candidate] = []
    for (task, dimension, band), counts in mined.bands.items():
        if not include_absent and is_absence(dimension, band):
            continue
        good = counts["success"]
        bad = counts["failure"]
        n = good + bad
        if n < min_episodes:
            continue
        base_counts = mined.outcomes[task]
        total = base_counts["success"] + base_counts["failure"]
        if not total:
            continue
        base = base_counts["failure"] / total
        lift = bad / n - base
        if abs(lift) < margin:
            continue
        direction = "bad" if lift > 0 else "good"
        opposite = "success" if direction == "bad" else "failure"
        evidence = mined.examples.get((task, dimension, band), ())
        counterexamples = tuple(where for where, outcome, _identity in evidence
                                if outcome == opposite)
        covers = tuple(sorted(identity for _where, _outcome, identity
                              in evidence))
        elsewhere: list[str] = []
        for other_task, other_counts in by_band[(dimension, band)]:
            if other_task == task:
                continue
            other_n = other_counts["success"] + other_counts["failure"]
            if other_n < min_task_support:
                continue
            other_total = sum(mined.outcomes[other_task].values())
            if not other_total:
                continue
            other_base = mined.outcomes[other_task]["failure"] / other_total
            other_lift = other_counts["failure"] / other_n - other_base
            if abs(other_lift) >= margin and (other_lift > 0) == (direction == "bad"):
                elsewhere.append(other_task)
        candidates.append(Candidate(
            task=task, dimension=dimension, band=band, n=n, bad=bad,
            base=base, lift=lift, direction=direction,
            counterexamples=counterexamples, covers=covers,
            elsewhere=tuple(elsewhere)))

    candidates.sort(key=lambda candidate: (-abs(candidate.lift),
                                           -candidate.n,
                                           candidate.task,
                                           candidate.dimension,
                                           candidate.band))
    return candidates


def _promotion_table(mined: Mined, min_episodes: int, margin: float,
                     min_task_support: int) -> list[str]:
    """The promotion candidates, with the verdict the evidence supports."""
    candidates = promotion_candidates(mined, min_episodes=min_episodes,
                                      margin=margin,
                                      min_task_support=min_task_support)
    every = promotion_candidates(mined, min_episodes=min_episodes,
                                 margin=margin,
                                 min_task_support=min_task_support,
                                 include_absent=True)
    absent = len(every) - len(candidates)
    lines = [f"## promotion candidates  "
             f"(k={min_episodes} episodes, margin {margin:.2f})"]
    if absent:
        lines.append(f"({absent} band{'s' if absent != 1 else ''} keyed on an "
                     f"absent reading left out -- absence is a validity fact, "
                     f"not a card condition; see the band table above)")
    if not candidates:
        lines.append("(no band separated outcomes by the margin)")
        return lines
    seen: dict[tuple[str, ...], str] = {}
    for candidate in candidates:
        label = f"{candidate.task} {candidate.dimension}={candidate.band}"
        lines.append(
            f"  {label}  "
            f"n={candidate.n}  bad {candidate.bad}  "
            f"bad share {candidate.bad / candidate.n:.2f} vs base "
            f"{candidate.base:.2f}  lift {candidate.lift:+.2f} "
            f"({candidate.direction})")
        twin = seen.get(candidate.covers)
        if twin is not None:
            lines.append(f"    same trials as {twin} -- one observation, "
                         f"not two")
        else:
            seen[candidate.covers] = label
        if candidate.clean:
            lines.append("    no counterexamples -- evidence supports marking "
                         "a card observed")
        else:
            lines.append(f"    {len(candidate.counterexamples)} counterexample"
                         f"{'s' if len(candidate.counterexamples) != 1 else ''}"
                         f" -- keep the card a hypothesis or narrow its scope")
            lines.append("    contradicted by: "
                         + ", ".join(candidate.counterexamples))
        if candidate.elsewhere:
            lines.append("    same direction on: "
                         + ", ".join(candidate.elsewhere))
    return lines


def report(mined: Mined, min_support: int, *,
           promote_episodes: int = 3, promote_margin: float = 0.20,
           min_task_support: int = 3) -> str:
    """The band table, plus the base rates it has to be read against."""
    lines: list[str] = []
    for task in sorted(mined.outcomes):
        base = mined.outcomes[task]
        total = base["success"] + base["failure"]
        if not total:
            continue
        silent = mined.keyless[task]["success"] + mined.keyless[task]["failure"]
        keyed = total - silent
        lines.append(f"\n## {task}  ({total} trials, "
                     f"{base['failure']} bad, base bad share "
                     f"{base['failure'] / total:.2f})")
        if silent:
            # Named, not buried: a silent trial is the most expensive kind of
            # failure to leave out of a denominator, because it looks like a
            # trial that was never run rather than one that never got a grip.
            lines.append(f"{silent} of {total} trials produced no tactile key "
                         f"(no reading carried a depth: never called the gel, "
                         f"crashed, or stopped before contact)")
        entries = {
            (dimension, band): counter
            for (owner, dimension, band), counter in mined.bands.items()
            if owner == task
        }
        block, _supported = _band_table(entries, min_support, keyed)
        if block:
            lines.extend(block)
        else:
            lines.append(f"(no band met --min-support {min_support})")
        action_lines = _action_table(task, mined, min_support)
        if action_lines:
            lines.append("")
            lines.extend(action_lines)
    if lines:
        lines.append("")
        lines.extend(_promotion_table(mined, promote_episodes, promote_margin,
                                      min_task_support))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", default=[],
                        help="a tree to walk for trial_* directories "
                             "(outcome taken from the directory name)")
    parser.add_argument("--out", help="write the report here as well as printing it")
    parser.add_argument("--min-support", type=int, default=3,
                        help="trials a band needs before it is reported")
    parser.add_argument("--promote-episodes", type=int, default=3,
                        help="independent trials a promotion candidate needs")
    parser.add_argument("--promote-margin", type=float, default=0.20,
                        help="how far a band's bad share must sit from the "
                             "task base before it is a candidate")
    parser.add_argument("--min-task-support", type=int, default=3,
                        help="trials a second task needs before it can "
                             "corroborate a cross-task candidate")
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("give at least one --root")

    mined = collect([Path(root) for root in args.root])
    text = (report(mined, args.min_support,
                   promote_episodes=args.promote_episodes,
                   promote_margin=args.promote_margin,
                   min_task_support=args.min_task_support).strip()
            or "(no labelled trials found)")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
