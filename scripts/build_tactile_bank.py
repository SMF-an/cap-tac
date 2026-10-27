#!/usr/bin/env python
"""Mine tactile keys out of traced trials: which bands belong to which outcome.

The library's cards are supposed to be earned, not invented.  This script is
the earning step: it reads the JSONL that ``CAPX_TACTILE_TRACE`` writes, keys
every reading with :func:`capx.tactile_memory.signature.key_from_readings`,
and reports, per task and per band, how many readings came from trials that
worked and how many from trials that did not.

It writes a report, never a bank.  A band that separates outcomes is a fact
about the sensor; the line a card carries is a judgement about what to do
with that fact, and a judgement is the one thing that cannot be mined.  Read
the report, pick the bands that separate, write the card by hand.

The gate this script exists to answer: if the bands do not separate outcomes
on the trials already run, a card bank cannot help, and the work stops here
rather than after a month of prompt tuning.

Example:
    python scripts/build_tactile_bank.py \\
        --success outputs/oracle/univtac_blind_pick_balls \\
        --failure outputs/gpt-6-astra/univtac_blind_pick_balls

Readings are not independent samples -- one trial contributes hundreds of
them, and they are correlated by construction.  So the counts answer "which
bands does a failing trial pass through" and nothing about significance; a
band that only ever appears in failures is worth a look, a band that appears
everywhere is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.tactile_memory.signature import DIMENSIONS, key_from_readings  # noqa: E402
from capx.tactile_memory.tasks import TASK_TYPES  # noqa: E402

TRACE_NAME = "tactile_trace.jsonl"


def traces(root: Path) -> list[Path]:
    """Every trace file under one run root, in a stable order."""
    return sorted(root.rglob(TRACE_NAME))


def task_of(path: Path) -> str:
    """The task a trace belongs to, from the directories above it.

    Run trees are laid out as ``<model>/univtac_<task>/...``, so the task is
    the nearest ancestor whose name is one the library knows.  A trace that
    sits nowhere near such a directory is attributed to no task, and its
    readings can only speak to cross-task bands.
    """
    for parent in path.parents:
        name = parent.name.removeprefix("univtac_")
        if name in TASK_TYPES:
            return name
    return ""


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written last line is not a reason to fail
    return rows


def tally(paths: list[Path], outcome: str,
          counts: dict, per_trial: dict) -> dict:
    """Fold every reading under these roots into the band counts."""
    for path in paths:
        task = task_of(path)
        per_trial.setdefault(str(path), Counter())[outcome] += 1
        for row in read_rows(path):
            key = key_from_readings(row.get("readings") or {})
            for dimension in DIMENSIONS:
                counts[(task or "(unknown)", dimension, key[dimension])][outcome] += 1
    return counts


def report(counts: dict, min_support: int) -> str:
    lines = ["task | dimension | band | from ok | from bad | bad share", "-" * 72]
    rows = []
    for (task, dimension, band), by_outcome in counts.items():
        good, bad = by_outcome["success"], by_outcome["failure"]
        if good + bad < min_support:
            continue
        rows.append((bad / (good + bad), task, dimension, band, good, bad))
    for share, task, dimension, band, good, bad in sorted(rows, reverse=True):
        lines.append(f"{task} | {dimension} | {band} | {good} | {bad} | {share:.2f}")
    if not rows:
        lines.append("(nothing reached --min-support -- run more traced trials)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--success", nargs="*", default=[],
                        help="run directories whose trials worked")
    parser.add_argument("--failure", nargs="*", default=[],
                        help="run directories whose trials did not")
    parser.add_argument("--min-support", type=int, default=3,
                        help="readings a band needs before it is reported")
    args = parser.parse_args(argv)

    if not args.success and not args.failure:
        parser.error("give at least one --success or --failure root")

    counts: dict = defaultdict(Counter)
    per_trial: dict = {}
    tally([p for root in args.success for p in traces(Path(root))],
          "success", counts, per_trial)
    tally([p for root in args.failure for p in traces(Path(root))],
          "failure", counts, per_trial)

    print(report(counts, args.min_support))
    print()
    print(f"{len(per_trial)} traced trials: "
          f"{sum(1 for c in per_trial.values() if c['success'])} ok, "
          f"{sum(1 for c in per_trial.values() if c['failure'])} bad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
