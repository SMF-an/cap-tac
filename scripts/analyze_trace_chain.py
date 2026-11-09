#!/usr/bin/env python3
"""Summarize a cross-seed trace chain.

Reads ``<chain_dir>/seed_*.json`` (one record per seed, written by
``capx.envs.trace_chain``), optionally annotates each seed with the simulator's
own ``truth=`` line for grasp_classify, and writes a CSV plus a Markdown table.

The success measure is ``task_completed`` -- the task's own verdict.  The
trial's ``sandbox_rc`` only says whether the generated code ran, and
``TraceRecord.reward`` mirrors the environment reward, not the classification.

Usage::

    python scripts/analyze_trace_chain.py \
        outputs/<model>/univtac_grasp_classify/trace_chain
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_FILE = re.compile(r"^seed_(\d+)\.json$")
INJECTED_FILE = re.compile(r"^seed_(\d+)_injected_block\.txt$")
RESET_LINE = re.compile(
    r"--- reset task=(?P<task>\S+) seed=(?P<seed>\d+) truth=(?P<truth>\S+) ---"
)


@dataclass
class Row:
    seed: int
    truth: str
    injected_from: int | None
    success: bool
    task_completed: bool | None
    reward: float
    sandbox_rc: int
    num_code_blocks: int
    num_regenerations: int
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    timed_out: bool
    complete: bool
    injected_chars: int
    cumulative_success: float
    check_detail: str


def _load_records(chain_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(chain_dir.iterdir()):
        match = TRACE_FILE.match(path.name)
        if not match:
            continue
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            print(f"[analyze] skipping unreadable {path}: {exc}")
    records.sort(key=lambda r: int(r.get("seed", 0)))
    return records


def _injected_lengths(chain_dir: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for path in chain_dir.iterdir():
        match = INJECTED_FILE.match(path.name)
        if match:
            lengths[int(match.group(1))] = len(path.read_text())
    return lengths


def _truth_by_seed(log_path: Path, task: str) -> dict[int, str]:
    """The last ``truth=`` line per seed for *task*.

    The calibration log is append-only across runs, so the last line for a
    given (task, seed) is the most recent episode.  This is analysis-only
    annotation; nothing here enters a prompt.
    """
    if not log_path.exists():
        return {}
    truth: dict[int, str] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        match = RESET_LINE.search(line)
        if match and match.group("task") == task:
            truth[int(match.group("seed"))] = match.group("truth")
    return truth


def _check_detail_text(record: dict[str, Any]) -> str:
    """Flatten the non-oracle check diagnostics into one cell."""
    detail = record.get("check_detail")
    if not isinstance(detail, dict):
        return ""
    parts = [str(reason) for reason in detail.get("reasons", [])]
    pos_error = detail.get("pos_error_cm")
    if isinstance(pos_error, (list, tuple)):
        parts.append("pos_error_cm=" + ", ".join(f"{float(v):.2f}" for v in pos_error))
    elif isinstance(pos_error, (int, float)):
        parts.append(f"pos_error_cm={float(pos_error):.2f}")
    tilt_cos = detail.get("tilt_cos")
    if isinstance(tilt_cos, (int, float)):
        parts.append(f"tilt_cos={float(tilt_cos):.4f}")
    return "; ".join(parts)


def _rows(
    records: list[dict[str, Any]],
    injected: dict[int, int],
    truth: dict[int, str],
) -> list[Row]:
    rows: list[Row] = []
    successes = 0
    for index, record in enumerate(records, start=1):
        seed = int(record.get("seed", 0))
        task_completed = record.get("task_completed")
        # JSON may carry a bool, 1/0 or null; only a present truthy verdict is
        # a task success.  sandbox_rc is deliberately not consulted here.
        success = task_completed is not None and bool(task_completed)
        successes += int(success)
        rows.append(
            Row(
                seed=seed,
                truth=truth.get(seed, "-"),
                injected_from=record.get("injected_from_seed"),
                success=success,
                task_completed=task_completed,
                reward=float(record.get("reward", 0.0)),
                sandbox_rc=int(record.get("sandbox_rc", 1)),
                num_code_blocks=int(record.get("num_code_blocks", 0)),
                num_regenerations=int(record.get("num_regenerations", 0)),
                input_tokens=int(record.get("input_tokens", 0)),
                output_tokens=int(record.get("output_tokens", 0)),
                elapsed_s=float(record.get("elapsed_s", 0.0)),
                timed_out=bool(record.get("timed_out", False)),
                complete=bool(record.get("complete", False)),
                injected_chars=injected.get(seed, 0),
                cumulative_success=successes / index,
                check_detail=_check_detail_text(record),
            )
        )
    return rows


def _write_csv(path: Path, rows: list[Row]) -> None:
    fields = list(Row.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def _rate(rows: list[Row], predicate) -> str:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return "n/a"
    wins = sum(row.success for row in selected)
    return f"{wins}/{len(selected)} ({wins / len(selected):.0%})"


def _write_markdown(path: Path, rows: list[Row], chain_dir: Path) -> None:
    lines = [
        "# Cross-seed trace chain",
        "",
        f"Chain directory: `{chain_dir}`",
        "",
        f"- Seeds recorded: **{len(rows)}**",
        f"- Task success (task_completed): **{_rate(rows, lambda r: True)}**",
    ]
    # Only grasp_classify's reset log carries a truth= label.  On any other
    # task these two lines would count a column of "-" and read as 0%.
    if any(row.truth != "-" for row in rows):
        lines += [
            f"- Rough seeds: {_rate(rows, lambda r: r.truth == 'rough')}",
            f"- Plain seeds: {_rate(rows, lambda r: r.truth == 'plain')}",
        ]
    lines += [
        f"- Seeds that received a trace: "
        f"**{sum(1 for r in rows if r.injected_from is not None)}/{len(rows)}**",
        f"- Timed out: {sum(1 for r in rows if r.timed_out)}",
        "",
        "| seed | truth | from | success | reward | rc | blocks | regen | in tok | wall s | trace chars | cum |",
        "|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.truth} | "
            f"{row.injected_from if row.injected_from is not None else '-'} | "
            f"{'Y' if row.success else 'N'} | {row.reward:.3f} | {row.sandbox_rc} | "
            f"{row.num_code_blocks} | {row.num_regenerations} | {row.input_tokens} | "
            f"{row.elapsed_s:.1f} | {row.injected_chars} | "
            f"{row.cumulative_success:.2f} |"
        )
    lines += [
        "",
        "## Failure details",
        "",
    ]
    failed = [row for row in rows if not row.success and row.check_detail]
    if failed:
        lines += [f"- seed {row.seed}: {row.check_detail}" for row in failed]
    else:
        lines.append("No failure diagnostics were recorded.")
    lines += [
        "",
        "`success` is the task's `task_completed`, not `sandbox_rc`.  `truth` is",
        "read from the simulator's calibration log for analysis only; it was",
        "never shown to the model.  `check_detail` is the non-oracle part of",
        "the task's check detail -- axis errors and tilt, no labels or poses.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain_dir", type=Path, help=".../<output_dir>/trace_chain")
    parser.add_argument(
        "--task",
        default="grasp_classify",
        help="task name used to match truth= lines (default: grasp_classify)",
    )
    parser.add_argument(
        "--calibration-log",
        type=Path,
        default=Path("outputs/univtac_tactile_calibration.log"),
        help="server reset log carrying truth=rough/plain lines",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="write <prefix>.csv and <prefix>.md (default: chain_dir/chain_summary)",
    )
    args = parser.parse_args()

    chain_dir = args.chain_dir.expanduser().resolve()
    if not chain_dir.is_dir():
        raise SystemExit(f"chain directory not found: {chain_dir}")

    records = _load_records(chain_dir)
    if not records:
        raise SystemExit(f"no seed_*.json records in {chain_dir}")

    rows = _rows(
        records,
        _injected_lengths(chain_dir),
        _truth_by_seed(args.calibration_log, args.task),
    )
    prefix = args.output_prefix or (chain_dir / "chain_summary")
    _write_csv(prefix.with_suffix(".csv"), rows)
    _write_markdown(prefix.with_suffix(".md"), rows, chain_dir)

    print(f"Wrote {prefix.with_suffix('.csv')}")
    print(f"Wrote {prefix.with_suffix('.md')}")
    print(f"Task success: {_rate(rows, lambda r: True)}")
    if any(row.truth != "-" for row in rows):
        print(f"Rough seeds:  {_rate(rows, lambda r: r.truth == 'rough')}")
        print(f"Plain seeds:  {_rate(rows, lambda r: r.truth == 'plain')}")


if __name__ == "__main__":
    main()
