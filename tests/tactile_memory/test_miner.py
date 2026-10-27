"""Tests for the miner: trace rows in, band/outcome table out.

The miner is the step that decides whether a card bank is worth building at
all, so what it must get right is attribution: which task a trace belongs to,
which outcome it belongs to, and which readings it is allowed to count.  The
numbers themselves are the operator's to read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_tactile_bank.py"

GOOD = {
    "left_contact": True, "right_contact": True,
    "left_depth_mm": 27.9, "right_depth_mm": 27.9,
    "left_marker_displacement_px": 2.4, "right_marker_displacement_px": 2.2,
    "left_marker_coherence": 0.52, "right_marker_coherence": 0.55,
}
BAD = {
    "left_contact": True, "right_contact": True,
    "left_depth_mm": 28.0, "right_depth_mm": 28.1,
    "left_marker_displacement_px": 22.8, "right_marker_displacement_px": 1.6,
    "left_marker_coherence": 0.09, "right_marker_coherence": 0.62,
}


@pytest.fixture
def miner():
    spec = importlib.util.spec_from_file_location("build_tactile_bank", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def traced(root: Path, task: str, readings: dict, rows: int = 4) -> Path:
    path = root / "univtac" / f"univtac_{task}" / "trial_01" / "tactile_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(
        json.dumps({"t": 0.0, "call": "/tactile", "args": {}, "readings": readings})
        for _ in range(rows)) + "\n", encoding="utf-8")
    return path


def test_readings_are_attributed_to_the_task_in_the_path(miner, tmp_path):
    path = traced(tmp_path, "blind_pick_balls", GOOD)
    assert miner.task_of(path) == "blind_pick_balls"


def test_a_trace_outside_the_naming_convention_names_no_task(miner, tmp_path):
    loose = tmp_path / "somewhere" / "tactile_trace.jsonl"
    loose.parent.mkdir(parents=True, exist_ok=True)
    loose.write_text("", encoding="utf-8")
    assert miner.task_of(loose) == ""


def test_success_and_failure_readings_are_counted_apart(miner, tmp_path, capsys):
    ok = traced(tmp_path / "ok", "blind_pick_balls", GOOD, rows=3)
    bad = traced(tmp_path / "bad", "blind_pick_balls", BAD, rows=2)
    miner.main(["--success", str(ok.parents[3]), "--failure", str(bad.parents[3]),
                "--min-support", "1"])
    out = capsys.readouterr().out
    assert "task | dimension | band | from ok | from bad" in out
    # The dead-pad trace is the only one with a split displacement gap.
    split = [line for line in out.splitlines() if "| disp_gap | split |" in line]
    assert split and split[0].endswith("| 0 | 2 | 1.00"), split
    kept = [line for line in out.splitlines() if "| disp | moving |" in line]
    assert kept and kept[0].endswith("| 3 | 0 | 0.00"), kept
    drag = [line for line in out.splitlines() if "| disp | heavy |" in line]
    assert drag and drag[0].endswith("| 0 | 2 | 1.00"), drag


def test_a_half_written_line_does_not_fail_the_run(miner, tmp_path):
    path = traced(tmp_path, "lift_can", GOOD)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"t": 1.0, "readings": {')
    assert len(miner.read_rows(path)) == 4


def test_a_band_below_the_support_floor_is_not_reported(miner, tmp_path, capsys):
    ok = traced(tmp_path / "ok", "lift_can", GOOD, rows=1)
    miner.main(["--success", str(ok.parents[3]), "--min-support", "5"])
    assert "nothing reached" in capsys.readouterr().out


def test_no_roots_is_a_usage_error(miner):
    with pytest.raises(SystemExit):
        miner.main([])
