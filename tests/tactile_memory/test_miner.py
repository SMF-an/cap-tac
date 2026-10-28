"""Tests for the miner: trial directories in, band/outcome table out.

The miner is the step that decides whether a card bank is worth building at
all (M0), so what it must get right is attribution: which task a trial
belongs to, which outcome it ended with, and which readings belong to it.
The numbers themselves are the operator's to read.
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
    "left_marker_displacement_px": 2.4, "right_marker_displacement_px": 2.3,
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


def trial(root: Path, task: str, outcome: str | None = "1") -> Path:
    """A trial directory laid out the way the runner writes them."""
    name = "trial_01" if outcome is None else f"trial_01_taskcompleted_{outcome}"
    path = root / "some-model" / f"univtac_{task}" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def traced(directory: Path, readings: dict, rows: int = 4) -> Path:
    path = directory / "tactile_trace.jsonl"
    path.write_text("\n".join(
        json.dumps({"t": 0.0, "call": "/tactile", "args": {}, "readings": readings})
        for _ in range(rows)) + "\n", encoding="utf-8")
    return path


def summarised(directory: Path, lines: list[str]) -> Path:
    path = directory / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


PRINTED = ("Tactile: L_contact=True R_contact=True L_depth=27.90mm "
           "R_depth=27.90mm L_disp=2.40px R_disp=2.30px L_coh=0.52 R_coh=0.55 ")


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_a_trace_is_attributed_to_the_task_in_the_path(miner, tmp_path):
    path = traced(trial(tmp_path, "blind_pick_balls"), GOOD)
    assert miner.task_of(path) == "blind_pick_balls"


def test_a_trace_outside_the_naming_convention_names_no_task(miner, tmp_path):
    loose = tmp_path / "somewhere" / "tactile_trace.jsonl"
    loose.parent.mkdir(parents=True, exist_ok=True)
    loose.write_text("", encoding="utf-8")
    assert miner.task_of(loose) == ""


def test_the_outcome_comes_from_the_directory_name(miner, tmp_path):
    assert miner.outcome_of(trial(tmp_path, "lift_can", "1")) == "success"
    assert miner.outcome_of(trial(tmp_path, "lift_can", "0")) == "failure"
    assert miner.outcome_of(trial(tmp_path, "lift_can", None)) is None


def test_success_and_failure_trials_are_counted_apart(miner, tmp_path):
    traced(trial(tmp_path, "blind_pick_balls", "1"), GOOD, rows=3)
    traced(trial(tmp_path, "blind_pick_balls", "0"), BAD, rows=2)
    outcomes, bands = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    # The dead pad is the only trial with a split displacement gap.
    assert bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp_gap", "even")] == {"success": 1}
    assert bands[("blind_pick_balls", "disp", "heavy")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp", "moving")] == {"success": 1}


def test_an_unlabelled_trial_directory_is_skipped(miner, tmp_path):
    traced(trial(tmp_path, "lift_can", None), GOOD)
    outcomes, _bands = miner.collect([tmp_path])
    assert dict(outcomes) == {}


def test_a_trial_with_no_readings_contributes_nothing(miner, tmp_path):
    trial(tmp_path, "lift_can", "1")
    outcomes, _bands = miner.collect([tmp_path])
    assert dict(outcomes) == {}


# ---------------------------------------------------------------------------
# Where the readings come from
# ---------------------------------------------------------------------------


def test_a_summary_fallback_carries_the_trial(miner, tmp_path):
    # No trace: the printed line is what a historical trial has left.
    directory = trial(tmp_path, "weight_classify", "1")
    summarised(directory, ["step 1: something", PRINTED])
    outcomes, bands = miner.collect([tmp_path])
    assert outcomes["weight_classify"] == {"success": 1}
    assert bands[("weight_classify", "depth", "face")] == {"success": 1}


def test_a_trace_wins_over_the_summary(miner, tmp_path):
    directory = trial(tmp_path, "weight_classify", "0")
    traced(directory, BAD)
    summarised(directory, [PRINTED])
    _outcomes, bands = miner.collect([tmp_path])
    assert bands[("weight_classify", "disp", "heavy")] == {"failure": 1}
    assert ("weight_classify", "disp", "moving") not in bands


def test_a_half_written_line_does_not_fail_the_run(miner, tmp_path):
    path = traced(trial(tmp_path, "lift_can"), GOOD)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"t": 1.0, "readings": {')
    assert len(miner.read_trace(path)) == 4


def test_readings_are_replayed_in_order_so_trends_appear(miner, tmp_path):
    # Two looks at a pad that has come in and stopped moving.
    directory = trial(tmp_path, "insert_hole", "0")
    rows = [{**GOOD, "left_depth_mm": 31.0, "right_depth_mm": 31.0},
            {**GOOD, "left_depth_mm": 27.0, "right_depth_mm": 27.0},
            {**GOOD, "left_depth_mm": 27.0, "right_depth_mm": 27.0}]
    path = directory / "tactile_trace.jsonl"
    path.write_text("\n".join(
        json.dumps({"t": float(index), "call": "/tactile", "args": {},
                    "readings": row}) for index, row in enumerate(rows)) + "\n",
        encoding="utf-8")
    assert "depth_trend=falling" in miner.bands_of(directory)
    assert "depth_trend=static" in miner.bands_of(directory)
    assert "depth_trend=unknown" in miner.bands_of(directory)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _rows_under(text: str, dimension: str) -> list[str]:
    """The band rows of one dimension's block in the report."""
    lines = text.splitlines()
    start = lines.index(f"{dimension}:")
    rows = []
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        rows.append(line.strip())
    return rows


def test_the_report_is_read_against_the_tasks_own_base_rate(miner, tmp_path, capsys):
    traced(trial(tmp_path, "blind_pick", "1"), GOOD)
    traced(trial(tmp_path, "blind_pick", "0"), BAD)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "## blind_pick  (2 trials, 1 bad, base bad share 0.50)" in out
    split = [row for row in _rows_under(out, "disp_gap") if row.startswith("split")]
    assert split and split[0].endswith("bad share 1.00"), split


def test_a_band_below_the_support_floor_is_not_reported(miner, tmp_path, capsys):
    traced(trial(tmp_path, "lift_can", "1"), GOOD, rows=1)
    miner.main(["--root", str(tmp_path), "--min-support", "5"])
    assert "no band met --min-support 5" in capsys.readouterr().out


def test_a_dimension_that_never_varies_is_not_ranked(miner, tmp_path, capsys):
    # Every trial saw the same shear band, so it separates nothing and
    # saying "bad share 0.50" about it would be noise dressed as a finding.
    traced(trial(tmp_path, "lift_can", "1"), GOOD)
    traced(trial(tmp_path, "lift_can", "0"), BAD)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "shear: no band excludes any trial" in out


def test_no_roots_is_a_usage_error(miner):
    with pytest.raises(SystemExit):
        miner.main([])
