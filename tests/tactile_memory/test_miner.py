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


def trial(root: Path, task: str, outcome: str | None = "1",
          number: int = 1) -> Path:
    """A trial directory laid out the way the runner writes them."""
    stem = f"trial_{number:02d}"
    name = stem if outcome is None else f"{stem}_taskcompleted_{outcome}"
    path = root / "some-model" / f"univtac_{task}" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def traced(directory: Path, readings: dict, rows: int = 4,
           trial: int | None = None) -> Path:
    path = directory / "tactile_trace.jsonl"
    path.write_text("\n".join(
        json.dumps({"t": 0.0, "trial": trial, "call": "/tactile",
                    "args": {}, "readings": readings})
        for _ in range(rows)) + "\n", encoding="utf-8")
    return path


def run_trace(run_dir: Path, per_trial: dict[int, tuple[dict, int]],
              run: str | None = None) -> Path:
    """One trace file for the whole run, split by its ``trial`` column."""
    lines = []
    for number, (readings, rows) in per_trial.items():
        lines += [json.dumps({"t": 0.0, "run": run, "trial": number,
                              "call": "/tactile", "args": {},
                              "readings": readings})] * rows
    path = run_dir / "tactile_trace.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    traced(trial(tmp_path, "blind_pick_balls", "1", number=1), GOOD, rows=3)
    traced(trial(tmp_path, "blind_pick_balls", "0", number=2), BAD, rows=2)
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    # The dead pad is the only trial with a split displacement gap.
    assert bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp_gap", "even")] == {"success": 1}
    assert bands[("blind_pick_balls", "disp", "heavy")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp", "moving")] == {"success": 1}


def test_two_runs_of_the_same_task_are_counted_separately(miner, tmp_path):
    # Every run numbers its trials from 01: the number is unique inside a
    # run, not across the tree, and two models' trial 01 are two trials.
    for model, readings, outcome in (("alpha", GOOD, "1"), ("beta", BAD, "0")):
        directory = (tmp_path / model / "univtac_lift_can"
                     / f"trial_01_taskcompleted_{outcome}")
        directory.mkdir(parents=True)
        traced(directory, readings)
    outcomes, _bands, _keyless = miner.collect([tmp_path])
    assert outcomes["lift_can"] == {"success": 1, "failure": 1}


def test_an_unlabelled_trial_directory_is_skipped(miner, tmp_path):
    traced(trial(tmp_path, "lift_can", None), GOOD)
    outcomes, _bands, _keyless = miner.collect([tmp_path])
    assert dict(outcomes) == {}


def test_a_trial_with_no_readings_is_still_a_labelled_trial(miner, tmp_path):
    # A crash leaves a directory and nothing else.  It is not a trial with a
    # clean record; it is a failure that never said why, and leaving it out
    # would flatter the base rate every band is read against.
    trial(tmp_path, "lift_can", "1")
    outcomes, _bands, keyless = miner.collect([tmp_path])
    assert dict(outcomes) == {"lift_can": {"success": 1}}
    assert dict(keyless) == {"lift_can": {"success": 1}}


# ---------------------------------------------------------------------------
# Where the readings come from
# ---------------------------------------------------------------------------


def test_a_summary_fallback_carries_the_trial(miner, tmp_path):
    # No trace: the printed line is what a historical trial has left.
    directory = trial(tmp_path, "weight_classify", "1")
    summarised(directory, ["step 1: something", PRINTED])
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["weight_classify"] == {"success": 1}
    assert bands[("weight_classify", "depth", "face")] == {"success": 1}


def test_a_trace_wins_over_the_summary(miner, tmp_path):
    directory = trial(tmp_path, "weight_classify", "0")
    traced(directory, BAD)
    summarised(directory, [PRINTED])
    _outcomes, bands, _keyless = miner.collect([tmp_path])
    assert bands[("weight_classify", "disp", "heavy")] == {"failure": 1}
    assert ("weight_classify", "disp", "moving") not in bands


def test_a_half_written_line_does_not_fail_the_run(miner, tmp_path):
    path = traced(trial(tmp_path, "lift_can"), GOOD)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"t": 1.0, "readings": {')
    assert len(miner.read_trace(path)) == 4


def test_the_key_is_taken_at_the_deepest_reading(miner, tmp_path):
    # Two looks at a pad that has come in and stopped moving.  The trial is
    # one situation, not three: the deepest reading, with the reading
    # before it as its "previous".
    directory = trial(tmp_path, "insert_hole", "0")
    rows = [{**GOOD, "left_depth_mm": 31.0, "right_depth_mm": 31.0},
            {**GOOD, "left_depth_mm": 27.0, "right_depth_mm": 27.0},
            {**GOOD, "left_depth_mm": 24.0, "right_depth_mm": 24.0}]
    path = directory / "tactile_trace.jsonl"
    path.write_text("\n".join(
        json.dumps({"t": float(index), "call": "/tactile", "args": {},
                    "readings": row}) for index, row in enumerate(rows)) + "\n",
        encoding="utf-8")
    readings = [row for _run, _trial, row in miner.read_trace(path)]
    assert miner.deepest(readings) == 2
    bands = miner.bands_of(readings)
    assert "depth=floor" in bands          # 24.5 mm, at the very end
    assert "depth=face" not in bands
    assert "depth_trend=falling" in bands  # still coming in when it stopped


# ---------------------------------------------------------------------------
# One trace for the whole run
# ---------------------------------------------------------------------------


def test_a_run_trace_is_split_by_its_trial_column(miner, tmp_path):
    # What the runner leaves today: one process, one file, every trial in it.
    run_dir = trial(tmp_path, "blind_pick_balls", "1", number=1).parent
    trial(tmp_path, "blind_pick_balls", "1", number=1)
    trial(tmp_path, "blind_pick_balls", "0", number=2)
    run_trace(run_dir, {1: (GOOD, 3), 2: (BAD, 2)})
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    assert bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp_gap", "even")] == {"success": 1}


def test_a_row_from_a_trial_with_no_directory_is_not_counted(miner, tmp_path):
    # A crashed trial leaves rows but no artifacts, so its outcome is
    # unknown; guessing one would put a made-up label behind a band.
    run_dir = trial(tmp_path, "blind_pick_balls", "1", number=1).parent
    run_trace(run_dir, {1: (GOOD, 2), 9: (BAD, 2)})
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1}
    assert ("blind_pick_balls", "disp_gap", "split") not in bands


def test_a_trial_in_the_run_trace_is_not_counted_twice_from_its_summary(
        miner, tmp_path):
    directory = trial(tmp_path, "blind_pick_balls", "1", number=1)
    run_trace(directory.parent, {1: (GOOD, 2)})
    summarised(directory, [PRINTED])
    outcomes, _bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1}


def test_a_rerun_appending_to_the_same_file_is_a_second_trial(miner, tmp_path):
    # The runner appends, and a re-run repeats its trial numbers, so both
    # attempts sit under trial=8 in one file.  They are two trials, and a
    # trial's rows must not be mixed with the other attempt's.
    directory = trial(tmp_path, "blind_pick_balls", "0", number=8)
    lines = []
    for run, readings in (("first", GOOD), ("second", BAD)):
        lines += [json.dumps({"t": 0.0, "run": run, "trial": 8,
                              "call": "/tactile", "args": {},
                              "readings": readings})] * 2
    (directory.parent / "tactile_trace.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"failure": 2}
    assert bands[("blind_pick_balls", "disp_gap", "even")] == {"failure": 1}
    assert bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}


def test_rows_naming_two_directories_are_dropped_for_their_summaries(
        miner, tmp_path):
    # A re-run that ends differently leaves two directories for trial 8.
    # The run column says which rows are one attempt, not which attempt a
    # directory is, so the rows cannot be labelled with an outcome.  The
    # printed lines sit inside the directory they belong to and can.
    failed = trial(tmp_path, "blind_pick_balls", "0", number=8)
    passed = trial(tmp_path, "blind_pick_balls", "1", number=8)
    run_trace(failed.parent, {8: (BAD, 2)})
    summarised(failed, [PRINTED])
    summarised(passed, [PRINTED])
    outcomes, bands, _keyless = miner.collect([tmp_path])
    assert outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    assert ("blind_pick_balls", "disp", "heavy") not in bands


def test_two_directories_for_one_number_are_two_trials(miner, tmp_path):
    summarised(trial(tmp_path, "lift_can", "0", number=3), [PRINTED])
    summarised(trial(tmp_path, "lift_can", "1", number=3), [PRINTED])
    outcomes, _bands, _keyless = miner.collect([tmp_path])
    assert outcomes["lift_can"] == {"success": 1, "failure": 1}


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
    traced(trial(tmp_path, "blind_pick", "1", number=1), GOOD)
    traced(trial(tmp_path, "blind_pick", "0", number=2), BAD)
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
    # Every trial read the same shear band, so it separates nothing and
    # saying "bad share 0.50" about it would be noise dressed as a finding.
    traced(trial(tmp_path, "lift_can", "1", number=1), GOOD)
    traced(trial(tmp_path, "lift_can", "0", number=2), BAD)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "shear: one band covers every trial that had a key" in out


def test_a_silent_trial_is_named_in_the_report(miner, tmp_path, capsys):
    # The number has to be visible next to the base rate, because a silent
    # trial is a failure that looks like a trial nobody ran.
    traced(trial(tmp_path, "lift_can", "1", number=1), GOOD)
    trial(tmp_path, "lift_can", "0", number=2)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "## lift_can  (2 trials, 1 bad, base bad share 0.50)" in out
    assert "1 of 2 trials produced no tactile key" in out


def test_no_roots_is_a_usage_error(miner):
    with pytest.raises(SystemExit):
        miner.main([])
