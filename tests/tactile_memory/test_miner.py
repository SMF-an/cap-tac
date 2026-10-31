"""Tests for the miner: trial directories in, band/outcome table out.

The miner is the step that decides whether a card bank is worth building at
all (M0), so what it must get right is attribution: which task a trial
belongs to, which outcome it ended with, and which readings belong to it.
The numbers themselves are the operator's to read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
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
    # Dataclasses resolve string annotations through sys.modules; a module
    # that is exec'd but never registered would fail at class creation.
    sys.modules[spec.name] = module
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
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    # The dead pad is the only trial with a split displacement gap.
    assert mined.bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}
    assert mined.bands[("blind_pick_balls", "disp_gap", "even")] == {"success": 1}
    assert mined.bands[("blind_pick_balls", "disp", "heavy")] == {"failure": 1}
    assert mined.bands[("blind_pick_balls", "disp", "moving")] == {"success": 1}


def test_two_runs_of_the_same_task_are_counted_separately(miner, tmp_path):
    # Every run numbers its trials from 01: the number is unique inside a
    # run, not across the tree, and two models' trial 01 are two trials.
    for model, readings, outcome in (("alpha", GOOD, "1"), ("beta", BAD, "0")):
        directory = (tmp_path / model / "univtac_lift_can"
                     / f"trial_01_taskcompleted_{outcome}")
        directory.mkdir(parents=True)
        traced(directory, readings)
    mined = miner.collect([tmp_path])
    assert mined.outcomes["lift_can"] == {"success": 1, "failure": 1}


def test_two_roots_with_the_same_relative_path_are_two_trials(miner, tmp_path):
    # Two roots mined in one run, each with its own copy of the layout.  The
    # printed paths coincide, so anything that decides whether two trials are
    # the same -- the promotion table's "one observation" note -- must use the
    # full path instead.
    left, right = tmp_path / "left", tmp_path / "right"
    traced(trial(left, "lift_can", "0"), BAD)
    traced(trial(right, "lift_can", "0"), GOOD)
    mined = miner.collect([left, right])
    assert mined.outcomes["lift_can"] == {"failure": 2}
    split = mined.examples[("lift_can", "disp_gap", "split")][0]
    even = mined.examples[("lift_can", "disp_gap", "even")][0]
    assert split[0] == even[0]          # same path relative to its root
    assert split[2] != even[2]          # different trials


def test_an_unlabelled_trial_directory_is_skipped(miner, tmp_path):
    traced(trial(tmp_path, "lift_can", None), GOOD)
    mined = miner.collect([tmp_path])
    assert dict(mined.outcomes) == {}


def test_a_trial_with_no_readings_is_still_a_labelled_trial(miner, tmp_path):
    # A crash leaves a directory and nothing else.  It is not a trial with a
    # clean record; it is a failure that never said why, and leaving it out
    # would flatter the base rate every band is read against.
    trial(tmp_path, "lift_can", "1")
    mined = miner.collect([tmp_path])
    assert dict(mined.outcomes) == {"lift_can": {"success": 1}}
    assert dict(mined.keyless) == {"lift_can": {"success": 1}}


# ---------------------------------------------------------------------------
# Where the readings come from
# ---------------------------------------------------------------------------


def test_a_summary_fallback_carries_the_trial(miner, tmp_path):
    # No trace: the printed line is what a historical trial has left.
    directory = trial(tmp_path, "weight_classify", "1")
    summarised(directory, ["step 1: something", PRINTED])
    mined = miner.collect([tmp_path])
    assert mined.outcomes["weight_classify"] == {"success": 1}
    assert mined.bands[("weight_classify", "depth", "face")] == {"success": 1}


def test_a_trace_wins_over_the_summary(miner, tmp_path):
    directory = trial(tmp_path, "weight_classify", "0")
    traced(directory, BAD)
    summarised(directory, [PRINTED])
    mined = miner.collect([tmp_path])
    assert mined.bands[("weight_classify", "disp", "heavy")] == {"failure": 1}
    assert ("weight_classify", "disp", "moving") not in mined.bands


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
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    assert mined.bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}
    assert mined.bands[("blind_pick_balls", "disp_gap", "even")] == {"success": 1}


def test_a_row_from_a_trial_with_no_directory_is_not_counted(miner, tmp_path):
    # A crashed trial leaves rows but no artifacts, so its outcome is
    # unknown; guessing one would put a made-up label behind a band.
    run_dir = trial(tmp_path, "blind_pick_balls", "1", number=1).parent
    run_trace(run_dir, {1: (GOOD, 2), 9: (BAD, 2)})
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"success": 1}
    assert ("blind_pick_balls", "disp_gap", "split") not in mined.bands


def test_a_trial_in_the_run_trace_is_not_counted_twice_from_its_summary(
        miner, tmp_path):
    directory = trial(tmp_path, "blind_pick_balls", "1", number=1)
    run_trace(directory.parent, {1: (GOOD, 2)})
    summarised(directory, [PRINTED])
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"success": 1}


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
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"failure": 2}
    assert mined.bands[("blind_pick_balls", "disp_gap", "even")] == {"failure": 1}
    assert mined.bands[("blind_pick_balls", "disp_gap", "split")] == {"failure": 1}


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
    mined = miner.collect([tmp_path])
    assert mined.outcomes["blind_pick_balls"] == {"success": 1, "failure": 1}
    assert ("blind_pick_balls", "disp", "heavy") not in mined.bands


def test_two_directories_for_one_number_are_two_trials(miner, tmp_path):
    summarised(trial(tmp_path, "lift_can", "0", number=3), [PRINTED])
    summarised(trial(tmp_path, "lift_can", "1", number=3), [PRINTED])
    mined = miner.collect([tmp_path])
    assert mined.outcomes["lift_can"] == {"success": 1, "failure": 1}


# ---------------------------------------------------------------------------
# Action attribution
# ---------------------------------------------------------------------------


def test_a_read_is_attributed_to_the_action_that_preceded_it(miner):
    # A read does not cause anything; it reports the state the last action
    # left behind.  Giving reads their own bucket would attribute the most
    # informative reading of a grasp to "/tactile", which says nothing.
    rows = [{"call": "/tactile"}, {"call": "/move"}, {"call": "/tactile"},
            {"call": "/gripper"}, {"call": "/tactile"}]
    assert miner.actions_of(rows) == [
        None, "/move", "/move", "/gripper", "/gripper"]


def test_a_row_with_no_call_keeps_the_current_action(miner):
    assert miner.actions_of([{"call": "/move"}, {}, {"call": "/tactile"}]) == [
        "/move", "/move", "/move"]


def action_trace(directory: Path,
                 per_trial: dict[int, list[tuple[str, float | None]]]) -> Path:
    """One trace file for several trials, one row per ``(call, depth)``."""
    path = directory.parent / "tactile_trace.jsonl"
    lines = []
    for trial, rows in per_trial.items():
        for index, (call, depth) in enumerate(rows):
            readings = {**GOOD}
            if depth is None:
                readings.pop("left_depth_mm")
                readings.pop("right_depth_mm")
            else:
                readings["left_depth_mm"] = depth
                readings["right_depth_mm"] = depth
            lines.append(json.dumps({"t": float(index), "run": "run",
                                     "trial": trial, "call": call, "args": {},
                                     "readings": readings}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_action_table_keys_the_deepest_reading_of_each_action(miner, tmp_path):
    directory = trial(tmp_path, "blind_pick_balls", "1", number=1)
    action_trace(directory, {1: [("/move", 29.0), ("/gripper", 24.0)]})
    mined = miner.collect([tmp_path])
    assert mined.action_outcomes[("blind_pick_balls", "/move")] == {"success": 1}
    assert mined.action_outcomes[("blind_pick_balls", "/gripper")] == {"success": 1}
    assert mined.action_bands[
        ("blind_pick_balls", "/gripper", "depth", "floor")] == {"success": 1}
    assert mined.action_bands[
        ("blind_pick_balls", "/move", "depth", "face")] == {"success": 1}


def test_a_read_does_not_get_its_own_action_block(miner, tmp_path, capsys):
    directory = trial(tmp_path, "blind_pick_balls", "1", number=1)
    trial(tmp_path, "blind_pick_balls", "0", number=2)
    action_trace(
        directory,
        {1: [("/move", 28.0), ("/tactile", 24.0), ("/gripper", 26.0)],
         2: [("/move", 29.0), ("/tactile", 28.5), ("/gripper", 26.0)]})
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "/move" in out and "/gripper" in out
    assert "/tactile" not in out
    # The read at 24 mm moved /move's deepest reading there: a read reports
    # what the last action left behind, so it belongs to the move's stretch.
    assert "floor" in out


def test_each_action_is_read_against_its_own_base_rate(miner, tmp_path, capsys):
    action_trace(trial(tmp_path, "blind_pick_balls", "1", number=1),
                 {1: [("/move", 28.0), ("/gripper", 24.0)]})
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "by action (deepest reading per trial per action):" in out
    assert "  /gripper  (1 trial, 0 bad, base bad share 0.00)" in out
    assert "  /move  (1 trial, 0 bad, base bad share 0.00)" in out


def test_a_summary_only_task_gets_no_action_block(miner, tmp_path, capsys):
    # Every trial recorded no action, so the block would repeat the overall
    # table under "(unknown)" and say nothing the reader did not just read.
    summarised(trial(tmp_path, "lift_can", "1"), [PRINTED])
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    assert "by action" not in capsys.readouterr().out


def test_an_action_whose_readings_carry_no_depth_is_named(miner, tmp_path, capsys):
    lines = []
    for number, outcome in ((1, "1"), (2, "0")):
        trial(tmp_path, "blind_pick_balls", outcome, number=number)
        lines += [json.dumps({"t": 0.0, "run": "run", "trial": number,
                              "call": "/move", "args": {},
                              "readings": {**GOOD, "left_depth_mm": 0.0,
                                           "right_depth_mm": 0.0}})] * 2
    (tmp_path / "some-model" / "univtac_blind_pick_balls"
     / "tactile_trace.jsonl").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "2 of 2 produced no key under this action" in out


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


# ---------------------------------------------------------------------------
# Promotion candidates
# ---------------------------------------------------------------------------


def balanced(root: Path, task: str, bad: int = 3, good: int = 3) -> None:
    """A task whose failures share one band and whose successes another."""
    number = 1
    for _ in range(bad):
        traced(trial(root, task, "0", number=number), BAD)
        number += 1
    for _ in range(good):
        traced(trial(root, task, "1", number=number), GOOD)
        number += 1


def test_a_clean_band_is_a_candidate_with_nothing_against_it(miner, tmp_path):
    balanced(tmp_path, "blind_pick")
    candidates = miner.promotion_candidates(miner.collect([tmp_path]))
    split = next(candidate for candidate in candidates
                 if candidate.dimension == "disp_gap"
                 and candidate.band == "split")
    assert split.direction == "bad"
    assert (split.n, split.bad) == (3, 3)
    assert split.lift == pytest.approx(0.5)
    assert split.clean
    assert split.counterexamples == ()
    assert len(split.covers) == split.n == 3


def test_a_success_inside_a_bad_band_keeps_its_name(miner, tmp_path):
    # Four successes and three failures, and one of the successes reads like
    # the failures.  The band still separates, but it is not clean, and the
    # trial that argues against it must travel with the candidate.
    balanced(tmp_path, "blind_pick")
    traced(trial(tmp_path, "blind_pick", "1", number=99), BAD)
    candidates = miner.promotion_candidates(miner.collect([tmp_path]))
    split = next(candidate for candidate in candidates
                 if candidate.dimension == "disp_gap"
                 and candidate.band == "split")
    assert not split.clean
    assert len(split.counterexamples) == 1
    assert "trial_99_taskcompleted_1" in split.counterexamples[0]


def test_a_band_needs_the_episode_floor_and_the_margin(miner, tmp_path):
    balanced(tmp_path, "blind_pick")
    mined = miner.collect([tmp_path])
    assert miner.promotion_candidates(mined, min_episodes=4) == []
    assert miner.promotion_candidates(mined, margin=0.6) == []


def test_a_band_that_only_says_a_reading_is_absent_is_not_a_candidate(
        miner, tmp_path, capsys):
    # One pad missing a displacement reading keys the failure trials to
    # disp_gap=unknown.  That band separates perfectly and still cannot
    # become a card condition: "this measurement does not exist" is a fact
    # about the record, not about the contact.
    absent = {**GOOD, "left_marker_displacement_px": 0.0}
    for number in (1, 2, 3):
        traced(trial(tmp_path, "lift_can", "0", number=number), absent)
    for number in (4, 5, 6):
        traced(trial(tmp_path, "lift_can", "1", number=number), GOOD)
    mined = miner.collect([tmp_path])
    kept = miner.promotion_candidates(mined)
    assert [candidate.band for candidate in kept] == ["even"]
    assert len(miner.promotion_candidates(mined, include_absent=True)) == 2
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "1 band keyed on an absent reading left out" in out
    promotion = out.split("## promotion candidates")[1]
    assert "disp_gap=unknown" not in promotion
    assert "disp_gap=even" in promotion


def test_a_second_task_with_the_same_band_says_so(miner, tmp_path):
    # Corroboration across tasks is the only evidence that a rule is not
    # about one workspace.  It is reported, not counted as an extra episode.
    balanced(tmp_path, "blind_pick")
    balanced(tmp_path, "lift_can")
    candidates = miner.promotion_candidates(miner.collect([tmp_path]))
    on_pick = next(candidate for candidate in candidates
                   if candidate.task == "blind_pick"
                   and candidate.dimension == "disp_gap"
                   and candidate.band == "split")
    assert on_pick.n == 3
    assert on_pick.elsewhere == ("lift_can",)


def test_the_report_prints_the_verdict_and_the_counterexample(
        miner, tmp_path, capsys):
    balanced(tmp_path, "blind_pick")
    traced(trial(tmp_path, "blind_pick", "1", number=99), BAD)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "## promotion candidates  (k=3 episodes, margin 0.20)" in out
    assert "keep the card a hypothesis or narrow its scope" in out
    assert "contradicted by: " in out
    # The complementary clean band is offered for keeping, not for acting on.
    assert "no counterexamples -- evidence supports marking a card observed" in out


def test_a_task_without_a_separating_band_says_so(miner, tmp_path, capsys):
    traced(trial(tmp_path, "lift_can", "1", number=1), GOOD)
    traced(trial(tmp_path, "lift_can", "0", number=2), GOOD)
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "(no band separated outcomes by the margin)" in out


def test_candidates_on_the_same_trials_are_named_as_one_observation(
        miner, tmp_path, capsys):
    # Depth, displacement and coherence all key the same three failures here.
    # Three lines over one set of trials is one observation, and a reader who
    # writes three cards from it is promoting the same evidence three times.
    balanced(tmp_path, "blind_pick")
    miner.main(["--root", str(tmp_path), "--min-support", "1"])
    out = capsys.readouterr().out
    assert "same trials as" in out
    assert "one observation, not two" in out


def test_the_promotion_floor_can_be_raised_from_the_command_line(
        miner, tmp_path, capsys):
    balanced(tmp_path, "blind_pick")
    miner.main(["--root", str(tmp_path), "--min-support", "1",
                "--promote-episodes", "4"])
    assert "(no band separated outcomes by the margin)" in capsys.readouterr().out
