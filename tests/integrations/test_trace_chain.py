"""Tests for the cross-seed trace chain.

No simulator, no model: the chain is JSON on disk plus a bounded formatter.
Covered here:

  1. A record round-trips and the next seed finds the latest earlier one.
  2. Seed one has nothing to read, and a missing seed does not break the chain.
  3. The formatted block carries the program, the outcome and both output
     streams, and stays under its character cap with a visible truncation mark.
  4. The framework-appended final check is stripped, and only that exact
     suffix -- a call the model wrote itself is part of its program.
  5. A chain needs an output directory.
  6. A reflection card travels instead of the program, with the raw trace as a
     fallback, and the trial loop writes one card per finished seed.
  7. The window arm carries the last k cards, adds the best success when it is
     older than the window, and states a repeated ending once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capx.envs.trace_chain import (
    TRACE_MODES,
    TraceRecord,
    build_reflection_prompt,
    filter_check_detail,
    format_card_trace,
    format_trace,
    format_window_cards,
    load_best_trace,
    load_previous_trace,
    load_recent_traces,
    strip_final_check,
    trace_chain_dir,
    write_injected_block,
    write_trace,
)


def test_only_the_four_supported_modes_remain() -> None:
    assert TRACE_MODES == ("off", "last", "card", "window_card")


def _record(seed: int, **overrides) -> TraceRecord:
    data = {
        "seed": seed,
        "complete": True,
        "sandbox_rc": 0,
        "reward": 0.0,
        "task_completed": False,
        "code": "print('hello')",
        "stdout": "hello\n",
        "stderr": "",
        "num_code_blocks": 1,
    }
    data.update(overrides)
    return TraceRecord(**data)


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    record = _record(4)
    path = write_trace(tmp_path, record)
    assert path.name == "seed_004.json"

    loaded = load_previous_trace(tmp_path, 5)
    assert loaded is not None
    assert loaded.seed == 4
    assert loaded.task_completed is False
    assert loaded.code == "print('hello')"


def test_seed_one_has_no_previous_trace(tmp_path: Path) -> None:
    assert load_previous_trace(tmp_path, 1) is None


def test_missing_seed_falls_back_to_the_latest_earlier_record(tmp_path: Path) -> None:
    write_trace(tmp_path, _record(2))
    write_trace(tmp_path, _record(4))

    loaded = load_previous_trace(tmp_path, 7)
    assert loaded is not None
    assert loaded.seed == 4, "the latest earlier seed, not the immediate predecessor"

    loaded = load_previous_trace(tmp_path, 4)
    assert loaded is not None and loaded.seed == 2


def test_format_trace_carries_program_outcome_and_streams() -> None:
    record = _record(
        3,
        code="close_gripper(force_threshold=27.5)",
        stdout="depth=26.4\n",
        stderr="RuntimeError: no contact\n",
    )
    block = format_trace(record)

    assert "seed 3" in block
    assert "FAILED" in block
    assert "task_completed=False" in block
    assert "close_gripper(force_threshold=27.5)" in block
    assert "depth=26.4" in block
    assert "RuntimeError: no contact" in block
    assert "different scene instance" in block


def test_format_trace_marks_a_timeout() -> None:
    block = format_trace(_record(5, complete=True, timed_out=True))
    assert "TIMED OUT" in block


def test_format_trace_stays_under_the_cap() -> None:
    record = _record(
        6,
        code="x = 1\n" * 5000,
        stdout="o" * 20000,
        stderr="e" * 20000,
    )
    block = format_trace(record, max_chars=4000)

    assert len(block) <= 4000
    assert "chars omitted" in block or "trace truncated" in block
    assert "different scene instance" in block


def test_format_trace_keeps_the_tail_of_output() -> None:
    record = _record(7, stdout="head\n" + "tail-line\n" * 500)
    block = format_trace(record, max_chars=4000)
    assert "tail-line" in block


def test_strip_final_check_removes_the_appended_suffix() -> None:
    code = "print('done')\n\ncheck_success()\n"
    assert strip_final_check(code, "check_success") == "print('done')"


def test_strip_final_check_leaves_a_model_written_call() -> None:
    code = "check_success()\nprint('after')\n"
    assert strip_final_check(code, "check_success") == code


def test_strip_final_check_is_a_no_op_without_a_name() -> None:
    code = "print('done')\n\ncheck_success()\n"
    assert strip_final_check(code, None) == code


def test_filter_check_detail_keeps_geometry_and_drops_the_oracle() -> None:
    detail = {
        "completed": False,
        "ground_truth_texture": "rough",
        "expected_pad": "orange_pad",
        "prism_pos": [0.3, 0.4, 0.1],
        "target_pos": [0.4, 0.0, 0.1],
        "pos_error_cm": [0.2, 3.2, 0.0],
        "tilt_cos": 0.91,
        "reasons": ["Y off by 3.2 cm (max 2 cm)", "tilted 24° (max 15°)"],
    }
    selected = filter_check_detail(detail)
    assert selected is not None
    assert selected["reasons"] == detail["reasons"]
    assert selected["pos_error_cm"] == detail["pos_error_cm"]
    assert selected["tilt_cos"] == detail["tilt_cos"]
    for oracle_key in (
        "ground_truth_texture",
        "expected_pad",
        "prism_pos",
        "target_pos",
    ):
        assert oracle_key not in selected


def test_filter_check_detail_drops_the_classify_label_and_target_plate() -> None:
    """hardness_classify and weight_classify also report the answer itself.

    The trace keeps the placement geometry and the reasons; the class label,
    the plate the block belonged on and the raw pose are the oracle's answer
    and stay out.
    """
    detail = {
        "hardness_label": "hard",
        "target_plate": "yellow_plate",
        "block_pose_in_target": [[1.0, 0.0, 0.0, 0.0]],
        "pos_error_cm": [7.0, 8.3, 0.99],
        "tilt_cos": 0.9999,
        "completed": False,
        "reasons": ["X off by 7.0 cm (max 4 cm)"],
    }
    assert filter_check_detail(detail) == {
        "reasons": ["X off by 7.0 cm (max 4 cm)"],
        "pos_error_cm": [7.0, 8.3, 0.99],
        "tilt_cos": 0.9999,
    }


def test_filter_check_detail_is_none_without_a_detail() -> None:
    assert filter_check_detail(None) is None
    assert filter_check_detail({}) is None
    assert filter_check_detail({"ground_truth_texture": "rough"}) is None


def test_formatted_trace_shows_the_check_detail() -> None:
    block = format_trace(
        _record(
            10,
            task_completed=False,
            check_detail={
                "reasons": ["Y off by 3.2 cm (max 2 cm)"],
                "pos_error_cm": [0.2, 3.2, 0.0],
                "tilt_cos": 0.91,
            },
        )
    )
    assert "CHECK DETAIL" in block
    assert "Y off by 3.2 cm (max 2 cm)" in block
    assert "position error (cm): 0.20, 3.20, 0.00" in block
    assert "tilt cosine: 0.9100" in block


def test_filter_check_detail_keeps_piano_progress() -> None:
    """The piano's progress counters travel; the note sequence does not.

    ``piano_sequence`` and ``piano_instruction`` are the answer for that seed
    and the next seed plays a different one; they stay out.  ``failure``,
    ``terminal_reason`` and the progress counters are what the next attempt
    can use.
    """
    detail = {
        "success": False,
        "failure": "wrong_key",
        "terminal_reason": "wrong_key",
        "notes_completed": 3,
        "notes_attempted": 4,
        "total_notes": 6,
        "piano_sequence": {"midi": [60, 62], "dynamics": ["light", "heavy"]},
        "piano_instruction": "play 60 then 62",
    }
    selected = filter_check_detail(detail)
    assert selected == {
        "failure": "wrong_key",
        "terminal_reason": "wrong_key",
        "notes_completed": 3,
        "notes_attempted": 4,
        "total_notes": 6,
    }
    block = format_trace(_record(11, task_completed=False, check_detail=selected))
    assert "notes completed: 3" in block
    assert "total notes: 6" in block
    assert "piano_sequence" not in block
    assert "play 60" not in block
    # The terminal reason is rendered when it says something the failure line
    # does not -- an episode that ran out of its own clock reports only here.
    timed = filter_check_detail({"terminal_reason": "timeout"})
    timed_block = format_trace(_record(12, task_completed=False, check_detail=timed))
    assert "terminal reason: timeout" in timed_block


def test_filter_check_detail_keeps_memory_match_placement() -> None:
    """memory_match's verdict is a bare bool, so placement is the only detail.

    The task reports where each candidate ended up; the hidden class keys and
    anything filed under another name stay out.
    """
    detail = {
        "candidates": {
            "candidate_left": {
                "placed": False,
                "on_match_pad": False,
                "xy_error_cm": 4.216,
                "z_error_cm": 1.104,
            },
            "candidate_right": {
                "placed": True,
                "on_match_pad": True,
                "xy_error_cm": 0.842,
                "z_error_cm": 0.203,
            },
            "reference_object": {"placed": True, "xy_error_cm": 0.0},
        },
        "reference_class_key": "heavy_rough_rigid",
    }
    filtered = filter_check_detail(detail)
    assert filtered == {
        "candidates": {
            "candidate_left": {
                "placed": False,
                "on_match_pad": False,
                "xy_error_cm": 4.22,
                "z_error_cm": 1.1,
            },
            "candidate_right": {
                "placed": True,
                "on_match_pad": True,
                "xy_error_cm": 0.84,
                "z_error_cm": 0.2,
            },
        }
    }


def test_filter_check_detail_keeps_strap_progress_and_drops_load() -> None:
    """tension_strap reports why it latched, never how hard the strap pulled.

    The load is the estimate the task asks the robot to make from the gel, so
    a ``tension_N`` field -- or anything nested that carries one -- must not
    survive the filter.
    """
    detail = {
        "failure": "strap_ruptured",
        "stages_completed": 1,
        "current_hold_s": 0.84,
        "elapsed_s": 12.3,
        "tension_N": 17.2,
        "measured": {"tension_N": 17.2},
    }
    assert filter_check_detail(detail) == {
        "failure": "strap_ruptured",
        "stages_completed": 1,
        "current_hold_s": 0.84,
        "elapsed_s": 12.3,
    }


def test_formatted_trace_shows_strap_progress() -> None:
    block = format_trace(
        _record(
            10,
            task_completed=False,
            check_detail={
                "failure": "strap_ruptured",
                "stages_completed": 1,
                "current_hold_s": 0.84,
                "elapsed_s": 12.3,
            },
        )
    )
    assert "CHECK DETAIL" in block
    assert "failure: strap_ruptured" in block
    assert "stages completed: 1" in block
    assert "current hold: 0.84 s" in block
    assert "elapsed: 12.30 s" in block
    assert "tension" not in block.lower()


def test_filter_check_detail_keeps_insertion_progress_and_drops_the_target() -> None:
    """bulb_tightening reports where the insertion stopped, not how long it was.

    ``nominal_free_travel_mm`` is this seed's thread lead -- the value the task
    prompt says has to be measured -- and ``target_torque_Nm`` is a scoring bar.
    Neither may reach the trace.
    """
    detail = {
        "completed": False,
        "failure": "bulb_damage",
        # Only the expert path ever writes these, so on a policy run they are
        # constant -- "setup" on every seed.  They must not reach the trace.
        "phase": "setup",
        "submitted": False,
        "held_steps": 12,
        "physics_step": 4210,
        "nominal_free_travel_mm": 9.43,
        "target_torque_Nm": 0.12,
        "torque_tolerance_Nm": 0.01,
        "measured": {"formal_advance_m": 0.0032, "formal_net_turns": 0.34},
        "reasons": [
            "advance: formal_advance_m 3.20 mm is short of this thread's full travel"
        ],
    }
    assert filter_check_detail(detail) == {
        "reasons": [
            "advance: formal_advance_m 3.20 mm is short of this thread's full travel"
        ],
        "failure": "bulb_damage",
        "held_steps": 12,
        "physics_step": 4210,
    }


def test_filter_check_detail_keeps_the_ball_failure_name() -> None:
    """blind_pick_balls' detail is its own step info: a failure and a clock."""
    detail = {
        "success": False,
        "failure": "wrong_material",
        "physics_step": 8800,
        "collected_count": 0,
        "instruction": "Retrieve one textured ball and release it in the bowl.",
    }
    assert filter_check_detail(detail) == {
        "failure": "wrong_material",
        "physics_step": 8800,
    }


def test_formatted_trace_shows_insertion_progress() -> None:
    block = format_trace(
        _record(
            10,
            task_completed=False,
            check_detail={
                "failure": "bulb_damage",
                "held_steps": 12,
                "physics_step": 4210,
            },
        )
    )
    assert "failure: bulb_damage" in block
    assert "held steps: 12" in block
    assert "physics step: 4210" in block


def test_formatted_trace_shows_candidate_placement() -> None:
    block = format_trace(
        _record(
            10,
            task_completed=False,
            check_detail={
                "candidates": {
                    "candidate_left": {
                        "placed": False,
                        "on_match_pad": False,
                        "xy_error_cm": 4.22,
                        "z_error_cm": 1.1,
                    },
                    "candidate_right": {
                        "placed": True,
                        "on_match_pad": True,
                        "xy_error_cm": 0.84,
                        "z_error_cm": 0.2,
                    },
                }
            },
        )
    )
    assert "CHECK DETAIL" in block
    assert (
        "candidate_left: placed=False, on_match_pad=False, "
        "xy error 4.22 cm, z error 1.10 cm"
    ) in block
    assert (
        "candidate_right: placed=True, on_match_pad=True, "
        "xy error 0.84 cm, z error 0.20 cm"
    ) in block


def test_formatted_trace_does_not_leak_the_hidden_check() -> None:
    code = strip_final_check(
        "print('done')\n\ncheck_success()\n", "check_success"
    )
    block = format_trace(_record(8, code=code))
    assert "check_success" not in block


def test_injected_block_is_kept_beside_the_record(tmp_path: Path) -> None:
    path = write_injected_block(tmp_path, 9, "PREVIOUS ATTEMPT")
    assert path.name == "seed_009_injected_block.txt"
    assert path.read_text() == "PREVIOUS ATTEMPT"


def test_chain_dir_requires_an_output_directory() -> None:
    with pytest.raises(ValueError, match="output directory"):
        trace_chain_dir({"output_dir": None})


def test_chain_dir_defaults_below_output_dir() -> None:
    assert trace_chain_dir({"output_dir": "/tmp/run"}) == Path("/tmp/run/trace_chain")


def test_generated_code_cannot_block_on_input() -> None:
    """``input()`` must fail fast, not wait on whatever launched the run.

    Seed 1 of the 2026-11-08 chain asked the operator for calibration slopes
    with ``json.loads(input(...))`` and parked the whole run on stdin.
    """
    from capx.envs.tasks.base import (
        CodeExecutionEnvBase,
        SimpleExecutor,
        _unattended_input,
    )

    result = SimpleExecutor(None, {}).run("input('Enter measured slopes: ')")
    assert result["ok"] is False
    assert "unattended" in result["error"]

    # The long-lived exec namespace used by env.step() must shadow it too.
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = None
    env._apis = {}
    env._init_exec_globals()
    assert env._exec_globals["input"] is _unattended_input


def test_the_trial_loop_injects_the_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook is actual trial-loop wiring, not just a formatter test.

    A fake env and a fake model call keep this off the simulator; what is
    checked is that seed 2's prompt carries seed 1's program and output, that
    seed 1's prompt does not, and that the hidden final check never reaches the
    trace.
    """
    import copy
    from types import SimpleNamespace

    import capx.envs.trial as trial_mod
    from capx.envs.launch import LaunchArgs

    class _FakeEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(append_final_check="check_success")
            self.low_level_env = SimpleNamespace()
            self.seed = None
            self.codes: list[str] = []

        def reset(self, *, seed=None, options=None):
            self.seed = seed
            return {
                "full_prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": "TASK"}]},
                ]
            }, {}

        def final_check_name(self):
            return "check_success"

        def step(self, code):
            self.codes.append(code)
            return (
                {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
                0.0,
                False,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": f"ran seed {self.seed}",
                    "stderr": "",
                    "task_completed": False,
                    "failure_detail": {
                        "completed": False,
                        "ground_truth_texture": "rough",
                        "expected_pad": "orange_pad",
                        "prism_pos": [0.3, 0.4, 0.1],
                        "target_pos": [0.4, 0.0, 0.1],
                        "pos_error_cm": [0.2, 3.2, 0.0],
                        "tilt_cos": 0.91,
                        "reasons": ["Y off by 3.2 cm (max 2 cm)"],
                    },
                },
            )

    seen: list[list] = []

    def _fake_query(_args, prompt, *_a, **_k):
        seen.append(copy.deepcopy(prompt))
        return {"content": "```python\nprint('hello')\n```", "reasoning": ""}

    monkeypatch.setattr(trial_mod, "_query_model", _fake_query)

    args = LaunchArgs(config_path="fake.yaml", model="fake-model", debug=False)
    config = {
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_oracle_code": False,
        "trace_mode": "last",
        "trace_max_chars": 12000,
        "deliver_tactile_skills": False,
        "exec_alarm_seconds": None,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
    }
    env = _FakeEnv()
    trial_mod._run_single_trial(env, 1, args, config, None)
    trial_mod._run_single_trial(env, 2, args, config, None)

    # The hidden check is not executed on a traced run: its stdout prints the
    # ground-truth texture and pad, which would travel to the next seed.
    assert all("check_success(" not in code for code in env.codes)

    chain = tmp_path / "trace_chain"
    seed1 = json.loads((chain / "seed_001.json").read_text())
    assert "check_success" not in seed1["code"], "the hidden check leaked"
    assert seed1["check_detail"]["reasons"] == ["Y off by 3.2 cm (max 2 cm)"]
    assert "ground_truth_texture" not in seed1["check_detail"]
    assert "expected_pad" not in seed1["check_detail"]

    assert "PREVIOUS ATTEMPT" not in json.dumps(seen[0])
    seed2_prompt = json.dumps(seen[1])
    assert "PREVIOUS ATTEMPT - seed 1" in seed2_prompt
    assert "ran seed 1" in seed2_prompt
    assert "Y off by 3.2 cm (max 2 cm)" in seed2_prompt
    assert "ground_truth_texture" not in seed2_prompt
    assert "orange_pad" not in seed2_prompt

    seed2 = json.loads((chain / "seed_002.json").read_text())
    assert seed2["injected_from_seed"] == 1


# ---------------------------------------------------------------------------
# Dual-slot and experience-card modes
# ---------------------------------------------------------------------------

def test_load_best_trace_picks_the_latest_success(tmp_path: Path) -> None:
    write_trace(tmp_path, _record(2, task_completed=False))
    write_trace(tmp_path, _record(3, task_completed=True, reward=1.0))
    write_trace(tmp_path, _record(4, task_completed=False))

    best = load_best_trace(tmp_path, 5)
    assert best is not None and best.seed == 3
    # The two slots are different records: the immediate predecessor failed.
    previous = load_previous_trace(tmp_path, 5)
    assert previous is not None and previous.seed == 4
    # Nothing below the success itself.
    assert load_best_trace(tmp_path, 3) is None


def test_load_best_trace_ignores_an_incomplete_success(tmp_path: Path) -> None:
    write_trace(
        tmp_path,
        _record(1, task_completed=True, reward=1.0, complete=False, timed_out=True),
    )
    assert load_best_trace(tmp_path, 2) is None


def test_card_trace_renders_the_card_and_the_check_detail() -> None:
    record = _record(
        6,
        task_completed=False,
        card=(
            "DID: pressed midi 60 with press_force_N=0.35\n"
            "MEASURED: left_force_N=0.36\n"
            "WHY IT FAILED: the last note was never released\n"
            "NEXT TIME: release after every note, including the last"
        ),
        check_detail={
            "failure": "wrong_key",
            "notes_completed": 3,
            "notes_attempted": 3,
            "total_notes": 6,
        },
    )
    block = format_card_trace(record)
    assert "EXPERIENCE CARD - seed 6 (FAILED)" in block
    assert "DID: pressed midi 60" in block
    # The exact measurement survives compression.
    assert "failure: wrong_key" in block
    assert "notes completed: 3" in block


def test_card_trace_falls_back_to_the_raw_trace() -> None:
    record = _record(7, card="", code="print('fallback')")
    block = format_card_trace(record)
    assert "UNAVAILABLE - RAW TRACE" in block
    assert "print('fallback')" in block


def test_reflection_prompt_uses_the_filtered_record() -> None:
    record = _record(
        8,
        code="press_key(60)",
        check_detail={"failure": "wrong_key", "notes_completed": 2},
    )
    text = json.dumps(build_reflection_prompt(record))
    assert "press_key(60)" in text
    assert "wrong_key" in text
    for prefix in ("DID:", "MEASURED:", "WHY IT FAILED:", "NEXT TIME:"):
        assert prefix in text


def test_card_mode_writes_and_injects_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One reflection call per seed; the next seed reads the card, not the code."""
    import copy
    from types import SimpleNamespace

    import capx.envs.trial as trial_mod
    from capx.envs.launch import LaunchArgs

    class _SeedEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(append_final_check="check_success")
            self.low_level_env = SimpleNamespace()
            self.seed = None

        def reset(self, *, seed=None, options=None):
            self.seed = seed
            return {
                "full_prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": "TASK"}]},
                ]
            }, {}

        def final_check_name(self):
            return "check_success"

        def step(self, code):
            return (
                {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
                0.0,
                False,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": f"ran seed {self.seed}",
                    "stderr": "",
                    "task_completed": False,
                    "failure_detail": {"failure": "incomplete"},
                },
            )

    seen: list[str] = []
    reflections: list[str] = []

    def _fake_query(_args, prompt, *_a, **_k):
        text = json.dumps(copy.deepcopy(prompt))
        # The reflection request is the one carrying the attempt under review;
        # an injected card also contains the words "experience card", so keying
        # on that phrase would misclassify the next seed's code query.
        if "ATTEMPT UNDER REVIEW" in text:
            reflections.append(text)
            return {
                "content": (
                    "DID: pressed one key\n"
                    "MEASURED: force 0.35 N\n"
                    "WHY IT FAILED: incomplete\n"
                    "NEXT TIME: release before the next note"
                ),
                "reasoning": "",
            }
        seen.append(text)
        return {"content": "```python\nprint('hello')\n```", "reasoning": ""}

    monkeypatch.setattr(trial_mod, "_query_model", _fake_query)

    args = LaunchArgs(config_path="fake.yaml", model="fake-model", debug=False)
    config = {
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_oracle_code": False,
        "trace_mode": "card",
        "trace_max_chars": 12000,
        "deliver_tactile_skills": False,
        "exec_alarm_seconds": None,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
    }
    env = _SeedEnv()
    trial_mod._run_single_trial(env, 1, args, config, None)
    trial_mod._run_single_trial(env, 2, args, config, None)

    assert len(reflections) == 2, "one reflection call per finished seed"
    record = json.loads((tmp_path / "trace_chain" / "seed_001.json").read_text())
    assert record["card"].startswith("DID: pressed one key")

    seed2_prompt = seen[1]
    assert "EXPERIENCE CARD - seed 1 (FAILED)" in seed2_prompt
    assert "DID: pressed one key" in seed2_prompt


# ---------------------------------------------------------------------------
# Window arm
# ---------------------------------------------------------------------------

def test_load_recent_traces_returns_newest_first(tmp_path: Path) -> None:
    for seed in (1, 2, 4):
        write_trace(tmp_path, _record(seed))
    assert [r.seed for r in load_recent_traces(tmp_path, 5, 3)] == [4, 2, 1]
    assert [r.seed for r in load_recent_traces(tmp_path, 5, 2)] == [4, 2]
    assert [r.seed for r in load_recent_traces(tmp_path, 2, 3)] == [1]


def test_window_card_renders_the_window_and_a_streak() -> None:
    records = [
        _record(4, card="DID: a", check_detail={"failure": "incomplete"}),
        _record(3, card="DID: b", check_detail={"failure": "incomplete"}),
        _record(2, card="DID: c", check_detail={"failure": "incomplete"}),
    ]
    block = format_window_cards(records, None)
    assert "STREAK: the last 3 attempts all ended with 'incomplete'." in block
    assert "PREVIOUS ATTEMPT - seed 4 (FAILED)" in block
    assert "EARLIER ATTEMPT - seed 3 (FAILED)" in block
    assert "EARLIER ATTEMPT - seed 2 (FAILED)" in block
    assert block.count("different scene instance") == 1


def test_window_card_adds_a_best_success_outside_the_window() -> None:
    records = [
        _record(5, card="DID: x", check_detail={"failure": "timeout"}),
        _record(4, card="DID: y", check_detail={"failure": "timeout"}),
    ]
    best = _record(2, task_completed=True, reward=1.0, card="DID: z")
    block = format_window_cards(records, best)
    assert "BEST SUCCESS SO FAR - seed 2 (SUCCEEDED)" in block
    assert "STREAK: the last 2 attempts all ended with 'timeout'." in block
    assert block.count("different scene instance") == 1


def test_window_card_has_no_streak_when_endings_differ() -> None:
    records = [
        _record(4, card="DID: a", check_detail={"failure": "wrong_key"}),
        _record(3, card="DID: b", check_detail={"failure": "timeout"}),
    ]
    assert "STREAK:" not in format_window_cards(records, None)


def test_window_mode_injects_the_last_k_and_records_the_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """window_card carries k cards and names the seeds on the new record."""
    import copy
    from types import SimpleNamespace

    import capx.envs.trial as trial_mod
    from capx.envs.launch import LaunchArgs

    class _SeedEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(append_final_check="check_success")
            self.low_level_env = SimpleNamespace()
            self.seed = None

        def reset(self, *, seed=None, options=None):
            self.seed = seed
            return {
                "full_prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": "TASK"}]},
                ]
            }, {}

        def final_check_name(self):
            return "check_success"

        def step(self, code):
            return (
                {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
                0.0,
                False,
                False,
                {
                    "sandbox_rc": 0,
                    "stdout": f"ran seed {self.seed}",
                    "stderr": "",
                    "task_completed": False,
                    "failure_detail": {"failure": "incomplete"},
                },
            )

    seen: list[str] = []

    def _fake_query(_args, prompt, *_a, **_k):
        text = json.dumps(copy.deepcopy(prompt))
        if "ATTEMPT UNDER REVIEW" in text:
            return {
                "content": (
                    "DID: pressed one key\n"
                    "MEASURED: force 0.35 N\n"
                    "WHY IT FAILED: incomplete\n"
                    "NEXT TIME: release before the next note"
                ),
                "reasoning": "",
            }
        seen.append(text)
        return {"content": "```python\nprint('hello')\n```", "reasoning": ""}

    monkeypatch.setattr(trial_mod, "_query_model", _fake_query)

    args = LaunchArgs(config_path="fake.yaml", model="fake-model", debug=False)
    config = {
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_oracle_code": False,
        "trace_mode": "window_card",
        "trace_window": 3,
        "trace_max_chars": 12000,
        "deliver_tactile_skills": False,
        "exec_alarm_seconds": None,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
    }
    env = _SeedEnv()
    for seed in (1, 2, 3, 4, 5):
        trial_mod._run_single_trial(env, seed, args, config, None)

    seed5_prompt = seen[4]
    assert "PREVIOUS ATTEMPT - seed 4" in seed5_prompt
    assert "EARLIER ATTEMPT - seed 3" in seed5_prompt
    assert "EARLIER ATTEMPT - seed 2" in seed5_prompt
    assert "EARLIER ATTEMPT - seed 1" not in seed5_prompt
    assert "STREAK: the last 3 attempts all ended with 'incomplete'." in seed5_prompt

    record = json.loads((tmp_path / "trace_chain" / "seed_005.json").read_text())
    assert record["injected_from_seed"] == 4
    assert record["injected_window_seeds"] == [4, 3, 2]
