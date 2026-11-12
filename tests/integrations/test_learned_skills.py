"""Tests for the run-local learned-skills overlay and its run document.

No simulator, no model: a document is markdown on disk, an overlay is JSON,
and the patch applier is pure.  Covered here:

  1. An overlay and a run document round-trip; the next seed finds the latest
     version below it.
  2. The delivered block is one line per rule plus an optional indented
     procedure body, in the same shape as the curated library, and is empty
     before anything has been learned.  It carries no heading and no warning:
     provenance stays in the JSON, not in the agent's prompt.
  3. The run document is cut at a line boundary, with a visible marker.
  4. Seed citations are read from both shapes the prompt asks for.
  5. The editor reply must carry a document and an operations list together.
  6. An add needs distinct seeds that the document or this attempt supports.
  7. A revise keeps the rule's earlier evidence; a delete removes it by name.
  8. The size cap drops the least-supported rule and says so.
  9. One attempt's evidence is the raw log plus check detail and parameters --
     never the solving model's card; the program and the task prompt are
     optional context.
 10. The editor prompt carries the document and the current attempt, and no
     card and no older attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

from capx.envs.learned_skills import (
    LearnedOverlay,
    LearnedSkill,
    apply_patch,
    build_editor_messages,
    clamp_doc,
    format_block,
    load_overlay,
    load_run_doc,
    parse_editor_reply,
    record_evidence,
    seeds_mentioned,
    write_overlay,
    write_run_doc,
)
from capx.envs.trace_chain import TraceRecord


def _record(seed: int, **overrides) -> TraceRecord:
    data = {
        "seed": seed,
        "complete": True,
        "sandbox_rc": 0,
        "reward": 0.0,
        "task_completed": False,
        "code": "",
        "stdout": "",
        "stderr": "",
        "num_code_blocks": 1,
    }
    data.update(overrides)
    return TraceRecord(**data)


def _cfg(tmp_path: Path) -> dict:
    return {"output_dir": str(tmp_path)}


def _rule(name: str, rule: str, seeds=(3, 5), created=5, body="") -> LearnedSkill:
    return LearnedSkill(
        name=name,
        rule=rule,
        body=body,
        evidence_seeds=tuple(seeds),
        created_seed=created,
    )


def test_overlay_roundtrip_and_latest_before_seed(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    first = LearnedOverlay(seed=3, revision=1, rules=[_rule("depth-ceiling", "pair")])
    second = LearnedOverlay(
        seed=5,
        revision=2,
        rules=[
            _rule("depth-ceiling", "pair"),
            _rule("contact-first", "check both pads at 25 mm before turning"),
        ],
    )
    write_overlay(config, first)
    write_overlay(config, second)

    assert load_overlay(config, 4).revision == 1
    assert load_overlay(config, 6).revision == 2
    assert load_overlay(config, 6).rules[1].name == "contact-first"


def test_run_doc_roundtrip_and_latest_before_seed(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    assert load_run_doc(config, 1) == ""
    write_run_doc(config, 3, "# RUN LOG\n## Confirmed facts\n- [seed 1,3] a\n")
    write_run_doc(config, 5, "# RUN LOG\n## Confirmed facts\n- [seed 3,5] b\n")
    assert "seed 3,5" in load_run_doc(config, 6)
    assert "seed 1,3" in load_run_doc(config, 4)
    assert "seed 3,5" not in load_run_doc(config, 4)
    assert (tmp_path / "learned_skills" / "run_doc.md").read_text().startswith(
        "# RUN LOG"
    )


def test_block_lists_rules_without_a_heading() -> None:
    overlay = LearnedOverlay(
        seed=5,
        revision=1,
        rules=[
            _rule("depth-ceiling", "keep the pad depth above 27 mm while seated"),
            _rule("contact-first", "confirm both pads at 25 mm before turning"),
        ],
    )
    block = format_block(overlay)
    assert block.startswith("- depth-ceiling: keep the pad depth above 27 mm")
    assert "- depth-ceiling: keep the pad depth above 27 mm while seated" in block
    assert "- contact-first: confirm both pads at 25 mm before turning" in block
    assert "LEARNED" not in block
    assert "hypotheses" not in block
    assert block.count("- ") == 2
    assert format_block(LearnedOverlay()) == ""


def test_block_includes_an_optional_procedure_body() -> None:
    overlay = LearnedOverlay(
        seed=5,
        revision=1,
        rules=[
            _rule(
                "sweep-the-rows",
                "Sweep x from 0.37 to 0.65 in 10 mm steps",
                body=(
                    "1. Move to x=0.37 at y=-0.115\n"
                    "2. Descend until the hand stops\n"
                    "3. Shift x by 10 mm and retry"
                ),
            )
        ],
    )
    block = format_block(overlay)
    assert "- sweep-the-rows: Sweep x from 0.37 to 0.65 in 10 mm steps" in block
    assert "  1. Move to x=0.37 at y=-0.115" in block
    assert "  3. Shift x by 10 mm and retry" in block


def test_add_body_applies_roundtrips_and_rejects_oversize() -> None:
    base = {
        "op": "add",
        "name": "sweep-the-rows",
        "rule": "Sweep x from 0.37 to 0.65 in 10 mm steps",
        "body": "Move to x=0.37; descend; shift x by 10 mm; retry.",
        "evidence_seeds": [3, 5],
    }
    overlay, net = apply_patch(
        overlay=LearnedOverlay(),
        patch={"operations": [base]},
        available_seeds={3, 5},
        seed=5,
        max_skills=6,
        min_evidence=2,
    )
    assert net == ["add sweep-the-rows"]
    assert overlay.rules[0].body.startswith("Move to x=0.37")
    assert LearnedSkill.from_dict(overlay.rules[0].to_dict()).body == (
        overlay.rules[0].body
    )

    too_long = dict(base, name="too-long", body="x" * 801)
    rejected, net = apply_patch(
        overlay=LearnedOverlay(),
        patch={"operations": [too_long]},
        available_seeds={3, 5},
        seed=5,
        max_skills=6,
        min_evidence=2,
    )
    assert net == []
    assert "at most 800 characters" in rejected.rejected[0]["reason"]


def test_revise_without_body_keeps_it_and_empty_body_clears_it() -> None:
    overlay = LearnedOverlay(
        seed=5,
        revision=1,
        rules=[
            _rule(
                "sweep-the-rows",
                "Sweep x from 0.37 to 0.65 in 10 mm steps",
                body="Move to x=0.37; descend; retry.",
            )
        ],
    )
    rule_only = {
        "op": "revise",
        "name": "sweep-the-rows",
        "rule": "Sweep x from 0.37 to 0.65 in 5 mm steps",
        "evidence_seeds": [3, 5, 7],
    }
    kept, _ = apply_patch(
        overlay=overlay,
        patch={"operations": [rule_only]},
        available_seeds={3, 5, 7},
        seed=7,
        max_skills=6,
        min_evidence=2,
    )
    assert kept.rules[0].body == "Move to x=0.37; descend; retry."

    cleared, _ = apply_patch(
        overlay=kept,
        patch={"operations": [dict(rule_only, body="")]},
        available_seeds={3, 5, 7},
        seed=7,
        max_skills=6,
        min_evidence=2,
    )
    assert cleared.rules[0].body == ""


def test_clamp_doc_cuts_at_a_line_boundary() -> None:
    lines = [f"- fact {i}: depth {i}.5 mm" for i in range(60)]
    text = "\n".join(lines)
    clamped, was_clamped = clamp_doc(text, 400)
    assert was_clamped
    assert len(clamped) <= 400 + 80
    assert "document compacted" in clamped
    assert "- fact 0" in clamped
    # The newest trial-log lines live at the tail and survive the cut.
    assert "- fact 59" in clamped
    # Short documents are untouched.
    short, was_clamped = clamp_doc("- fact: depth 27 mm", 400)
    assert short == "- fact: depth 27 mm"
    assert not was_clamped


def test_seeds_mentioned_reads_both_shapes() -> None:
    doc = (
        "## Confirmed facts\n"
        "- [seed 3,5,7] depth stays above 27 mm\n"
        "- [1, 11] closes at 20.5 crush the glass\n"
        "- seed_12 repeats the same failure\n"
    )
    assert seeds_mentioned(doc) == {1, 3, 5, 7, 11, 12}


def test_parse_editor_reply_requires_doc_and_operations() -> None:
    fenced = """Here is the update:
```json
{"doc": "# RUN LOG\\n## Trial log\\n- seed 1: failed", "operations": [{"op": "delete", "name": "old-rule", "reason": "contradicted", "evidence_seeds": [1]}]}
```
"""
    parsed = parse_editor_reply(fenced)
    assert parsed is not None
    assert parsed["doc"].startswith("# RUN LOG")
    assert parsed["operations"][0]["name"] == "old-rule"
    # The two artefacts travel together: missing either rejects the reply.
    assert parse_editor_reply('{"operations": []}') is None
    assert parse_editor_reply('{"doc": ""}') is None
    assert parse_editor_reply("no json here") is None


def test_add_needs_two_distinct_seeds_the_run_has_recorded() -> None:
    patch = {
        "operations": [
            {
                "op": "add",
                "name": "depth-ceiling",
                "rule": "keep the pad depth above 27 mm while seated",
                "evidence_seeds": [3],
            }
        ]
    }
    overlay, net = apply_patch(
        overlay=LearnedOverlay(),
        patch=patch,
        available_seeds={3, 5},
        seed=5,
        max_skills=6,
        min_evidence=2,
    )
    assert overlay.rules == []
    assert net == []
    assert overlay.rejected[0]["reason"] == "needs at least 2 distinct evidence seeds"

    patch["operations"][0]["evidence_seeds"] = [3, 5]
    overlay, net = apply_patch(
        overlay=LearnedOverlay(),
        patch=patch,
        available_seeds={3, 5},
        seed=5,
        max_skills=6,
        min_evidence=2,
    )
    assert [skill.name for skill in overlay.rules] == ["depth-ceiling"]
    assert net == ["add depth-ceiling"]
    assert overlay.revision == 1


def test_add_rejects_seeds_outside_the_document_or_attempt() -> None:
    patch = {
        "operations": [
            {
                "op": "add",
                "name": "invented-history",
                "rule": "keep the pad depth above 27 mm while seated",
                "evidence_seeds": [1, 2],
            }
        ]
    }
    overlay, net = apply_patch(
        overlay=LearnedOverlay(),
        patch=patch,
        available_seeds={3, 5},
        seed=5,
        max_skills=6,
        min_evidence=2,
    )
    assert net == []
    assert "not in the run document" in overlay.rejected[0]["reason"]


def test_revise_keeps_earlier_evidence_and_delete_removes() -> None:
    overlay = LearnedOverlay(
        seed=5,
        revision=1,
        rules=[_rule("depth-ceiling", "keep the pad depth above 27 mm", seeds=(3, 5))],
    )
    revise = {
        "operations": [
            {
                "op": "revise",
                "name": "depth-ceiling",
                "rule": "keep the pad depth above 27.5 mm while seated",
                "evidence_seeds": [7, 8],
            }
        ]
    }
    overlay, net = apply_patch(
        overlay=overlay,
        patch=revise,
        available_seeds={7, 8},
        seed=8,
        max_skills=6,
        min_evidence=2,
    )
    assert net == ["revise depth-ceiling"]
    assert overlay.rules[0].rule.endswith("27.5 mm while seated")
    # The rule's own earlier seeds are still allowed on a revise.
    assert overlay.rules[0].evidence_seeds == (3, 5, 7, 8)

    delete = {
        "operations": [
            {
                "op": "delete",
                "name": "depth-ceiling",
                "reason": "a 24 mm grasp still succeeded",
                "evidence_seeds": [8],
            }
        ]
    }
    overlay, net = apply_patch(
        overlay=overlay,
        patch=delete,
        available_seeds={7, 8},
        seed=8,
        max_skills=6,
        min_evidence=2,
    )
    assert net == ["delete depth-ceiling"]
    assert overlay.rules == []


def test_cap_drops_the_least_supported_rule() -> None:
    overlay = LearnedOverlay(
        seed=5,
        revision=1,
        rules=[_rule("kept-rule", "keep the pad depth above 27 mm", seeds=(3, 5, 7))],
    )
    patch = {
        "operations": [
            {
                "op": "add",
                "name": "weak-rule",
                "rule": "confirm both pads at 25 mm before turning",
                "evidence_seeds": [7, 8],
            }
        ]
    }
    overlay, net = apply_patch(
        overlay=overlay,
        patch=patch,
        available_seeds={3, 5, 7, 8},
        seed=8,
        max_skills=1,
        min_evidence=2,
    )
    assert net == []
    assert [skill.name for skill in overlay.rules] == ["kept-rule"]
    assert overlay.rejected[-1]["op"] == "cap"
    assert overlay.rejected[-1]["name"] == "weak-rule"


def test_record_evidence_is_raw_and_has_no_card() -> None:
    record = _record(
        7,
        reward=0.0,
        task_completed=False,
        card="DID: closed at 20.5\nWHY IT FAILED: crushed",
        check_detail={"failure": "glass_crushed", "held_steps": 0},
        code="close_gripper(force_threshold=20.5)\nmove_delta(dz=0.03)",
        stdout="depth=27.41 contact=True\n",
        stderr="RuntimeError: Motion failed: glass_crushed\n",
    )
    evidence = record_evidence(record)
    assert "seed 7 (FAILED)" in evidence
    assert "glass_crushed" in evidence
    assert "force_threshold=20.5" in evidence
    assert "depth=27.41" in evidence
    assert "RuntimeError" in evidence
    # The editor works from measurements, never from the solver's own card.
    assert "DID: closed at 20.5" not in evidence
    assert "CARD" not in evidence


def test_record_evidence_keeps_head_and_tail_of_long_logs() -> None:
    record = _record(7, stdout="A" * 3000 + "B" * 3000)
    evidence = record_evidence(record, log_max_chars=400)
    assert "chars omitted" in evidence
    assert evidence.count("A") > 100
    assert evidence.count("B") > 100


def test_record_evidence_includes_the_program_only_when_asked() -> None:
    record = _record(7, stdout="done", code="close_gripper(force_threshold=20.5)")
    lean = record_evidence(record)
    # Without the program the digest is the only code-derived material.
    assert "PARAMETERS: force_threshold=20.5" in lean
    assert "PROGRAM" not in lean
    rich = record_evidence(record, include_code=True)
    assert "PROGRAM" in rich
    assert "close_gripper(force_threshold=20.5)" in rich
    # With the program in view the digest is a lossy duplicate, so it is gone.
    assert "PARAMETERS:" not in rich


def test_editor_prompt_can_carry_the_task_prompt_and_program() -> None:
    record = _record(7, code="press_key(60)", stdout="note=60 force=0.35")
    messages = build_editor_messages(
        task="bulb_tightening",
        overlay=LearnedOverlay(),
        base_skills="TACTILE SKILLS",
        run_doc="",
        record=record,
        max_skills=6,
        min_evidence=2,
        doc_max_chars=12000,
        task_prompt="Goal: turn the bulb until it seats.",
        include_code=True,
    )
    system = messages[0]["content"][0]["text"]
    user = messages[1]["content"][0]["text"]
    # The prompt says which material decides a fact.
    assert "EVIDENCE ORDER" in system
    assert "never as proof that a step happened" in system
    # The per-scene goal and the executed program reach the editor.
    assert "TASK PROMPT THE AGENT RECEIVED" in user
    assert "Goal: turn the bulb until it seats." in user
    assert "PROGRAM" in user
    assert "press_key(60)" in user


def test_editor_prompt_carries_doc_and_this_attempt_only() -> None:
    doc = "# RUN LOG\n## Confirmed facts\n- [seed 3,5] depth above 27 mm\n"
    record = _record(
        7,
        card="DID: closed at 20.5",
        stdout="depth=27.41 contact=True",
        check_detail={"failure": "glass_crushed"},
    )
    messages = build_editor_messages(
        task="bulb_tightening",
        overlay=LearnedOverlay(seed=5, revision=1, rules=[_rule("depth-ceiling", "x")]),
        base_skills="TACTILE SKILLS:\n- bulb-close-force: ...",
        run_doc=doc,
        record=record,
        max_skills=6,
        min_evidence=2,
        doc_max_chars=12000,
        task_prompt=(
            "Solve bulb_tightening.\n\nTACTILE SKILLS:\n- bulb-close-force: ..."
        ),
    )
    assert messages[0]["role"] == "system"
    user = messages[1]["content"][0]["text"]
    assert "TASK: bulb_tightening" in user
    assert "CURRENT RUN DOCUMENT" in user
    assert "- [seed 3,5] depth above 27 mm" in user
    assert "CURRENT LEARNED SKILLS" in user
    assert "BASE TACTILE SKILLS" in user
    # The curated block already sits inside the task prompt: the editor is
    # pointed at it instead of being shown a second copy.
    assert user.count("- bulb-close-force: ...") == 1
    assert "block inside the task prompt" in user
    assert "THIS ATTEMPT" in user
    assert "seed 7" in user
    # No card and no older attempt: history is the document's job.
    assert "DID: closed at 20.5" not in user


def test_write_overlay_writes_the_exact_block(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    overlay = LearnedOverlay(
        seed=4, revision=2, rules=[_rule("depth-ceiling", "keep depth above 27 mm")]
    )
    path = write_overlay(config, overlay)
    assert path.exists()
    md = (path.parent / "seed_004.md").read_text()
    assert md == format_block(overlay)
    data = json.loads(path.read_text())
    assert data["rules"][0]["name"] == "depth-ceiling"
