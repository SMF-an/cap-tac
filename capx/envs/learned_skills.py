"""Run-local learned skills, backed by a shared run document.

The curated library in ``capx.tactile_skills`` is shared by every task and is
never written to during a run.  This module adds the layer between that
library and the per-seed traces:

  * a **run document** the editor rewrites after every attempt: confirmed
    facts, dead ends, open questions and a one-line log per seed.  It is the
    editor's only history and is never delivered to the agent;
  * a short **learned skill list** distilled from that document and appended
    to the curated TACTILE SKILLS block in the next seed's prompt, in the same
    one-line-per-skill form.  There is no separate heading: the agent reads
    them as skills, and the run-document history behind them stays on disk.

The editor sees exactly one attempt in full: the raw stdout/stderr of the
seed that just finished and its check detail.  It can also be given the
program itself -- or, when that is withheld, a digest of its numeric
parameters -- and the task prompt that seed received: the prompt states which
of them decides a fact -- the streams and check detail do, the program only
recovers control flow, and the task prompt only explains the per-scene goal.
It does not see cards: a card is the solving model's own compression and can
be wrong, while the raw log is the measurement.  History reaches the editor
through the run document, not through a window of old attempts.

Each editor call returns both artefacts at once::

    {"doc": "<complete new run document>", "operations": [...]}

The code writes the document first, then applies the patch under the evidence
and size gates, so the two always describe the same state.  Every version is
kept on disk below ``<output_dir>/learned_skills/``::

    run_doc_seed_003.md   # the document after seed 3 (editor-only)
    seed_003.json         # rules + patch + usage (the injected list)
    seed_003.md           # exact text injected into seed 4's prompt
    run_doc.md            # copy of the newest document, for convenience

Config keys (all top-level in the run's YAML)::

    learned_skills: true            # master switch
    learned_skills_inject: true     # false = editor runs, block withheld
    learned_skills_max: 6           # hard cap on one-line rules
    learned_skills_evidence: 2      # distinct seeds a new rule needs
    learned_skills_model: ...       # editor model, defaults to the run's model
    learned_skills_include_prompt: true   # task prompt text, images excluded
    learned_skills_include_code: true     # the attempt's program
    learned_skills_code_max_chars: 20000  # program budget
    learned_skills_log_max_chars: 120000  # raw log budget per attempt
    learned_skills_doc_max_chars: 12000   # run-document size cap

The agent arm this is meant for is ``trace_mode=off`` plus
``learned_skills: true``: the prompt carries the curated skills and the
learned list, and no cards or trace window.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capx.envs.trace_chain import TraceRecord, trace_chain_dir

DIRNAME = "learned_skills"
FILE_TEMPLATE = "seed_{seed:03d}.json"
MD_TEMPLATE = "seed_{seed:03d}.md"
DOC_TEMPLATE = "run_doc_seed_{seed:03d}.md"
DOC_LATEST = "run_doc.md"

_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
# A learned rule stays one line; a procedure that cannot fit there may carry
# this much body (shown verbatim under the rule).
_BODY_MAX_CHARS = 800
_BODY_MAX_LINES = 12
_PARAM_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\b"
)
# Seed citations the editor writes in the run document.  Both "seed 3,5,7"
# and "[3, 5, 7]" are accepted because both shapes are natural in the
# section format the prompt asks for.
_SEED_RE = re.compile(r"(?i)\bseeds?[_\s-]*(\d+(?:\s*[,，]\s*\d+)*)")
_BRACKET_SEEDS_RE = re.compile(r"\[\s*(\d+(?:\s*[,，]\s*\d+)*)\s*\]")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedSkill:
    """One provisional rule.

    Attributes:
        name: Kebab-case handle, stable across revisions.
        rule: One line stating what a reading means, including the number or
            channel that makes it checkable.
        body: Optional calculation or measurement sequence the one-line rule
            cannot carry.  Shown verbatim to the next agent, indented under
            the rule; it must not contain an abort or recovery procedure.
        evidence_seeds: Seeds whose attempts support the current wording.
        created_seed: Seed whose editor first added the rule.
        revised_seed: Seed whose editor last rewrote it, if any.
    """

    name: str
    rule: str
    body: str = ""
    evidence_seeds: tuple[int, ...] = ()
    created_seed: int | None = None
    revised_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule": self.rule,
            "body": self.body,
            "evidence_seeds": list(self.evidence_seeds),
            "created_seed": self.created_seed,
            "revised_seed": self.revised_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedSkill":
        seeds = data.get("evidence_seeds") or []
        return cls(
            name=str(data.get("name", "")),
            rule=str(data.get("rule", "")),
            body=str(data.get("body") or ""),
            evidence_seeds=tuple(int(s) for s in seeds),
            created_seed=data.get("created_seed"),
            revised_seed=data.get("revised_seed"),
        )


@dataclass
class LearnedOverlay:
    """The whole provisional list after one seed's editor call.

    Attributes:
        seed: Seed whose editor produced this version; 0 is the empty state.
        revision: Counts applied changes, not editor calls, so a no-op run
            keeps the previous number.
        rules: The current list, in injection order.
        rejected: Operations the code refused, with the reason.  Kept for
            analysis, never injected.
        patch: The editor's raw JSON, for audit.
        editor: Token usage of the editor call, mirroring ``meta["reflection"]``.
        doc_chars: Length of the run document written at the same time.
        doc_clamped: Whether the document had to be cut to the size cap.
    """

    seed: int = 0
    revision: int = 0
    rules: list[LearnedSkill] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    patch: dict[str, Any] | None = None
    editor: dict[str, int] = field(default_factory=dict)
    doc_chars: int = 0
    doc_clamped: bool = False

    def by_name(self) -> dict[str, LearnedSkill]:
        return {skill.name: skill for skill in self.rules}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "revision": self.revision,
            "rules": [skill.to_dict() for skill in self.rules],
            "rejected": self.rejected,
            "patch": self.patch,
            "editor": self.editor,
            "doc_chars": self.doc_chars,
            "doc_clamped": self.doc_clamped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedOverlay":
        return cls(
            seed=int(data.get("seed", 0)),
            revision=int(data.get("revision", 0)),
            rules=[LearnedSkill.from_dict(d) for d in data.get("rules") or []],
            rejected=list(data.get("rejected") or []),
            patch=data.get("patch"),
            editor=dict(data.get("editor") or {}),
            doc_chars=int(data.get("doc_chars", 0)),
            doc_clamped=bool(data.get("doc_clamped", False)),
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def overlay_dir(config: dict[str, Any]) -> Path:
    """``<output_dir>/learned_skills``, beside the trace chain."""
    return trace_chain_dir(config).parent / DIRNAME


def overlay_path(config: dict[str, Any], seed: int) -> Path:
    return overlay_dir(config) / FILE_TEMPLATE.format(seed=int(seed))


def load_overlay(config: dict[str, Any], before_seed: int) -> LearnedOverlay:
    """The latest overlay written below *before_seed*, or the empty one."""
    directory = overlay_dir(config)
    for seed in range(int(before_seed) - 1, 0, -1):
        path = directory / FILE_TEMPLATE.format(seed=seed)
        if not path.exists():
            continue
        try:
            return LearnedOverlay.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            continue
    return LearnedOverlay()


def write_overlay(config: dict[str, Any], overlay: LearnedOverlay) -> Path:
    """Write one version as JSON plus its exact injected text."""
    directory = overlay_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILE_TEMPLATE.format(seed=int(overlay.seed))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(overlay.to_dict(), indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    text = format_block(overlay)
    md_path = directory / MD_TEMPLATE.format(seed=int(overlay.seed))
    md_path.write_text(text if text else "(no learned skills yet)\n")
    return path


def load_run_doc(config: dict[str, Any], before_seed: int) -> str:
    """The run document written below *before_seed*, or "".

    A missing snapshot means the editor failed on that seed; the latest
    earlier document stays in force, exactly like the overlay.
    """
    directory = overlay_dir(config)
    for seed in range(int(before_seed) - 1, 0, -1):
        path = directory / DOC_TEMPLATE.format(seed=seed)
        if not path.exists():
            continue
        try:
            return path.read_text()
        except OSError:
            continue
    return ""


def write_run_doc(config: dict[str, Any], seed: int, text: str) -> Path:
    """Snapshot one seed's run document and refresh the ``run_doc.md`` copy."""
    directory = overlay_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DOC_TEMPLATE.format(seed=int(seed))
    path.write_text(text)
    (directory / DOC_LATEST).write_text(text)
    return path


def clamp_doc(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut *text* at line boundaries to *max_chars*, keeping both ends.

    The document is model-written and can grow; the cap is a hard budget, not
    a suggestion.  The head holds the facts and dead ends, the tail holds the
    newest trial-log lines, so the cut is taken out of the middle.  The marker
    tells a later editor that something was dropped.
    """
    text = (text or "").strip()
    limit = max(400, int(max_chars))
    if len(text) <= limit:
        return text, False
    marker = "\n... [document compacted to the size cap] ...\n"
    room = limit - len(marker)
    head_budget = int(room * 0.65)
    tail_budget = room - head_budget
    head = text[:head_budget]
    newline = head.rfind("\n")
    if newline > 0:
        head = head[:newline]
    tail = text[-tail_budget:] if tail_budget > 0 else ""
    newline = tail.find("\n")
    if newline >= 0:
        tail = tail[newline + 1 :]
    return head.rstrip() + marker + tail.lstrip(), True


# ---------------------------------------------------------------------------
# Prompt block
# ---------------------------------------------------------------------------


def format_block(overlay: LearnedOverlay) -> str:
    """The exact lines appended to the next seed's prompt, or "" when empty.

    The shape matches the curated library's compact render -- ``- name: rule``
    -- so the two read as one list.  Provenance (which seed wrote it, which
    seeds support it) stays in the JSON and the per-seed markdown file; the
    prompt carries rules, not history.
    """
    if not overlay.rules:
        return ""
    lines = []
    for skill in overlay.rules:
        lines.append(f"- {skill.name}: {skill.rule}")
        if skill.body:
            lines.extend(f"  {line}" for line in skill.body.splitlines())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evidence and editor prompt
# ---------------------------------------------------------------------------


def _verdict(record: TraceRecord) -> str:
    if record.timed_out:
        return "TIMED OUT"
    if record.task_completed is True:
        return "SUCCEEDED"
    return "FAILED"


def parameter_digest(code: str, limit: int = 32) -> str:
    """The numeric keyword arguments the program used, deduplicated.

    Only for attempts whose program is not shown: it keeps the constants
    comparable across attempts without shipping the program itself.  When the
    editor sees the program, this digest is strictly worse than the source
    (it drops branches, order and repetition) and is not emitted.
    """
    seen: list[str] = []
    for name, value in _PARAM_RE.findall(code or ""):
        item = f"{name}={value}"
        if item not in seen:
            seen.append(item)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def _truncate_stream(text: str, max_chars: int, label: str) -> str:
    """Keep the head and the tail of a long stream, with a visible cut.

    The opening commands and the final state are both evidence; the middle of
    a long tactile dump is usually repetition, so the middle is what goes.
    """
    text = text or ""
    limit = max(400, int(max_chars))
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n... [{label}: {omitted} chars omitted] ...\n"
        + text[-tail:]
    )


def record_evidence(
    record: TraceRecord,
    log_max_chars: int = 60000,
    stderr_max_chars: int = 4000,
    include_code: bool = False,
    code_max_chars: int = 20000,
) -> str:
    """One attempt for the editor: verdict, detail, parameters, raw streams.

    No card: the editor works from the measurements, not from the solving
    model's own summary of them.  The program is optional on purpose: it says
    what the attempt tried to run, while the streams say what happened, and a
    long program only earns its tokens when the control flow is in doubt.
    """
    lines = [
        f"=== seed {record.seed} ({_verdict(record)}) "
        f"reward={record.reward:.3f} ===",
    ]
    detail = record.check_detail
    if detail:
        lines.append("CHECK DETAIL: " + json.dumps(detail, ensure_ascii=False))
    code = (record.code or "").strip()
    show_code = bool(include_code and code)
    if not show_code:
        # Fallback only: with the program in view, the digest is a lossy copy
        # of it (deduplicated, no branches, no order) and reads like a second
        # source of evidence when it is not one.
        params = parameter_digest(record.code)
        if params:
            lines.append("PARAMETERS: " + params)
    if show_code:
        lines.append(
            "PROGRAM (what the attempt tried to run; not evidence that a step "
            "happened):\n```python\n"
            + _truncate_stream(code, code_max_chars, "program")
            + "\n```"
        )
    stdout = _truncate_stream(record.stdout or "", log_max_chars, "stdout")
    stderr = _truncate_stream(record.stderr or "", stderr_max_chars, "stderr")
    if stdout.strip():
        lines.append("STDOUT:\n" + stdout.strip())
    if stderr.strip():
        lines.append("STDERR:\n" + stderr.strip())
    return "\n".join(lines)


def seeds_mentioned(text: str) -> set[int]:
    """Every seed id the run document cites, in either accepted shape."""
    found: set[int] = set()
    for pattern in (_SEED_RE, _BRACKET_SEEDS_RE):
        for match in pattern.finditer(text or ""):
            for part in re.split(r"[,，]", match.group(1)):
                part = part.strip()
                if part.isdigit():
                    found.add(int(part))
    return found


def build_editor_messages(
    *,
    task: str,
    overlay: LearnedOverlay,
    base_skills: str,
    run_doc: str,
    record: TraceRecord,
    max_skills: int,
    min_evidence: int,
    doc_max_chars: int,
    log_max_chars: int = 60000,
    task_prompt: str = "",
    include_code: bool = False,
    code_max_chars: int = 20000,
) -> list[dict[str, Any]]:
    """Messages for the one editor call that updates both artefacts.

    The editor sees the previous run document and exactly one new attempt:
    no cards, no window of older attempts.  History reaches it only through
    the document it is about to rewrite.  The attempt is the raw log and
    check detail; its program and the task prompt are optional context, and
    the prompt says which of the three decides a fact.
    """
    current = format_block(overlay) or "(no learned skills yet)"
    document = run_doc.strip() or "(the run document is empty)"
    attempt = record_evidence(
        record,
        log_max_chars=log_max_chars,
        include_code=include_code,
        code_max_chars=code_max_chars,
    )
    prompt_section = ""
    if task_prompt.strip():
        prompt_section = (
            "\nTASK PROMPT THE AGENT RECEIVED FOR THIS SCENE "
            "(text only; the scene image is omitted):\n"
            + task_prompt.strip()
            + "\n"
        )
    system = f"""You maintain two things for a robot agent that solves one task by reading GelSight tactile sensors.

1. RUN DOCUMENT: a compact, structured record of what the whole run has established. It is the only history you will see, and it is never shown to the agent.
2. LEARNED SKILLS: at most {max_skills} one-line rules distilled from that document and injected into the agent's prompt.

You receive the previous run document and exactly one new attempt: its raw stdout/stderr and the check detail, plus -- when shown -- the program it ran (or, if the program is withheld, a digest of its numeric parameters) and the task prompt the agent was given. There are no cards; trust the measurements, not any summary of them.

EVIDENCE ORDER. The check detail and the raw streams decide every fact. The program says only what the attempt tried to run: use it to recover the control flow (which branch, which loop, which order), never as proof that a step happened. The task prompt is the goal for this scene, including any seed-specific target: it explains the check detail, but its goal or API text is not a skill. A constant that happened to work in this attempt belongs in the run document, not in a skill; a skill must still hold after the scene changes.

Before you write anything, answer this for yourself: why did this attempt stop? If an earlier seed stopped the same way, which seeds were they, and why has no rule covered it yet? Then check the other direction: was this stop caused by the attempt treating a learned rule as a hard condition it had to satisfy? A rule that was obeyed and still failed is a defective rule, not a missing one.

STEP 1 - UPDATE THE DOCUMENT. Rewrite it; do not append endlessly. Keep it under {doc_max_chars} characters. Use exactly these sections:

# RUN LOG - {task}
## Confirmed facts
- [seed 3,5] <a measured relationship, with the number>
## Dead ends
- [seed 1,11] <what was tried and the failure it produced>
## Open questions
- <what the run still does not know>
## Trial log
- seed N: <action -> outcome -> key measurement>

Facts and dead ends must cite the seeds that support them. Keep every fact or dead end that later evidence has not contradicted. Compact older trial-log lines before dropping facts.

STEP 2 - UPDATE THE SKILLS. A skill's `rule` states what a reading means: which channel or condition, what the number implies, and what it does not imply. It is background for the next agent's own decisions, not a condition on whether the attempt may continue. It must be about how to read a measurement, not about what the task is or how to call a function. A skill may carry an optional `body` (at most {_BODY_MAX_CHARS} characters and {_BODY_MAX_LINES} lines) only for a calculation or a measurement sequence -- how to obtain or compute a quantity; the next agent sees it verbatim under the rule.

Rules for a skill:
- The `rule` is one line, at most 240 characters, no newlines.
- Aim to include the number or the measurement channel that makes it checkable (depth, shear, coherence, force, relief, anisotropy, contact, texture, hardness, load, ...). A rule that names neither is hard for the next agent to verify.
- No history and no narrative ("we tried", "in the last attempt"). Write the rule as it should be followed next time.
- A rule may call a reading invalid or a measurement unavailable; say what that means for the measurement, not what it means for the attempt. Never write a rule or body that tells the agent to abort, raise, stop the trial, require a condition before proceeding, or give up. If a procedure is unavoidable, describe what to measure next (which channel, pose, or reference), not what to do after a failure.

Rules for changes:
- Add or revise a skill only when at least {min_evidence} distinct seeds support it, and cite them in evidence_seeds. Every cited seed must already appear in the run document you were given, or be this attempt's seed.
- Delete a skill when the evidence contradicts it; cite the contradicting seeds. If this attempt stopped because of a rule in the current learned list, that rule is wrong: delete it, or replace it with a more accurate statement about the reading. Do not tighten it into a stricter condition. A rule that two seeds obeyed without changing the outcome should be deleted.
- Keep at most {max_skills} skills. Revise or delete before adding when the list is full.
- If the evidence is not enough for any change, return an empty operations list.

Return strict JSON only, in this shape:
{{"doc": "the complete new run document", "operations": [{{"op": "add", "name": "kebab-name", "rule": "...", "body": "optional measurement steps", "evidence_seeds": [3, 5]}}, {{"op": "revise", "name": "existing-name", "rule": "...", "body": "optional measurement steps", "evidence_seeds": [3, 5, 7]}}, {{"op": "delete", "name": "existing-name", "reason": "...", "evidence_seeds": [7]}}]}}"""
    # The curated block is already inside the task prompt text; showing it a
    # second time only spends tokens.  When the task prompt is withheld, it is
    # the editor's only view of the curated library and must be printed here.
    if not base_skills.strip():
        base_note = "(none delivered)"
    elif task_prompt.strip():
        base_note = (
            "the TACTILE SKILLS block inside the task prompt above -- do not "
            "repeat it in the learned list; revise the learned list only when "
            "the evidence contradicts it."
        )
    else:
        base_note = base_skills.strip()
    user = f"""TASK: {task}
{prompt_section}
CURRENT RUN DOCUMENT (through the seed before this one):
{document}

CURRENT LEARNED SKILLS (revision {overlay.revision}, written after seed {overlay.seed}):
{current}

BASE TACTILE SKILLS ALREADY IN THE AGENT'S PROMPT:
{base_note}

THIS ATTEMPT:
{attempt}

Return the JSON now."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


# ---------------------------------------------------------------------------
# Reply parsing and patch application
# ---------------------------------------------------------------------------


def parse_editor_reply(text: str) -> dict[str, Any] | None:
    """Extract ``{"doc": ..., "operations": [...]}`` from the editor reply.

    Models sometimes wrap JSON in a fenced block and occasionally add a
    sentence around it; the first object brace to the last closing brace is
    the whole payload in both cases.  A reply without a usable document is
    rejected as a whole: the two artefacts are updated together or not at all.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("doc"), str) or not data["doc"].strip():
        return None
    if not isinstance(data.get("operations"), list):
        return None
    return data


def _validate_rule(rule: str) -> str | None:
    """Return a rejection reason, or None when the rule is deliverable.

    Naming a checkable number or channel is a prompt instruction, not a gate:
    a keyword whitelist cannot cover every task's own property names, and on
    2026-11-11 it refused the one memory_match rule that addressed the run's
    dominant error (equalize-property-groups, whose text names texture,
    hardness and load -- none of which were on the list).
    """
    if not isinstance(rule, str):
        return "rule is not a string"
    rule = rule.strip()
    if "\n" in rule:
        return "rule must be one line"
    if not 15 <= len(rule) <= 240:
        return "rule must be 15-240 characters"
    return None


def _validate_body(body: Any) -> str | None:
    """Return a rejection reason, or None when the body is deliverable."""
    if body is None:
        return None
    if not isinstance(body, str):
        return "body must be a string"
    body = body.strip()
    if not body:
        return None
    if len(body) > _BODY_MAX_CHARS:
        return f"body must be at most {_BODY_MAX_CHARS} characters"
    if len(body.splitlines()) > _BODY_MAX_LINES:
        return f"body must be at most {_BODY_MAX_LINES} lines"
    return None


def _validate_name(name: str) -> str | None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return "name must be kebab-case"
    if not 3 <= len(name) <= 40:
        return "name must be 3-40 characters"
    return None


def _clean_seeds(
    raw: Any,
    *,
    available: set[int],
    minimum: int,
) -> tuple[tuple[int, ...], str | None]:
    if not isinstance(raw, list):
        return (), "evidence_seeds must be a list"
    try:
        seeds = tuple(sorted({int(s) for s in raw}))
    except (TypeError, ValueError):
        return (), "evidence_seeds must be integers"
    missing = [s for s in seeds if s not in available]
    if missing:
        return (), (
            f"evidence seeds {missing} are not in the run document "
            "and are not this attempt"
        )
    if len(seeds) < minimum:
        return (), f"needs at least {minimum} distinct evidence seeds"
    return seeds, None


def apply_patch(
    *,
    overlay: LearnedOverlay,
    patch: dict[str, Any],
    available_seeds: set[int],
    seed: int,
    max_skills: int,
    min_evidence: int,
) -> tuple[LearnedOverlay, list[str]]:
    """Apply the editor's skill operations under the evidence and size gates.

    The gates are enforced here rather than trusted to the editor: an
    operation that cannot cite seeds the run has actually recorded, names an
    unknown rule, or pushes the list past the cap is refused and recorded in
    ``rejected``.

    Args:
        overlay: The list this seed started with.
        patch: The parsed reply; only ``operations`` is read here.
        available_seeds: Seeds the editor has actually seen -- the previous
            run document's citations plus this attempt.
        seed: The seed the new version is written for.
        max_skills: Hard cap on delivered rules.
        min_evidence: Distinct attempts an add or revise needs.

    Returns:
        ``(new_overlay, net_changes)``.
    """
    rules = list(overlay.rules)
    by_name = {skill.name: skill for skill in rules}
    rejected: list[dict[str, Any]] = []

    for raw in patch.get("operations", [])[:32]:
        if not isinstance(raw, dict):
            rejected.append({"op": "?", "reason": "operation is not an object"})
            continue
        op = str(raw.get("op", "")).lower()
        name = str(raw.get("name", ""))
        if op not in ("add", "revise", "delete"):
            rejected.append({"op": op, "name": name, "reason": "unknown op"})
            continue
        name_error = _validate_name(name)
        if name_error:
            rejected.append({"op": op, "name": name, "reason": name_error})
            continue
        if op == "add" and name in by_name:
            rejected.append(
                {"op": op, "name": name, "reason": "name already exists; use revise"}
            )
            continue
        if op in ("revise", "delete") and name not in by_name:
            rejected.append(
                {"op": op, "name": name, "reason": "no such learned skill"}
            )
            continue
        # A revise may carry forward the evidence the rule already had; an
        # add may only cite seeds the editor has actually seen.
        allowed = (
            set(available_seeds) | set(by_name[name].evidence_seeds)
            if op == "revise"
            else set(available_seeds)
        )
        minimum = 1 if op == "delete" else int(min_evidence)
        seeds, seed_error = _clean_seeds(
            raw.get("evidence_seeds"), available=allowed, minimum=minimum
        )
        if seed_error:
            rejected.append({"op": op, "name": name, "reason": seed_error})
            continue

        if op == "delete":
            rules = [skill for skill in rules if skill.name != name]
            by_name.pop(name, None)
            continue

        rule = str(raw.get("rule", "")).strip()
        rule_error = _validate_rule(rule)
        if rule_error:
            rejected.append({"op": op, "name": name, "reason": rule_error})
            continue

        raw_body = raw.get("body")
        body_error = _validate_body(raw_body)
        if body_error:
            rejected.append({"op": op, "name": name, "reason": body_error})
            continue
        body = str(raw_body).strip() if isinstance(raw_body, str) else None

        if op == "add":
            skill = LearnedSkill(
                name=name,
                rule=rule,
                body=body or "",
                evidence_seeds=seeds,
                created_seed=int(seed),
            )
            rules.append(skill)
            by_name[name] = skill
        else:
            old = by_name[name]
            merged = tuple(sorted(set(old.evidence_seeds) | set(seeds)))
            skill = LearnedSkill(
                name=name,
                rule=rule,
                # A rule-only revise keeps the body it was carrying; an
                # explicit empty string clears it.
                body=old.body if body is None else body,
                evidence_seeds=merged,
                created_seed=old.created_seed,
                revised_seed=int(seed),
            )
            rules = [skill if s.name == name else s for s in rules]
            by_name[name] = skill

    # A cap violation is resolved by dropping the least-supported rule: the
    # point of the list is that it stays short enough to read every seed.
    while len(rules) > int(max_skills):
        weakest = min(
            rules,
            key=lambda s: (len(s.evidence_seeds), s.created_seed or 0),
        )
        rules = [skill for skill in rules if skill.name != weakest.name]
        rejected.append(
            {
                "op": "cap",
                "name": weakest.name,
                "reason": f"dropped to keep at most {max_skills} rules",
            }
        )

    # Report the net change, not the operations attempted: an add that the
    # cap then drops is not a change, and a no-op seed keeps its revision.
    before = {skill.name: skill.rule for skill in overlay.rules}
    after = {skill.name: skill.rule for skill in rules}
    net: list[str] = []
    for name in sorted(set(before) | set(after)):
        if name not in before:
            net.append(f"add {name}")
        elif name not in after:
            net.append(f"delete {name}")
        elif before[name] != after[name]:
            net.append(f"revise {name}")
    revision = overlay.revision + (1 if net else 0)
    new_overlay = LearnedOverlay(
        seed=int(seed),
        revision=revision,
        rules=rules,
        rejected=rejected,
        patch=dict(patch),
        editor={},
    )
    return new_overlay, net
