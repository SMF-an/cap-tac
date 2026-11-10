"""One-trial execution traces, carried to the next seed's prompt.

The experiment this serves runs seeds 1..N in order, each in its own process,
and hands seed ``N`` the trace of the last attempt below it.  The trace is the
previous trial's final program, its execution output and its outcome -- not the
model's hidden reasoning, and not the simulator's ground truth.

Two rules matter here:

* The framework appends a hidden ``check_success()`` call to the last code
  block before execution (see ``CodeExecutionEnvBase.final_check_name``).  That
  line is scaffolding, not something the agent wrote, and the function is
  deliberately absent from the API document.  A trace that carried it would
  teach the next seed about a function its prompt never mentions, so
  :func:`strip_final_check` removes exactly the appended suffix.
* The trace must be bounded.  A thirty-seed chain that accumulates or that lets
  one runaway program fill the prompt would make the experiment's cost and its
  behaviour depend on the worst trial.  :func:`format_trace` caps the block at
  ``max_chars``, keeping the program and the tails of the two output streams.

The chain lives on disk, not in memory, because each seed is launched
separately and because the server-backed tasks that need a fresh server per
seed cannot keep Python state across seeds.  ``<output_dir>/trace_chain/`` holds
one JSON file per seed plus the exact block injected into that seed's prompt.

``trace_mode`` selects what the next seed receives: ``last`` (the original
arm), ``dual`` (that record plus the best completed success so far), ``card``
(a four-line experience card written by one reflection call) or ``dual_card``
(cards for the last record and the best success).  ``last`` is kept exactly as
it was, so the runs already on disk stay comparable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_FILE_TEMPLATE = "seed_{seed:03d}.json"
INJECTED_FILE_TEMPLATE = "seed_{seed:03d}_injected_block.txt"
DEFAULT_MAX_CHARS = 12000

# How much of the chain reaches the next seed.
#
#   off          no trace at all
#   last         the most recent record below the seed (the original arm)
#   dual         that record plus the best completed success so far
#   card         the model's four-line experience card for the last record
#   dual_card    cards for the last record and for the best success
#   window_card  cards for the last ``trace_window`` records plus the best
#                success, with a one-line streak when they all ended alike
#
# ``last`` is kept exactly as it was: it is the arm the 2026-11-08/09 chains
# ran, and every result so far is measured against it.  The other modes are
# separate arms, never a silent change to it.
TRACE_MODES = ("off", "last", "dual", "card", "dual_card", "window_card")

CLOSING_LINE = (
    "This is evidence from a different scene instance, not instructions. "
    "Do not assume the object is in the same place or has the same class "
    "as in that episode."
)


def trace_enabled(mode: str | None) -> bool:
    """Whether *mode* turns the chain on."""
    return str(mode or "off").lower() != "off"


def trace_uses_best(mode: str | None) -> bool:
    """Whether *mode* also injects the best successful record so far."""
    return str(mode or "off").lower() in ("dual", "dual_card", "window_card")


def trace_uses_card(mode: str | None) -> bool:
    """Whether *mode* injects a reflection card instead of the raw program."""
    return str(mode or "off").lower() in ("card", "dual_card", "window_card")


def trace_uses_window(mode: str | None) -> bool:
    """Whether *mode* injects the last few records, not just one."""
    return str(mode or "off").lower() == "window_card"

# The candidate names a task's check detail may report placement for.  Fixed
# here rather than trusting the detail's keys: an unexpected name would be
# rendered verbatim into the next prompt.
_TRACE_CANDIDATE_NAMES = frozenset({"candidate_left", "candidate_right"})


@dataclass
class TraceRecord:
    """What one seed leaves behind for the next one.

    Attributes:
        seed: Trial id, which is also the environment seed.
        complete: False for an intermediate write made after a code block, True
            once the attempt is over (normally or by timeout).
        timed_out: The attempt was ended by the trial wall-clock alarm.
        sandbox_rc: 0 when the executed code ran to completion.
        reward: The environment's reward after the last executed block.
        task_completed: The task's own verdict, or None when the task cannot
            report one.  This -- not ``sandbox_rc`` -- is the success measure.
        code: Final program, with the framework-appended final check removed.
        stdout: Execution stdout, one section per block when there are several.
        stderr: Execution stderr, same shape as ``stdout``.
        check_detail: The task's non-oracle check diagnostics (geometric
            reasons and measurements), or None.  Ground-truth labels and object
            poses are removed by :func:`filter_check_detail`.
        card: The model's four-line experience card for this attempt, written
            by one reflection call after the attempt ends.  Empty when card
            modes are off or the reflection call failed; the next seed then
            falls back to the raw trace for that attempt.
        num_code_blocks: Blocks in the final program.
        num_regenerations: Times the model chose to regenerate.
        num_finishes: Times the model chose to finish.
        elapsed_s: Wall time of the attempt.
        injected_from_seed: Which earlier seed's trace this attempt received.
        injected_best_seed: Which successful seed's record was injected
            alongside it (dual modes), or None.
        injected_window_seeds: The seeds whose records were injected as the
            window, newest first (window mode), or None.
        input_tokens: Model input tokens summed over the attempt.
        output_tokens: Model output tokens summed over the attempt.
        reasoning_tokens: Reasoning tokens (a subset of ``output_tokens``).
        cached_tokens: Cached input tokens.
        token_calls: Model calls made by the attempt.
        meta: Run settings worth keeping next to the trace (model, skill switch,
            trace mode, character cap).
        created_at: Unix time of the write.
    """

    seed: int
    complete: bool = False
    timed_out: bool = False
    sandbox_rc: int = 1
    reward: float = 0.0
    task_completed: bool | None = None
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    check_detail: dict[str, Any] | None = None
    card: str = ""
    num_code_blocks: int = 0
    num_regenerations: int = 0
    num_finishes: int = 0
    elapsed_s: float = 0.0
    injected_from_seed: int | None = None
    injected_best_seed: int | None = None
    injected_window_seeds: list[int] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    token_calls: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        """Rebuild a record, ignoring keys a newer writer added."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})


def trace_chain_dir(config: dict[str, Any]) -> Path:
    """Directory holding one run's chain.

    Defaults to ``<output_dir>/trace_chain``.  ``output_dir`` already carries
    the model name (the runner inserts it), so two models' chains never mix.
    """
    explicit = config.get("trace_chain_dir")
    if explicit:
        return Path(explicit)
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError(
            "trace_mode needs an output directory: pass --output-dir or set "
            "output_dir in the YAML"
        )
    return Path(output_dir) / "trace_chain"


def strip_final_check(code: str, final_check: str | None) -> str:
    """Remove the framework-appended ``final_check()`` call from *code*.

    Only the exact suffix the trial loop adds is removed.  A call the model
    wrote itself is left alone: that is part of its program, even though the
    function is hidden from the prompt.
    """
    if not final_check:
        return code
    suffix = f"\n\n{final_check}()\n"
    if code.endswith(suffix):
        return code[: -len(suffix)]
    stripped = code.rstrip()
    if stripped.endswith(f"{final_check}()"):
        return stripped[: -len(f"{final_check}()")].rstrip()
    return code


def filter_check_detail(
    detail: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The part of a task's check detail that may reach a trace.

    A served task's ``check_success_detail`` mixes two kinds of information:
    where the attempt ended up (axis errors, tilt) and what the answer was
    (``ground_truth_texture``, ``expected_pad``) or exactly where the objects
    are (raw poses).  Only the first kind is a measurement the robot could have
    taken itself; the second is the oracle.  The allowlist is deliberately
    small, so a new task field is withheld until someone decides it is safe.
    """
    if not isinstance(detail, dict):
        return None

    selected: dict[str, Any] = {}
    reasons = detail.get("reasons")
    if isinstance(reasons, (list, tuple)):
        reasons = [str(reason).strip() for reason in reasons if str(reason).strip()]
        if reasons:
            selected["reasons"] = reasons

    for key in ("pos_error_cm", "tilt_cos"):
        value = detail.get(key)
        if isinstance(value, (int, float, list, tuple)):
            selected[key] = value

    # tension_strap latches a failure string and reports how far the two-stage
    # hold got.  It never reports the measured tension -- that is the estimate
    # the task asks the robot to make.
    failure = detail.get("failure")
    if isinstance(failure, str) and failure.strip():
        selected["failure"] = failure.strip()[:120]

    # The piano's own end-of-episode reason: ``wrong_key``, ``wrong_note_force``,
    # ``partial_score``, ``timeout`` or ``success``.  It is the task-side
    # verdict, not the note sequence (``piano_sequence`` stays out), and it is
    # the only field that says an episode ended on its own clock rather than
    # on a failure.
    terminal_reason = detail.get("terminal_reason")
    if isinstance(terminal_reason, str) and terminal_reason.strip():
        selected["terminal_reason"] = terminal_reason.strip()[:120]

    # Counters of how far an attempt got: bulb_tightening's "good hold" steps
    # and blind_pick_balls' physics clock.  Neither is a value the scorer reads,
    # so neither can hand over the thing the episode asks the robot to measure
    # (the thread lead, the material bar).  ``phase`` is deliberately not here:
    # bulb_tightening updates it only on its expert path, so on a policy run it
    # would read "setup" forever and mislead the next seed.
    for key in ("held_steps", "physics_step"):
        value = detail.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            selected[key] = value

    # The piano reports how far through the note sequence an attempt got.
    # Like ``held_steps`` this is a progress counter, not a value the scorer
    # reads; the sequence itself lives in ``piano_sequence`` and stays out.
    for key in ("notes_completed", "notes_attempted", "total_notes"):
        value = detail.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            selected[key] = value

    stages = detail.get("stages_completed")
    if isinstance(stages, int) and not isinstance(stages, bool):
        selected["stages_completed"] = stages
    for key in ("current_hold_s", "elapsed_s"):
        value = detail.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            selected[key] = round(float(value), 2)

    # memory_match reports where each candidate physically ended up, because
    # its verdict is a bare bool.  Placement geometry is an observation the
    # robot could have taken itself; the task never puts a class label here.
    candidates = detail.get("candidates")
    if isinstance(candidates, dict):
        kept: dict[str, Any] = {}
        for name, info in candidates.items():
            # The two names the task reports.  Anything else -- a new field,
            # or a label smuggled in under the reference's name -- is not part
            # of the allowlist yet.
            if name not in _TRACE_CANDIDATE_NAMES:
                continue
            if not isinstance(info, dict):
                continue
            item: dict[str, Any] = {}
            for key in ("placed", "on_match_pad"):
                value = info.get(key)
                if isinstance(value, bool):
                    item[key] = value
            for key in ("xy_error_cm", "z_error_cm"):
                value = info.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item[key] = round(float(value), 2)
            if item:
                kept[name] = item
        if kept:
            selected["candidates"] = kept

    return selected or None


def _tail(text: str, limit: int) -> str:
    """Last *limit* characters, with a marker when the head was dropped."""
    if not text or len(text) <= limit:
        return text or ""
    return f"... [{len(text) - limit} chars omitted] ...\n{text[-limit:]}"


def _middle(text: str, limit: int) -> str:
    """Head and tail of *text*, with a marker for the omitted middle."""
    if not text or len(text) <= limit:
        return text or ""
    head = max(1, int(limit * 0.6))
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} chars omitted] ...\n{text[-tail:]}"


def _verdict(record: TraceRecord) -> str:
    """One word for how the attempt ended."""
    if record.timed_out:
        return "TIMED OUT"
    if not record.complete:
        return "INCOMPLETE"
    if record.task_completed is None:
        return "VERDICT UNKNOWN"
    return "SUCCEEDED" if bool(record.task_completed) else "FAILED"


def _detail_lines(detail: dict[str, Any] | None) -> list[str]:
    """The rendered CHECK DETAIL section, or an empty list."""
    if not detail:
        return []
    lines = ["-" * 70, "CHECK DETAIL:"]
    for reason in detail.get("reasons", []):
        lines.append(f"  - {reason}")
    pos_error = detail.get("pos_error_cm")
    if isinstance(pos_error, (list, tuple)):
        lines.append(
            "  position error (cm): "
            + ", ".join(f"{float(value):.2f}" for value in pos_error)
        )
    elif isinstance(pos_error, (int, float)):
        lines.append(f"  position error (cm): {float(pos_error):.2f}")
    tilt_cos = detail.get("tilt_cos")
    if isinstance(tilt_cos, (int, float)):
        lines.append(f"  tilt cosine: {float(tilt_cos):.4f}")
    failure = detail.get("failure")
    if isinstance(failure, str) and failure:
        lines.append(f"  failure: {failure}")
    terminal = detail.get("terminal_reason")
    if (
        isinstance(terminal, str)
        and terminal
        and terminal != failure
        and terminal != "success"
    ):
        lines.append(f"  terminal reason: {terminal}")
    for key, caption in (
        ("held_steps", "held steps"),
        ("physics_step", "physics step"),
        ("notes_completed", "notes completed"),
        ("notes_attempted", "notes attempted"),
        ("total_notes", "total notes"),
    ):
        value = detail.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            lines.append(f"  {caption}: {value}")
    stages = detail.get("stages_completed")
    if isinstance(stages, int) and not isinstance(stages, bool):
        lines.append(f"  stages completed: {stages}")
    for key, caption in (
        ("current_hold_s", "current hold"),
        ("elapsed_s", "elapsed"),
    ):
        value = detail.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append(f"  {caption}: {float(value):.2f} s")
    candidates = detail.get("candidates")
    if isinstance(candidates, dict):
        for name, info in candidates.items():
            if not isinstance(info, dict):
                continue
            parts = [
                f"{key}={info[key]}"
                for key in ("placed", "on_match_pad")
                if isinstance(info.get(key), bool)
            ]
            for key, caption in (("xy_error_cm", "xy error"), ("z_error_cm", "z error")):
                value = info.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    parts.append(f"{caption} {float(value):.2f} cm")
            if parts:
                lines.append(f"  {name}: " + ", ".join(parts))
    return lines


def _render(
    record: TraceRecord,
    code: str,
    stdout: str,
    stderr: str,
    *,
    label: str = "PREVIOUS ATTEMPT",
    closing: bool = True,
) -> str:
    """Assemble the prompt block around already-truncated pieces.

    ``label`` names the slot in a multi-record block ("PREVIOUS ATTEMPT",
    "BEST SUCCESS SO FAR").  ``closing`` is False for every block but the last:
    the "different scene instance" instruction belongs at the end of the whole
    block, not between two records.
    """
    lines = [
        "=" * 70,
        f"{label} - seed {record.seed} ({_verdict(record)})",
        (
            f"task_completed={record.task_completed}, reward={record.reward:.3f}, "
            f"sandbox_rc={record.sandbox_rc}, blocks={record.num_code_blocks}, "
            f"regenerations={record.num_regenerations}"
        ),
        "-" * 70,
        "PROGRAM:",
        "```python",
        code,
        "```",
    ]
    if stdout:
        lines += ["-" * 70, "STDOUT (tail):", stdout]
    if stderr:
        lines += ["-" * 70, "STDERR (tail):", stderr]
    lines += _detail_lines(record.check_detail)
    lines.append("=" * 70)
    if closing:
        lines.append(CLOSING_LINE)
    return "\n".join(lines).rstrip() + "\n"


def format_trace(
    record: TraceRecord,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    label: str = "PREVIOUS ATTEMPT",
    closing: bool = True,
) -> str:
    """The block injected into the next seed's prompt, bounded by *max_chars*.

    The program is kept first; output streams keep their tails.  Truncation is
    deterministic and marked, so two runs with the same record produce the same
    prompt.  The closing instruction is part of the fixed overhead and is never
    the part that gets cut.

    ``label`` and ``closing`` let :func:`format_dual_trace` stack two records:
    only the last one carries the closing instruction.
    """
    max_chars = max(2000, int(max_chars))
    code = record.code.strip() or "(no program was produced)"
    stdout = record.stdout or ""
    stderr = record.stderr or ""

    # Header, labels and the closing instruction are not negotiable; what is
    # left is shared by the program and the two streams.
    overhead = len(_render(record, "", "", "", label=label, closing=closing))
    budget = max(600, max_chars - overhead)

    has_stdout = bool(stdout)
    has_stderr = bool(stderr)
    if has_stdout and has_stderr:
        code_share, stdout_share, stderr_share = 0.60, 0.20, 0.20
    elif has_stdout or has_stderr:
        code_share, stdout_share, stderr_share = 0.70, 0.30, 0.30
    else:
        code_share, stdout_share, stderr_share = 1.0, 0.0, 0.0

    code_budget = max(400, int(budget * code_share))
    stdout_budget = max(200, int(budget * stdout_share)) if has_stdout else 0
    stderr_budget = max(200, int(budget * stderr_share)) if has_stderr else 0

    stdout_text = _tail(stdout, stdout_budget) if has_stdout else ""
    stderr_text = _tail(stderr, stderr_budget) if has_stderr else ""
    block = _render(
        record,
        _middle(code, code_budget),
        stdout_text,
        stderr_text,
        label=label,
        closing=closing,
    )

    excess = len(block) - max_chars
    if excess > 0:
        # Marker text and the per-stream floors can push the first estimate
        # over; take the difference out of the program, not the instruction.
        code_budget = max(200, code_budget - excess - 40)
        block = _render(
            record,
            _middle(code, code_budget),
            stdout_text,
            stderr_text,
            label=label,
            closing=closing,
        )

    if len(block) > max_chars:
        # The floors alone exceed a very small cap.  Keep the framing and let
        # the bound win; at the default cap this path is unreachable.
        block = block[: max_chars - 40] + "\n... [trace truncated] ...\n"
    return block


def load_best_trace(
    chain_dir: Path | str,
    seed: int,
) -> TraceRecord | None:
    """The best completed success below *seed*, latest on ties.

    "Best" is the highest reward among records whose ``task_completed`` is
    True; an incomplete (timed-out) attempt is never eligible.  None means the
    chain has not succeeded yet, and the caller renders the last record alone.
    """
    chain_dir = Path(chain_dir)
    best: TraceRecord | None = None
    for earlier in range(int(seed) - 1, 0, -1):
        path = chain_dir / TRACE_FILE_TEMPLATE.format(seed=earlier)
        if not path.exists():
            continue
        try:
            record = TraceRecord.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            continue
        if not record.complete or record.task_completed is not True:
            continue
        if best is None or (record.reward, record.seed) > (best.reward, best.seed):
            best = record
    return best


def format_dual_trace(
    last: TraceRecord,
    best: TraceRecord | None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """The last attempt and the best success so far, in one block.

    ``trace_mode=last`` overwrites the only working program with the next
    seed's failure.  This keeps a second slot: the most recent attempt (which
    carries the failure detail) and the best completed success (which carries
    the procedure that worked).  With no success yet -- or when the two slots
    are the same record -- the block is exactly ``format_trace(last)``.
    """
    max_chars = max(4000, int(max_chars))
    if best is None or best.seed == last.seed:
        return format_trace(last, max_chars)
    best_budget = max(2000, int(max_chars * 0.40))
    last_budget = max(2000, max_chars - best_budget)
    last_block = format_trace(last, last_budget, closing=False)
    best_block = format_trace(best, best_budget, label="BEST SUCCESS SO FAR")
    block = last_block + best_block
    if len(block) > max_chars:
        block = block[: max_chars - 40] + "\n... [trace truncated] ...\n"
    return block


CARD_LABEL = "EXPERIENCE CARD"

REFLECTION_SYSTEM = (
    "You review one robot-control attempt and write a compact experience card "
    "for the next attempt."
)

REFLECTION_INSTRUCTIONS = """\
The next attempt runs the same task on a freshly randomized scene instance and
will see only this card -- not the program, and not the output below.  Write
four lines, each starting with exactly one of these prefixes:

DID: the commands the program ran, in order, naming exact API calls and values.
MEASURED: the readings the program acted on, with units.
WHY IT FAILED: why the attempt ended this way, taken from the check detail or
the exception.  If the attempt succeeded, use WHY IT PASSED: instead and name
the steps the output shows completed.
NEXT TIME: name one step to keep.  If the attempt failed, name one concrete
step to change.  If it succeeded, keep the program and change only what the
new scene forces; never remove a step that completed the task.

Rules:
- Use only what the attempt below shows; never invent a number.
- Never state a class label, an object pose, or any answer the attempt was not
  given.  If the attempt does not show it, write "unknown".
- task_completed=True is the verdict: the attempt succeeded even when stderr
  shows a traceback after the task had already ended.  That error is expected;
  do not advise removing the action that completed the task to avoid it.
- One short sentence per line, under 220 characters.
"""


def build_reflection_prompt(
    record: TraceRecord,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Messages for the reflection call that writes *record*'s card.

    The body is the same filtered trace the next seed would receive in ``last``
    mode, so the card cannot carry anything the raw trace could not.  Content
    is a text-parts list, the same shape the task prompt uses, so the
    Responses-API conversion treats this prompt like every other one.
    """
    body = format_trace(
        record,
        max_chars=max_chars,
        label="ATTEMPT UNDER REVIEW",
        closing=False,
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": REFLECTION_SYSTEM}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": REFLECTION_INSTRUCTIONS + "\n" + body,
                }
            ],
        },
    ]


def parse_card(text: str, max_chars: int = 2000) -> str:
    """Trim a reflection reply to the card text.

    Models sometimes wrap a non-code answer in a fenced block anyway; strip
    one pair of fences, then bound the result.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned[: int(max_chars)]


def format_card_trace(
    record: TraceRecord,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    label: str = CARD_LABEL,
    closing: bool = True,
) -> str:
    """One attempt as its experience card, plus the raw check detail.

    The card is the model's compression of the attempt; the check detail is
    appended verbatim because it is the measurement the card summarizes and
    the only place exact numbers survive compression.  A record whose card was
    never written (a timed-out seed, or a record from before card mode) falls
    back to the raw trace, marked as such, so the chain never goes blind.
    """
    text = (record.card or "").strip()
    if not text:
        return format_trace(
            record,
            max_chars,
            label=f"{label} UNAVAILABLE - RAW TRACE",
            closing=closing,
        )
    lines = [
        "=" * 70,
        f"{label} - seed {record.seed} ({_verdict(record)})",
        (
            f"task_completed={record.task_completed}, reward={record.reward:.3f}, "
            f"blocks={record.num_code_blocks}, "
            f"regenerations={record.num_regenerations}"
        ),
        "-" * 70,
        text,
    ]
    lines += _detail_lines(record.check_detail)
    lines.append("=" * 70)
    if closing:
        lines.append(CLOSING_LINE)
    block = "\n".join(lines).rstrip() + "\n"
    max_chars = max(1200, int(max_chars))
    if len(block) > max_chars:
        block = block[: max_chars - 40] + "\n... [card truncated] ...\n"
    return block


def format_cards(
    last: TraceRecord,
    best: TraceRecord | None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Cards for the last attempt and the best success, in one block."""
    # Two slots, and either may have to fall back to a raw trace (whose own
    # floor is 2000 chars); keep the combined cap above both floors.
    max_chars = max(4000, int(max_chars))
    if best is None or best.seed == last.seed:
        return format_card_trace(last, max_chars)
    half = max(2000, max_chars // 2)
    first = format_card_trace(last, half, closing=False)
    second = format_card_trace(
        best,
        half,
        label="BEST SUCCESS SO FAR - EXPERIENCE CARD",
    )
    block = first + second
    if len(block) > max_chars:
        block = block[: max_chars - 40] + "\n... [card truncated] ...\n"
    return block


def _streak_line(records: list[TraceRecord]) -> str | None:
    """One line for a window whose attempts all ended the same way.

    This is the only thing a window adds over ``dual``: two records can show
    what just happened and what worked, but only a run of records can say the
    same ending keeps repeating.  Everything else about the window is more
    copies of the same evidence.
    """
    if len(records) < 2:
        return None
    verdicts = {_verdict(record) for record in records}
    if verdicts == {"SUCCEEDED"}:
        return f"STREAK: the last {len(records)} attempts all succeeded."
    reasons: list[str] = []
    for record in records:
        detail = record.check_detail or {}
        reason = detail.get("failure") or detail.get("terminal_reason")
        if not isinstance(reason, str) or not reason:
            return None
        reasons.append(reason)
    if len(set(reasons)) == 1:
        return (
            f"STREAK: the last {len(records)} attempts all ended with "
            f"{reasons[0]!r}."
        )
    return None


def format_window_cards(
    records: list[TraceRecord],
    best: TraceRecord | None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Cards for the last few attempts, plus the best success if it is older.

    ``last``/``dual`` answer "what just happened" and "what worked"; this
    answers "what has been happening".  Only the streak line is new evidence;
    the rest is the same cards, so the window cannot flood the prompt the way
    a window of raw programs would.

    Each slot may fall back to a raw trace (2000-character floor), so the
    oldest records are dropped until the cap can hold every slot.
    """
    if not records:
        return ""
    max_chars = max(4000, int(max_chars))
    window = list(records)
    best_seed = best.seed if best is not None else None
    separate_best = best is not None and all(
        record.seed != best_seed for record in window
    )
    slots = len(window) + int(separate_best)
    while slots > 1 and max_chars < 2000 * slots:
        window.pop()
        slots = len(window) + int(separate_best)
    per_slot = max(2000, max_chars // slots)

    labeled: list[tuple[TraceRecord, str]] = [
        (record, "PREVIOUS ATTEMPT" if index == 0 else "EARLIER ATTEMPT")
        for index, record in enumerate(window)
    ]
    if separate_best:
        labeled.append((best, "BEST SUCCESS SO FAR"))

    parts = [
        format_card_trace(
            record,
            per_slot,
            label=label,
            closing=(index == len(labeled) - 1),
        )
        for index, (record, label) in enumerate(labeled)
    ]
    streak = _streak_line(window)
    block = (streak + "\n" if streak else "") + "".join(parts)
    if len(block) > max_chars:
        block = block[: max_chars - 40] + "\n... [card truncated] ...\n"
    return block


def write_trace(chain_dir: Path | str, record: TraceRecord) -> Path:
    """Atomically write one seed's record into *chain_dir*.

    A partial write after a code block is replaced by the final one; because
    the replace is atomic, the next seed never reads half a JSON document.
    """
    chain_dir = Path(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    path = chain_dir / TRACE_FILE_TEMPLATE.format(seed=record.seed)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def load_previous_trace(
    chain_dir: Path | str,
    seed: int,
) -> TraceRecord | None:
    """The most recent record for a seed below *seed*, or None.

    Falling back past a missing seed (one killed before it wrote anything, for
    example) keeps the chain alive; the record names which seed was actually
    used, and that name travels into the next record.
    """
    chain_dir = Path(chain_dir)
    for earlier in range(int(seed) - 1, 0, -1):
        path = chain_dir / TRACE_FILE_TEMPLATE.format(seed=earlier)
        if not path.exists():
            continue
        try:
            return TraceRecord.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError) as exc:
            print(f"[trace] skipping unreadable {path}: {exc}")
    return None


def load_recent_traces(
    chain_dir: Path | str,
    seed: int,
    limit: int = 3,
) -> list[TraceRecord]:
    """The most recent *limit* records below *seed*, newest first.

    A missing seed is skipped rather than ending the window, so one killed
    attempt does not shorten the evidence the next seed receives.
    """
    chain_dir = Path(chain_dir)
    limit = max(1, int(limit))
    records: list[TraceRecord] = []
    for earlier in range(int(seed) - 1, 0, -1):
        path = chain_dir / TRACE_FILE_TEMPLATE.format(seed=earlier)
        if not path.exists():
            continue
        try:
            records.append(TraceRecord.from_dict(json.loads(path.read_text())))
        except (OSError, ValueError, TypeError) as exc:
            print(f"[trace] skipping unreadable {path}: {exc}")
            continue
        if len(records) >= limit:
            break
    return records


def write_injected_block(
    chain_dir: Path | str,
    seed: int,
    block: str,
) -> Path:
    """Keep the exact text appended to *seed*'s initial prompt."""
    chain_dir = Path(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    path = chain_dir / INJECTED_FILE_TEMPLATE.format(seed=seed)
    path.write_text(block)
    return path
