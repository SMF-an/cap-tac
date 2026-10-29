"""The resident half of the tactile library: principles and task lines.

Two kinds of text live here, and they are separated because they fail
differently.

A *principle* holds for every task that can read the gel.  It is the kind of
thing the API docstring cannot say, because the docstring describes what a
field is and a principle describes what it is not -- that a single reading is
not a result, that a dead pad does not read as dead.  These are delivered
every time, and the budget is six: a seventh would make the block a document
the model skims.

A *task line* is true on one task and cheap to carry.  It earns residency
only when missing it costs an entire run -- not "it would help to know", but
"a run was lost to not knowing".

Everything narrower belongs in the card bank
(:mod:`capx.tactile_memory.bank`), which is retrieved by the live reading and
therefore only costs words when it applies.  That split is the whole design:
the resident text is what a careful engineer would say before touching the
robot, and the bank is what they would say while looking at the gel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from capx.tactile_memory.tasks import TASK_TYPES

SCHEMA = "tactile_core.v1"

CORE_PATH = Path(__file__).with_name("core.yaml")


class CoreError(ValueError):
    """The core block is not usable as written."""


@dataclass(frozen=True)
class Core:
    """Everything delivered before the robot moves.

    Attributes:
        principles: ``(id, text)`` pairs, in delivery order.
        task_lines: Task name to the lines that only that task carries.
    """

    principles: tuple[tuple[str, str], ...] = ()
    task_lines: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def render(self, task: str = "") -> str:
        """The block for one task, or "" when there is nothing to say.

        Args:
            task: The simulator's task name.  An unknown or empty name still
                gets the principles -- they are about the sensor, not the
                task -- but no task lines.

        Returns:
            The prompt block, ending without a trailing newline.
        """
        lines = [f"- {text}" for _id, text in self.principles]
        lines += [f"- {line}" for line in self.task_lines.get(task, ())]
        if not lines:
            return ""
        return "TACTILE READING:\n" + "\n".join(lines)


def _text(raw: object, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CoreError(f"{where}: text must be a non-empty string")
    return " ".join(raw.split())


def load_core(path: str | Path | None = None) -> Core:
    """Read and validate the core block.

    Args:
        path: YAML file with ``schema: tactile_core.v1``.  Defaults to the
            ``core.yaml`` shipped beside this module.

    Returns:
        The validated block.  An absent file is an empty block, so a task
        with no core text still runs.

    Raises:
        CoreError: The file exists and any part of it is unusable.
    """
    path = Path(path) if path is not None else CORE_PATH
    if not path.exists():
        return Core()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise CoreError(f"{path}: schema must be {SCHEMA}")
    unknown = set(data) - {"schema", "principles", "task_lines"}
    if unknown:
        raise CoreError(f"{path}: unknown top-level fields {sorted(unknown)}")

    principles: list[tuple[str, str]] = []
    for index, raw in enumerate(data.get("principles") or []):
        if not isinstance(raw, Mapping) or set(raw) != {"id", "text"}:
            raise CoreError(f"{path}: principle {index} must be id/text")
        principles.append((str(raw["id"]), _text(raw["text"], str(path))))
    ids = [identifier for identifier, _text_ in principles]
    if len(set(ids)) != len(ids):
        raise CoreError(f"{path}: duplicate principle id")

    task_lines: dict[str, tuple[str, ...]] = {}
    for task, lines in (data.get("task_lines") or {}).items():
        if task not in TASK_TYPES:
            raise CoreError(f"{path}: task line for unknown task {task!r}")
        if isinstance(lines, str) or not isinstance(lines, list):
            raise CoreError(f"{path}: task lines for {task!r} must be a list")
        task_lines[str(task)] = tuple(
            _text(line, f"{path}/{task}") for line in lines)
    return Core(tuple(principles), task_lines)


@lru_cache(maxsize=1)
def default_core() -> Core:
    """The shipped core block, loaded once."""
    return load_core()
