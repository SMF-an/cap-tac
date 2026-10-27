"""The card bank: tactile situations that earned a line, retrieved by key.

A card is what a skill becomes when it is allowed to say *when* it applies.
The old library carried one paragraph per idea and delivered it by task name,
so every task got everything routed to it and the model had to decide which
of fifty paragraphs were about the reading in front of it.  A card instead
declares a partial key -- two or three dimensions of
:mod:`capx.tactile_memory.signature` -- and only surfaces when the live
reading matches it.

What a card is not: a reward, a success label, or an answer.  Cards are
written from what the gel reported, never from the simulator's verdict, so
the same bank can be handed to a task whose ground truth it has no access
to.  Text that names ``check_success``, a reward or a ground-truth field is
rejected at load, not filtered later.

Most cards start as ``hypothesis``: a pattern someone saw often enough to
write down, not one that was measured against an outcome.  ``observed`` is
reserved for cards whose evidence line names trials that were actually run.
Both are delivered -- a hypothesis that says so is still cheaper than
rediscovering the situation from scratch -- but the status travels with the
text so the model can weigh it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from capx.tactile_memory import signature

SCHEMA = "tactile_keys.v1"

#: The status a card may claim.
STATUSES = ("observed", "hypothesis")

#: A card that names any of these is reading the answer sheet.
_FORBIDDEN = (
    "check_success", "ground_truth", "ground truth", "reward", "oracle",
)

_MAX_DO = 240
_ID = re.compile(r"[a-z][a-z0-9-]{2,63}")


class CardError(ValueError):
    """A card, or the file holding it, is not usable as written."""


@dataclass(frozen=True)
class Card:
    """One tactile situation and the line that belongs with it.

    Attributes:
        id: Kebab-case identity, unique in the bank.
        when: Partial key -- dimension to required band.  Every named
            dimension must match the live reading.
        do: The line delivered to the model when the card fires.  Written
            action-first: what to do with this reading, then the number that
            says why.
        tasks: Task names this card is scoped to; empty means any task.
        why: Optional provenance prose for a human reader.  Not delivered.
        evidence: Where the pattern was seen -- trial ids or run names.
        status: ``observed`` when an outcome was measured, ``hypothesis``
            when the pattern was only seen.
    """

    id: str
    when: dict[str, str]
    do: str
    tasks: tuple[str, ...] = ()
    why: str = ""
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status: str = "hypothesis"

    @property
    def specificity(self) -> int:
        """How many dimensions this card pins down."""
        return len(self.when)

    def render(self) -> str:
        """One line for the model: what to do, and the shape that fired it."""
        return f"- {self.do} ({signature.describe(self.when)})"


def _card_from(raw: Mapping, where: str) -> Card:
    if not isinstance(raw, Mapping):
        raise CardError(f"{where}: a card must be a mapping")
    known = {"id", "when", "do", "tasks", "why", "evidence", "status"}
    unknown = set(raw) - known
    if unknown:
        raise CardError(f"{where}: unknown card fields {sorted(unknown)}")

    identifier = raw.get("id")
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise CardError(f"{where}: invalid id {identifier!r}")

    when = raw.get("when")
    if not isinstance(when, Mapping) or not when:
        raise CardError(f"{where}/{identifier}: 'when' must name at least one dimension")
    conditions: dict[str, str] = {}
    for dimension, band in when.items():
        if dimension not in signature.BANDS:
            raise CardError(f"{where}/{identifier}: unknown dimension {dimension!r}")
        if band not in signature.BANDS[dimension]:
            raise CardError(
                f"{where}/{identifier}: {dimension}={band!r} is not one of "
                f"{signature.BANDS[dimension]}")
        conditions[str(dimension)] = str(band)

    do = raw.get("do")
    if not isinstance(do, str) or not 1 <= len(do) <= _MAX_DO:
        raise CardError(f"{where}/{identifier}: 'do' must be 1..{_MAX_DO} chars")

    tasks = raw.get("tasks") or ()
    if isinstance(tasks, str) or not isinstance(tasks, Iterable):
        raise CardError(f"{where}/{identifier}: 'tasks' must be a list")
    tasks = tuple(str(task) for task in tasks)

    why = raw.get("why") or ""
    if not isinstance(why, str):
        raise CardError(f"{where}/{identifier}: 'why' must be a string")

    evidence = raw.get("evidence") or ()
    if isinstance(evidence, str) or not isinstance(evidence, Iterable):
        raise CardError(f"{where}/{identifier}: 'evidence' must be a list")
    evidence = tuple(str(item) for item in evidence)

    status = raw.get("status", "hypothesis")
    if status not in STATUSES:
        raise CardError(f"{where}/{identifier}: status {status!r} is not one of {STATUSES}")

    blob = " ".join((identifier, do, why)).lower()
    if any(marker in blob for marker in _FORBIDDEN):
        raise CardError(f"{where}/{identifier}: names the answer sheet")

    return Card(id=identifier, when=conditions, do=do, tasks=tasks,
                why=why, evidence=evidence, status=status)


@dataclass(frozen=True)
class CardBank:
    """Every card that can fire, and the ordering they fire in."""

    cards: tuple[Card, ...] = ()

    def __len__(self) -> int:
        return len(self.cards)

    def scope(self, task: str) -> list[Card]:
        """Cards that may fire on this task, in any key."""
        return [card for card in self.cards if not card.tasks or task in card.tasks]

    def recall(self, key: Mapping[str, str], task: str = "",
               limit: int = 4) -> list[Card]:
        """The cards whose condition this reading satisfies.

        Args:
            key: A full key from :func:`signature.key_from_readings`.
            task: The task name the reading was taken on; cards scoped to
                other tasks are not candidates.
            limit: Most cards to return.  The specific ones win: a card that
                names three dimensions is a statement about this reading,
                while a card that names one is a statement about the band.

        Returns:
            Matching cards, most specific first, ties broken by id so the
            order is the same on every run.
        """
        matched = [card for card in self.scope(task)
                   if signature.matches(key, card.when)]
        matched.sort(key=lambda card: (-card.specificity, card.id))
        return matched[:limit] if limit else matched

    def render(self, cards: Iterable[Card]) -> str:
        """The block delivered with a reading, or "" when nothing matched."""
        lines = [card.render() for card in cards]
        if not lines:
            return ""
        return "TACTILE MATCHES (this reading):\n" + "\n".join(lines)


def load_bank(path: str | Path) -> CardBank:
    """Read and validate one bank file.

    Args:
        path: YAML file with ``schema: tactile_keys.v1`` and a ``cards`` list.

    Returns:
        The validated bank.  An absent file is an empty bank, so the branch
        runs -- with no retrieval -- before anything has been mined.

    Raises:
        CardError: The file exists and any part of it is unusable.
    """
    path = Path(path)
    if not path.exists():
        return CardBank()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise CardError(f"{path}: schema must be {SCHEMA}")
    unknown = set(data) - {"schema", "cards"}
    if unknown:
        raise CardError(f"{path}: unknown top-level fields {sorted(unknown)}")
    raw_cards = data.get("cards") or []
    if not isinstance(raw_cards, list):
        raise CardError(f"{path}: 'cards' must be a list")
    cards = [_card_from(raw, str(path)) for raw in raw_cards]
    seen: set[str] = set()
    for card in cards:
        if card.id in seen:
            raise CardError(f"{path}: duplicate card id {card.id!r}")
        seen.add(card.id)
    return CardBank(tuple(cards))
