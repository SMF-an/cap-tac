"""Glue between a live reading and the cards that belong with it.

The retrieval is deliberately small: key the reading, ask the bank, render
the lines.  What it must not grow is a second source of truth -- no cached
"current situation" that drifts from the last reading, no scoring model that
has to be tuned, no state carried between calls.  A recall answers the
question "given this gel reading, which of the things we already know apply"
and answers it the same way every time the same reading arrives.

The bank is a file, and which file is an experiment parameter: setting
``CAPX_TACTILE_BANK`` points the process at a different bank, which is what
lets an A/B run hold the code fixed and vary only the mined knowledge.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from capx.tactile_memory import signature
from capx.tactile_memory.bank import Card, CardBank, load_bank

ENV_VAR = "CAPX_TACTILE_BANK"
BANK_PATH = Path(__file__).with_name("bank.yaml")

#: Most cards a single reading may pull in.  Four is about as many lines as a
#: model will act on at one decision point; past that the last ones are read
#: as boilerplate.
DEFAULT_LIMIT = 4


@dataclass(frozen=True)
class Match:
    """What one reading matched.

    Attributes:
        key: The full key, for a log line or a trace row -- never for the
            model, which needs the situation, not the table.
        cards: The cards that fired, most specific first.
        text: The rendered block, empty when nothing fired.
    """

    key: dict[str, str] = field(default_factory=dict)
    cards: tuple[Card, ...] = ()
    text: str = ""


def bank_path() -> Path:
    """Which bank file this process reads."""
    raw = os.environ.get(ENV_VAR, "").strip()
    return Path(raw).expanduser() if raw else BANK_PATH


@lru_cache(maxsize=4)
def _bank_at(path: Path) -> CardBank:
    return load_bank(path)


def current_bank() -> CardBank:
    """The bank for this process, reloaded only when the path changes."""
    return _bank_at(bank_path())


def match(readings: Mapping[str, Any], task: str = "",
          limit: int = DEFAULT_LIMIT,
          previous: Mapping[str, Any] | None = None) -> Match:
    """Key one tactile reading and select the cards it fires.

    Args:
        readings: The mapping ``get_tactile_data`` returns, before any
            images are stripped (both paths handle them; they are ignored).
        task: The simulator's task name, for scoping.  Empty means only
            cross-task cards can fire.
        limit: Most cards to return; 0 returns every match.
        previous: The last reading taken before this one, for the
            ``*_trend`` dimensions.  None leaves them ``unknown``; the
            caller decides what "before" means, because a reading from the
            previous episode is not a previous reading.

    Returns:
        A Match.  Never raises on a reading that is missing fields -- an
        incomplete reading keys to ``missing`` bands and matches only cards
        that asked for them.
    """
    key = signature.key_from_readings(readings, previous=previous)
    bank = current_bank()
    cards = bank.recall(key, task=task, limit=limit)
    return Match(key=key, cards=tuple(cards), text=bank.render(cards))
