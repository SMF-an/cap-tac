"""Tactile memory: contact records, resident principles, and cards.

The pieces live in two layers, because they answer different questions.

Records -- what happened, reduced to something that outlives one row:

  * :mod:`capx.tactile_memory.events` -- a trace cut into bounded contacts,
    each carrying the change across the contact and the action that made it.
  * :mod:`capx.tactile_memory.signature` -- one reading reduced to bands, so
    a card can say *when* it applies without naming a number that only held
    on one seed.

Delivery -- what the model sees, and only when it applies:

  * :mod:`capx.tactile_memory.core` -- the few things that are always true
    about reading this sensor.  Delivered with every prompt.
  * :mod:`capx.tactile_memory.bank` -- the cards themselves, surfaced by
    :mod:`capx.tactile_memory.recall` at the moment a reading matches.

The old ``capx.tactile_skills`` delivered every paragraph routed to a task,
every time, and left the model to work out which ones described the reading
in front of it.  This library delivers the same kind of knowledge, but the
narrow part waits until the sensor says it applies.
"""

from capx.tactile_memory.bank import Card, CardBank, CardError, Support, load_bank
from capx.tactile_memory.core import Core, CoreError, default_core, load_core
from capx.tactile_memory.events import ContactEvent, PadDelta, segment_contact_events
from capx.tactile_memory.recall import Match, current_bank, match
from capx.tactile_memory.signature import BANDS, DIMENSIONS, key_from_readings
from capx.tactile_memory.tasks import TASK_TYPES

__all__ = [
    "BANDS",
    "Card",
    "CardBank",
    "CardError",
    "ContactEvent",
    "Core",
    "CoreError",
    "DIMENSIONS",
    "Match",
    "PadDelta",
    "Support",
    "TASK_TYPES",
    "current_bank",
    "default_core",
    "key_from_readings",
    "load_bank",
    "load_core",
    "match",
    "segment_contact_events",
]
