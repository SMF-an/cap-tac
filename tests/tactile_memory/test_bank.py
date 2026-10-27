"""Tests for the card bank: what it accepts, and what it retrieves.

A card is text an LLM will act on, so the schema refuses the things that
would make it unactionable or unfair: a condition naming a dimension that
does not exist, a band outside that dimension's range, an id that repeats,
or a line that reads the answer sheet.  Retrieval is then a plain filter,
and the tests pin its ordering -- most specific first, ties by id -- because
a bank that returns a different order on every call is a bank whose A/B
results cannot be compared.
"""

from __future__ import annotations

import pytest

from capx.tactile_memory.bank import CardError, load_bank
from capx.tactile_memory.signature import key_from_readings


def card_yaml(*cards: str, schema: str = "tactile_keys.v1") -> str:
    return f"schema: {schema}\ncards:\n" + "".join(cards)


DEAD_PAD = """\
  - id: dead-pad-after-close
    tasks: [blind_pick_balls]
    when: {disp_gap: split, coh_gap: split}
    do: Read the pad that is still coherent; the quiet one describes the drag.
    evidence: [blind_pick_balls/seed_3/trial_12]
    status: observed
"""

ANY_MOVING = """\
  - id: any-moving
    when: {disp: moving}
    do: The gel is being dragged, so compare two readings before believing a depth.
"""


def seated_key() -> dict:
    return key_from_readings({
        "left_contact": True, "right_contact": True,
        "left_depth_mm": 27.9, "right_depth_mm": 27.9,
        "left_marker_displacement_px": 2.4, "right_marker_displacement_px": 2.2,
        "left_marker_coherence": 0.5, "right_marker_coherence": 0.55,
    })


def write(tmp_path, text: str):
    path = tmp_path / "bank.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_bank_is_empty_and_that_is_supported():
    from capx.tactile_memory.recall import BANK_PATH

    bank = load_bank(BANK_PATH)
    assert len(bank) == 0
    assert bank.recall(seated_key(), task="blind_pick_balls") == []
    assert bank.render([]) == ""


def test_an_absent_bank_file_loads_as_empty(tmp_path):
    assert len(load_bank(tmp_path / "nope.yaml")) == 0


def test_recall_returns_the_matching_cards_most_specific_first(tmp_path):
    bank = load_bank(write(tmp_path, card_yaml(ANY_MOVING, DEAD_PAD)))
    key = dict(seated_key(), disp_gap="split", coh_gap="split")
    cards = bank.recall(key, task="blind_pick_balls")
    assert [card.id for card in cards] == ["dead-pad-after-close", "any-moving"]


def test_a_card_scoped_to_another_task_does_not_fire(tmp_path):
    bank = load_bank(write(tmp_path, card_yaml(DEAD_PAD)))
    key = dict(seated_key(), disp_gap="split", coh_gap="split")
    assert bank.recall(key, task="lift_can") == []
    assert bank.recall(key, task="blind_pick_balls")


def test_an_unscoped_card_fires_on_any_task(tmp_path):
    bank = load_bank(write(tmp_path, card_yaml(ANY_MOVING)))
    assert bank.recall(seated_key(), task="wipe_vase")
    assert bank.recall(seated_key(), task="")


def test_the_limit_keeps_the_specific_cards(tmp_path):
    bank = load_bank(write(tmp_path, card_yaml(ANY_MOVING, DEAD_PAD)))
    key = dict(seated_key(), disp_gap="split", coh_gap="split")
    assert [card.id for card in bank.recall(key, task="blind_pick_balls", limit=1)] \
        == ["dead-pad-after-close"]


def test_render_is_one_line_per_card_and_names_the_condition(tmp_path):
    bank = load_bank(write(tmp_path, card_yaml(ANY_MOVING)))
    text = bank.render(bank.recall(seated_key(), task=""))
    assert text.startswith("TACTILE MATCHES")
    assert "(disp=moving)" in text
    assert text.count("\n") == 1


@pytest.mark.parametrize("body,message", [
    # A dimension nobody computes would silently never fire.
    ("""\
  - id: bad-dimension
    when: {weight: heavy}
    do: x
""", "unknown dimension"),
    # A band outside its dimension is the same failure one level down.
    ("""\
  - id: bad-band
    when: {depth: deep}
    do: x
""", "is not one of"),
    # The whole point of the bank is that it is not reading the answer sheet.
    ("""\
  - id: answer-sheet
    when: {depth: face}
    do: Call check_success to see whether the grasp worked.
""", "answer sheet"),
    ("""\
  - id: no-condition
    when: {}
    do: x
""", "at least one dimension"),
    ("""\
  - id: Bad_ID
    when: {depth: face}
    do: x
""", "invalid id"),
    ("""\
  - id: wrong-status
    when: {depth: face}
    do: x
    status: proven
""", "status"),
    ("""\
  - id: extra-field
    when: {depth: face}
    do: x
    priority: 2
""", "unknown card fields"),
])
def test_a_card_that_could_not_be_acted_on_fails_at_load(tmp_path, body, message):
    with pytest.raises(CardError, match=message):
        load_bank(write(tmp_path, card_yaml(body)))


def test_a_duplicate_id_fails_at_load(tmp_path):
    with pytest.raises(CardError, match="duplicate"):
        load_bank(write(tmp_path, card_yaml(ANY_MOVING, ANY_MOVING)))


def test_the_schema_line_is_enforced(tmp_path):
    with pytest.raises(CardError, match="schema"):
        load_bank(write(tmp_path, card_yaml(ANY_MOVING, schema="something_else.v9")))
