"""The one place the library speaks mid-trial: get_tactile_data.

A card is retrieved by a reading, so the answer to "when does the model hear
about it" is "when it looks at the gel".  These tests pin that: the block
appears under the scalars, the reading itself is unchanged, and the shipped
(empty) bank prints nothing at all.
"""

from __future__ import annotations

from unittest import mock

import pytest

from capx.integrations.tactile.univtac_api import UniVTACTactileApi
from capx.tactile_memory import recall


SEATED = {
    "left_contact": True, "right_contact": True,
    "left_depth_mm": 28.0, "right_depth_mm": 28.1,
    "left_marker_displacement_px": 22.8, "right_marker_displacement_px": 1.6,
    "left_marker_coherence": 0.09, "right_marker_coherence": 0.62,
    "left_marker_shear_dx": -0.2, "left_marker_shear_dy": 0.0,
    "right_marker_shear_dx": 0.2, "right_marker_shear_dy": 0.0,
    "left_marker_anisotropy": 1.02, "right_marker_anisotropy": 1.01,
}

BANK = """\
schema: tactile_keys.v1
cards:
  - id: dead-pad-after-close
    when: {disp_gap: split, coh_gap: split}
    do: Read the pad that is still coherent; the quiet one describes the drag.
"""


class _Env:
    task_name = "blind_pick_balls"


def _client(env=None) -> UniVTACTactileApi:
    return UniVTACTactileApi(env if env is not None else object(),
                             server_url="http://test:8120")


def _get(api, readings: dict, capsys):
    response = mock.Mock()
    response.json.return_value = dict(readings)
    response.raise_for_status.return_value = None
    with mock.patch("capx.integrations.tactile.univtac_api.requests.get",
                    return_value=response):
        data = api.get_tactile_data()
    return data, capsys.readouterr().out


@pytest.fixture
def mined_bank(tmp_path, monkeypatch):
    path = tmp_path / "bank.yaml"
    path.write_text(BANK, encoding="utf-8")
    monkeypatch.setenv(recall.ENV_VAR, str(path))
    return path


def test_a_matching_card_is_printed_under_the_reading(mined_bank, capsys):
    data, out = _get(_client(_Env()), SEATED, capsys)
    assert "TACTILE MATCHES" in out
    assert "the quiet one describes the drag" in out
    # The block is output, not return: the reading's shape is unchanged.
    assert not any("MATCH" in key for key in data)


def test_the_reading_is_printed_before_the_cards(mined_bank, capsys):
    _data, out = _get(_client(_Env()), SEATED, capsys)
    assert out.index("Tactile:") < out.index("TACTILE MATCHES")


def test_a_reading_that_matches_nothing_prints_no_block(mined_bank, capsys):
    quiet = {**SEATED, "left_marker_displacement_px": 2.0,
             "left_marker_coherence": 0.55}
    _data, out = _get(_client(_Env()), quiet, capsys)
    assert "TACTILE MATCHES" not in out


def test_the_shipped_bank_is_silent(monkeypatch, capsys):
    monkeypatch.delenv(recall.ENV_VAR, raising=False)
    _data, out = _get(_client(_Env()), SEATED, capsys)
    assert "TACTILE MATCHES" not in out


def test_a_task_scoped_card_needs_the_task_name(mined_bank, capsys):
    scoped = BANK.replace("    when:", "    tasks: [blind_pick_balls]\n    when:")
    mined_bank.write_text(scoped, encoding="utf-8")
    # The env the client is built with is where the name comes from.
    _data, out = _get(_client(object()), SEATED, capsys)
    assert "TACTILE MATCHES" not in out


def test_the_api_reads_the_task_name_off_its_env():
    assert _client(_Env())._task_name() == "blind_pick_balls"
    assert _client(object())._task_name() == ""


# ---------------------------------------------------------------------------
# The reading before this one
# ---------------------------------------------------------------------------

TREND_BANK = """\
schema: tactile_keys.v1
cards:
  - id: surface-backing-away
    when: {depth_trend: rising}
    do: The surface is backing away; the pads are past the object.
"""


def _reads(api, readings: list[dict], capsys):
    response = mock.Mock()
    response.json.side_effect = [dict(reading) for reading in readings]
    response.raise_for_status.return_value = None
    with mock.patch("capx.integrations.tactile.univtac_api.requests.get",
                    return_value=response):
        data = [api.get_tactile_data() for _ in readings]
    return data, capsys.readouterr().out


def test_a_card_can_be_about_a_change(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bank.yaml"
    path.write_text(TREND_BANK, encoding="utf-8")
    monkeypatch.setenv(recall.ENV_VAR, str(path))
    api = _client(_Env())
    deep = {**SEATED, "left_depth_mm": 27.0, "right_depth_mm": 27.0}
    out_of = {**SEATED, "left_depth_mm": 28.6, "right_depth_mm": 28.6}
    _data, out = _reads(api, [deep, out_of], capsys)
    # The first reading has nothing to compare against, so it claims no
    # trend and fires nothing; only the second one knows the pad moved out.
    assert out.count("TACTILE MATCHES") == 1
    assert "backing away" in out


def test_a_motions_tactile_payload_becomes_the_previous_reading():
    api = _client(_Env())
    api._remember({"success": True, "tactile": {**SEATED, "left_depth_mm": 27.0}})
    assert api._last_reading["left_depth_mm"] == 27.0


def test_a_payload_without_tactile_leaves_the_previous_reading_alone():
    api = _client(_Env())
    api._remember({"success": True, "position": [0.4, 0.1, 0.2]})
    assert api._last_reading is None
