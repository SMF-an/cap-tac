"""Tests for the resident half: the core block and the prompt hook.

What the hook must get right is narrower than it used to be -- the block is
now the same principles for every task, plus the few lines that are true on
one -- but the two failure modes it was written against are unchanged:

  * delivering gel principles to a robot whose config swapped the tactile
    API for the motion-only one (the ablation configs keep the task name),
    and
  * resolving the task name to "" for every shipped config, because the
    configs hand over the *path* to the low-level YAML rather than a parsed
    dict.  The second one reached no trial at all while every stub-based
    test passed, so it is checked through the configs themselves.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.tactile_memory import core as core_module
from capx.tactile_memory.core import CoreError, default_core, load_core
from capx.tactile_memory.tasks import TASK_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "env_configs" / "univtac"


class _TactileApi:
    def functions(self) -> dict:
        return {"get_tactile_data": object()}


class _MotionOnlyApi:
    def functions(self) -> dict:
        return {"goto_pose": object()}


class _StubTask:
    """Just enough of an env to exercise the hook without building one."""

    _task_name = CodeExecutionEnvBase._task_name
    _reads_the_gel = CodeExecutionEnvBase._reads_the_gel
    _tactile_prompt_block = CodeExecutionEnvBase._tactile_prompt_block

    def __init__(self, task: str, **attrs) -> None:
        self.cfg = SimpleNamespace(low_level=SimpleNamespace(task_name=task))
        self._apis = {"api": _TactileApi()}
        for key, value in attrs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# The shipped block
# ---------------------------------------------------------------------------


def test_the_shipped_core_validates_and_carries_the_principles():
    block = default_core()
    assert len(block.principles) == 6
    assert block.principles[0][0] == "no-single-reading"


def test_a_task_gets_its_own_lines_on_top_of_the_principles():
    text = default_core().render("blind_pick_balls")
    assert text.startswith("TACTILE READING:")
    assert "A gripper close that raises is the ball seated" in text
    plain = default_core().render("lift_can")
    assert plain.startswith("TACTILE READING:")
    assert "ball seated" not in plain
    assert plain.count("\n") == 6  # the principles and nothing else


def test_an_unknown_task_still_gets_the_principles():
    # The principles are about the sensor, not the task, so a task outside
    # the routing table is not a reason to withhold them.
    assert default_core().render("some_task_we_have_never_seen") == \
        default_core().render("")


@pytest.mark.parametrize("body,message", [
    ("""\
schema: tactile_core.v1
principles:
  - {id: a, text: one}
  - {id: a, text: two}
""", "duplicate principle"),
    ("""\
schema: tactile_core.v1
task_lines:
  not_a_task: [something true]
""", "unknown task"),
    ("""\
schema: tactile_core.v1
principles:
  - {id: a, text: ""}
""", "non-empty"),
    ("""\
schema: wrong.v2
principles: []
""", "schema"),
])
def test_a_core_that_would_misroute_fails_at_load(tmp_path, body, message):
    path = tmp_path / "core.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(CoreError, match=message):
        load_core(path)


def test_an_absent_core_file_renders_nothing(tmp_path):
    assert load_core(tmp_path / "nope.yaml").render("lift_can") == ""


def test_the_shipped_core_is_the_one_this_module_loads():
    assert core_module.CORE_PATH.name == "core.yaml"
    assert core_module.CORE_PATH.exists()


# ---------------------------------------------------------------------------
# The prompt hook
# ---------------------------------------------------------------------------


def test_the_hook_delivers_the_block():
    block = _StubTask("blind_pick_balls")._tactile_prompt_block()
    assert block.startswith("TACTILE READING:")
    assert "ball seated" in block


def test_a_task_without_a_tactile_api_gets_nothing():
    task = _StubTask("blind_pick_balls", _apis={"api": _MotionOnlyApi()})
    assert task._tactile_prompt_block() == ""


def test_a_task_can_opt_out():
    assert _StubTask("blind_pick_balls", tactile_memory=False)._tactile_prompt_block() == ""


def test_an_empty_api_set_is_read_as_cannot_tell():
    # Delivering is the old behaviour and the safe one: a task that has no
    # recorded APIs is a test harness, not an ablation.
    assert _StubTask("lift_can", _apis={})._tactile_prompt_block()


# ---------------------------------------------------------------------------
# Routing through the real configs
# ---------------------------------------------------------------------------


def test_a_shipped_config_resolves_its_task_name() -> None:
    """Through the configs themselves, because the stub above is the one shape
    production never builds.

    A shipped config hands `cfg.low_level` over as the *path* to the
    low-level YAML, not as a parsed dict -- so an implementation that read
    `cfg.low_level.task_name` found nothing, the name resolved to "", and the
    block reached no trial at all while every stub-based test here passed.
    """
    from capx.envs.configs.instantiate import instantiate as cfg_instantiate
    from capx.envs.configs.loader import DictLoader

    configs = sorted(CONFIG_ROOT.glob("*/univtac_*.yaml"))
    assert len(configs) >= 30, len(configs)  # a broken glob must not pass

    for cfg_path in configs:
        envcfg = cfg_instantiate(DictLoader.load(str(cfg_path))["env"]["cfg"])
        stub = _StubTask.__new__(_StubTask)
        stub.cfg = envcfg
        low = DictLoader.load(envcfg.low_level)
        assert stub._task_name() == low["task_name"], cfg_path.name
        assert stub._task_name() in TASK_TYPES, cfg_path.name
