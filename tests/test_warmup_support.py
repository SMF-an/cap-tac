"""The warmup record says how much evidence each learned value rests on.

A warmup number is one observation, not a task constant.  Recording it as
n=1 together with the parameters that produced it is the difference between
"this probe stopped at 27.9 mm" and "this task's first contact is 27.9 mm" --
the second claim has no support and would be delivered as if it did.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "univtac_skill" / "tools"


def load_warmup():
    """Load ``tools/warmup.py`` without running the tools package __init__.

    The package imports the recorder, which pulls in imaging dependencies the
    test does not need; warmup itself only needs ``robo`` at call time, and
    the helper under test never touches it.  A stub keeps the import honest
    without dragging the whole skill stack in.
    """
    sys.modules.setdefault("robo", types.ModuleType("robo"))
    package = types.ModuleType("univtac_skill.tools")
    package.__path__ = [str(TOOLS)]
    root = types.ModuleType("univtac_skill")
    root.__path__ = [str(TOOLS.parent)]
    sys.modules.setdefault("univtac_skill", root)
    sys.modules.setdefault("univtac_skill.tools", package)
    spec = importlib.util.spec_from_file_location("univtac_skill.tools.warmup",
                                                  TOOLS / "warmup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_probe_is_one_episode_under_the_parameters_that_produced_it():
    support = load_warmup()._probe_support("lift_can", 0.005, 20)
    assert support["n"] == 1
    assert support["counterexamples"] == []
    assert "step_down 5.0 mm" in support["conditions"]
    assert "max_contact_steps 20" in support["conditions"]


def test_an_initial_hold_says_it_ran_no_probe():
    support = load_warmup()._probe_support("insert_hdmi", None, None)
    assert support["n"] == 1
    assert support["conditions"] == ["initial state, no probe"]


def test_the_support_block_names_its_source_and_serialises():
    support = load_warmup()._probe_support("lift_can", 0.005, 20)
    assert support["source"].startswith("warmup lift_can ")
    json.dumps(support)  # the record is written as JSON, so it must be plain
