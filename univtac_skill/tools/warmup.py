"""Run the mandatory UniVTAC warmup and save per-task memory.

In the Octopi branch, tactile scalars (depth_mm, contact flags, etc.) are no
longer available.  The warmup is simplified to:

  1. Call describe_tactile() to get an Octopi text description of the
     unloaded gel (fingertips open, nothing grasped).
  2. Record the result and the Octopi service URL so the memory is traceable.

The full scalar warmup (contact probe, far-plane depth measurement, etc.) is
deleted because it depends entirely on get_tactile_data(), which is removed.
"""

from __future__ import annotations

import robo
from ._config import load_config, task_name, server_url, save_memory


def run_warmup(
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Execute the Octopi-branch warmup and (by default) save memory.

    Describes the unloaded gel with Octopi, records the result.

    Args:
        save: write ``memory/warmup_<task>.json`` when True.
        verbose: print progress lines.

    Returns:
        The full warmup record dict.
    """
    task = task_name()
    record = {
        "fingerprint": {
            "task": task,
            "server_url": server_url(),
            "api_module": load_config().get("api_module"),
        },
        "task": task,
        "notes": [],
        "steps": {},
        "learned": {},
        "api_rejections": [],
    }

    def note(msg):
        if verbose:
            print(f"[NOTE] {msg}")
        record["notes"].append(str(msg))

    def safe(fn, *a, **kw):
        try:
            return fn(*a, **kw), None
        except Exception as e:
            record["api_rejections"].append(
                f"{getattr(fn, '__name__', fn)}: {type(e).__name__}: {str(e)[:200]}"
            )
            return None, e

    if verbose:
        print(f"=== warmup {task} ===")

    # Open gripper to ensure nothing is held
    _, err = safe(robo.open_gripper)
    if err:
        note(f"open_gripper failed: {err}")

    # Describe the unloaded gel
    result, err = safe(robo.describe_tactile, side="both",
                       prompt="Describe the object in the following tactile video(s).")
    if err:
        note(f"describe_tactile failed: {err}")
        desc = ""
    else:
        desc = (result or {}).get("description", "")
        note(f"unloaded gel description: {desc[:120]}")

    record["steps"]["unloaded_gel"] = {"description": desc}
    record["learned"]["unloaded_baseline"] = {
        "description": desc,
        "note": "Octopi branch: no scalar depth/contact baseline available.",
    }

    if save:
        p = save_memory(record, task)
        if verbose:
            print(f"\nSaved warmup memory -> {p}")
            print(f"api_rejections: {record['api_rejections']}")
            print("=== warmup done ===")
    return record


def ensure_warmup(force: bool = False, **kw) -> dict:
    """Return existing warmup memory, or run warmup when it is missing.

    Mirrors SKILL.md: if ``memory/warmup_<task>.json`` exists, read and use
    it; otherwise run :func:`run_warmup` and save. Pass ``force=True`` to
    re-run.
    """
    from ._config import load_memory

    mem = None if force else load_memory()
    if mem is not None:
        if kw.pop("verbose", True):
            print(f"[warmup] using existing memory for {task_name()}")
        return mem
    return run_warmup(**kw)
