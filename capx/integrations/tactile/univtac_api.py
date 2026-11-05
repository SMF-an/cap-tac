"""UniVTAC tactile manipulation API for LLM code generation.

Functions in this class are called directly by LLM-generated Python code
during trial execution.  Each function's docstring appears verbatim in the
LLM prompt (via :meth:`ApiBase.combined_doc`), so it describes what the
function and its arguments ARE.

**Tactile representation:** this branch replaces the scalar tactile-vector
layer (depth_mm, displacement_px, coherence, shear, anisotropy, roughness)
with natural-language descriptions produced by the Octopi-1.5 visual-tactile
language model.  The purpose is an ablation experiment: both the original
scalar arm and this text arm run the same UniVTAC eval harness so the numbers
are directly comparable.

Motion primitives return only the physics clock ``{"step": int}``.
The one tactile-sensing call is ``describe_tactile()``.

All motion/perception calls are forwarded to the UniVTAC API server over HTTP.
The Octopi service runs as a separate process on port 8130 (override with
``OCTOPI_SERVER_URL``).  Start it with::

    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/octopi/bin/python \\
        /mnt/nas/shimufan/octopi-1.5/octopi_serve.py \\
        --weights /mnt/nas/shimufan/octopi-1.5/octopi_s/data/weights \\
        --port 8130
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

_DEFAULT_SERVER = "http://127.0.0.1:8120"
_DEFAULT_OCTOPI  = "http://127.0.0.1:8130"
_READ_TIMEOUT    = 1800.0
_OCTOPI_TIMEOUT  = 30.0


def _settled_note(data: dict) -> str:
    note = (data or {}).get("message") or ""
    return f" - {note}" if note and note != "OK" else ""


def _step_from(data: dict) -> int:
    return int((data.get("tactile") or {}).get("step", 0))


class UniVTACTactileApi(ApiBase):
    """Robot control API for a Franka arm with GelSight Mini tactile sensors.

    Provides motion, tactile description, and perception primitives for
    contact-rich manipulation tasks (insertion, grasping, placement).
    The robot is controlled via Cartesian end-effector poses solved
    through inverse kinematics.  Tactile feedback is available as natural-
    language descriptions via describe_tactile().
    """

    _prompt_hidden = {"check_success"}
    _ground_truth_grasp = True

    def __init__(self, env: BaseEnv, server_url: str | None = None) -> None:
        super().__init__(env)
        self._server = server_url or os.environ.get(
            "UNIVTAC_SERVER_URL", _DEFAULT_SERVER
        )
        self._octopi = os.environ.get("OCTOPI_SERVER_URL", _DEFAULT_OCTOPI)

    # ------------------------------------------------------------------
    # ApiBase interface
    # ------------------------------------------------------------------

    def functions(self) -> dict[str, Any]:
        return {
            "goto_pose":         self.goto_pose,
            "move_delta":        self.move_delta,
            "open_gripper":      self.open_gripper,
            "close_gripper":     self.close_gripper,
            "describe_tactile":  self.describe_tactile,
            "get_object_pose":   self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "rotate_gripper":    self.rotate_gripper,
            "check_success":     self.check_success,
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _server_health(self, timeout: float = 3.0) -> bool:
        try:
            r = requests.get(f"{self._server}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _post(self, path: str, body: dict | None = None) -> dict:
        started = time.monotonic()
        try:
            r = requests.post(
                f"{self._server}{path}",
                json=body or {},
                timeout=_READ_TIMEOUT,
            )
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            waited = time.monotonic() - started
            if waited < _READ_TIMEOUT * 0.9:
                raise RuntimeError(
                    f"POST {path} was abandoned after {waited:.0f}s, far short\n"
                    f"of the {_READ_TIMEOUT:.0f}s read timeout — an interrupted\n"
                    f"wait, not a slow server.  The usual cause is the harness's\n"
                    f"own wall-clock alarm (SIGALRM -> TimeoutError -> urllib3's\n"
                    f"socket.timeout -> ReadTimeout).  The sim op is still\n"
                    f"running on the server's main thread and will still write\n"
                    f"its rows to physical_trace.jsonl; do not assume it did\n"
                    f"nothing, and do not retry it blind.\n"
                    f"Request: {body}"
                )
            if self._server_health():
                raise RuntimeError(
                    f"Server did not respond within {_READ_TIMEOUT:.0f}s for "
                    f"POST {path}, but\n"
                    f"/health now answers — the sim op was extremely slow or\n"
                    f"stuck-and-recovered; its effect on the scene is unknown.\n"
                    f"Request: {body}\n"
                    f"This usually means the motion planner (cuRobo IK) took a\n"
                    f"degenerate query.  Re-query the object pose before going\n"
                    f"on; do not assume the motion did or did not happen."
                )
            raise RuntimeError(
                f"Server did not respond within {_READ_TIMEOUT:.0f}s for POST "
                f"{path} and is\n"
                f"FROZEN (/health silent) — a C++/CUDA hang holding the GIL.\n"
                f"Request: {body}\n"
                f"The external watchdog (120 s grace) will SIGKILL it and the\n"
                f"supervisor respawns it; Isaac cold-start takes minutes.\n"
                f"Treat this trial as lost and do NOT retry the motion blindly:\n"
                f"the sim state at the freeze is undefined."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to simulation server at {self._server}{path}.\n"
                f"The server may not be started, may have crashed, or the freeze\n"
                f"watchdog may have just SIGKILLed it (the supervisor respawns it\n"
                f"automatically, but Isaac cold-start takes minutes) — wait for\n"
                f"/health to answer before retrying."
            )
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"Server returned HTTP {e.response.status_code} for POST {path}.\n"
                f"Request: {body}\n"
                f"Server response: {detail}"
            )
        data = r.json()
        if not data.get("success", True):
            msg = data.get("message", "Unknown motion error")
            raise RuntimeError(f"Motion failed: {msg}")
        self._note_step(data)
        return data

    def _note_step(self, data: dict) -> None:
        """Remember the episode clock from any reply that carries it.

        Every motion and tactile reply carries the physics tick count, but the
        generated code usually drops it.  Keep the last one on the API and on
        the low-level env so the trial harness can still report a step count
        when the server is busy: a wall-clock alarm can leave a motion running
        on its main thread, and the post-trial /tactile query then times out.
        """
        try:
            tactile = data.get("tactile") or {}
            step = int(tactile.get("step", data.get("step", 0)) or 0)
        except Exception:
            step = 0
        if step:
            self._last_step = step
            env = getattr(self, "_env", None)
            if env is not None:
                setattr(env, "_last_api_step", step)

    def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            r = requests.get(
                f"{self._server}{path}",
                params=params or {},
                timeout=40,
            )
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            state = ("now answers (a slow op recovered)" if self._server_health()
                     else "FROZEN (/health silent — the watchdog will SIGKILL "
                          "and the supervisor respawn it)")
            raise RuntimeError(
                f"Server did not respond within 30s for GET {path}; server "
                f"{state}.\n"
                f"The simulation may be stuck — try resetting the environment."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to simulation server at {self._server}{path}.\n"
                f"The server may have crashed or not been started yet."
            )
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"Server returned HTTP {e.response.status_code} for GET {path}.\n"
                f"Server response: {detail}"
            )
        data = r.json()
        self._note_step(data)
        return data

    # ------------------------------------------------------------------
    # Motion API
    # ------------------------------------------------------------------

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> dict:
        """Move the end-effector to a target pose via inverse kinematics.

        This commands the END-EFFECTOR, not a held object: the two differ by
        the grasp offset, so a target for a held object is not a target for
        this call.  The position is the wrist flange; the gripper centre lies
        0.131 m from it along the hand's own z axis, toward the fingers, and
        the pads reach about 2.5 mm further along that axis.  With the fingers
        pointing down, the lowest point of the pads is therefore about 0.134 m
        below the commanded position.  When the commanded pose would drive a
        held object into the table, the planner refuses with
        "Motion planning failed".

        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis approach distance in meters. The
                end-effector first moves to position + z_approach above the
                target, then descends to the requested pose. Default is 0.0.

        Returns:
            dict with: step (int — physics ticks since reset, 120 per
            simulated second; use it to measure how long a motion took or
            to time a hold).
        """
        data = self._post(
            "/move",
            {
                "position": np.asarray(position, dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(quaternion_wxyz, dtype=float).tolist(),
                "z_approach": float(z_approach),
                "tag": "goto_pose",
            },
        )
        print(f"Moved to position={np.round(position, 3)} "
              f"quat_w={quaternion_wxyz[0]:.3f} z_approach={z_approach}"
              + _settled_note(data))
        return {"step": _step_from(data)}

    def move_delta(
        self,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
        drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
        coord: str = "world",
    ) -> dict:
        """Move the end-effector by a small delta. Primary primitive for
        contact-rich tasks (insertion, alignment).

        Limits per call: max 0.10 m translation per axis, max 0.5 rad rotation;
        larger moves MUST be split into multiple calls. There is also a MINIMUM
        of 0.002 m (2 mm) translation and 0.02 rad per call: a delta below it on
        EVERY axis cannot be executed and raises instead of moving nothing, so
        no closed-loop descent can converge to a tolerance finer than 2 mm. The
        target Z must stay above the table (Z >= 0); violating these limits is
        rejected.

        Args:
            dx: (float) X translation in meters.
            dy: (float) Y translation in meters.
            dz: (float) Z translation in meters.
            drx: (float) X rotation in radians.
            dry: (float) Y rotation in radians.
            drz: (float) Z rotation in radians.
            coord: ("world" or "ee") Reference frame for the whole delta,
                rotation included. Under "world" a rotation turns the
                end-effector's position about the world origin as well as its
                orientation, so the hand sweeps an arc whose radius is its
                distance from that origin; under "ee" the same rotation is
                about the end-effector itself and its position is left alone.

        Returns:
            dict with: step (int — physics ticks since reset).
        """
        data = self._post(
            "/move_delta",
            {
                "dx": float(dx), "dy": float(dy), "dz": float(dz),
                "drx": float(drx), "dry": float(dry), "drz": float(drz),
                "coord": coord,
                "tag": "move_delta",
            },
        )
        print(f"Moved delta: dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} "
              f"drx={drx:.3f} dry={dry:.3f} drz={drz:.3f} coord={coord}"
              + _settled_note(data))
        return {"step": _step_from(data)}

    def move_reflexive(
        self,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
        drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
        coord: str = "world",
        stop_on: dict | None = None,
    ) -> dict:
        """Move by a small delta, stopping early when a tactile event fires.

        The fast loop runs INSIDE the motion: the server evaluates tactile
        events on every simulation step and freezes the arm at the first
        enabled one.

        stop_on keys (all ORed; a key absent from the dict stays disabled):
            contact=True         stop when either fingertip touches
            hard_depth=<mm>      stop when either depth <= threshold
            shear_spike=<px>     stop when per-step marker displacement grows
            slip=<deg>           stop when the shear direction turns >= deg
            asymmetry=<mm>       stop when |L_depth - R_depth| >= mm
            plateau={"mm":0.5,"steps":5}  stop when contact is flat
            max_steps=<n>        safety net: stop after n evaluations

        Args:
            dx: (float) X translation in meters.
            dy: (float) Y translation in meters.
            dz: (float) Z translation in meters.
            drx: (float) X rotation in radians.
            dry: (float) Y rotation in radians.
            drz: (float) Z rotation in radians.
            coord: ("world" or "ee") Reference frame; see move_delta.
            stop_on: (dict) Conditions to stop on (see above).

        Returns:
            dict with: step (int), terminated_by (event that stopped the
            motion, or None), event (full digest).
        """
        data = self._post(
            "/move_reflexive",
            {
                "dx": float(dx), "dy": float(dy), "dz": float(dz),
                "drx": float(drx), "dry": float(dry), "drz": float(drz),
                "coord": coord,
                "tag": "move_reflexive",
                "stop_on": stop_on,
            },
        )
        event = data.get("event") or {}
        term = event.get("terminated_by")
        print(f"Reflexive move: dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} "
              f"terminated_by={term}")
        return {"step": _step_from(data), "terminated_by": term, "event": event}

    def open_gripper(self) -> dict:
        """Open the gripper fully.

        Returns:
            dict with: step (int — physics ticks since reset).
        """
        data = self._post("/gripper", {"action": "open"})
        print("Gripper opened")
        return {"step": _step_from(data)}

    def close_gripper(self, position: float) -> dict:
        """Close the gripper to a commanded finger opening.

        The opening is a fraction of the pads' full 0.08 m travel: 1.0 is fully
        open, 0.5 puts the pad faces 40 mm apart, and a smaller number presses
        the pads into the object.  This arm's close is positional: it has no
        gel-depth target, because the depth servo is the scalar tactile
        layer's channel and the text arm reads the gel only as text.  Size the
        opening from the object's own width, or from the scene's published
        geometry.

        Args:
            position: (float) Gripper opening, 0.0 (fully closed) to 1.0
                (fully open).

        Returns:
            dict with: step (int — physics ticks since reset).
        """
        data = self._post("/gripper", {
            "action": "close",
            "use_adaptive_grasp": False,
            "position": float(position),
        })
        print("Gripper closed")
        return {"step": _step_from(data)}

    # ------------------------------------------------------------------
    # Tactile description API  (replaces the scalar tactile layer)
    # ------------------------------------------------------------------

    def describe_tactile(
        self,
        side: str = "both",
        prompt: str = (
            "Describe the object in the following tactile video(s)."
        ),
    ) -> dict:
        """Describe what the fingertip gel(s) currently feel using text.

        Fetches the current GelSight gel image(s) from the UniVTAC server and
        sends them to the Octopi-1.5 visual-tactile language model, which
        returns a natural-language description of the surface texture and
        compliance.

        Call this at DECISION POINTS only — each call triggers a full gel
        render on the sim server (~0.5 s) and an Octopi inference (~1–2 s).
        Do not call it inside motion loops.

        Typical description vocabulary: hard/soft, rough/smooth/textured,
        bumpy, flexible, sticky.

        Args:
            side: ("left", "right", or "both") Which fingertip to read.
                "both" concatenates the two descriptions separated by " | ".
            prompt: (str) Instruction sent to Octopi. Default asks for a
                free description.  For a forced choice, ask the question
                and name the alternatives.

        Returns:
            dict with:
            step (int — physics ticks since reset, same clock as motion
                primitives);
            description (str — Octopi's text description of the surface;
                empty string when the Octopi service is unavailable);
            octopi_seconds (float — wall-clock time inside the Octopi service,
                for latency accounting; 0.0 on fallback).
        """
        raw = self._get("/tactile")
        step = raw.get("step", 0)

        sides_to_read = (["left", "right"] if side == "both" else [side])
        descriptions: list[str] = []
        total_secs = 0.0

        for s in sides_to_read:
            b64 = raw.get(f"{s}_rgb_base64", "")
            if not b64:
                descriptions.append(f"({s} gel image unavailable)")
                continue
            desc, secs = self._octopi_describe(b64, prompt)
            descriptions.append(f"{s}: {desc}" if side == "both" else desc)
            total_secs += secs

        description = " | ".join(descriptions) if side == "both" else descriptions[0]
        print(f"Tactile description ({side}): {description[:120]}")
        return {
            "step": step,
            "description": description,
            "octopi_seconds": round(total_secs, 2),
        }

    def _octopi_describe(self, b64: str, prompt: str) -> tuple[str, float]:
        """POST one gel-image base64 to the Octopi service.

        Falls back to an empty string if the service is unreachable so that a
        downed Octopi process does not crash the trial.
        """
        t0 = time.monotonic()
        try:
            r = requests.post(
                f"{self._octopi}/describe",
                json={"frames": [b64], "prompt": prompt, "max_new_tokens": 150},
                timeout=_OCTOPI_TIMEOUT,
            )
            r.raise_for_status()
            text = r.json().get("text", "")
            secs = time.monotonic() - t0
            # Strip the "Object 1:" prefix Octopi may prepend
            text = text.split("Object 1:")[-1].strip()
            text = text.split("\n")[0].strip()
            # Octopi was trained on multi-object scenes; a single gel frame
            # still makes it continue into a hallucinated "Object 2:" of the
            # same surface.  Keep only the object actually on the gel.
            text = re.split(r"\bobject\s*2\b", text, maxsplit=1, flags=re.I)[0].strip()
            return text, secs
        except Exception as exc:
            secs = time.monotonic() - t0
            print(f"  [Octopi unavailable: {exc}]")
            return "", secs

    # ------------------------------------------------------------------
    # Perception API
    # ------------------------------------------------------------------

    def sample_grasp_pose(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object resting on the table.

        The pose is the object's pose in the simulator's ground-truth scene
        plus a hand-relative offset, both read on the FIRST call for that
        object and cached from then on.  A task that would rather not take the
        cached offset has its own rule, which the API applies instead: where a
        rule exists the pose is rebuilt from the object's pose on every call
        rather than cached, so it follows the object after it has been moved or
        re-grasped, and the FIRST-call caveat above does not apply.

        The cached offset is only meaningful for the object the hand was left
        above (the one the pre-move anchored).  For any other object it is the
        hand's own position rather than that object's grasp, so the returned
        pose lands wherever the hand happens to be.  Build such a grasp from a
        measured offset instead: grasp_pos = object_pos + offset, with offset
        taken from a real grasp (grasp_pos - object_pos).

        Args:
            object_name: The simulator PRIM name of the object (e.g.
                "object_a", "object_b") — NOT a perception phrase.  This
                queries the simulator's ground-truth scene, so a SAM3
                phrase like "orange can" returns a 404.  (Perception
                prompts only apply to get_object_pose.)

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        data = self._get("/grasp_pose", {
            "object_name": object_name,
            "allow_ground_truth": self._ground_truth_grasp,
        })
        position = np.array(data["position"], dtype=np.float64)
        quat = np.array(data["quaternion_wxyz"], dtype=np.float64)
        print(f"Grasp pose for '{object_name}': pos={np.round(position, 3)} "
              f"quat_w={quat[0]:.3f}")
        return position, quat

    def rotate_gripper(
        self, object_name: str, angle_rad: float, steps: int = 4,
        feed_lead_m: float | None = None,
    ) -> dict:
        """Rotate the gripper while holding an object, about the world z axis.

        feed_lead_m selects the implementation.  Omitted, the environment's
        expert rotate runs one planner sub-move per step and fails the call
        when the object turns by less than 30 % of the requested angle (or
        about 3 deg, whichever is larger).
        Given, zero included, each step is one differential-IK iteration that
        turns the object by angle_rad / steps and advances the hand by the
        matched feed; the twist and the advance stay locked together.

        Args:
            object_name: The name of the object being held.
            angle_rad: (float) Total rotation angle in radians.
            steps: (int) Number of increments to split the rotation into.
            feed_lead_m: (float) Axial advance per full turn, in metres.

        Returns:
            dict with: step (int — physics ticks since reset).  With
            feed_lead_m given, a "helix" entry is added: completed /
            requested (servo iterations run), feed_lead_m, reached (bool),
            aborted (always None in a returned dict), grip_released (bool)
            and trace (the per-iteration record).
        """
        payload: dict[str, Any] = {
            "object_name": object_name, "angle_rad": angle_rad, "steps": steps,
        }
        if feed_lead_m is not None:
            payload["feed_lead_m"] = feed_lead_m
        data = self._post("/rotate_gripper", payload)
        step = _step_from(data)
        if feed_lead_m is None:
            print(f"Rotated gripper holding '{object_name}' by {angle_rad:.2f} rad "
                  f"in {steps} steps")
            return {"step": step}
        result: dict = {"step": step, "helix": {
            "feed_lead_m":   data.get("feed_lead_m"),
            "completed":     data.get("completed"),
            "requested":     data.get("requested"),
            "grip_released": data.get("grip_released"),
            "aborted":       data.get("aborted"),
            "reached":       data.get("reached"),
            "trace":         data.get("trace"),
        }}
        print(f"Twisted '{object_name}' along lead {feed_lead_m * 1000:.3f} mm: "
              f"{data.get('completed')}/{data.get('requested')} iterations, "
              f"{'seated' if data.get('reached') else 'stopped: ' + str(data.get('aborted')) if data.get('aborted') else 'OK'}")
        return result

    def check_success(self) -> bool:
        """Check whether the task is complete. Call at the end of the code to
        verify.

        Returns:
            bool, True if the success criteria are met.
        """
        data = self._get("/check_success")
        result = bool(data.get("completed", False))
        if result:
            print("Task check: SUCCESS")
        else:
            detail = data.get("detail") or {}
            print("Task check: FAILED")
            for reason in detail.get("reasons", []):
                print(f"  - {reason}")
            if detail.get("ground_truth_texture"):
                print(f"  ground truth: {detail['ground_truth_texture']}, "
                      f"expected pad: {detail['expected_pad']}")
                print(f"  prism at {detail.get('prism_pos')}, "
                      f"target at {detail.get('target_pos')}")
        return result

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the scene. position is the prim's world
        origin, not its geometric center. Task objects are authored with their
        origin at the bottom face (they rest on the table, so position[2] is
        the resting height above it). When return_bbox_extent=True, bbox_extent
        is the full 3D bounding-box size (width/depth/height in metres), not
        the half-extent.

        Args:
            object_name: The name of the object to get the pose of — the
                scene's PRIM name, verbatim (prefix or substring matches,
                e.g. "prism"; when several match, the one nearest the
                workspace wins).  Spaces are not part of any name: pass
                "orange_pad", never "orange pad".
            return_bbox_extent: Whether to fill in bbox_extent, the third return
                value, with the full 3D bounding-box size. Default is False,
                in which case bbox_extent comes back as None.

        Returns:
            Three values, always - the tuple is never shortened:
            position: (3,) XYZ in meters (prim world origin, not the center).
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length
                extent), or None when return_bbox_extent is False.
        """
        params: dict = {"object_name": object_name}
        if return_bbox_extent:
            params["return_bbox_extent"] = "true"
        data = self._get("/object_pose", params)
        if not data.get("found", False):
            raise RuntimeError(f"Object '{object_name}' not found in scene")
        position = np.array(data["position"], dtype=np.float64)
        quat = np.array(data["quaternion_wxyz"], dtype=np.float64)
        print(f"Object '{object_name}': pos={np.round(position, 3)} "
              f"quat_w={quat[0]:.3f}")
        bbox = None
        if return_bbox_extent:
            raw = data.get("bbox_extent")
            if raw is not None:
                bbox = np.array(raw, dtype=np.float64)
            if bbox is None:
                raise NotImplementedError(
                    "bbox_extent is not available for this object. "
                    "Try calling get_object_pose() without return_bbox_extent."
                )
        return position, quat, bbox

    # ------------------------------------------------------------------
    # Agent-facing helpers (NOT in functions() — not shown in LLM prompt)
    # ------------------------------------------------------------------

    def get_observation(self, include_depth: bool = True) -> dict:
        """Return the current head/wrist camera images for interactive agents."""
        data = self._get("/observation")
        result: dict = {}
        for cam in ("head", "wrist"):
            rgb_b64 = data.get(f"{cam}_rgb_base64", "")
            if rgb_b64:
                img = Image.open(io.BytesIO(base64.b64decode(rgb_b64)))
                result[f"{cam}_rgb"] = np.asarray(img.convert("RGB"), dtype=np.uint8)
            else:
                result[f"{cam}_rgb"] = None
            depth = None
            if include_depth:
                depth_b64 = data.get(f"{cam}_depth_float32_base64", "")
                shape = data.get(f"{cam}_image_shape")
                if depth_b64 and shape:
                    depth = np.frombuffer(
                        base64.b64decode(depth_b64), dtype=np.float32
                    ).reshape(shape[0], shape[1])
            result[f"{cam}_depth"] = depth
        for cam in ("head", "wrist"):
            for key in (f"{cam}_intrinsics", f"{cam}_cam_pos_w", f"{cam}_cam_quat_w_ros"):
                result[key] = data.get(key)
        shapes = ", ".join(
            f"{cam}="
            + (str(result[f"{cam}_rgb"].shape) if result[f"{cam}_rgb"] is not None else "None")
            for cam in ("head", "wrist")
        )
        print(f"Observation: {shapes}")
        return result

    def get_ee_pose(self) -> dict:
        """Return the end-effector pose in the world frame (agent-facing only)."""
        return self._get("/ee_pose")
