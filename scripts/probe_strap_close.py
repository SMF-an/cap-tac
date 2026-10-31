#!/usr/bin/env python3
"""Is the strap's grip reproducible?  Close it the same way, repeatedly.

The gel's absolute calibration belongs to the grasp: its zero, its slope and
whether a pad's marker field survives the close all move with how the pads
land on the tab.  The depth servo lands differently every time (27.31/26.99,
27.39/27.10, 27.70/27.38, 27.80/27.32 for the same command), so this asks
whether commanding the *opening* instead -- the finger separation, which is
what a position close sets -- lands the same way twice.

Each cycle is a fresh reset and a fresh grasp from the pose /grasp_pose hands
out, then one close, and the pads are read back.  A mode is reproducible if
its owns rows agree; it is usable if its depths are near 27.0-27.4, the band
the depth close works in.

Usage:
    .venv/bin/python scripts/probe_strap_close.py --seed 1
"""
from __future__ import annotations

import argparse
import sys

import requests

OBJECT = "elastic_strap"
# (label, payload-extra).  "depth" is the servoed close the API hands out
# when force_threshold is given; "position" commands the opening directly.
SCHEDULE = [
    ("depth 27.2", {"force_threshold": 27.2, "use_adaptive_grasp": True}),
    ("position 0.10", {"position": 0.10, "use_adaptive_grasp": False}),
    ("position 0.14", {"position": 0.14, "use_adaptive_grasp": False}),
    ("position 0.18", {"position": 0.18, "use_adaptive_grasp": False}),
    ("position 0.22", {"position": 0.22, "use_adaptive_grasp": False}),
    ("best x2", None),   # filled in below once the sweep picks a winner
    ("best x3", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8120)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def get(path, timeout=60):
        r = requests.get(base + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def post(path, body, timeout=1800):
        r = requests.post(base + path, json=body, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"{path}: {r.status_code} {r.text[:200]}")
        return r.json()

    rows = []
    for label, payload in SCHEDULE:
        if payload is None:
            # Repeat the position close whose depth landed nearest 27.2.
            usable = [r for r in rows if r["mode"].startswith("position")]
            if not usable:
                continue
            best = min(usable, key=lambda r: abs(r["depth"] - 27.2))
            payload = {"position": best["value"], "use_adaptive_grasp": False}
            label = f"{label} ({best['value']})"
        print(f"--- {label} ---", flush=True)
        post("/reset", {"seed": args.seed}, timeout=900)
        pose = get("/grasp_pose?object_name=%s&allow_ground_truth=true" % OBJECT)
        quat = pose["quaternion_wxyz"]
        post("/move", {"position": [pose["position"][0], pose["position"][1],
                                    pose["position"][2] + 0.05],
                       "quaternion_wxyz": quat, "tag": "above"})
        post("/move", {"position": pose["position"], "quaternion_wxyz": quat,
                       "tag": "at_tab"})
        body = {"action": "close", **payload}
        try:
            post("/gripper", body)
            raised = ""
        except RuntimeError as exc:
            raised = str(exc)[:60]
        t = get("/tactile")
        fs = get("/force_state")
        depth = 0.5 * (t["left_depth_mm"] + t["right_depth_mm"])
        row = {
            "mode": label.split(" (")[0],
            "label": label,
            "value": payload.get("position"),
            "depth": depth,
            "left_depth": t["left_depth_mm"], "right_depth": t["right_depth_mm"],
            "ldisp": t["left_marker_displacement_px"],
            "rdisp": t["right_marker_displacement_px"],
            "ratio": (max(t["left_marker_displacement_px"],
                          t["right_marker_displacement_px"])
                      / max(min(t["left_marker_displacement_px"],
                                t["right_marker_displacement_px"]), 1e-6)),
            "lshear": (t["left_marker_shear_dx"], t["left_marker_shear_dy"]),
            "rshear": (t["right_marker_shear_dx"], t["right_marker_shear_dy"]),
            "released": fs.get("grip_released"),
            "raised": raised,
        }
        rows.append(row)
        print("  depths %.2f/%.2f  disp %.1f/%.1f (ratio %.1f)  released %s %s"
              % (row["left_depth"], row["right_depth"], row["ldisp"], row["rdisp"],
                 row["ratio"], row["released"], ("RAISED " + raised) if raised else ""),
              flush=True)

    print("\n=== summary ===")
    print("%-18s %-7s %-7s %-7s %-6s %s" % ("mode", "depthL", "depthR", "ratio", "rel", "disp L/R"))
    for r in rows:
        print("%-18s %-7.2f %-7.2f %-7.1f %-6s %.1f/%.1f"
              % (r["label"], r["left_depth"], r["right_depth"], r["ratio"],
                 r["released"], r["ldisp"], r["rdisp"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
