#!/usr/bin/env python3
"""One strap calibration: grasp the given way, raise, read the true tension.

The gel's differential shear is the only non-privileged force channel this
task has, and its two numbers -- newtons per pixel, and the take-up that sets
where zero sits -- belong to a grip and, as it turns out, to the strap the seed
draws.  This measures both, against the task's own tension, in one short
episode, so the same procedure can be repeated on each seed and the results
compared.

The server must already be running on the seed under test: one process serves
one physics seed, and the strap's stiffness is drawn before the task is built.

Usage:
    .venv/bin/python scripts/strap_calibration.py --seed 0
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import requests

OBJECT = "elastic_strap"
# The close that reproduces: position=0.22 lands the pads at 25.70/25.25 mm on
# every strap tried, where a depth close lands somewhere new each time.
CLOSE_OPENING = 0.22
STEPS_MM = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--port", type=int, default=8120)
    ap.add_argument("--opening", type=float, default=CLOSE_OPENING)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def get(path, timeout=120):
        r = requests.get(base + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def post(path, body, timeout=1800):
        r = requests.post(base + path, json=body, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"{path}: {r.status_code} {r.text[:200]}")
        return r.json()

    def tension() -> float:
        measured = get("/force_state").get("measured") or {}
        return float(measured.get("tension_N") or 0.0)

    post("/reset", {"seed": args.seed}, timeout=900)
    pose = get(f"/grasp_pose?object_name={OBJECT}&allow_ground_truth=true")
    quat = pose["quaternion_wxyz"]
    grasp = pose["position"]
    print("seed %d: grasp %s" % (args.seed, [round(v, 4) for v in grasp]))

    post("/move", {"position": [grasp[0], grasp[1], grasp[2] + 0.05],
                   "quaternion_wxyz": quat, "tag": "above"})
    post("/move", {"position": grasp, "quaternion_wxyz": quat, "tag": "at_tab"})
    try:
        post("/gripper", {"action": "close", "position": args.opening,
                          "use_adaptive_grasp": False})
    except RuntimeError as exc:
        print("close raised:", exc)

    t = get("/tactile")
    zero = (t["left_marker_shear_dx"] - t["right_marker_shear_dx"],
            t["left_marker_shear_dy"] - t["right_marker_shear_dy"])
    released = bool(get("/force_state").get("grip_released"))
    print("  close: depths %.2f/%.2f  disp %.1f/%.1f  |zero| %.2f  released %s"
          % (t["left_depth_mm"], t["right_depth_mm"],
             t["left_marker_displacement_px"], t["right_marker_displacement_px"],
             math.hypot(*zero), released))
    if not released:
        print("ABORT: fixture never released; a pull here measures the fixture")
        return 1

    px, n = [], []
    for step in STEPS_MM:
        z = grasp[2] + 0.001 * step
        post("/move", {"position": [grasp[0], grasp[1], z],
                       "quaternion_wxyz": quat, "tag": f"pull{step:g}"})
        t = get("/tactile")
        dx = t["left_marker_shear_dx"] - t["right_marker_shear_dx"]
        dy = t["left_marker_shear_dy"] - t["right_marker_shear_dy"]
        d = math.hypot(dx - zero[0], dy - zero[1])
        truth = tension()
        px.append(d)
        n.append(truth)
        print("  +%4.1f mm  |d|=%6.2f px  truth=%7.3f N" % (step, d, truth))
        if truth > 20.0:
            # The task ruptures the strap past its overload, and a ruptured
            # strap needs a new server process; stop with the points we have.
            print("  stopping: past 20 N, and the task tears the strap at 24")
            break

    px_a, n_a = np.array(px), np.array(n)
    good = n_a > 0.5
    if good.sum() >= 3:
        slope, intercept = np.polyfit(px_a[good], n_a[good], 1)
        take_up = -intercept / slope
        worst = float(np.max(np.abs(n_a[good] - (slope * px_a[good] + intercept))))
        print("\nFIT seed=%d slope=%.3f N/px take_up=%.2f px worst_residual=%.2f N points=%d"
              % (args.seed, slope, take_up, worst, int(good.sum())))
    else:
        print("\nnot enough loaded points to fit (truth never passed 0.5 N)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
