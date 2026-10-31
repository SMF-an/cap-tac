#!/usr/bin/env python3
"""Measure the strap's gel calibration at the grasp the task actually hands out.

The gel's newtons-per-pixel and its zero both belong to a particular grasp:
the pads' contact patch sets how far the markers move per newton, so a grasp
a few millimetres off changes the slope as well as the zero.  The first
calibration was taken at a pose 4.5 mm from the one `sample_grasp_pose`
returns, and on seed 1 that difference was a slope of 0.83 N/px against the
calibrated 1.03.

So this does the whole thing at the delivered grasp pose: reset, call
/grasp_pose with ground truth the way the API client does, close at the
task's own 27.6 mm, then raise 2 mm at a time and print, for every step, the
close-anchored differential against the task's own tension.

Usage:
    .venv/bin/python scripts/check_strap_grasp.py --seed 1 --use-grasp-pose
"""
from __future__ import annotations

import argparse
import math
import sys

import requests

XX_LABEL = "elastic_strap"
CLOSE_DEPTH_MM = 27.6
DEFAULT_GRASP_Z = 0.005 + 0.174 + 0.131


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8120)
    ap.add_argument("--close-depth", type=float, default=CLOSE_DEPTH_MM)
    ap.add_argument("--pulls", type=int, default=6)
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--require-release", action="store_true", default=True,
                    help="stop if the fixture never released (default on)")
    ap.add_argument("--use-grasp-pose", action="store_true",
                    help="take the pose from /grasp_pose, as the agent does")
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

    print("reset:", post("/reset", {"seed": args.seed}, timeout=900))
    if args.use_grasp_pose:
        pose = get("/grasp_pose?object_name=%s&allow_ground_truth=true" % XX_LABEL)
        grasp = pose["position"]
        quat = pose["quaternion_wxyz"]
    else:
        quat = get("/ee_pose")["quaternion_wxyz"]
        grasp = [0.55, 0.0, DEFAULT_GRASP_Z]
    print("grasp target:", [round(v, 4) for v in grasp])

    post("/move", {"position": [grasp[0], grasp[1], grasp[2] + 0.05],
                   "quaternion_wxyz": quat, "tag": "above"})
    post("/move", {"position": grasp, "quaternion_wxyz": quat, "tag": "at_tab"})
    try:
        post("/gripper", {"action": "close", "force_threshold": args.close_depth,
                          "use_adaptive_grasp": True})
    except RuntimeError as exc:
        print("close raised:", exc)

    t = get("/tactile")
    zero = (t["left_marker_shear_dx"] - t["right_marker_shear_dx"],
            t["left_marker_shear_dy"] - t["right_marker_shear_dy"])
    print("close: zero differential = (%.2f, %.2f) |z|=%.2f  depths %.2f/%.2f"
          % (zero[0], zero[1], math.hypot(*zero), t["left_depth_mm"], t["right_depth_mm"]))
    state = get("/force_state").get("measured") or {}
    print("  truth: tension %.3f N" % (state.get("tension_N") or 0.0))

    # The task pins the strap to its rest shape until BOTH pads read at or
    # under grasp_depth_mm + 0.1, and it checks that while the fingers are
    # still moving.  A close that lands a tenth of a millimetre shallower
    # leaves the strap pinned: the pull then buys almost no tension and the
    # tab is dragged through the gel instead, which reads as a huge slide and
    # means nothing.  Everything below is void in that state.
    released = bool(get("/force_state").get("grip_released"))
    print("  fixture released:", released)
    if args.require_release and not released:
        print("ABORT: the strap never left its fixture; a pull here measures the fixture")
        return 1

    samples = []
    for i in range(1, args.pulls + 1):
        z = grasp[2] + 0.001 * args.step_mm * i
        post("/move", {"position": [grasp[0], grasp[1], z],
                       "quaternion_wxyz": quat, "tag": f"pull{i}"})
        t = get("/tactile")
        dx = t["left_marker_shear_dx"] - t["right_marker_shear_dx"]
        dy = t["left_marker_shear_dy"] - t["right_marker_shear_dy"]
        load_px = math.hypot(dx - zero[0], dy - zero[1])
        truth = (get("/force_state").get("measured") or {}).get("tension_N") or 0.0
        samples.append((load_px, truth))
        print("  +%4.1f mm  |d|=%.2f px  truth=%.3f N  N/px=%s"
              % (args.step_mm * i, load_px, truth,
                 ("%.3f" % (truth / load_px)) if load_px > 1e-6 else "-"))

    good = [(p, n) for p, n in samples if p > 1.0]
    if len(good) >= 2:
        slope = sum(n for _, n in good) / sum(p for p, _ in good)
        print("\nleast-squares-through-zero slope: %.3f N/px over %d points"
              % (slope, len(good)))
        worst = max(abs(n - slope * p) for p, n in good)
        print("worst residual: %.2f N" % worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
