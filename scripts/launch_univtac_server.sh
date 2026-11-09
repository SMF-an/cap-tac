#!/bin/bash
# Usage: bash scripts/launch_univtac_server.sh <task> [port] [gpu_id] [seed] [extra args...]
# Example: bash scripts/launch_univtac_server.sh lift_can 8120 0 7
#          bash scripts/launch_univtac_server.sh weight_classify 8120 2 0 --cfg-override weight_label=light

TASK="${1:?Usage: $0 <task> [port] [gpu_id] [seed] [extra args...]}"
PORT="${2:-8120}"
GPU="${3:-0}"
SEED="${4:-0}"

cd "$(dirname "$0")/.."

# UniVTAC root: sibling directory by default, override with UNIVTAC_ROOT
export UNIVTAC_ROOT="${UNIVTAC_ROOT:-${PWD}/../UniVTAC}"
export PYTHONPATH="${PWD}:${UNIVTAC_ROOT}:${PYTHONPATH:-}"

# ViTaForge root: sibling directory by default, override with VITAFORGE_ROOT.
# Deliberately NOT added to PYTHONPATH — the server puts the active task's
# backing root first on sys.path itself (_activate_backing_root).  Listing it
# here would place it *after* UNIVTAC_ROOT, inverting the priority and
# producing a mixed-up envs package.
export VITAFORGE_ROOT="${VITAFORGE_ROOT:-${PWD}/../ViTaForge}"

# Isaac Sim's usdrt scenegraph only supports cuda:0 ("GPUs other than
# cuda:0 are not currently supported").  To run on GPU N, hide the other
# GPUs from CUDA so the chosen one appears as cuda:0 — the same pattern
# UniVTAC's own scripts/collect_data.py uses (--gpu).
export CUDA_VISIBLE_DEVICES="$GPU"

# The piano's Shadow Hand USD normally comes from the Isaac Nucleus S3 URL.
# Direct egress on this host lands on the campus captive portal and the
# Omniverse asset resolver ignores the shell's HTTP proxy, so piano servers
# died at startup with "USD file not found".  Point the task at a local copy
# of the same two files when it exists; other tasks ignore the variable.
export SHADOW_HAND_USD="${SHADOW_HAND_USD:-${HOME}/assets/shadow_hand/shadow_hand_instanceable.usd}"
# The carrying Panda arm is a second remote tree (links, hand, materials);
# same fix.
export PANDA_USD="${PANDA_USD:-${HOME}/assets/franka_emika/panda_instanceable.usd}"

# Isaac Sim's omni.services HTTP transport takes a fixed port (8083) from its
# carb settings, and when it loses the bind race it calls kit_exit and kills
# the whole server process.  On 2026-11-09 two ViTaForge servers started at
# once and hardness_classify died with "address already in use" on 8083.  Give
# each server its own port in the 9000 range; nothing in this harness talks to
# that service.
KIT_ARGS="--/exts/omni.services.transport.server.http/port=$((PORT + 1000))"

# Supervisor loop (shared with launch_sam3_server.sh, see _supervise.sh): a
# cuRobo hang — which the server's freeze watchdog terminates with SIGKILL
# (exit 137) — auto-respawns the server; Ctrl+C (graceful exit 0) breaks it.
source "$(dirname "$0")/_supervise.sh"
supervise "UniVTAC server" \
    "${UNIVTAC_PYTHON:-${HOME}/miniconda3/envs/UniVTAC/bin/python}" \
    capx/serving/launch_univtac_server.py \
    --task "$TASK" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --headless \
    --device "cuda:0" \
    --seed "$SEED" \
    --kit_args="$KIT_ARGS" \
    "${@:5}"
