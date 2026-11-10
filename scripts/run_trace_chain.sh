#!/bin/bash
# Run one task's seeds in order, carrying each seed's execution trace into the
# next seed's prompt.  Each seed is its own process: the only state that
# crosses a seed boundary is <output_dir>/trace_chain/.
#
# Usage:
#   bash scripts/run_trace_chain.sh <task> <start_seed> <end_seed> <model> \
#       [port] [max_chars] [config_path]
#
# Example:
#   bash scripts/run_trace_chain.sh grasp_classify 1 30 gpt-6-astra 8120 12000
#
# The trace arm is selected with $CAPX_TRACE_MODE (default "last"):
#   last       the most recent record only (the original arm)
#   dual       last record + best completed success so far
#   card       a four-line experience card for the last record
#   dual_card  cards for the last record and the best success
#   window_card  cards for the last $CAPX_TRACE_WINDOW (default 3) records
#                plus the best success, with a streak line when they agree
#
# By default the UniVTAC server must already be running on the same task:
#   bash scripts/launch_univtac_server.sh grasp_classify 8120 <gpu> 0
#
# A config may set `server_lifecycle: per_seed` (tension_strap and the other
# force tasks must).  Then this script owns the server: before each seed it
# stops whatever is on the port, starts a server on that seed, waits for
# /health to report the right task AND seed, and stops it again when the chain
# ends.  GPUs come from $CAPX_GPU (default 0), the startup budget from
# $CAPX_SERVER_START_TIMEOUT (default 900 s -- a force task's first reset runs
# the full scene build).
#
# The model proxy comes from $CAPX_SERVER_URL, defaulting to
# http://127.0.0.1:8112/chat/completions.  launch.py's own default is port 8110,
# which on this host belongs to another user's viser server: it accepts the
# connection and closes it without an HTTP reply, so the first model query dies
# as RemoteDisconnected.  The proxy itself is started separately, e.g.
#   ./.venv/bin/python3 capx/serving/openrouter_server.py \
#       --base-url https://api.codexzh.com/v1 --port 8112
#
# By default this uses univtac_<task>_trace.yaml, whose class names the skills
# the experiment keeps.  For a run with every skill withheld, invoke
# capx/envs/launch.py directly with --deliver-tactile-skills False; this script
# always uses the config as given.
set -euo pipefail

TASK="${1:?Usage: $0 <task> <start_seed> <end_seed> <model> [port] [max_chars] [config_path]}"
START="${2:?start seed}"
END="${3:?end seed}"
MODEL="${4:?model name}"
PORT="${5:-8120}"
MAX_CHARS="${6:-12000}"
CONFIG_PATH="${7:-env_configs/univtac/${TASK}/univtac_${TASK}_trace.yaml}"
LLM_SERVER_URL="${CAPX_SERVER_URL:-http://127.0.0.1:8112/chat/completions}"
# last  - the original arm: the most recent record only
# dual  - last record + best success so far
# card  - the model's four-line experience card for the last record
# dual_card - cards for both slots
# window_card - cards for the last $CAPX_TRACE_WINDOW records + best success
TRACE_MODE="${CAPX_TRACE_MODE:-last}"
TRACE_WINDOW="${CAPX_TRACE_WINDOW:-3}"

cd "$(dirname "$0")/.."

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found: $CONFIG_PATH" >&2
    echo "Pass a config path as the 7th argument." >&2
    exit 1
fi

MODEL_DIR="${MODEL//\//_}"
# Take output_dir from the config instead of assuming univtac_<task>: the trace
# arm has its own directory.  runner._setup_output_dir inserts the model before
# the last path component, so mirror that here.
CONFIG_VALUES=$(./.venv/bin/python3 - "$CONFIG_PATH" "$TASK" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(str(cfg.get("output_dir") or f"./outputs/univtac_{sys.argv[2]}"))
print(str(cfg.get("server_lifecycle") or "external"))
PY
)
BASE_OUT_DIR=$(printf '%s\n' "$CONFIG_VALUES" | sed -n 1p)
SERVER_LIFECYCLE=$(printf '%s\n' "$CONFIG_VALUES" | sed -n 2p)
BASE_OUT_DIR="${BASE_OUT_DIR%/}"
RUN_DIR="$(dirname "$BASE_OUT_DIR")/${MODEL_DIR}/$(basename "$BASE_OUT_DIR")"
LOG_DIR="${RUN_DIR}/chain_logs"
mkdir -p "$LOG_DIR"

export UNIVTAC_SERVER_URL="http://127.0.0.1:${PORT}"

# Per-seed server start/stop lives in its own file so it can be tested
# without running a chain.
source "$(dirname "$0")/_server_lifecycle.sh"

if [ "$SERVER_LIFECYCLE" = "per_seed" ]; then
    # Stop the last server on any exit, including Ctrl+C and a failed trial.
    trap 'stop_univtac_server "$PORT" >/dev/null 2>&1 || true' EXIT
else
    HEALTH=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
    if [ -z "$HEALTH" ]; then
        echo "ERROR: no server answering on http://127.0.0.1:${PORT}/health" >&2
        echo "Start it first (same task, same port):" >&2
        echo "  bash scripts/launch_univtac_server.sh ${TASK} ${PORT} <gpu> 0" >&2
        exit 1
    fi

    # The port is reused across tasks (8120 by default), so a healthy server is
    # not necessarily the right one.  Running the chain against another task's
    # scene would reset into the wrong world and score nonsense.
    SERVER_TASK=$(printf '%s' "$HEALTH" | ./.venv/bin/python3 -c \
        'import json,sys; print(json.load(sys.stdin).get("task",""))' 2>/dev/null || true)
    if [ -n "$SERVER_TASK" ] && [ "$SERVER_TASK" != "$TASK" ]; then
        echo "ERROR: the server on port ${PORT} is running '${SERVER_TASK}', not '${TASK}'." >&2
        echo "  Stop it and start the right one:" >&2
        echo "    bash scripts/launch_univtac_server.sh ${TASK} ${PORT} <gpu> 0" >&2
        exit 1
    fi
fi

# Probe the model proxy before the first (expensive) query.  An empty body
# makes a FastAPI proxy answer 422, which is enough to prove the port speaks
# HTTP; a wrong service answers with no HTTP at all (curl code 000).  Without
# this the chain dies on seed 1 with a RemoteDisconnected traceback instead.
LLM_CODE=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
    -X POST "$LLM_SERVER_URL" -H 'Content-Type: application/json' -d '{}' \
    2>/dev/null || true)
case "$LLM_CODE" in
    200|400|422) ;;
    *)
        echo "ERROR: $LLM_SERVER_URL did not answer as an HTTP API (curl code '$LLM_CODE')." >&2
        echo "  Expected 422 from a FastAPI proxy given an empty body." >&2
        echo "  Start the proxy, or point CAPX_SERVER_URL at the one already running:" >&2
        echo "    ./.venv/bin/python3 capx/serving/openrouter_server.py \\" >&2
        echo "        --base-url https://api.codexzh.com/v1 --port 8112" >&2
        exit 1
        ;;
esac

for seed in $(seq "$START" "$END"); do
    if [ "$SERVER_LIFECYCLE" = "per_seed" ]; then
        echo "------------------------------------------------------------"
        echo "  [server] seed ${seed}: restarting the UniVTAC server"
        stop_univtac_server "$PORT" || exit 1
        start_univtac_server "$TASK" "$PORT" "$seed" \
            "${LOG_DIR}/server_seed_$(printf '%03d' "$seed").log" || exit 1
    fi
    echo "============================================================"
    echo "  ${TASK} seed ${seed}  mode=${TRACE_MODE} window=${TRACE_WINDOW}  ($(date +%H:%M:%S))"
    echo "============================================================"
    ./.venv/bin/python3 capx/envs/launch.py \
        --config-path "$CONFIG_PATH" \
        --model "$MODEL" \
        --server-url "$LLM_SERVER_URL" \
        --total-trials "$seed" \
        --resume-idx "$seed" \
        --trace-mode "$TRACE_MODE" \
        --trace-max-chars "$MAX_CHARS" \
        --trace-window "$TRACE_WINDOW" \
        2>&1 | tee "${LOG_DIR}/seed_$(printf '%03d' "$seed").log"
done

echo "Chain complete: ${START}..${END}"
if [ "$SERVER_LIFECYCLE" = "per_seed" ]; then
    stop_univtac_server "$PORT" || true
fi
./.venv/bin/python3 scripts/analyze_trace_chain.py \
    "${RUN_DIR}/trace_chain" \
    --task "$TASK" \
    --calibration-log outputs/univtac_tactile_calibration.log
