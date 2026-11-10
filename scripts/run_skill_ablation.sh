#!/bin/bash
# Skill-family ablation: withhold one family of the tactile skill library
# (sense / act / place) and run the same seeds as the full-library baseline.
#
# Usage:
#   bash scripts/run_skill_ablation.sh [task|all] [model] [first_seed] [last_seed] [gpu] [port]
#
# Examples:
#   # one cell, one seed, first GPU
#   bash scripts/run_skill_ablation.sh bulb_tightening gpt-6-astra 20 20 0 8120
#
#   # four cells in parallel: one GPU and one port per process
#   bash scripts/run_skill_ablation.sh tactile_memory_match gpt-6-astra 1 5 0 8120 &
#   bash scripts/run_skill_ablation.sh bulb_tightening      gpt-6-astra 1 5 1 8121 &
#   bash scripts/run_skill_ablation.sh blind_pick_balls     gpt-6-astra 1 5 2 8122 &
#   bash scripts/run_skill_ablation.sh tension_strap        gpt-6-astra 1 5 3 8123 &
#   wait
#
# Seeds are trial ids (seed = trial id).  Every arm of a task runs exactly the
# same seed range, so an arm differs from the baseline in one thing only.
#
# Arms: act / place / sense.  `place` delivers no skill to bulb_tightening or
# tension_strap, so those two arms are skipped.  INCLUDE_FULL=1 adds the
# full-library baseline arm; it is off by default.
#
# Output layout (the runner inserts the model name before the last part):
#   outputs/ablation/<task>/<model>/<arm>/trial_XX_taskcompleted_N/...
# and next to it, per arm: _command.txt, run.log (or seed_XXX.log and
# server_seed_XXX.log when the server is restarted per seed).
#
# Server lifecycle comes from the config's `server_lifecycle` key:
#   external  the runner starts one server for the whole seed range (default);
#   per_seed  this script owns the server: before each seed it stops whatever
#             is on the port, starts a server on that seed, waits for /health
#             to report the right task AND seed, runs the one trial, and stops
#             it again.  Required by bulb_tightening, blind_pick_balls and
#             tension_strap, whose scenes draw per-seed state before UIPC
#             initialises.
#
# Prerequisites:
#   - the model proxy on $CAPX_SERVER_URL (default 8112) is up;
#   - each parallel process uses its own port and GPU;
#   - for `external`, if a server already answers on the port it must be this
#     task's, or the run is refused (a wrong scene scores nonsense).
#
# Env knobs: INCLUDE_FULL=1 adds the baseline arm, DRY_RUN=1 prints the matrix
# without launching anything, CAPX_SERVER_START_TIMEOUT and CAPX_WATCHDOG_GRACE
# are passed through to the server starter.

set -euo pipefail

TASK="${1:-all}"
MODEL="${2:-gpt-6-astra}"
FIRST_SEED="${3:-1}"
LAST_SEED="${4:-1}"
GPU="${5:-${CAPX_GPU:-0}}"
PORT="${6:-8120}"
LLM_SERVER_URL="${CAPX_SERVER_URL:-http://127.0.0.1:8112/chat/completions}"
INCLUDE_FULL="${INCLUDE_FULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."

if [ "$FIRST_SEED" -gt "$LAST_SEED" ]; then
    echo "first_seed ($FIRST_SEED) > last_seed ($LAST_SEED)" >&2
    exit 1
fi

# The task client reads this at import time; without it the trial would talk to
# whatever owns the config's port instead of this process's server.
export UNIVTAC_SERVER_URL="http://127.0.0.1:${PORT}"
# Isaac Sim only sees cuda:0, so the server launcher hides the other GPUs
# rather than selecting one; the runner-spawned server inherits this too.
export CUDA_VISIBLE_DEVICES="$GPU"
export CAPX_GPU="$GPU"

source scripts/_server_lifecycle.sh

# task -> config
declare -A CONFIG=(
    [tactile_memory_match]="env_configs/univtac/tactile_memory_match/univtac_tactile_memory_match.yaml"
    [bulb_tightening]="env_configs/univtac/bulb_tightening/univtac_bulb_tightening.yaml"
    [blind_pick_balls]="env_configs/univtac/blind_pick_balls/univtac_blind_pick_balls.yaml"
    [tension_strap]="env_configs/univtac/tension_strap/univtac_tension_strap.yaml"
)
# task -> arms that have a non-zero dose on it
declare -A ARMS=(
    [tactile_memory_match]="act place sense"
    [bulb_tightening]="act sense"
    [blind_pick_balls]="act place sense"
    [tension_strap]="act sense"
)
TASKS=(tactile_memory_match bulb_tightening blind_pick_balls tension_strap)

if [ "$TASK" != "all" ] && [ -z "${CONFIG[$TASK]:-}" ]; then
    echo "unknown task: $TASK (known: all ${TASKS[*]})" >&2
    exit 1
fi

server_lifecycle() {
    ./.venv/bin/python3 - "$1" <<'PY'
import sys
from omegaconf import OmegaConf
print(str(OmegaConf.load(sys.argv[1]).get("server_lifecycle") or "external"))
PY
}

check_live_server() {
    local task="$1" health server_task
    health=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
    [ -n "$health" ] || return 0
    server_task=$(printf '%s' "$health" | ./.venv/bin/python3 -c \
        'import json,sys; print(json.load(sys.stdin).get("task",""))' 2>/dev/null || true)
    if [ -n "$server_task" ] && [ "$server_task" != "$task" ]; then
        echo "ERROR: port ${PORT} is serving '${server_task}', not '${task}'." >&2
        echo "  Stop it, or give this process another port." >&2
        return 1
    fi
}

launch_range() {
    local config="$1" out="$2" first="$3" last="$4" log="$5" arm="$6"
    local -a exclusion=()
    if [ "$arm" != "full" ]; then
        exclusion=(--tactile-skills-exclude-families "$arm")
    fi
    ./.venv/bin/python capx/envs/launch.py \
        --config-path "$config" \
        --model "$MODEL" \
        --server-url "$LLM_SERVER_URL" \
        --api-port "$PORT" \
        --total-trials "$last" \
        --resume-idx "$first" \
        --output-dir "$out" \
        "${exclusion[@]}" 2>&1 | tee "$log"
}

run_arm() {
    local task="$1" arm="$2"
    local config="${CONFIG[$task]}"
    local out="./outputs/ablation/${task}/${arm}"
    local run_dir="./outputs/ablation/${task}/${MODEL//\//_}/${arm}"
    local lifecycle exclude_arg
    lifecycle=$(server_lifecycle "$config")
    exclude_arg=""
    [ "$arm" = "full" ] || exclude_arg="--tactile-skills-exclude-families ${arm}"

    echo ""
    echo "========================================================================"
    echo "  ${task} / ${arm}   seeds ${FIRST_SEED}-${LAST_SEED}   gpu ${GPU}   port ${PORT}   (${lifecycle})"
    echo "  output: ${run_dir}"
    echo "========================================================================"

    mkdir -p "$run_dir"
    {
        echo "config:    ${config}"
        echo "model:     ${MODEL}"
        echo "seeds:     ${FIRST_SEED}-${LAST_SEED}"
        echo "gpu:       ${GPU}"
        echo "port:      ${PORT}"
        echo "lifecycle: ${lifecycle}"
        echo "arm:       ${arm}"
        echo "command:   ./.venv/bin/python capx/envs/launch.py --config-path ${config} --model ${MODEL} --server-url ${LLM_SERVER_URL} --api-port ${PORT} --output-dir ${out} ${exclude_arg} --resume-idx <seed> --total-trials <seed>"
    } > "${run_dir}/_command.txt"

    if [ "$lifecycle" = "per_seed" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            echo "  (dry run: restart the server per seed, ${FIRST_SEED}-${LAST_SEED})"
            return 0
        fi
        # Stop the last server on any exit, Ctrl+C and a failed trial included.
        trap 'stop_univtac_server "$PORT" >/dev/null 2>&1 || true' EXIT
        local seed
        for seed in $(seq "$FIRST_SEED" "$LAST_SEED"); do
            echo "---- seed ${seed}: restarting the UniVTAC server on port ${PORT} ----"
            stop_univtac_server "$PORT"
            start_univtac_server "$task" "$PORT" "$seed" \
                "${run_dir}/server_seed_$(printf '%03d' "$seed").log"
            launch_range "$config" "$out" "$seed" "$seed" \
                "${run_dir}/seed_$(printf '%03d' "$seed").log" "$arm"
        done
        stop_univtac_server "$PORT" || true
        trap - EXIT
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  (dry run: one launch for the whole range)"
        return 0
    fi
    check_live_server "$task"
    launch_range "$config" "$out" "$FIRST_SEED" "$LAST_SEED" "${run_dir}/run.log" "$arm"
}

if [ "$DRY_RUN" != "1" ]; then
    # Probe the model proxy before the first expensive query: a wrong service
    # on the port accepts the connection and closes it without an HTTP reply.
    LLM_CODE=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
        -X POST "$LLM_SERVER_URL" -H 'Content-Type: application/json' -d '{}' \
        2>/dev/null || true)
    case "$LLM_CODE" in
        200|400|422) ;;
        *)
            echo "ERROR: $LLM_SERVER_URL did not answer as an HTTP API (curl code '$LLM_CODE')." >&2
            echo "  Start the proxy, or point CAPX_SERVER_URL at the one running." >&2
            exit 1
            ;;
    esac
fi

for task in "${TASKS[@]}"; do
    if [ "$TASK" != "all" ] && [ "$TASK" != "$task" ]; then
        continue
    fi
    arms="${ARMS[$task]}"
    if [ "$INCLUDE_FULL" = "1" ]; then
        arms="full $arms"
    fi
    for arm in $arms; do
        run_arm "$task" "$arm"
    done
done

echo ""
echo "Ablation complete: ${TASK} / ${MODEL} / seeds ${FIRST_SEED}-${LAST_SEED} / gpu ${GPU} / port ${PORT}"
