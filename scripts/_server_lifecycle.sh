#!/bin/bash
# Start/stop the UniVTAC server for a chain that needs one process per seed.
#
# Sourced by scripts/run_trace_chain.sh when the config sets
# `server_lifecycle: per_seed` (tension_strap and the other force tasks: the
# band modulus / thread lead is drawn from the seed before UIPC initialises,
# and the task raises on any other combination).  Source it from the repo root;
# the health parser uses ./.venv/bin/python3.
#
# The stop is deliberately keyed on the PORT, not the task, and kills the
# supervisor before the server: the supervisor respawns a dead child in 2 s, so
# the other order races.  Both processes are SIGKILLed -- the sim server does
# not exit on SIGTERM.

_health_field() {
    # _health_field <port> <key> -> the value, or empty when the server is down
    curl -sf "http://127.0.0.1:${1}/health" 2>/dev/null \
        | ./.venv/bin/python3 -c \
            'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' \
            "${2}" 2>/dev/null || true
}

_pids_on_port() {
    # _pids_on_port <port> <sh|py>
    # PIDs whose argv holds the launcher (or server) path as its own argument
    # AND the port as its own argument.  A plain `pgrep -f` is not enough: it
    # also matches a shell whose -c script merely *mentions* the launcher, and
    # killing that is killing the caller.
    local port="$1" kind="$2" pid arg has_script has_port suffix
    case "$kind" in
        sh) suffix="/scripts/launch_univtac_server.sh" ;;
        py) suffix="/capx/serving/launch_univtac_server.py" ;;
        *) return 1 ;;
    esac
    for pid in $(pgrep -f "launch_univtac_server\.${kind}" || true); do
        [ -r "/proc/$pid/cmdline" ] || continue
        has_script=0
        has_port=0
        while IFS= read -r arg; do
            case "$arg" in
                *"$suffix"|"${suffix#/}") has_script=1 ;;
            esac
            if [ "$arg" = "$port" ]; then
                has_port=1
            fi
        done < <(tr '\0' '\n' < "/proc/$pid/cmdline")
        if [ "$has_script" -eq 1 ] && [ "$has_port" -eq 1 ]; then
            echo "$pid"
        fi
    done
    return 0
}

stop_univtac_server() {
    local port="$1" pids i
    pids=$(_pids_on_port "$port" sh)
    if [ -n "$pids" ]; then
        echo "  [server] stopping supervisor(s) on ${port}: $(echo "$pids" | tr '\n' ' ')"
        kill -9 $pids 2>/dev/null || true
    fi
    pids=$(_pids_on_port "$port" py)
    if [ -n "$pids" ]; then
        echo "  [server] stopping server process(es) on ${port}: $(echo "$pids" | tr '\n' ' ')"
        kill -9 $pids 2>/dev/null || true
    fi
    for i in $(seq 1 30); do
        if ! curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: port ${port} still answers after the stop." >&2
    return 1
}

start_univtac_server() {
    # start_univtac_server <task> <port> <seed> <log>
    local task="$1" port="$2" seed="$3" log="$4"
    local gpu="${CAPX_GPU:-0}"
    local budget="${CAPX_SERVER_START_TIMEOUT:-900}"
    local started=$SECONDS launcher_pid server_task server_seed
    local grace_args=()
    if [ -n "${CAPX_WATCHDOG_GRACE:-}" ]; then
        # The default 120 s grace kills a server whose /health is silent
        # during a long GIL-holding reset under load (seen on bulb seed 6 on
        # 2026-11-09); 540 s still sits below the server's own 600 s reset
        # budget.
        grace_args=(--watchdog-grace "$CAPX_WATCHDOG_GRACE")
    fi

    echo "  [server] starting ${task} on ${port} with seed ${seed} (log: ${log})"
    nohup bash scripts/launch_univtac_server.sh \
        "$task" "$port" "$gpu" "$seed" "${grace_args[@]}" > "$log" 2>&1 &
    launcher_pid=$!

    while [ $((SECONDS - started)) -lt "$budget" ]; do
        server_task=$(_health_field "$port" task)
        if [ -n "$server_task" ]; then
            server_seed=$(_health_field "$port" seed)
            # Older servers do not report a seed; accept them and let the
            # task's own seed guard fail loudly at reset if it is wrong.
            if [ "$server_task" = "$task" ] \
                && { [ -z "$server_seed" ] || [ "$server_seed" = "$seed" ]; }; then
                echo "  [server] ready after $((SECONDS - started)) s"
                return 0
            fi
        fi
        if ! kill -0 "$launcher_pid" 2>/dev/null; then
            echo "ERROR: the launcher exited before ${task} answered /health." >&2
            tail -20 "$log" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "ERROR: ${task} did not answer /health within ${budget}s." >&2
    tail -20 "$log" >&2 || true
    return 1
}
