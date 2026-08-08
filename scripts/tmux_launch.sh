#!/usr/bin/env bash
# Launch a station's experiment stages inside its own tmux session.
#
#   ./scripts/tmux_launch.sh <STATION> <STAGE> [STAGE ...]
#   ./scripts/tmux_launch.sh 004-CangFang S2 S3
#
# One session per station, named after it:
#   tmux ls                    list sessions
#   tmux attach -t CangFang    watch one
#   Ctrl-b d                   detach without stopping anything
#
# The pane stays open after the stages finish (or fail) so the tail of the run
# is still readable; exit it with Ctrl-d.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-PVDetection}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"

STATION="${1:-}"
if [[ -z "$STATION" ]]; then
    echo "usage: $0 <STATION> <STAGE> [STAGE ...]" >&2
    exit 1
fi
shift
if [[ $# -eq 0 ]]; then
    echo "usage: $0 <STATION> <STAGE> [STAGE ...]" >&2
    exit 1
fi
STAGES=("$@")

# Short session name: 004-CangFang -> CangFang.
# SESSION_SUFFIX lets one station occupy several sessions at once, which is how
# an independent stage set (e.g. disjoint S5_CONFIGS) gets split across cards.
# Without it the second launch sees a busy session of the same name and refuses.
SESSION="${STATION#*-}${SESSION_SUFFIX:-}"

# A tmux session does not inherit this shell's environment, so any driver knob
# set by the caller has to be baked into the command string itself. Without
# this, OUT_ROOT silently reverts to the default and the run lands on top of a
# previous one instead of beside it.
ENV_PREFIX=""
for v in OUT_ROOT DATA_ROOT CONFIG_PATH DOM_CONFIG IMAGE_GLOB ITER_MAX \
         SKIP_EXISTING BATCH_SIZE OVERLAP GPUS NPROC_PER_NODE MASTER_PORT \
         PIXEL_CELL_SIZE PIXEL_MIN_VOTES PIXEL_MIN_AREA PIXEL_SUFFIX \
         S5_CONFIGS STAGE_LOG_SUFFIX; do
    [[ -n "${!v:-}" ]] && ENV_PREFIX+="${v}=$(printf '%q' "${!v}") "
done

CMD="${ENV_PREFIX}./scripts/run_station_exp.sh '$STATION' ${STAGES[*]}"
[[ -n "$ENV_PREFIX" ]] && echo "  env: ${ENV_PREFIX}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Reuse the session rather than refusing: after a stage set finishes the
    # pane drops to an interactive bash and is ready for the next batch.
    # Anything still running there must not be disturbed.
    if pgrep -f "run_station_exp.sh $STATION" > /dev/null; then
        echo "session '$SESSION' is busy running $STATION -- attach with: tmux attach -t $SESSION" >&2
        exit 1
    fi

    # That interactive bash re-sourced the profile and is back on (base), so
    # the env has to be re-activated before the command will import anything.
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    sleep 3
    tmux send-keys -t "$SESSION" "$CMD" Enter

    echo "reused tmux session '$SESSION': $STATION ${STAGES[*]}"
    echo "  attach: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" \
    "source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
     echo \"[\$(date +%F' '%T)] $STATION stages: ${STAGES[*]} (env=\$CONDA_DEFAULT_ENV)\" && \
     ${CMD}; \
     echo; echo '=== stages exited (rc='\$?') -- Ctrl-d to close ==='; exec bash"

echo "started tmux session '$SESSION': $STATION ${STAGES[*]}"
echo "  attach: tmux attach -t $SESSION"
