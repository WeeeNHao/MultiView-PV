#!/usr/bin/env bash
# One-shot progress view across all three stations. Safe to run at any time --
# it only reads the output tree and never touches a running experiment.
#
#   ./scripts/exp_status.sh            # snapshot
#   watch -n30 ./scripts/exp_status.sh # live
set -uo pipefail

ZS_ROOT="${ZS_ROOT:-/data/dataset/PV/ZS_PV}"
STATIONS=("001-BeiOu" "003-XinXie" "004-CangFang")
ITER_MAX="${ITER_MAX:-3}"

count_shp() { compgen -G "$1/*.shp" > /dev/null && ls -1 "$1"/*.shp 2>/dev/null | wc -l || echo 0; }

printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'

printf '\n=== running pipelines ===\n'
if ! pgrep -af "run_pipeline.py" | sed 's/^/  /' ; then
    echo "  (none)"
fi

printf '\n=== stations ===\n'
for st in "${STATIONS[@]}"; do
    out="${ZS_ROOT}/${st}-exp"
    printf '\n  %s  (%s)\n' "$st" "$out"
    if [[ ! -d "$out" ]]; then
        echo "    not started"
        continue
    fi

    # S0
    if [[ -f "${out}/_preflight/smoke.shp" ]]; then s0="done"; else s0="--"; fi

    # S1 shared cache
    n_infer=$(count_shp "${out}/iter_0/shared/infer")
    n_proj=$(count_shp "${out}/iter_0/shared/proj")

    # S2 iter_0 postprocess over the full direction set
    if [[ -f "${out}/iter_0/full/final.shp" ]]; then s2="done"; else s2="--"; fi

    # S3 iterations
    iters=""
    for ((t = 1; t <= ITER_MAX; t++)); do
        [[ -f "${out}/iter_${t}/full/final.shp" ]] && iters="${iters}${t} "
    done
    [[ -z "$iters" ]] && iters="--"

    # S4 camera-direction sets (table 7)
    n_dirs=0
    for d in "${out}"/iter_0/dirs/*/; do
        [[ -f "${d}final.shp" ]] && n_dirs=$((n_dirs + 1))
    done

    printf '    S0 preflight : %s\n' "$s0"
    printf '    S1 cache     : infer=%s proj=%s\n' "$n_infer" "$n_proj"
    printf '    S2 iter_0    : %s\n' "$s2"
    printf '    S3 iters     : %s(of 1..%s)\n' "$iters" "$ITER_MAX"
    printf '    S4 dir sets  : %s/5 done\n' "$n_dirs"

    # Last line of whichever stage log was touched most recently.
    last_log=$(ls -t "${out}/_logs/"*.log 2>/dev/null | head -1)
    if [[ -n "$last_log" ]]; then
        printf '    last log     : %s\n' "$(basename "$last_log")"
        tail -n 1 "$last_log" 2>/dev/null | sed 's/^/      | /'
    fi
done

printf '\n=== eval ===\n'
for f in "${ZS_ROOT}"/eval_*/all_stations_summary.csv; do
    [[ -f "$f" ]] || continue
    printf '  %s (%s rows, %s)\n' "$f" "$(($(wc -l < "$f") - 1))" \
        "$(date -r "$f" '+%m-%d %H:%M')"
done

printf '\n'
