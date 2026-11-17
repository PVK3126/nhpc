#!/bin/bash
# Predictions with LogGP (baseline) and LogGPO + error tables / plots.
# Runs on the login node too (pure Python):  bash scripts/analyze.sh
# Extra simulator options can be passed with SIMOPTS, e.g.
#   SIMOPTS="--set S=12288 --set progress=DCP" bash scripts/analyze.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
SIMOPTS=${SIMOPTS:-}
P_INTER="$RES/params_inter.json"
P_INTRA="$RES/params_intra.json"
[ -f "$P_INTER" ] || { echo "missing $P_INTER (run the params step)"; exit 1; }
INTRA=""; [ -f "$P_INTRA" ] && INTRA="--params-intra $P_INTRA"
mkdir -p "$RES/report"

for place in inter intra; do
    for kind in pingpong overlap; do
        d="$RES/$kind/$place"
        [ -f "$d/pingpong.csv" ] || continue
        echo "== $kind ($place)"
        $PY analysis/pingpong_report.py --csv "$d/pingpong.csv" --trace "$d/trace" \
            --params "$P_INTER" $INTRA $SIMOPTS --out "$RES/report/${kind}_$place" \
            | tee "$RES/report/${kind}_$place.txt"
    done
done

if [ -d "$RES/apps" ]; then
    echo "== applications"
    $PY analysis/apps_report.py --root "$RES/apps" --params "$P_INTER" $INTRA $SIMOPTS \
        --out "$RES/report/apps" | tee "$RES/report/apps.txt"
fi
echo "== reports in $RES/report"
