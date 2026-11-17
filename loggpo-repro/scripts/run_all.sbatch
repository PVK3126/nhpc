#!/bin/bash
#SBATCH --job-name=loggpo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=loggpo-%j.out
#
# Full reproduction on 4 nodes:
#   1. measure LogGP (baseline) and LogGPO parameters   (inter- and intra-node)
#   2. Ping-Pong, measured and traced                    (paper Fig. 5)
#   3. CG (P = 8 16 32 64) and Sweep (P = 4 8 16 32), untraced + traced (Fig. 6)
#   4. predictions with LogGP (baseline) and LogGPO, error tables, plots,
#      comparison with the paper's reported errors
#
#   sbatch scripts/run_all.sbatch
#   STEPS="apps analyze" CG_PLIST="8 16" sbatch scripts/run_all.sbatch
#
# --ntasks-per-node must be >= the largest P / NODES (64/4 = 16).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/common.sh
load_env
STEPS=${STEPS:-"params pingpong apps analyze"}
mkdir -p "$RES"
echo "== LogGPO reproduction: steps [$STEPS], nodes $NODES, CG P=[$CG_PLIST], Sweep P=[$SWEEP_PLIST], results in $RES"
command -v mpicc >/dev/null && mpicc --version | head -1 || true

has() { [[ " $STEPS " == *" $1 "* ]]; }

# 1. parameters ------------------------------------------------------------------
if has params; then
    echo "== [1] parameters, inter-node (2 nodes x 1 rank)"
    launch 2 2 1 bin/params_bench "$RES/params_inter.json" | tee "$RES/params_inter.log"
    echo "== [1] parameters, intra-node (1 node x 2 ranks)"
    launch 1 2 2 bin/params_bench "$RES/params_intra.json" | tee "$RES/params_intra.log"
fi

# 2. ping-pong ----------------------------------------------------------------------
if has pingpong; then
    for place in inter intra; do
        if [ $place = inter ]; then N=2; ppn=1; else N=1; ppn=2; fi
        d="$RES/pingpong/$place"; rm -rf "$d"; mkdir -p "$d"
        echo "== [2] ping-pong ($place)"
        launch $N 2 $ppn env LOGGPO_TRACE_DIR="$d/trace" \
            bin/pingpong_traced -m pingpong -x "$PINGPONG_MAX" -o "$d/pingpong.csv"
        d2="$RES/overlap/$place"; rm -rf "$d2"; mkdir -p "$d2"
        echo "== [2] overlap exchange ($place, ${OVERLAP_COMP_US} us computation)"
        launch $N 2 $ppn env LOGGPO_TRACE_DIR="$d2/trace" \
            bin/pingpong_traced -m overlap -c "$OVERLAP_COMP_US" -x "$PINGPONG_MAX" -o "$d2/pingpong.csv"
    done
fi

# 3. applications ------------------------------------------------------------------
if has apps; then
    for app in cg sweep; do
        if [ $app = cg ]; then exe=cg_mini; args=$CG_ARGS; plist=$CG_PLIST
        else exe=sweep_mini; args=$SWEEP_ARGS; plist=$SWEEP_PLIST; fi
        for P in $plist; do
            ppn=$(( (P + NODES - 1) / NODES ))
            N=$(( P < NODES ? P : NODES ))
            d="$RES/apps/${app}_P$P"; rm -rf "$d"; mkdir -p "$d"
            echo "== [3] $app P=$P ($N nodes x $ppn): untraced baseline run"
            launch $N "$P" $ppn env LOGGPO_TRACE=0 bin/$exe $args | tee "$d/untraced.log"
            grep RESULT "$d/untraced.log" | sed 's/.*time=\([0-9.eE+-]*\).*/\1/' > "$d/untraced_time.txt"
            echo "== [3] $app P=$P: traced run"
            launch $N "$P" $ppn env LOGGPO_TRACE_DIR="$d" bin/${exe}_traced $args | tee "$d/traced.log"
        done
    done
fi

# 4. analysis -------------------------------------------------------------------------
if has analyze; then
    bash scripts/analyze.sh
fi
echo "== done"
