#!/bin/bash
# Phase 1 -- obtain, inspect, build and smoke-test the ORIGINAL SIM-MPI.
# Run on the cluster login node:   bash scripts/get_original_simmpi.sh
# Everything goes into ./original/ ; logs into ./original/logs/.
#
# Sources:
#   paper link (2009): http://www.hpctest.org.cn/resources/sim-mpi.tgz   (HTTP 522 in Nov 2026)
#   SourceForge      : https://sourceforge.net/projects/sim-mpi/  -> SIM-MPI-1.0.tgz (2007, GPLv2)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
load_env
O=$PWD/original; mkdir -p "$O/logs" "$O/download"; cd "$O"

# ---- 1. obtain --------------------------------------------------------------
fetch() { curl -fL --connect-timeout 20 -o "$2" "$1" && tar tzf "$2" >/dev/null 2>&1; }
if [ ! -s download/sim-mpi.tgz ]; then
    fetch http://www.hpctest.org.cn/resources/sim-mpi.tgz download/sim-mpi.tgz \
    || fetch "https://sourceforge.net/projects/sim-mpi/files/latest/download" download/sim-mpi.tgz \
    || { echo "download failed: fetch SIM-MPI-1.0.tgz by hand from"
         echo "  https://sourceforge.net/projects/sim-mpi/files/  into $O/download/sim-mpi.tgz"; exit 1; }
fi
sha256sum download/sim-mpi.tgz | tee logs/sha256.txt
rm -rf src && mkdir src && tar xzf download/sim-mpi.tgz -C src
SRC=$(find src -maxdepth 2 -type d | sort | sed -n 2p); SRC=${SRC:-src}
echo "source tree: $SRC"

# ---- 2. understand: inventory + where the communication cost is computed -----
{
  echo "== files";            find "$SRC" -type f | sort
  echo "== docs";             find "$SRC" -iname 'readme*' -o -iname '*.txt' -o -iname '*.pdf' -o -iname '*.doc*' | sort
  echo "== build files";      find "$SRC" -name 'Makefile*' -o -name 'configure*' -o -name '*.mk' | sort
  echo "== PMPI wrappers (trace collection)"
  grep -rnE '\bPMPI_[A-Z][a-z_]+\s*\(' "$SRC" --include='*.[ch]' | head -40
  echo "== model / cost functions (LogGP, LogGPO, overhead, latency, gap, protocol)"
  grep -rniE 'loggpo|loggp|logp|overhead|latency|\bgap\b|eager|rendez|progress|\b(Os|Or|Oi|Oc|Tc)\b' \
       "$SRC" --include='*.[ch]' --include='*.cpp' --include='*.cc' | head -150
  echo "== parameter / config files"
  find "$SRC" -iname '*.conf' -o -iname '*.cfg' -o -iname '*param*' -o -iname '*.ini' | sort
  echo "== main()"
  grep -rn 'int main' "$SRC" --include='*.[ch]' --include='*.c*'
} > logs/inventory.txt 2>&1
echo "inventory -> $O/logs/inventory.txt"
# Decision rule (condition A):
#   original contains LogGPO -> run the original as the reference, and use
#                               analysis/loggposim.py only as a cross-check
#   original lacks LogGPO    -> the LogGPO results are from OUR implementation
#                               (analysis/loggposim.py, class LogGPO); the
#                               original is used only as the LogGP / SIM-MPI reference
# The keyword hit is a hint only: confirm by reading the cost code before claiming either.
if grep -qiE 'loggpo|rendez|eager' logs/inventory.txt; then HINT=yes; else HINT=no; fi
cat > logs/decision.txt <<EOF
archive_sha256      : $(cut -d' ' -f1 logs/sha256.txt)
loggpo_keyword_hint : $HINT
original_has_loggpo : UNVERIFIED   <- set to yes/no after reading the cost code (docs/PHASE1, step 3)
loggpo_used         : $( [ $HINT = yes ] && echo "original (pending verification)" || echo "reimplementation analysis/loggposim.py:LogGPO" )
EOF
cat logs/decision.txt
if [ $HINT = no ]; then
    echo "NOTE: no LogGPO / protocol code found -- this release likely predates the paper (LogGP only)."
    echo "      LogGPO results will come from the reimplementation; say so in the report."
fi

# ---- 3. compile ---------------------------------------------------------------
cd "$SRC"
( if [ -x configure ]; then ./configure CC=mpicc; fi
  make CC="${MPICC:-mpicc}" MPICC="${MPICC:-mpicc}" ) > "$O/logs/build.log" 2>&1
echo "build exit $? -> $O/logs/build.log   (2007 code: expect fixes for modern gcc/MPI, see docs/PHASE1_original_simmpi.md)"
find . -type f \( -perm -u+x -o -name '*.a' -o -name '*.so' \) -newer "$O/download/sim-mpi.tgz" | tee "$O/logs/built_artifacts.txt"
