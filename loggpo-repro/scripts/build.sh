#!/bin/bash
# Build everything and run the simulator self-test.  Run on the login node.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
load_env
make -j "${MAKEFLAGS_J:-4}" MPICC="${MPICC:-mpicc}"
python3 tests/test_sim.py
echo "build OK -- now: sbatch scripts/run_all.sbatch"
