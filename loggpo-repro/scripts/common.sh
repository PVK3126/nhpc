# Shared settings for the job scripts.  Edit this file for your cluster.

# --- environment ----------------------------------------------------------
# Load your MPI + Python here, e.g.
#   module load openmpi/4.1.5 python/3.10
# (left empty on purpose: module names differ between clusters)
load_env() {
    :
}

# --- how MPI jobs are started -----------------------------------------------
#   srun   : Slurm (default; if srun cannot start your MPI, try SRUN_MPI=pmix)
#   ompi   : Open MPI mpirun        (set HOSTFILE when not under Slurm)
#   mpich  : MPICH / Intel MPI mpirun (Hydra)
LAUNCHER=${LAUNCHER:-srun}
SRUN_MPI=${SRUN_MPI:-}
HOSTFILE=${HOSTFILE:-}

# launch NODES NTASKS TASKS_PER_NODE command...
launch() {
    local N=$1 n=$2 ppn=$3
    shift 3
    case "$LAUNCHER" in
        srun)
            srun ${SRUN_MPI:+--mpi=$SRUN_MPI} --nodes="$N" --ntasks="$n" --ntasks-per-node="$ppn" "$@" ;;
        ompi)
            mpirun -np "$n" --map-by "ppr:${ppn}:node" --bind-to core ${HOSTFILE:+--hostfile "$HOSTFILE"} "$@" ;;
        mpich|intel)
            mpirun -n "$n" -ppn "$ppn" ${HOSTFILE:+-f "$HOSTFILE"} "$@" ;;
        *)
            echo "unknown LAUNCHER=$LAUNCHER" >&2; return 1 ;;
    esac
}

# --- experiment sizes -------------------------------------------------------
NODES=${NODES:-4}
# process counts for the application runs, as in the paper's Fig. 6:
#   CG: 8 16 32 64        Sweep3D: 4 8 16 32
# spread evenly over $NODES nodes (P=64 on 4 nodes = 16 ranks/node, so the
# nodes need >= 16 cores).  CG needs a power of two.
CG_PLIST=${CG_PLIST:-"8 16 32 64"}
SWEEP_PLIST=${SWEEP_PLIST:-"4 8 16 32"}
CG_ARGS=${CG_ARGS:-"-c C"}                    # NPB class C, 75 iterations (paper)
SWEEP_ARGS=${SWEEP_ARGS:-"-g 50,50,50 -i 50"}  # 50x50x50 (paper)
PINGPONG_MAX=${PINGPONG_MAX:-4194304}
OVERLAP_COMP_US=${OVERLAP_COMP_US:-500}       # computation inserted in overlap mode

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RES=${RES:-$ROOT/results}
PY=${PY:-python3}
