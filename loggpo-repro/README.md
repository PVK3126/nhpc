# LogGPO reproduction

Reproduces Chen, Zhai, Zhang, Zheng, *"LogGPO: An accurate communication model for performance prediction of MPI programs"*, Sci China Ser F 52(10), 2009, on a Slurm cluster with 4 nodes.

**Phase 1** (getting the original SIM-MPI, baseline results, basic experiments): see [`docs/PHASE1_original_simmpi.md`](docs/PHASE1_original_simmpi.md).

## What is reproduced

| Paper | Here |
|---|---|
| SIM-MPI instrumentation library | `src/tracer.c`, a PMPI tracer (linked into `bin/*_traced` or `LD_PRELOAD=lib/libloggpotrace.so`) |
| SIM-MPI trace-driven simulator | `analysis/loggposim.py` |
| LogGP model (**baseline**), parameters from logp_mpi | class `LogGP` + `os(K)`, `or(K)`, `L`, `G` from `bin/params_bench` |
| LogGPO model (section 2) | class `LogGPO`: eager / rendezvous, DCP (RAT) / ICP (RTA), `O`, `Oi(K)`, `Oc(K)`, `S` |
| IMB-3.0 Ping-Pong (Fig. 5) | `bin/pingpong -m pingpong` |
| Overlap example (Fig. 1) | `bin/pingpong -m overlap -c <us>` |
| NPB CG class C (Fig. 6a) | `bin/cg_mini -c C`: NPB CG structure and message pattern, in C |
| Sweep3D 50×50×50 (Fig. 6b) | `bin/sweep_mini`: KBA wavefront, 2–3 KB messages |

## On the cluster

```bash
# 1. edit scripts/common.sh: load_env() (module load ...), LAUNCHER, PLIST
bash scripts/build.sh                         # make + simulator self-test
sbatch scripts/run_all.sbatch                 # params, ping-pong, apps, analysis
# results/report/{pingpong,overlap}_{inter,intra}.{txt,csv,png}, results/report/apps.*
```

- Run individual steps with `STEPS="params"`, `STEPS="apps analyze"`, and so on.
- Process counts follow the paper's Fig. 6: `CG_PLIST="8 16 32 64"` and `SWEEP_PLIST="4 8 16 32"`. P=64 means 16 ranks per node. Problem sizes are set with `CG_ARGS` and `SWEEP_ARGS`.
- **LogGPO source:** if `original/logs/decision.txt` shows that SIM-MPI-1.0 has no LogGPO, all LogGPO results come from this repo's implementation (`analysis/loggposim.py`, class `LogGPO`). State that in the report.
- `bash scripts/analyze.sh` re-runs the predictions on the login node (Python only). To use the eager limit and progress strategy from your MPI's documentation instead of the measured ones, add `SIMOPTS="--set S=... --set progress=DCP"`.
- Plots need matplotlib (`pip install --user matplotlib`). Without it you still get the CSV and text tables.

Single prediction:
```bash
python3 analysis/predict.py --trace results/apps/cg_P16 \
    --params results/params_inter.json --params-intra results/params_intra.json
```

Tracing your own C/C++ MPI program:
```bash
srun -N4 -n16 env LD_PRELOAD=$PWD/lib/libloggpotrace.so LOGGPO_TRACE_DIR=$PWD/mytrace ./app
```

## Layout

```
src/        tracer.c params_bench.c pingpong.c cg_mini.c sweep_mini.c
analysis/   loggposim.py predict.py pingpong_report.py apps_report.py
scripts/    common.sh build.sh run_all.sbatch analyze.sh get_original_simmpi.sh
tests/      test_sim.py      (simulator vs. paper's closed-form formulas)
docs/       PHASE1_original_simmpi.md
```

## Deviations and limitations (be explicit about these in your report)

- **Not the original SIM-MPI.** This is a reimplementation of the 2009 simulator, whose download link no longer works (see Phase 1).
- **Paper formulas.** The paper's per-end formulas and its total cost `Tc` differ by one `O` in places. The simulator keeps `Tc` exact.
- **Paper case split.** The DCP case split (ACK sent in `MPI_Irecv`, or in the next blocking MPI call) is generalized to "the next interval the receiver spends inside MPI".
- **CG and Sweep are not NPB CG or Sweep3D.** NPB CG is Fortran and Sweep3D isn't distributed here, so `cg_mini` and `sweep_mini` reproduce their communication structure. CG uses a random SPD matrix with NPB's sizes, so its zeta does not match NPB's verification value.
- **Collectives** on `MPI_COMM_WORLD` are modelled as point-to-point steps: dissemination for barrier/allreduce, binomial trees for bcast/reduce. Collectives on sub-communicators keep their measured time.
- **Tracer scope.** The tracer covers C/C++ calls, not Fortran bindings or persistent requests.
- **Intra- vs inter-node.** Mixed placements use separate parameter sets, chosen by hostname. The paper used one set.
- **Testing done so far.** The C code was compile-checked, and the whole pipeline was run once on a fake socket-based MPI (CG/Sweep give identical results for P=1–8). It has **not** run on a real MPI yet, and no accuracy numbers exist yet.
