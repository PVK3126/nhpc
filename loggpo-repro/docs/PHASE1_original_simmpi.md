# Phase 1: Original SIM-MPI / LogGPO, Baseline

**Goal:** get a working copy of the original LogGPO implementation, find where it computes communication cost, record baseline results, and run basic MPI communication experiments on the 4-node cluster.

> **Status (Nov 2026).** Nothing below has been run yet; you run it on the cluster.
> - **The paper's own link is dead.** `http://www.hpctest.org.cn/resources/sim-mpi.tgz` returns HTTP 522.
> - **The only public copy predates the paper.** It is [SIM-MPI-1.0.tgz on SourceForge](https://sourceforge.net/projects/sim-mpi/files/) (GPLv2, 193 KB, files dated 2007). The LogGPO paper is from 2009, so this release may only have the **LogGP** model. Step 1 checks this.
> - **If LogGPO is missing**, use the reimplementation in this repo (`analysis/loggposim.py`, class `LogGPO`) as the LogGPO model. You can also email the corresponding author (cwg@tsinghua.edu.cn, given in the paper) and ask for the 2009 version.

---

## Step 1: Obtain the original SIM-MPI source

```bash
bash scripts/get_original_simmpi.sh
```

The script:
1. Tries the paper's URL, then SourceForge.
2. Saves the archive and its sha256 to `original/download/` and `original/logs/sha256.txt`.
3. Extracts it to `original/src/`.

If the cluster has no internet, download `SIM-MPI-1.0.tgz` in a browser, copy it to `original/download/sim-mpi.tgz`, and run the script again.

**Record:** source URL, sha256, release date, and whether `LogGPO` appears anywhere (the script prints this).

## Step 2: Understand the implementation and execution flow

`original/logs/inventory.txt` lists files, docs, build files, PMPI wrappers, cost/model functions and `main()`. Fill in this table from it:

| Component (from the paper) | What to look for in the original | Equivalent in this repo |
|---|---|---|
| Instrumentation library (collects comm + comp traces) | files that define `MPI_Send` and call `PMPI_Send` | `src/tracer.c` |
| Trace format | `fprintf` / `fwrite` in the wrappers | `src/tracer.c` → `write_trace()` |
| Trace reader | `fopen`/`fscanf` in the simulator | `analysis/loggposim.py:149` `load_trace` |
| Matching of sends/receives | queues keyed by (src, dst, tag) | `load_trace` (`send_msg` / `recv_msg`) |
| Event loop (replays per-rank clocks) | main loop over ranks/events | `loggposim.py:458` `_advance`, `:518` `simulate` |
| **Communication model** | functions using L, o, g, G (and O, S) | `loggposim.py:345` `LogGP`, `:365` `LogGPO` |
| Parameter input | config file or command-line options | `params_*.json` from `src/params_bench.c` |

The execution flow to confirm is the paper's (section 3.2.2):

```
app linked with instrumentation lib ──run on cluster──> per-rank traces
    (MPI events + timestamps; computation = gap between MPI calls)
traces + model parameters ──simulator──> predicted time per rank
    each rank: clock += computation;  at each MPI event: clock += model cost,
    blocking calls wait for the matching message's arrival time
```

## Step 3: Find where LogGPO calculates communication cost

Look in the original for code that implements the section 2.2 formulas. The variable names will differ, so grep for the arithmetic. The rendezvous total has the shape `4*O + 3*L + Oi + G*(K-1)`.

| Paper formula | Meaning | Reimplementation (`analysis/loggposim.py`) |
|---|---|---|
| Eager `K ≤ S`: `Os = Oi`, `Or = Oc`, `Tc = Oi + G(K−1) + L + Oc` | send cost, recv copy, arrival | `post_send` (l.369) cost `Oi(K)`; `item` (l.403) receive branch `arr = Ts_end + G(K-1) + L`, cost `Oc(K)` |
| Rendezvous, DCP/RAT, case (a) `Tr > Ts+O+L` | ACK sent by `MPI_Irecv` | `_dcp_progress` (l.384): `prog = Tr` |
| Rendezvous, DCP/RAT, case (b) `Tr ≤ Ts+O+L` | ACK only when receiver is inside MPI (its `MPI_Wait`) | `_dcp_progress`: `prog = max(Trw, req_arr)` |
| `Osw = (Tr−Tsw) + 2O + L + Oi + G(K−1)` | sender blocked in `MPI_Wait` until ACK, then transfers | `item` send branch: ready `ack_arr`, cost `O + Oi + G(K-1)` |
| Rendezvous, ICP/RTA: `Os = 2O`, `Or = 2O + Oi`, `Tc = 4O + 3L + Oi + G(K−1)` | handshake progressed independently | `_icp_times` (l.376) |
| `Ro = comp / (comp + O)` | overlap ratio | follows from the simulated clocks; not a separate input |
| LogGP baseline | `os(K) + G(K−1) + L + or(K)`, full overlap assumed | class `LogGP` (l.345–363) |

The simulator is checked against these closed-form formulas by `tests/test_sim.py` (11 checks).

## Step 4: Compile and execute the existing simulator

`get_original_simmpi.sh` tries `./configure` (if present) and then `make` with `mpicc`. Output goes to `original/logs/build.log`, and the list of built binaries and libraries to `original/logs/built_artifacts.txt`.

The code dates from 2007, so modern compilers may reject it. Common fixes:

| Build error | Fix |
|---|---|
| `conflicting types for 'MPI_Send'` (missing `const` in wrapper prototypes) | add `const` to send buffers, or build with `-DMPICH_IGNORE_CXX_SEEK` / older `-std=gnu89` |
| multiple definition of globals (gcc ≥ 10) | add `CFLAGS += -fcommon` |
| implicit function declaration errors | `CFLAGS += -Wno-error=implicit-function-declaration` |
| `mpi.h` not found | `module load` your MPI and build with `CC=mpicc` |

Write down every change you make. Then run its bundled example, or a ping-pong traced with its instrumentation library, exactly as its README describes.

**Repo-side simulator (always available):**
```bash
make                      # bin/, lib/libloggpotrace.so
python3 tests/test_sim.py # formula self-test, no MPI needed
```

## Step 5: Record baseline results

"Baseline" means two things here, and the paper compares against both:

1. **Measured times.** Real runs without tracing (`LOGGPO_TRACE=0`), plus traced runs to check the tracing overhead.
2. **The LogGP model.** The paper's comparison model, with parameters measured the way logp_mpi does it (`os(K)`, `or(K)`, `L`, `G`).

```bash
STEPS="params" sbatch scripts/run_all.sbatch            # LogGP + LogGPO parameters
STEPS="pingpong apps analyze" sbatch scripts/run_all.sbatch
```

Fill this table from `results/params_inter.log`, `results/report/*.txt`, and the original simulator's output if it built:

| Item | Value |
|---|---|
| Cluster / interconnect / MPI version | |
| L, G (MB/s), O, S (bytes), progress (DCP/ICP), inter-node | |
| same, intra-node | |
| Ping-pong avg error: LogGP / LogGPO (paper Explorer3: ~3x over for >4 KB / 3.2 %) | |
| CG class C, P=8,16,32,64: measured, LogGP err, LogGPO err (paper: 38.3 % / 2.4 % avg) | |
| Sweep 50³, P=4,8,16,32: measured, LogGP err, LogGPO err (paper: 9.1 % / 2.0 % avg) | |
| Tracing overhead (traced vs untraced time) | |
| Original SIM-MPI prediction for the same traces (if it ran) | |

## Step 6: Basic MPI communication experiments

All are in `scripts/run_all.sbatch`:

| Experiment | Binary | Placement | What it shows |
|---|---|---|---|
| Parameter benchmark | `bin/params_bench` | 2 nodes × 1, 1 node × 2 | L, G, O, Oi(K), Oc(K), os/or(K), eager limit S, DCP vs ICP |
| IMB-style ping-pong, 0 B–4 MB | `bin/pingpong_traced -m pingpong` | inter + intra | paper Fig. 5 |
| Overlap exchange (Isend/Irecv, computation, Waitall) | `bin/pingpong_traced -m overlap -c 500` | inter + intra | paper Fig. 1: how much of the computation actually overlaps |

What to expect:
- Below `S`, LogGP and LogGPO should be close.
- Above `S` with **DCP**, LogGP over-predicts the ping-pong, because it counts the transfer inside both os(K) and G·K.
- With DCP, the overlap experiment shows the transfer happening **after** the computation, which LogGP cannot capture.
- Many modern stacks (UCX, RDMA-read rendezvous) behave like **ICP**. In that case the gap between LogGP and LogGPO will be smaller than in the 2009 paper. That is a real result to report, not a bug.

## Conditions for calling this a reproduction

| | Condition | How it is met | Evidence file |
|---|---|---|---|
| A | **Source verification:** does SIM-MPI-1.0 contain LogGPO? | Run Step 1. Then read the cost code (Step 3) and set `original_has_loggpo` to yes/no | `original/logs/decision.txt` |
| B | **Working baseline** | If **yes**: the original builds and runs (Step 4), and our simulator is a cross-check. If **no**: LogGPO results come from **our implementation** (`analysis/loggposim.py`, class `LogGPO`), verified by `tests/test_sim.py`. The original serves only as the LogGP / SIM-MPI reference. The report must say which. | `original/logs/build.log`, test output |
| C | **Experimental results** | `run_all.sbatch`: ping-pong, overlap, CG at P = 8 16 32 64, Sweep at P = 4 8 16 32 (the paper's Fig. 6 counts) | `results/report/*` |
| D | **Comparison with the paper** | Printed automatically at the end of each report (table below) | `results/report/apps.txt`, `pingpong_*.txt` |

| Experiment | Paper LogGPO error | Paper LogGP error | This cluster |
|---|---|---|---|
| Ping-pong, Explorer3 (InfiniBand) | 3.2 % avg | ~3x over above 4 KB | |
| Ping-pong, Dawning 4000A (Myrinet) | 3.7 % avg | large above 32 KB | |
| CG class C, P = 8–64 | 2.4 % avg, 6.6 % max | 38.3 % avg, 65.3 % max | |
| Sweep3D 50³, P = 4–32 | 2.0 % avg, 6.8 % max | 9.1 % avg, 14.1 % max | |

The original 2009 platforms (Explorer3, Dawning 4000A) can't be reproduced, so the absolute errors will differ. What should carry over:
- LogGPO stays accurate.
- LogGP's error grows when messages are large and above the eager limit (CG).
- LogGP's error stays small for small messages (Sweep).
- With an ICP-like modern MPI, the gap may shrink. Report that finding rather than tuning parameters to match the paper.

P=64 on 4 nodes needs at least 16 cores per node. If your nodes have fewer, run `CG_PLIST="8 16 32"` and note the missing point.

## Deliverable checklist

- [ ] Archive, sha256 and source URL recorded (`original/logs/`)
- [ ] Component table (Step 2) filled in
- [ ] LogGPO cost code located in the original, **or** documented as absent from SIM-MPI-1.0
- [ ] Original builds and runs (with patch notes), **or** build failure documented
- [ ] `results/params_*.json` + baseline table (Step 5)
- [ ] Ping-pong and overlap reports (`results/report/`)
