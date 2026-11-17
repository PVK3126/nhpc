#!/usr/bin/env python3
"""Predict the run time of one traced run with LogGP (baseline) and LogGPO.

example:
  python3 analysis/predict.py --trace results/apps/cg_P16 \
      --params results/params_inter.json --params-intra results/params_intra.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loggposim as ls  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="trace directory (trace.<r>.txt + meta.txt)")
    ap.add_argument("--params", required=True, help="inter-node parameters (params_bench output)")
    ap.add_argument("--params-intra", help="intra-node parameters (optional)")
    ap.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="override a parameter, e.g. --set S=65536 --set progress=ICP")
    ap.add_argument("--comp-scale", type=float, default=1.0,
                    help="scale computation (e.g. 0.5 predicts a CPU twice as fast)")
    ap.add_argument("--measured", type=float, help="override the measured time (seconds)")
    ap.add_argument("--json", help="append the result as one JSON line to this file")
    a = ap.parse_args()

    res = ls.predict(a.trace, a.params, a.params_intra, ls.parse_overrides(a.set), a.comp_scale)
    meas = a.measured if a.measured is not None else res["measured"]
    for w in res["warnings"]:
        print("warning:", w)
    print("trace     : %s  (P=%d)" % (a.trace, res["nprocs"]))
    print("measured  : %s" % ("%.6f s" % meas if meas else "n/a"))
    for name in ("LogGP", "LogGPO"):
        r = res[name]
        extra = ("  (%d DCP handshakes resolved while both peers were blocked in MPI)" % r["forced"]
                 if r["forced"] else "")
        print("%-9s : %.6f s   error %6.2f %%%s" % (name, r["time"], ls.rel_err(r["time"], meas), extra))
    print("max comp  : %.6f s" % res["LogGPO"]["comp_time"])
    if a.json:
        row = {"trace": a.trace, "P": res["nprocs"], "measured": meas,
               "LogGP": res["LogGP"]["time"], "LogGPO": res["LogGPO"]["time"]}
        with open(a.json, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
