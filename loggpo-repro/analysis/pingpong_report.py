#!/usr/bin/env python3
"""Paper Fig. 5: measured vs predicted IMB-style Ping-Pong time per message
size, LogGP (baseline) vs LogGPO.  Uses the MPI_Pcontrol markers written by
bin/pingpong_traced around every message size.

example:
  python3 analysis/pingpong_report.py --csv results/pingpong/inter/pingpong.csv \
      --trace results/pingpong/inter/trace --params results/params_inter.json \
      --out results/pingpong_inter
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loggposim as ls  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--params-intra")
    ap.add_argument("--set", action="append", metavar="KEY=VAL")
    ap.add_argument("--out", required=True, help="output prefix (.csv / .png)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    res = ls.predict(a.trace, a.params, a.params_intra, ls.parse_overrides(a.set))
    for w in res["warnings"]:
        print("warning:", w)
    marks = {name: res[name]["marks"][0] for name in ("LogGP", "LogGPO")}

    out = []
    for r in rows:
        reps, div = int(r["reps"]), int(r["divisor"])
        ms, me = int(r["mark_start"]), int(r["mark_end"])
        row = {"bytes": int(r["bytes"]), "measured_us": float(r["t_usec"])}
        for name in ("LogGP", "LogGPO"):
            mk = marks[name]
            row[name + "_us"] = (mk[me] - mk[ms]) / reps / div * 1e6 if ms in mk and me in mk else float("nan")
            row[name + "_err"] = ls.rel_err(row[name + "_us"], row["measured_us"])
        out.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)

    print("%10s %12s %12s %8s %12s %8s" % ("bytes", "measured", "LogGP", "err%", "LogGPO", "err%"))
    for r in out:
        print("%10d %12.3f %12.3f %8.1f %12.3f %8.1f" % (r["bytes"], r["measured_us"], r["LogGP_us"],
                                                        r["LogGP_err"], r["LogGPO_us"], r["LogGPO_err"]))
    for name in ("LogGP", "LogGPO"):
        errs = [r[name + "_err"] for r in out if r[name + "_err"] == r[name + "_err"]]
        if errs:
            print("%-7s average relative error %.1f %%, max %.1f %%" % (name, sum(errs) / len(errs), max(errs)))
    if rows and rows[0]["mode"] == "pingpong":
        print("paper (Fig. 5) LogGPO average error: 3.2 % (Explorer3, InfiniBand), 3.7 % (Dawning 4000A, Myrinet);")
        print("               LogGP ~3x too high above 4 KB on Explorer3, large errors above 32 KB on Dawning")
    print("wrote", a.out + ".csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available: no plot)")
        return
    xs = [max(r["bytes"], 1) for r in out]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [r["measured_us"] for r in out], "k-o", ms=3, label="Measured")
    ax.plot(xs, [r["LogGP_us"] for r in out], "--s", ms=3, label="LogGP (baseline)")
    ax.plot(xs, [r["LogGPO_us"] for r in out], "-^", ms=3, label="LogGPO")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Time (us)")
    ax.set_title("Ping-Pong: measured vs predicted")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150)
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
