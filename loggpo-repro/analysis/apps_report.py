#!/usr/bin/env python3
"""Paper Fig. 6: measured vs predicted application run time for several
process counts, LogGP (baseline) vs LogGPO.

Looks for trace directories named <app>_P<n> under --root, e.g.
results/apps/cg_P4, results/apps/cg_P8, results/apps/sweep_P16 ...
If <dir>/untraced_time.txt exists (written by the job script from the
untraced run) it is reported as well, to show the tracing overhead.

example:
  python3 analysis/apps_report.py --root results/apps \
      --params results/params_inter.json --params-intra results/params_intra.json \
      --out results/apps_summary
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loggposim as ls  # noqa: E402

# reported in the paper (avg %, max %) and its process counts (Fig. 6)
PAPER = {
    "cg":    {"LogGP": (38.3, 65.3), "LogGPO": (2.4, 6.6), "P": "8 16 32 64"},
    "sweep": {"LogGP": (9.1, 14.1),  "LogGPO": (2.0, 6.8), "P": "4 8 16 32"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--params-intra")
    ap.add_argument("--set", action="append", metavar="KEY=VAL")
    ap.add_argument("--out", required=True, help="output prefix (.csv / .png)")
    a = ap.parse_args()

    runs = []
    for d in sorted(os.listdir(a.root)):
        mt = re.match(r"(.+)_P(\d+)$", d)
        path = os.path.join(a.root, d)
        if mt and os.path.exists(os.path.join(path, "trace.0.txt")):
            runs.append((mt.group(1), int(mt.group(2)), path))
    if not runs:
        sys.exit("no <app>_P<n> trace directories under " + a.root)
    runs.sort()

    rows = []
    for app, P, path in runs:
        res = ls.predict(path, a.params, a.params_intra, ls.parse_overrides(a.set))
        unt = os.path.join(path, "untraced_time.txt")
        untraced = float(open(unt).read().split()[0]) if os.path.exists(unt) else float("nan")
        row = {"app": app, "P": P, "measured_s": res["measured"], "untraced_s": untraced,
               "LogGP_s": res["LogGP"]["time"], "LogGPO_s": res["LogGPO"]["time"]}
        row["LogGP_err"] = ls.rel_err(row["LogGP_s"], row["measured_s"])
        row["LogGPO_err"] = ls.rel_err(row["LogGPO_s"], row["measured_s"])
        rows.append(row)
        for w in res["warnings"]:
            print("warning (%s P=%d): %s" % (app, P, w))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("%-8s %5s %11s %11s %11s %8s %11s %8s" % ("app", "P", "measured", "untraced", "LogGP", "err%",
                                                    "LogGPO", "err%"))
    for r in rows:
        print("%-8s %5d %11.4f %11.4f %11.4f %8.1f %11.4f %8.1f" % (
            r["app"], r["P"], r["measured_s"], r["untraced_s"], r["LogGP_s"], r["LogGP_err"],
            r["LogGPO_s"], r["LogGPO_err"]))
    apps = sorted(set(r["app"] for r in rows))
    ours = {}
    for app in apps:
        for name in ("LogGP", "LogGPO"):
            e = [r[name + "_err"] for r in rows if r["app"] == app]
            ours[(app, name)] = (sum(e) / len(e), max(e))
            print("%-8s %-7s average error %.1f %%, max %.1f %%" % (app, name, sum(e) / len(e), max(e)))

    # Paper, section 3.2.2 / Fig. 6 (Explorer3: Itanium2 + InfiniBand, MVAPICH-0.9.2)
    print("\nComparison with the paper (average / max relative error, %)")
    print("%-8s %-7s %-16s %-16s %s" % ("app", "model", "paper", "this cluster", "process counts (paper | here)"))
    for app in apps:
        ref = PAPER.get(app)
        here_p = " ".join(str(r["P"]) for r in rows if r["app"] == app)
        for name in ("LogGP", "LogGPO"):
            avg, mx = ours[(app, name)]
            p = "%.1f / %.1f" % ref[name] if ref else "n/a"
            print("%-8s %-7s %-16s %-16s %s | %s" % (app, name, p, "%.1f / %.1f" % (avg, mx),
                                                     ref["P"] if ref else "-", here_p))
    print("(different hardware and MPI: compare the LogGP-vs-LogGPO gap, not exact numbers)")
    print("wrote", a.out + ".csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available: no plot)")
        return
    fig, axes = plt.subplots(1, len(apps), figsize=(5 * len(apps), 4), squeeze=False)
    for ax, app in zip(axes[0], apps):
        rs = [r for r in rows if r["app"] == app]
        x = list(range(len(rs)))
        wdt = 0.27
        ax.bar([i - wdt for i in x], [r["measured_s"] for r in rs], wdt, label="Measured", color="0.3")
        ax.bar(x, [r["LogGP_s"] for r in rs], wdt, label="LogGP (baseline)")
        ax.bar([i + wdt for i in x], [r["LogGPO_s"] for r in rs], wdt, label="LogGPO")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["P"]) for r in rs])
        ax.set_xlabel("Number of processes")
        ax.set_ylabel("Time (s)")
        ax.set_title(app)
        ax.legend()
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150)
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
