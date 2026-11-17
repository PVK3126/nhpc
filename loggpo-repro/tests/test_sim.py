#!/usr/bin/env python3
"""Checks the simulator against the closed-form formulas of the paper
(section 2.2) on small synthetic traces.  Run:  python3 tests/test_sim.py"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import loggposim as ls  # noqa: E402

us = 1e-6
P = dict(L=5 * us, G=1e-9, O=1 * us, S=8192, progress="DCP",
         sizes=[1, 1024, 8192, 8193, 65536, 1048576],
         os=[2 * us, 3 * us, 6 * us, 30 * us, 80 * us, 1000 * us],
         Oi=[1.5 * us, 2 * us, 4 * us, 3 * us, 5 * us, 20 * us],
         Oc=[0.5 * us, 1 * us, 3 * us, 0, 0, 0])
P["or"] = [2 * us, 3 * us, 6 * us, 30 * us, 80 * us, 1000 * us]
P["oneway"] = [0] * 6


def write_trace(d, ranks, hosts=None):
    os.makedirs(d, exist_ok=True)
    for r, lines in enumerate(ranks):
        with open(os.path.join(d, "trace.%d.txt" % r), "w") as f:
            f.write("# test\n" + "\n".join(lines) + "\n")
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write("nprocs %d\nmeasured_time 1.0\n" % len(ranks))
        for r in range(len(ranks)):
            f.write("host %d %s\n" % (r, (hosts or ["n%d" % i for i in range(len(ranks))])[r]))


def run(tmp, ranks, **over):
    d = tempfile.mkdtemp(dir=tmp)
    write_trace(d, ranks)
    pp = dict(P, **over)
    pf = os.path.join(d, "p.json")
    json.dump(pp, open(pf, "w"))
    return ls.predict(d, pf)


def one_message(K, comp_s=0.0, recv_delay=0.0):
    """rank0: Isend(K) ; comp ; Wait      rank1: [delay] Irecv ; Wait"""
    r0 = ["ISEND 0 0 1 7 %d 1" % K, "WAIT %.9f %.9f 1 1 -2 -2" % (comp_s, comp_s), "END %.9f" % comp_s]
    r1 = ["IRECV %.9f %.9f 0 7 %d 2" % (recv_delay, recv_delay, K),
          "WAIT %.9f %.9f 1 2 0 7" % (recv_delay, recv_delay), "END %.9f" % recv_delay]
    return [r0, r1]


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol + 1e-9 * abs(b)


def main():
    tmp = tempfile.mkdtemp()
    ps = ls.ParamSet(dict(P))
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = close(got, want)
        fails += not ok
        print("%-58s %s  got %.3f us  want %.3f us" % (name, "ok  " if ok else "FAIL", got / us, want / us))

    # eager: Tc = Oi + G(K-1) + L + Oc
    K = 1024
    res = run(tmp, one_message(K))
    check("LogGPO eager Tc", res["LogGPO"]["rank_times"][1], ps.Oi(K) + P["G"] * (K - 1) + P["L"] + ps.Oc(K))
    check("LogGP  eager Tc", res["LogGP"]["rank_times"][1], ps.os(K) + P["G"] * (K - 1) + P["L"] + ps.or_(K))

    # rendezvous, DCP, receiver early (case b): Tc = 4O + 3L + Oi + G(K-1)
    K = 65536
    Tc = 4 * P["O"] + 3 * P["L"] + ps.Oi(K) + P["G"] * (K - 1)
    res = run(tmp, one_message(K))
    check("LogGPO rendezvous DCP Tc (receiver waiting)", res["LogGPO"]["rank_times"][1], Tc)

    # DCP: sender computes 1 ms between Isend and Wait -> no overlap of transfer
    D = 1e-3
    res = run(tmp, one_message(K, comp_s=D))
    want = D + 2 * P["O"] + ps.Oi(K) + P["G"] * (K - 1) + P["L"]   # Isend O + ACK processing O
    check("LogGPO DCP: transfer delayed until sender's Wait", res["LogGPO"]["rank_times"][1], want)
    check("LogGP  assumes full overlap (receiver)", res["LogGP"]["rank_times"][1],
          ps.os(K) + P["G"] * (K - 1) + P["L"] + ps.or_(K))

    # ICP: transfer overlaps with the sender's computation
    res = run(tmp, one_message(K, comp_s=D), progress="ICP")
    check("LogGPO ICP Tc with sender computing", res["LogGPO"]["rank_times"][1], Tc)
    check("LogGPO ICP sender: Os = 2O (Wait costs O)", res["LogGPO"]["rank_times"][0], D + 2 * P["O"])

    # DCP case (a): receiver posts Irecv after REQ arrived -> ACK from Irecv
    late = 100 * us
    res = run(tmp, one_message(K, recv_delay=late))
    want = late + 2 * P["O"] + P["L"] + P["O"] + ps.Oi(K) + P["G"] * (K - 1) + P["L"]
    check("LogGPO DCP case (a) receiver end", res["LogGPO"]["rank_times"][1], want)

    # blocking ping-pong of 10 round trips, rendezvous: per one-way time = Tc
    n = 10
    r0, r1 = [], []
    for i in range(n):
        r0 += ["SEND 0 0 1 1 %d" % K, "RECV 0 0 1 1 %d" % K]
        r1 += ["RECV 0 0 0 1 %d" % K, "SEND 0 0 0 1 %d" % K]
    r0.append("END 0")
    r1.append("END 0")
    res = run(tmp, [r0, r1])
    check("LogGPO blocking ping-pong one-way = Tc", res["LogGPO"]["time"] / (2 * n), Tc)

    # collectives expand and finish without deadlock on 4 and 6 ranks
    for nr in (4, 6):
        ranks = [["ALLREDUCE 0 0 8", "BARRIER 0 0", "BCAST 0 0 2 100000", "REDUCE 0 0 1 100000", "END 0"]
                 for _ in range(nr)]
        res = run(tmp, ranks)
        ok = res["LogGPO"]["time"] > 0 and res["LogGP"]["time"] > 0
        fails += not ok
        print("%-58s %s" % ("collectives on %d ranks" % nr, "ok" if ok else "FAIL"))

    shutil.rmtree(tmp)
    print("\n%d failure(s)" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
