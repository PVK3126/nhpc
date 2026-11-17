#!/usr/bin/env python3
"""
loggposim.py -- trace-driven simulator (the SIM-MPI of Chen et al., 2009)
with two pluggable communication models:

  LogGP  (baseline)  Alexandrov et al.; parameters measured logp_mpi style:
                     sender pays os(K) in the send call, the message arrives
                     after G(K-1)+L, receiver pays or(K).  Computation and
                     communication are assumed to overlap perfectly.

  LogGPO (the paper) LogGP + O, Oi(K), Oc(K), switch point S, progress
                     strategy Scp (DCP/ICP) and transfer mechanism Tm.
                     Section 2.2 formulas:
     eager (K<=S):        Os = Oi            Or = Oc
                          Tc = Oi + G(K-1) + L + Oc
     rendezvous, DCP/RAT: REQ (O+L) -> ACK sent when the receiver is inside
                          MPI (in MPI_Irecv if REQ already arrived, else in
                          its next blocking call) -> data moved inside the
                          sender's MPI_Wait (O + Oi + G(K-1)) -> +L
                          Tc = 4O + 3L + Oi + G(K-1) + blocking time
     rendezvous, ICP/RTA: same handshake but progressed independently:
                          Os = 2O, Or = 2O + Oi, Tc = 4O + 3L + Oi + G(K-1)

Where the paper's per-end formulas and its Tc are not mutually consistent
(it states Tc but drops one O at one end), we keep Tc exact.

Collectives on MPI_COMM_WORLD are expanded into point-to-point messages
(dissemination for barrier/allreduce, binomial trees for bcast/reduce), so
both models price them with the same parameters.
"""
import bisect
import json
import os
import sys
from collections import defaultdict

# op codes
C, S, R, W, M = 0, 1, 2, 3, 4


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
class Table:
    """Piecewise-linear function of the message size in bytes."""

    def __init__(self, xs, ys):
        pts = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pts]
        self.ys = [p[1] for p in pts]

    def __call__(self, k):
        xs, ys = self.xs, self.ys
        if not xs:
            return 0.0
        if k <= xs[0]:
            return ys[0]
        if k >= xs[-1]:
            if len(xs) >= 2 and xs[-1] > xs[-2]:
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                return max(0.0, ys[-1] + slope * (k - xs[-1]))
            return ys[-1]
        i = bisect.bisect_right(xs, k)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (k - x0) / (x1 - x0)


class ParamSet:
    def __init__(self, d, overrides=None):
        d = dict(d)
        for k, v in (overrides or {}).items():
            d[k] = v
        self.raw = d
        self.L = float(d["L"])
        self.G = float(d["G"])
        self.O = float(d["O"])
        self.S = int(float(d["S"]))
        self.progress = str(d.get("progress", "DCP")).upper()
        sizes = d["sizes"]
        self.os = Table(sizes, d["os"])
        self.or_ = Table(sizes, d["or"])
        e = [(k, oi, oc) for k, oi, oc in zip(sizes, d["Oi"], d["Oc"]) if k <= self.S]
        r = [(k, oi) for k, oi in zip(sizes, d["Oi"]) if k > self.S]
        self.Oi_e = Table([x[0] for x in e], [x[1] for x in e])
        self.Oc = Table([x[0] for x in e], [x[2] for x in e])
        self.Oi_r = Table([x[0] for x in r], [x[1] for x in r])

    def Oi(self, k):
        return self.Oi_e(k) if k <= self.S else self.Oi_r(k)

    @staticmethod
    def load(path, overrides=None):
        with open(path) as f:
            return ParamSet(json.load(f), overrides)


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v if k == "progress" else float(v)
    return out


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------
class Msg:
    __slots__ = ("src", "dst", "K", "has_send", "has_recv", "Ts", "Ts_end", "Tr",
                 "Tr_end", "Tsw", "Trw", "prog", "data_arr", "sdone", "ps", "eager",
                 "rwait")

    def __init__(self, src, dst):
        self.src, self.dst = src, dst
        self.K = 0
        self.has_send = self.has_recv = False
        self.reset()

    def reset(self):
        self.Ts = self.Ts_end = self.Tr = self.Tr_end = None
        self.Tsw = self.Trw = self.prog = self.data_arr = self.sdone = None
        self.rwait = False


class Trace:
    def __init__(self, nprocs, ops, msgs, hosts, measured, warnings):
        self.nprocs, self.ops, self.msgs = nprocs, ops, msgs
        self.hosts, self.measured, self.warnings = hosts, measured, warnings


def _read_meta(d):
    meta = {"hosts": {}}
    p = os.path.join(d, "meta.txt")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "nprocs":
                    meta["nprocs"] = int(t[1])
                elif t[0] == "measured_time":
                    meta["measured"] = float(t[1])
                elif t[0] == "host":
                    meta["hosts"][int(t[1])] = t[2] if len(t) > 2 else "?"
    return meta


def load_trace(d):
    meta = _read_meta(d)
    if "nprocs" in meta:
        P = meta["nprocs"]
    else:
        P = len([f for f in os.listdir(d) if f.startswith("trace.") and f.endswith(".txt")])
    raw = []
    for r in range(P):
        with open(os.path.join(d, "trace.%d.txt" % r)) as f:
            raw.append([line.split() for line in f if line.strip() and not line.startswith("#")])

    msgs = {}
    scount = defaultdict(int)
    rcount = defaultdict(int)
    warnings = []

    def send_msg(src, dst, tag, K):
        key = (src, dst, tag)
        seq = scount[key]
        scount[key] += 1
        m = msgs.get(key + (seq,))
        if m is None:
            m = msgs[key + (seq,)] = Msg(src, dst)
        m.K = K
        m.has_send = True
        return m

    def recv_msg(src, dst, tag):
        key = (src, dst, tag)
        seq = rcount[key]
        rcount[key] += 1
        m = msgs.get(key + (seq,))
        if m is None:
            m = msgs[key + (seq,)] = Msg(src, dst)
        m.has_recv = True
        return m

    def exchange(r, ops, to, frm, tag, K):
        items = []
        if frm is not None:
            mr = recv_msg(frm, r, tag)
            ops.append((R, mr))
            items.append((mr, False))
        if to is not None:
            ms = send_msg(r, to, tag, K)
            ops.append((S, ms))
            items.append((ms, True))
        if items:
            ops.append((W, items))

    all_ops = []
    for r in range(P):
        evs = raw[r]
        # resolve actual source/tag of every receive request from its completion
        resolved = {}
        for e in evs:
            if e[0] == "WAIT":
                n = int(e[3])
                for k in range(n):
                    rid, src, tag = int(e[4 + 3 * k]), int(e[5 + 3 * k]), int(e[6 + 3 * k])
                    if src != -2:
                        resolved[rid] = (src, tag)
        ops = []
        reqs = {}
        last = 0.0
        ncoll = 0
        for e in evs:
            kind = e[0]
            if kind == "END":
                gap = float(e[1]) - last
                if gap > 0:
                    ops.append((C, gap))
                break
            t0, t1 = float(e[1]), float(e[2])
            gap = t0 - last
            if gap > 0:
                ops.append((C, gap))
            last = max(last, t1)
            if kind == "SEND":
                m = send_msg(r, int(e[3]), int(e[4]), int(e[5]))
                ops.append((S, m))
                ops.append((W, [(m, True)]))
            elif kind == "RECV":
                m = recv_msg(int(e[3]), r, int(e[4]))
                ops.append((R, m))
                ops.append((W, [(m, False)]))
            elif kind == "ISEND":
                m = send_msg(r, int(e[3]), int(e[4]), int(e[5]))
                reqs[int(e[6])] = (m, True)
                ops.append((S, m))
            elif kind == "IRECV":
                rid = int(e[6])
                if rid not in resolved:
                    continue  # never completed inside the traced region
                src, tag = resolved[rid]
                m = recv_msg(src, r, tag)
                reqs[rid] = (m, False)
                ops.append((R, m))
            elif kind == "WAIT":
                n = int(e[3])
                items = []
                for k in range(n):
                    rid = int(e[4 + 3 * k])
                    if rid in reqs:
                        items.append(reqs.pop(rid))
                if items:
                    ops.append((W, items))
                else:
                    ops.append((C, t1 - t0))
            elif kind == "SENDRECV":
                dst, stag, sb, src, rtag = int(e[3]), int(e[4]), int(e[5]), int(e[6]), int(e[7])
                items = []
                if src >= 0:
                    mr = recv_msg(src, r, rtag)
                    ops.append((R, mr))
                    items.append((mr, False))
                if dst >= 0:
                    ms = send_msg(r, dst, stag, sb)
                    ops.append((S, ms))
                    items.append((ms, True))
                if items:
                    ops.append((W, items))
            elif kind in ("BARRIER", "ALLREDUCE"):
                K = 0 if kind == "BARRIER" else int(e[3])
                d = 1
                rnd = 0
                while d < P:   # dissemination
                    exchange(r, ops, (r + d) % P, (r - d) % P, ("coll", ncoll, rnd), K)
                    d *= 2
                    rnd += 1
                ncoll += 1
            elif kind in ("BCAST", "REDUCE"):
                root, K = int(e[3]), int(e[4])
                vr = (r - root) % P
                tag = ("coll", ncoll, 0)
                if kind == "BCAST":   # binomial tree, root outwards
                    mask = 1
                    while mask < P:
                        if vr & mask:
                            exchange(r, ops, None, (vr - mask + root) % P, tag, K)
                            break
                        mask <<= 1
                    mask >>= 1
                    while mask > 0:
                        if vr + mask < P:
                            exchange(r, ops, (vr + mask + root) % P, None, tag, K)
                        mask >>= 1
                else:                 # binomial tree towards the root
                    mask = 1
                    while mask < P:
                        if vr & mask:
                            exchange(r, ops, (vr - mask + root) % P, None, tag, K)
                            break
                        if vr + mask < P:
                            exchange(r, ops, None, (vr + mask + root) % P, tag, K)
                        mask <<= 1
                ncoll += 1
            elif kind == "MARK":
                ops.append((M, int(e[3])))
            else:  # OTHER: keep the measured duration
                ops.append((C, t1 - t0))
        if reqs:
            warnings.append("rank %d: %d requests never completed in the traced region" % (r, len(reqs)))
        all_ops.append(ops)

    nos = sum(1 for m in msgs.values() if not m.has_recv)
    nor = sum(1 for m in msgs.values() if not m.has_send)
    if nos or nor:
        warnings.append("%d sends without matching receive, %d receives without matching send" % (nos, nor))
    hosts = [meta["hosts"].get(r, "node%d" % r) for r in range(P)]
    return Trace(P, all_ops, list(msgs.values()), hosts, meta.get("measured"), warnings)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Model:
    name = "?"

    def __init__(self, inter, intra=None):
        self.inter, self.intra = inter, intra or inter
        self.forcing = False
        self.ranks = None

    def prepare(self, trace):
        for m in trace.msgs:
            m.reset()
            same = trace.hosts[m.src] == trace.hosts[m.dst]
            m.ps = self.intra if same else self.inter
            m.eager = m.K <= m.ps.S

    @staticmethod
    def k1(m):
        return m.K - 1 if m.K > 0 else 0


class LogGP(Model):
    """Baseline: LogGP with logp_mpi-measured os(K), or(K)."""
    name = "LogGP"

    def post_send(self, m):
        return m.ps.os(m.K)

    def post_recv(self, m):
        return 0.0

    def item(self, m, is_send, tw):
        if is_send or not m.has_send:
            return (tw, 0.0, 0.0, None)
        if m.Ts_end is None:
            return None
        ps = m.ps
        arr = m.Ts_end + ps.G * self.k1(m) + ps.L
        return (arr, ps.or_(m.K), 0.0, None)


class LogGPO(Model):
    """The paper's LogGPO model."""
    name = "LogGPO"

    def post_send(self, m):
        return m.ps.Oi(m.K) if m.eager else m.ps.O

    def post_recv(self, m):
        return 0.0 if m.eager else m.ps.O

    # ----- helpers -----
    def _icp_times(self, m):
        if m.data_arr is None:
            ps = m.ps
            match = max(m.Ts_end + ps.L, m.Tr_end)        # REQ arrived and recv posted
            ack_arr = match + 2 * ps.O + ps.L              # process REQ, send ACK
            m.sdone = ack_arr + ps.O + ps.Oi(m.K) + ps.G * self.k1(m)
            m.data_arr = m.sdone + ps.L

    def _dcp_progress(self, m):
        """Time at which the receiver's MPI library answers the REQ (DCP)."""
        if m.prog is not None:
            return m.prog
        req_arr = m.Ts_end + m.ps.L
        if req_arr <= m.Tr:                       # case (a): MPI_Irecv finds the REQ
            m.prog = m.Tr
            return m.prog
        rr = self.ranks[m.dst]                    # case (b): next time inside MPI
        i = bisect.bisect_left(rr.iv_te, req_arr)
        if i < len(rr.iv_te):
            m.prog = max(rr.iv_tw[i], req_arr)
            return m.prog
        w = rr.wait
        if w is not None and w[0] >= m.Tr_end and (m.rwait or req_arr <= w[1] or self.forcing):
            m.prog = max(w[0], req_arr)
            return m.prog
        return None

    def item(self, m, is_send, tw):
        ps = m.ps
        if is_send:
            if not m.has_recv:
                return (tw, 0.0, 0.0, None)
            if m.eager:
                return (tw, 0.0, 0.0, None)
            if m.Tr is None:
                return None
            if ps.progress == "ICP":
                self._icp_times(m)
                return (m.sdone, 0.0, ps.O, None)
            prog = self._dcp_progress(m)
            if prog is None:
                return None
            ack_arr = prog + 2 * ps.O + ps.L
            cost = ps.O + ps.Oi(m.K) + ps.G * self.k1(m)

            def done(t, m=m):
                m.data_arr = t + m.ps.L
            return (ack_arr, cost, 0.0, done)
        # receive side
        if not m.has_send:
            return (tw, 0.0, 0.0, None)
        if m.Ts_end is None:
            return None
        if m.eager:
            arr = m.Ts_end + ps.G * self.k1(m) + ps.L
            return (arr, ps.Oc(m.K), 0.0, None)
        if ps.progress == "ICP":
            self._icp_times(m)
            return (m.data_arr, 0.0, ps.O + ps.Oi(m.K), None)
        if m.data_arr is None:
            return None
        return (m.data_arr, 0.0, 0.0, None)


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class RankState:
    __slots__ = ("id", "ops", "pc", "t", "wait", "iv_tw", "iv_te", "marks", "tcomp")

    def __init__(self, i, ops):
        self.id, self.ops, self.pc, self.t = i, ops, 0, 0.0
        self.wait = None          # [t_wait_start, current_time, pending_items]
        self.iv_tw, self.iv_te = [], []
        self.marks = []
        self.tcomp = 0.0


class SimDeadlock(RuntimeError):
    pass


def _advance(rk, model, comp_scale):
    moved = False
    ops = rk.ops
    n = len(ops)
    while rk.pc < n:
        op = ops[rk.pc]
        k = op[0]
        if k == C:
            dt = op[1] * comp_scale
            rk.t += dt
            rk.tcomp += dt
        elif k == S:
            m = op[1]
            m.Ts = rk.t
            rk.t += model.post_send(m)
            m.Ts_end = rk.t
        elif k == R:
            m = op[1]
            m.Tr = rk.t
            rk.t += model.post_recv(m)
            m.Tr_end = rk.t
        elif k == M:
            rk.marks.append((op[1], rk.t))
        else:  # W
            if rk.wait is None:
                rk.wait = [rk.t, rk.t, list(op[1])]
                for m, is_send in op[1]:
                    if is_send:
                        m.Tsw = rk.t
                    else:
                        m.Trw = rk.t
                        m.rwait = True
            w = rk.wait
            ready = []
            for it in w[2]:
                res = model.item(it[0], it[1], w[0])
                if res is not None:
                    ready.append((res, it))
            if ready:
                ready.sort(key=lambda x: x[0][0])
                for (rdy, cost, minb, cb), it in ready:
                    w[1] = max(w[1] + minb, rdy) + cost
                    if cb is not None:
                        cb(w[1])
                    w[2].remove(it)
                moved = True
            if w[2]:
                return moved
            rk.t = w[1]
            rk.iv_tw.append(w[0])
            rk.iv_te.append(w[1])
            for m, is_send in op[1]:
                if not is_send:
                    m.rwait = False
            rk.wait = None
        rk.pc += 1
        moved = True
    return moved


def simulate(trace, model, comp_scale=1.0):
    model.prepare(trace)
    ranks = [RankState(i, ops) for i, ops in enumerate(trace.ops)]
    model.ranks = ranks
    forced = 0
    while True:
        progress = False
        alldone = True
        for rk in ranks:
            if rk.pc < len(rk.ops):
                if _advance(rk, model, comp_scale):
                    progress = True
                if rk.pc < len(rk.ops):
                    alldone = False
        if alldone:
            break
        if not progress:
            model.forcing = True
            for rk in ranks:
                if rk.pc < len(rk.ops) and _advance(rk, model, comp_scale):
                    progress = True
            model.forcing = False
            if not progress:
                blocked = ["rank %d at op %d/%d (t=%.6f)" % (rk.id, rk.pc, len(rk.ops), rk.t)
                           for rk in ranks if rk.pc < len(rk.ops)]
                raise SimDeadlock("simulation deadlock: " + "; ".join(blocked[:8]))
            forced += 1
    total = max(rk.t for rk in ranks) if ranks else 0.0
    return {
        "model": model.name,
        "time": total,
        "rank_times": [rk.t for rk in ranks],
        "comp_time": max(rk.tcomp for rk in ranks) if ranks else 0.0,
        "marks": [dict(rk.marks) for rk in ranks],
        "forced": forced,
    }


def build_models(params, params_intra=None, overrides=None):
    inter = ParamSet.load(params, overrides)
    intra = ParamSet.load(params_intra, overrides) if params_intra else None
    return LogGP(inter, intra), LogGPO(inter, intra)


def predict(trace_dir, params, params_intra=None, overrides=None, comp_scale=1.0):
    tr = load_trace(trace_dir)
    out = {"trace": trace_dir, "nprocs": tr.nprocs, "measured": tr.measured, "warnings": tr.warnings}
    for model in build_models(params, params_intra, overrides):
        out[model.name] = simulate(tr, model, comp_scale)
    return out


def rel_err(pred, meas):
    if not meas:
        return float("nan")
    return abs(pred - meas) / meas * 100.0


if __name__ == "__main__":
    sys.exit("use predict.py, pingpong_report.py or apps_report.py")
