/*
 * tracer.c -- PMPI tracing library (the "SIM-MPI instrumentation library"
 * of Chen et al., LogGPO, Sci China 2009, section 3.2.2).
 *
 * Records, per rank, every point-to-point call, wait/test completion and
 * MPI_COMM_WORLD collective with start/end timestamps.  The time between
 * two consecutive MPI calls is the computation that the simulator replays.
 *
 * Usage
 *   - link the object into the executable (the bin/..._traced binaries), or
 *   - LD_PRELOAD=lib/libloggpotrace.so ./app   (C/C++ applications)
 *
 * Environment
 *   LOGGPO_TRACE_DIR   output directory          (default ./loggpo_trace)
 *   LOGGPO_TRACE=0     disable tracing completely (same binary, untraced)
 *
 * Trace region control (optional, from the application):
 *   MPI_Pcontrol(1)      restart trace here (collective: does a barrier)
 *   MPI_Pcontrol(0)      stop trace here
 *   MPI_Pcontrol(100+i)  record marker i (used by the ping-pong benchmark)
 * Without Pcontrol the region is MPI_Init .. MPI_Finalize.
 *
 * Output: <dir>/trace.<rank>.txt and <dir>/meta.txt (see README).
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(MPI_VERSION) && MPI_VERSION >= 3
#define MPI3CONST const
#else
#define MPI3CONST
#endif

enum { E_SEND, E_RECV, E_ISEND, E_IRECV, E_WAIT, E_SENDRECV, E_BARRIER,
       E_ALLREDUCE, E_REDUCE, E_BCAST, E_MARK, E_OTHER };

typedef struct {
    int op;
    double t0, t1;
    int a, b;            /* peer, tag            (SENDRECV: dst, send tag) */
    long long c;         /* bytes                                         */
    long long d;         /* request id / root / marker id                 */
    int e, f;            /* SENDRECV: src, recv tag                       */
    long long g;         /* SENDRECV: recv bytes                          */
    long long woff;      /* WAIT: offset into wrec (triples)              */
    int wn;              /* WAIT: number of completed requests            */
    const char *name;    /* OTHER: call name                              */
} ev_t;

typedef struct { MPI_Request h; long long id; int is_recv; MPI_Comm comm; } rq_t;

static ev_t *evs;        static size_t nev, evcap;
static long long *wrec;  static size_t nw, wcap;
static rq_t *rqs;        static int nrq, rqcap;
static long long next_req = 1;
static int  enabled = 1, active = 0, inside = 0, started = 0;
static int  myrank = 0, nprocs = 1;
static double tbase = 0.0, tend = 0.0;

/* ------------------------------------------------------------------ */
static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) { fprintf(stderr, "[loggpo-trace] out of memory\n"); PMPI_Abort(MPI_COMM_WORLD, 1); }
    return q;
}

static ev_t *newev(int op, double a, double b) {
    if (nev == evcap) { evcap = evcap ? 2 * evcap : 65536; evs = xrealloc(evs, evcap * sizeof(ev_t)); }
    ev_t *e = &evs[nev++];
    memset(e, 0, sizeof *e);
    e->op = op; e->t0 = a - tbase; e->t1 = b - tbase;
    return e;
}

static void wpush(long long id, long long src, long long tag) {
    if (nw + 3 > wcap) { wcap = wcap ? 2 * wcap : 65536; wrec = xrealloc(wrec, wcap * sizeof(long long)); }
    wrec[nw++] = id; wrec[nw++] = src; wrec[nw++] = tag;
}

static void rq_add(MPI_Request h, long long id, int is_recv, MPI_Comm comm) {
    if (nrq == rqcap) { rqcap = rqcap ? 2 * rqcap : 256; rqs = xrealloc(rqs, rqcap * sizeof(rq_t)); }
    rqs[nrq].h = h; rqs[nrq].id = id; rqs[nrq].is_recv = is_recv; rqs[nrq].comm = comm; nrq++;
}
static int rq_find(MPI_Request h) {
    if (h == MPI_REQUEST_NULL) return -1;
    for (int i = nrq - 1; i >= 0; i--) if (rqs[i].h == h) return i;
    return -1;
}
static void rq_del(int i) { rqs[i] = rqs[--nrq]; }

static long long nbytes(int count, MPI_Datatype dt) {
    int s = 0; PMPI_Type_size(dt, &s); return (long long)count * s;
}

/* translate a rank of `comm` to MPI_COMM_WORLD (negative values pass through) */
static int wrank(MPI_Comm comm, int r) {
    if (r < 0 || comm == MPI_COMM_WORLD) return r;
    int res; PMPI_Comm_compare(comm, MPI_COMM_WORLD, &res);
    if (res == MPI_IDENT || res == MPI_CONGRUENT) return r;
    MPI_Group g, wg; int out;
    PMPI_Comm_group(comm, &g); PMPI_Comm_group(MPI_COMM_WORLD, &wg);
    PMPI_Group_translate_ranks(g, 1, &r, wg, &out);
    PMPI_Group_free(&g); PMPI_Group_free(&wg);
    return out;
}

/* collectives are modelled only when they span all of MPI_COMM_WORLD */
static int is_world(MPI_Comm comm) {
    if (comm == MPI_COMM_WORLD) return 1;
    int res; PMPI_Comm_compare(comm, MPI_COMM_WORLD, &res);
    return res == MPI_IDENT || res == MPI_CONGRUENT;
}

static void tracer_start(void) {
    if (!enabled) return;
    PMPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    nev = 0; nw = 0; nrq = 0;          /* restart: drop previous events */
    PMPI_Barrier(MPI_COMM_WORLD);       /* align the time origin of all ranks */
    tbase = PMPI_Wtime();
    active = 1; started = 1;
}

static void tracer_stop(void) {
    if (!active) return;
    tend = PMPI_Wtime();
    active = 0;
}

static void mkdir_p(const char *dir) {
    char buf[4096]; size_t n = strlen(dir);
    if (n >= sizeof buf) return;
    strcpy(buf, dir);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\0') {
            char c = buf[i]; buf[i] = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST)
                fprintf(stderr, "[loggpo-trace] cannot create %s: %s\n", buf, strerror(errno));
            buf[i] = c;
        }
    }
}

static void write_trace(void) {
    const char *dir = getenv("LOGGPO_TRACE_DIR");
    if (!dir || !*dir) dir = "loggpo_trace";
    if (myrank == 0) mkdir_p(dir);
    PMPI_Barrier(MPI_COMM_WORLD);

    char host[MPI_MAX_PROCESSOR_NAME + 1]; int hl = 0;
    memset(host, 0, sizeof host);
    PMPI_Get_processor_name(host, &hl);
    for (int i = 0; host[i]; i++) if (host[i] == ' ') host[i] = '_';

    char path[4200];
    snprintf(path, sizeof path, "%s/trace.%d.txt", dir, myrank);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "[loggpo-trace] cannot write %s\n", path); return; }
    fprintf(f, "# LOGGPO-TRACE v1 rank %d size %d host %s\n", myrank, nprocs, host);
    for (size_t i = 0; i < nev; i++) {
        ev_t *e = &evs[i];
        switch (e->op) {
        case E_SEND:  fprintf(f, "SEND %.9f %.9f %d %d %lld\n", e->t0, e->t1, e->a, e->b, e->c); break;
        case E_RECV:  fprintf(f, "RECV %.9f %.9f %d %d %lld\n", e->t0, e->t1, e->a, e->b, e->c); break;
        case E_ISEND: fprintf(f, "ISEND %.9f %.9f %d %d %lld %lld\n", e->t0, e->t1, e->a, e->b, e->c, e->d); break;
        case E_IRECV: fprintf(f, "IRECV %.9f %.9f %d %d %lld %lld\n", e->t0, e->t1, e->a, e->b, e->c, e->d); break;
        case E_WAIT:
            fprintf(f, "WAIT %.9f %.9f %d", e->t0, e->t1, e->wn);
            for (int k = 0; k < e->wn; k++)
                fprintf(f, " %lld %lld %lld", wrec[e->woff + 3 * k], wrec[e->woff + 3 * k + 1], wrec[e->woff + 3 * k + 2]);
            fputc('\n', f);
            break;
        case E_SENDRECV:
            fprintf(f, "SENDRECV %.9f %.9f %d %d %lld %d %d %lld\n", e->t0, e->t1, e->a, e->b, e->c, e->e, e->f, e->g);
            break;
        case E_BARRIER:   fprintf(f, "BARRIER %.9f %.9f\n", e->t0, e->t1); break;
        case E_ALLREDUCE: fprintf(f, "ALLREDUCE %.9f %.9f %lld\n", e->t0, e->t1, e->c); break;
        case E_REDUCE:    fprintf(f, "REDUCE %.9f %.9f %lld %lld\n", e->t0, e->t1, e->d, e->c); break;
        case E_BCAST:     fprintf(f, "BCAST %.9f %.9f %lld %lld\n", e->t0, e->t1, e->d, e->c); break;
        case E_MARK:      fprintf(f, "MARK %.9f %.9f %lld\n", e->t0, e->t1, e->d); break;
        case E_OTHER:     fprintf(f, "OTHER %.9f %.9f %s\n", e->t0, e->t1, e->name); break;
        }
    }
    fprintf(f, "END %.9f\n", tend - tbase);
    fclose(f);

    double dur = tend - tbase, mx = 0.0;
    PMPI_Reduce(&dur, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    char *hosts = NULL;
    if (myrank == 0) hosts = malloc((size_t)nprocs * (MPI_MAX_PROCESSOR_NAME + 1));
    PMPI_Gather(host, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, hosts, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        snprintf(path, sizeof path, "%s/meta.txt", dir);
        FILE *m = fopen(path, "w");
        if (m) {
            fprintf(m, "nprocs %d\nmeasured_time %.9f\n", nprocs, mx);
            for (int r = 0; r < nprocs; r++) fprintf(m, "host %d %s\n", r, hosts + (size_t)r * (MPI_MAX_PROCESSOR_NAME + 1));
            fclose(m);
        }
        free(hosts);
        fprintf(stderr, "[loggpo-trace] wrote %d rank traces to %s (traced region %.6f s)\n", nprocs, dir, mx);
    }
}

/* ------------------------------------------------------------------ */
/* init / finalize / pcontrol                                          */
/* ------------------------------------------------------------------ */
static void read_env(void) {
    const char *s = getenv("LOGGPO_TRACE");
    if (s && s[0] == '0') enabled = 0;
}

int MPI_Init(int *argc, char ***argv) {
    int rc = PMPI_Init(argc, argv);
    read_env(); tracer_start();
    return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int rc = PMPI_Init_thread(argc, argv, required, provided);
    read_env(); tracer_start();
    return rc;
}

int MPI_Finalize(void) {
    if (active) tracer_stop();
    if (enabled && started) write_trace();
    free(evs); free(wrec); free(rqs);
    evs = NULL; wrec = NULL; rqs = NULL;
    return PMPI_Finalize();
}

int MPI_Pcontrol(const int level, ...) {
    if (!enabled) return MPI_SUCCESS;
    if (level == 1) tracer_start();
    else if (level == 0) tracer_stop();
    else if (level >= 100 && active) {
        double t = PMPI_Wtime();
        ev_t *e = newev(E_MARK, t, t); e->d = level - 100;
    }
    return MPI_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* point-to-point                                                      */
/* ------------------------------------------------------------------ */
#define PASSTHRU (!active || inside)

int MPI_Send(MPI3CONST void *buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Send(buf, count, dt, dest, tag, comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Send(buf, count, dt, dest, tag, comm);
    double b = PMPI_Wtime();
    if (dest != MPI_PROC_NULL) {
        ev_t *e = newev(E_SEND, a, b); e->a = wrank(comm, dest); e->b = tag; e->c = nbytes(count, dt);
    }
    inside = 0; return rc;
}

/* synchronous / ready / buffered sends are recorded as plain sends */
int MPI_Ssend(MPI3CONST void *buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Ssend(buf, count, dt, dest, tag, comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Ssend(buf, count, dt, dest, tag, comm);
    double b = PMPI_Wtime();
    if (dest != MPI_PROC_NULL) {
        ev_t *e = newev(E_SEND, a, b); e->a = wrank(comm, dest); e->b = tag; e->c = nbytes(count, dt);
    }
    inside = 0; return rc;
}

int MPI_Recv(void *buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status *status) {
    if (PASSTHRU) return PMPI_Recv(buf, count, dt, src, tag, comm, status);
    inside = 1;
    MPI_Status st; MPI_Status *sp = (status == MPI_STATUS_IGNORE) ? &st : status;
    double a = PMPI_Wtime();
    int rc = PMPI_Recv(buf, count, dt, src, tag, comm, sp);
    double b = PMPI_Wtime();
    if (src != MPI_PROC_NULL) {
        ev_t *e = newev(E_RECV, a, b); e->a = wrank(comm, sp->MPI_SOURCE); e->b = sp->MPI_TAG; e->c = nbytes(count, dt);
    }
    inside = 0; return rc;
}

int MPI_Isend(MPI3CONST void *buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm, MPI_Request *req) {
    if (PASSTHRU) return PMPI_Isend(buf, count, dt, dest, tag, comm, req);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Isend(buf, count, dt, dest, tag, comm, req);
    double b = PMPI_Wtime();
    if (dest != MPI_PROC_NULL) {
        long long id = next_req++;
        ev_t *e = newev(E_ISEND, a, b); e->a = wrank(comm, dest); e->b = tag; e->c = nbytes(count, dt); e->d = id;
        rq_add(*req, id, 0, comm);
    }
    inside = 0; return rc;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Request *req) {
    if (PASSTHRU) return PMPI_Irecv(buf, count, dt, src, tag, comm, req);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Irecv(buf, count, dt, src, tag, comm, req);
    double b = PMPI_Wtime();
    if (src != MPI_PROC_NULL) {
        long long id = next_req++;
        ev_t *e = newev(E_IRECV, a, b);
        e->a = (src == MPI_ANY_SOURCE) ? -1 : wrank(comm, src); e->b = tag; e->c = nbytes(count, dt); e->d = id;
        rq_add(*req, id, 1, comm);
    }
    inside = 0; return rc;
}

static void record_completion(int idx, MPI_Status *sp) {
    rq_t *q = &rqs[idx];
    if (q->is_recv) wpush(q->id, wrank(q->comm, sp->MPI_SOURCE), sp->MPI_TAG);
    else            wpush(q->id, -2, -2);
}

int MPI_Wait(MPI_Request *req, MPI_Status *status) {
    if (PASSTHRU) return PMPI_Wait(req, status);
    inside = 1;
    int i = rq_find(*req);
    MPI_Status st; MPI_Status *sp = (status == MPI_STATUS_IGNORE) ? &st : status;
    double a = PMPI_Wtime();
    int rc = PMPI_Wait(req, sp);
    double b = PMPI_Wtime();
    if (i >= 0) {
        ev_t *e = newev(E_WAIT, a, b); e->woff = (long long)nw; e->wn = 1;
        record_completion(i, sp); rq_del(i);
    } else {
        ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Wait";
    }
    inside = 0; return rc;
}

int MPI_Waitall(int n, MPI_Request reqs[], MPI_Status stats[]) {
    if (PASSTHRU) return PMPI_Waitall(n, reqs, stats);
    inside = 1;
    int *idx = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    rq_t *snap = malloc((size_t)(n > 0 ? n : 1) * sizeof(rq_t));
    for (int k = 0; k < n; k++) { idx[k] = rq_find(reqs[k]); if (idx[k] >= 0) snap[k] = rqs[idx[k]]; }
    MPI_Status *sp = (stats == MPI_STATUSES_IGNORE) ? malloc((size_t)(n > 0 ? n : 1) * sizeof(MPI_Status)) : stats;
    double a = PMPI_Wtime();
    int rc = PMPI_Waitall(n, reqs, sp);
    double b = PMPI_Wtime();
    int known = 0;
    for (int k = 0; k < n; k++) if (idx[k] >= 0) known++;
    if (known) {
        ev_t *e = newev(E_WAIT, a, b); e->woff = (long long)nw; e->wn = known;
        for (int k = 0; k < n; k++) {
            if (idx[k] < 0) continue;
            if (snap[k].is_recv) wpush(snap[k].id, wrank(snap[k].comm, sp[k].MPI_SOURCE), sp[k].MPI_TAG);
            else                 wpush(snap[k].id, -2, -2);
        }
        /* remove completed entries (search by id: indices shift on delete) */
        for (int k = 0; k < n; k++) {
            if (idx[k] < 0) continue;
            for (int j = nrq - 1; j >= 0; j--) if (rqs[j].id == snap[k].id) { rq_del(j); break; }
        }
    } else if (n > 0) {
        ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Waitall";
    }
    if (sp != stats) free(sp);
    free(idx); free(snap);
    inside = 0; return rc;
}

int MPI_Waitany(int n, MPI_Request reqs[], int *index, MPI_Status *status) {
    if (PASSTHRU) return PMPI_Waitany(n, reqs, index, status);
    inside = 1;
    MPI_Request *h = malloc((size_t)(n > 0 ? n : 1) * sizeof(MPI_Request));
    for (int k = 0; k < n; k++) h[k] = reqs[k];
    MPI_Status st; MPI_Status *sp = (status == MPI_STATUS_IGNORE) ? &st : status;
    double a = PMPI_Wtime();
    int rc = PMPI_Waitany(n, reqs, index, sp);
    double b = PMPI_Wtime();
    int i = (*index != MPI_UNDEFINED) ? rq_find(h[*index]) : -1;
    if (i >= 0) {
        ev_t *e = newev(E_WAIT, a, b); e->woff = (long long)nw; e->wn = 1;
        record_completion(i, sp); rq_del(i);
    } else {
        ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Waitany";
    }
    free(h);
    inside = 0; return rc;
}

int MPI_Test(MPI_Request *req, int *flag, MPI_Status *status) {
    if (PASSTHRU) return PMPI_Test(req, flag, status);
    inside = 1;
    int i = rq_find(*req);
    MPI_Status st; MPI_Status *sp = (status == MPI_STATUS_IGNORE) ? &st : status;
    double a = PMPI_Wtime();
    int rc = PMPI_Test(req, flag, sp);
    double b = PMPI_Wtime();
    if (*flag && i >= 0) {   /* a successful test is a completed wait */
        ev_t *e = newev(E_WAIT, a, b); e->woff = (long long)nw; e->wn = 1;
        record_completion(i, sp); rq_del(i);
    }
    inside = 0; return rc;
}

int MPI_Request_free(MPI_Request *req) {
    if (active && !inside) { int i = rq_find(*req); if (i >= 0) rq_del(i); }
    return PMPI_Request_free(req);
}

int MPI_Sendrecv(MPI3CONST void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                 void *rbuf, int rcount, MPI_Datatype rtype, int src, int rtag,
                 MPI_Comm comm, MPI_Status *status) {
    if (PASSTHRU) return PMPI_Sendrecv(sbuf, scount, stype, dest, stag, rbuf, rcount, rtype, src, rtag, comm, status);
    inside = 1;
    MPI_Status st; MPI_Status *sp = (status == MPI_STATUS_IGNORE) ? &st : status;
    double a = PMPI_Wtime();
    int rc = PMPI_Sendrecv(sbuf, scount, stype, dest, stag, rbuf, rcount, rtype, src, rtag, comm, sp);
    double b = PMPI_Wtime();
    ev_t *e = newev(E_SENDRECV, a, b);
    e->a = (dest == MPI_PROC_NULL) ? -9 : wrank(comm, dest); e->b = stag; e->c = nbytes(scount, stype);
    e->e = (src == MPI_PROC_NULL) ? -9 : wrank(comm, sp->MPI_SOURCE); e->f = sp->MPI_TAG; e->g = nbytes(rcount, rtype);
    inside = 0; return rc;
}

/* ------------------------------------------------------------------ */
/* collectives on MPI_COMM_WORLD (others are kept as fixed-cost calls)  */
/* ------------------------------------------------------------------ */
int MPI_Barrier(MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Barrier(comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Barrier(comm);
    double b = PMPI_Wtime();
    if (is_world(comm)) newev(E_BARRIER, a, b);
    else { ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Barrier(subcomm)"; }
    inside = 0; return rc;
}

int MPI_Allreduce(MPI3CONST void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Allreduce(sb, rb, count, dt, op, comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Allreduce(sb, rb, count, dt, op, comm);
    double b = PMPI_Wtime();
    if (is_world(comm)) { ev_t *e = newev(E_ALLREDUCE, a, b); e->c = nbytes(count, dt); }
    else { ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Allreduce(subcomm)"; }
    inside = 0; return rc;
}

int MPI_Reduce(MPI3CONST void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Reduce(sb, rb, count, dt, op, root, comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Reduce(sb, rb, count, dt, op, root, comm);
    double b = PMPI_Wtime();
    if (is_world(comm)) { ev_t *e = newev(E_REDUCE, a, b); e->c = nbytes(count, dt); e->d = root; }
    else { ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Reduce(subcomm)"; }
    inside = 0; return rc;
}

int MPI_Bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    if (PASSTHRU) return PMPI_Bcast(buf, count, dt, root, comm);
    inside = 1;
    double a = PMPI_Wtime();
    int rc = PMPI_Bcast(buf, count, dt, root, comm);
    double b = PMPI_Wtime();
    if (is_world(comm)) { ev_t *e = newev(E_BCAST, a, b); e->c = nbytes(count, dt); e->d = root; }
    else { ev_t *e = newev(E_OTHER, a, b); e->name = "MPI_Bcast(subcomm)"; }
    inside = 0; return rc;
}
