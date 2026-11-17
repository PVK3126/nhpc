/*
 * params_bench.c -- measures the LogGP (baseline) and LogGPO parameters
 * between rank 0 and rank 1 (paper section 2 / 3.1).
 *
 *   LogGP  (baseline, Kielmann logp_mpi style):  L, G, os(K), or(K)
 *   LogGPO (this paper):  L, G, O, Oi(K), Oc(K), S (eager/rendezvous switch),
 *                         Scp (DCP / ICP), Tm (T for eager; RAT or RTA)
 *
 * Run with exactly 2 ranks:
 *   2 nodes x 1 rank  -> inter-node parameters   (params_inter.json)
 *   1 node  x 2 ranks -> intra-node parameters   (params_intra.json)
 *
 * usage: params_bench [out.json] [maxbytes]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int rank, nproc;
static MPI_Comm pc;              /* communicator of ranks 0 and 1 */
static char *sbuf, *rbuf;
static long MAXK = 4L << 20;

static double now(void) { return MPI_Wtime(); }
static void spin(double s) { double t = now(); while (now() - t < s) ; }

static int reps_for(long K) {
    if (K <= 4096) return 1000;
    if (K <= 65536) return 300;
    if (K <= (1L << 20)) return 60;
    return 20;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static double median(double *v, int n) { qsort(v, n, sizeof(double), cmp_d); return v[n / 2]; }

/* one-way time = round trip / 2 (blocking ping-pong, as IMB / logp_mpi) */
static double oneway(long K) {
    int reps = reps_for(K), warm = 5;
    MPI_Barrier(pc);
    double t = 0;
    for (int i = -warm; i < reps; i++) {
        if (i == 0) t = now();
        if (rank == 0) { MPI_Send(sbuf, K, MPI_CHAR, 1, 1, pc); MPI_Recv(rbuf, K, MPI_CHAR, 1, 1, pc, MPI_STATUS_IGNORE); }
        else           { MPI_Recv(rbuf, K, MPI_CHAR, 0, 1, pc, MPI_STATUS_IGNORE); MPI_Send(sbuf, K, MPI_CHAR, 0, 1, pc); }
    }
    double r = (now() - t) / reps / 2.0;
    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, pc);
    return r;
}

/* os(K): time spent in MPI_Send while the receiver is ready (logp_mpi os(m));
 * isw(K): time spent in MPI_Isend + MPI_Wait, same setting (LogGPO Os = Oi + Osw). */
static void send_overheads(long K, double *os, double *isw) {
    int reps = reps_for(K), warm = 3;
    double acc1 = 0, acc2 = 0;
    MPI_Barrier(pc);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = -warm; i < reps; i++) {
            if (rank == 0) {
                double t = now();
                if (pass == 0) MPI_Send(sbuf, K, MPI_CHAR, 1, 2, pc);
                else { MPI_Request rq; MPI_Isend(sbuf, K, MPI_CHAR, 1, 2, pc, &rq); MPI_Wait(&rq, MPI_STATUS_IGNORE); }
                double d = now() - t;
                if (i >= 0) { if (pass == 0) acc1 += d; else acc2 += d; }
                MPI_Recv(rbuf, 0, MPI_CHAR, 1, 3, pc, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(rbuf, K, MPI_CHAR, 0, 2, pc, MPI_STATUS_IGNORE);
                MPI_Send(sbuf, 0, MPI_CHAR, 0, 3, pc);
            }
        }
    }
    double v[2] = { acc1 / reps, acc2 / reps };
    MPI_Bcast(v, 2, MPI_DOUBLE, 0, pc);
    *os = v[0]; *isw = v[1];
}

/* or(K): time spent in MPI_Recv when the receiver arrives late, i.e. the
 * message (eager) or its request (rendezvous) is already there. */
static double recv_overhead(long K, double delay) {
    int reps = reps_for(K) / 2 + 5, warm = 3;
    double acc = 0;
    MPI_Barrier(pc);
    for (int i = -warm; i < reps; i++) {
        if (rank == 0) {
            MPI_Send(sbuf, K, MPI_CHAR, 1, 4, pc);
            MPI_Recv(rbuf, 0, MPI_CHAR, 1, 5, pc, MPI_STATUS_IGNORE);
        } else {
            spin(delay);
            double t = now();
            MPI_Recv(rbuf, K, MPI_CHAR, 0, 4, pc, MPI_STATUS_IGNORE);
            if (i >= 0) acc += now() - t;
            MPI_Send(sbuf, 0, MPI_CHAR, 0, 5, pc);
        }
    }
    double r = acc / reps;
    MPI_Bcast(&r, 1, MPI_DOUBLE, 1, pc);
    return r;
}

/* eager if MPI_Send returns long before the (late) receiver posts its recv */
static int is_eager(long K) {
    double D = 1e-3 + 20e-9 * (double)K;   /* far above any transfer time */
    double d[5];
    MPI_Barrier(pc);
    for (int i = 0; i < 5; i++) {
        MPI_Barrier(pc);
        if (rank == 0) {
            double t = now();
            MPI_Send(sbuf, K, MPI_CHAR, 1, 6, pc);
            d[i] = now() - t;
            MPI_Recv(rbuf, 0, MPI_CHAR, 1, 7, pc, MPI_STATUS_IGNORE);
        } else {
            spin(D);
            MPI_Recv(rbuf, K, MPI_CHAR, 0, 6, pc, MPI_STATUS_IGNORE);
            MPI_Send(sbuf, 0, MPI_CHAR, 0, 7, pc);
        }
    }
    int eager = 0;
    if (rank == 0) eager = median(d, 5) < 0.5 * D;
    MPI_Bcast(&eager, 1, MPI_INT, 0, pc);
    return eager;
}

static long find_switch_point(void) {
    long lo = 0, hi = -1;
    for (long K = 1; K <= MAXK; K *= 2) {
        if (is_eager(K)) lo = K; else { hi = K; break; }
    }
    if (hi < 0) return -1;                  /* never switched to rendezvous */
    while (hi - lo > (lo / 64 > 64 ? lo / 64 : 64)) {
        long mid = (lo + hi) / 2;
        if (is_eager(mid)) lo = mid; else hi = mid;
    }
    return lo;
}

/* DCP vs ICP: does a rendezvous transfer finish while the sender computes
 * (no MPI calls) between MPI_Isend and MPI_Wait? */
static int independent_progress(long K, double ow) {
    double D = 20 * ow + 2e-3, done[5];
    for (int i = 0; i < 5; i++) {
        if (rank == 1) {
            MPI_Request rq;
            MPI_Irecv(rbuf, K, MPI_CHAR, 0, 8, pc, &rq);
            MPI_Barrier(pc);
            double t = now();
            MPI_Wait(&rq, MPI_STATUS_IGNORE);
            done[i] = now() - t;
        } else {
            MPI_Request rq;
            MPI_Barrier(pc);
            MPI_Isend(sbuf, K, MPI_CHAR, 1, 8, pc, &rq);
            spin(D);
            MPI_Wait(&rq, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(pc);
    }
    int icp = 0;
    if (rank == 1) icp = median(done, 5) < 0.5 * D;
    MPI_Bcast(&icp, 1, MPI_INT, 1, pc);
    return icp;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const char *out = argc > 1 ? argv[1] : "params.json";
    if (argc > 2) MAXK = atol(argv[2]);
    if (nproc < 2) { if (rank == 0) fprintf(stderr, "params_bench needs 2 ranks\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm_split(MPI_COMM_WORLD, rank < 2 ? 0 : MPI_UNDEFINED, rank, &pc);

    char host[MPI_MAX_PROCESSOR_NAME], peer[MPI_MAX_PROCESSOR_NAME]; int hl;
    memset(host, 0, sizeof host); memset(peer, 0, sizeof peer);
    MPI_Get_processor_name(host, &hl);

    if (rank < 2) {
        sbuf = malloc(MAXK + 1); rbuf = malloc(MAXK + 1);
        memset(sbuf, 1, MAXK + 1); memset(rbuf, 0, MAXK + 1);
        MPI_Sendrecv(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 0,
                     peer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 0, pc, MPI_STATUS_IGNORE);

        /* 1. eager/rendezvous switch point S */
        long S = find_switch_point();
        if (rank == 0) printf("[params] host %s <-> %s, switch point S = %ld bytes%s\n",
                              host, peer, S, S < 0 ? " (no rendezvous seen)" : "");

        /* 2. size grid: powers of two plus both sides of S */
        long sizes[64]; int n = 0;
        for (long K = 1; K <= MAXK; K *= 2) sizes[n++] = K;
        if (S > 0) {
            int hasS = 0, hasS1 = 0;
            for (int i = 0; i < n; i++) { if (sizes[i] == S) hasS = 1; if (sizes[i] == S + 1) hasS1 = 1; }
            if (!hasS) sizes[n++] = S;
            if (!hasS1 && S + 1 <= MAXK) sizes[n++] = S + 1;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
                if (sizes[j] < sizes[i]) { long t = sizes[i]; sizes[i] = sizes[j]; sizes[j] = t; }
        }

        double ow[64], os[64], isw[64], orr[64];
        for (int i = 0; i < n; i++) {
            ow[i] = oneway(sizes[i]);
            send_overheads(sizes[i], &os[i], &isw[i]);
            orr[i] = recv_overhead(sizes[i], 4 * ow[i] + 50e-6);
            if (rank == 0) printf("[params] K=%8ld  oneway=%10.3f us  os=%10.3f us  isend+wait=%10.3f us  or=%10.3f us\n",
                                  sizes[i], ow[i] * 1e6, os[i] * 1e6, isw[i] * 1e6, orr[i] * 1e6);
        }

        /* 3. progress strategy (needs a rendezvous size) */
        int icp = 0, tested = 0;
        if (S > 0) {
            long K = 4 * S; if (K < (1L << 20)) K = 1L << 20; if (K > MAXK) K = MAXK;
            if (K > S) {
                double owK = ow[n - 1] * (double)K / (double)sizes[n - 1];
                icp = independent_progress(K, owK); tested = 1;
            }
        }

        if (rank == 0) {
            /* 4. derived parameters */
            double O = 0.5 * (os[0] + orr[0]);
            double L = ow[0] - os[0] - orr[0]; if (L < 0) L = 0;
            /* G: least squares slope of one-way time over large sizes */
            double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
            for (int i = 0; i < n; i++) {
                if (sizes[i] < 65536 || (S > 0 && sizes[i] <= S)) continue;
                sx += sizes[i]; sy += ow[i]; sxx += (double)sizes[i] * sizes[i]; sxy += (double)sizes[i] * ow[i]; m++;
            }
            if (m < 2) {   /* fall back to the two largest sizes */
                m = 0; sx = sy = sxx = sxy = 0;
                for (int i = n - 2; i < n; i++) { sx += sizes[i]; sy += ow[i]; sxx += (double)sizes[i] * sizes[i]; sxy += (double)sizes[i] * ow[i]; m++; }
            }
            double G = (m * sxy - sx * sy) / (m * sxx - sx * sx);
            if (!(G > 0)) G = (ow[n - 1] - ow[n - 2]) / (double)(sizes[n - 1] - sizes[n - 2]);

            long Seff = S < 0 ? (1L << 62) : S;
            double Oi[64], Oc[64];
            for (int i = 0; i < n; i++) {
                double K1 = sizes[i] > 0 ? (double)(sizes[i] - 1) : 0.0;
                if (sizes[i] <= Seff) {           /* eager:  Os = Oi,  Or = Oc */
                    Oi[i] = isw[i];
                    Oc[i] = orr[i];
                } else {                          /* rendezvous: Tc = 4O + 3L + Oi + G(K-1) */
                    Oi[i] = ow[i] - (4 * O + 3 * L + G * K1); if (Oi[i] < 0) Oi[i] = 0;
                    Oc[i] = 0;
                }
            }

            FILE *f = fopen(out, "w");
            if (!f) { perror(out); MPI_Abort(MPI_COMM_WORLD, 1); }
            fprintf(f, "{\n  \"hosts\": [\"%s\", \"%s\"],\n", host, peer);
            fprintf(f, "  \"placement\": \"%s\",\n", strcmp(host, peer) == 0 ? "intra" : "inter");
            fprintf(f, "  \"L\": %.6e,\n  \"G\": %.6e,\n  \"O\": %.6e,\n  \"S\": %ld,\n", L, G, O, Seff);
            fprintf(f, "  \"progress\": \"%s\",\n  \"progress_tested\": %s,\n", icp ? "ICP" : "DCP", tested ? "true" : "false");
            fprintf(f, "  \"Tm_eager\": \"T\",\n  \"Tm_rendezvous\": \"%s\",\n", icp ? "RTA" : "RAT");
            #define ARR(name, v) do { fprintf(f, "  \"%s\": [", name); \
                for (int i_ = 0; i_ < n; i_++) fprintf(f, "%s%.6e", i_ ? ", " : "", (double)(v)[i_]); \
                fprintf(f, "],\n"); } while (0)
            fprintf(f, "  \"sizes\": [");
            for (int i = 0; i < n; i++) fprintf(f, "%s%ld", i ? ", " : "", sizes[i]);
            fprintf(f, "],\n");
            ARR("oneway", ow); ARR("os", os); ARR("or", orr); ARR("isend_wait", isw); ARR("Oi", Oi); ARR("Oc", Oc);
            fprintf(f, "  \"units\": \"seconds, bytes; G in seconds per byte\"\n}\n");
            fclose(f);
            printf("[params] L=%.3f us  O=%.3f us  G=%.4f ns/B (%.1f MB/s)  S=%ld  progress=%s\n",
                   L * 1e6, O * 1e6, G * 1e9, 1e-6 / G, Seff, icp ? "ICP" : "DCP");
            printf("[params] wrote %s\n", out);
        }
        free(sbuf); free(rbuf);
        MPI_Comm_free(&pc);
    }
    MPI_Finalize();
    return 0;
}
