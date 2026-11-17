/*
 * cg_mini.c -- conjugate-gradient kernel with the NAS NPB CG structure and
 * communication pattern (paper section 3.2.2, "NPB CG class C").
 *
 *  - inverse power method; each outer iteration runs 25 CG iterations
 *  - P = 2^k ranks on an nprows x npcols grid (npcols = nprows or 2*nprows)
 *  - sparse mat-vec on a 2D block decomposition:
 *        local SpMV  ->  row reduction (log2(npcols) Irecv/Send/Wait steps)
 *                    ->  transpose exchange with partner (Irecv/Send/Wait)
 *  - dot products reduced across the process row with Irecv/Send/Wait
 *
 * The matrix is a deterministic random symmetric, diagonally dominant
 * (hence SPD) matrix with NPB's size parameters (na, nonzer); it is NOT
 * the NPB makea() matrix, so zeta differs from the NPB verification value.
 * Message sizes and counts follow NPB CG (e.g. class C on 16 ranks: 300 KB).
 *
 * usage: cg_mini [-c S|W|A|B|C] [-n niter]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

typedef struct { const char *name; long na; int nonzer, niter; double shift; } cls_t;
static const cls_t classes[] = {
    {"S", 1400, 7, 15, 10.0}, {"W", 7000, 8, 15, 12.0}, {"A", 14000, 11, 15, 20.0},
    {"B", 75000, 13, 75, 60.0}, {"C", 150000, 15, 75, 110.0},
};

#define CGITMAX 25
#define TAG_ROW 100
#define TAG_TR  200

static int me, P, nprows, npcols, prow, pcol, m_per_row;
static long na, row_lo, row_hi, col_lo, col_hi, nrows, ncols;
static long *rowptr; static int *colidx; static double *val;
static double *wbuf, *tmpbuf;

static long blk(long n, int nb, int b) { return (long)((long long)n * b / nb); }

static inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
/* k-th generated off-diagonal partner of row i, and its value in (0,1] */
static inline void gen(long i, int k, long *j, double *v) {
    uint64_t h = mix64(((uint64_t)i << 20) ^ (uint64_t)k ^ 0x5DEECE66DULL);
    *j = (long)(h % (uint64_t)na);
    *v = ((double)(h >> 11) + 1.0) / 9007199254740992.0;
}

/* sum a vector across the process row (recursive doubling, NPB style p2p) */
static void row_allreduce(double *buf, long n) {
    for (int d = 1; d < npcols; d <<= 1) {
        int partner = prow * npcols + (pcol ^ d);
        MPI_Request rq;
        MPI_Irecv(tmpbuf, (int)n, MPI_DOUBLE, partner, TAG_ROW + d, MPI_COMM_WORLD, &rq);
        MPI_Send(buf, (int)n, MPI_DOUBLE, partner, TAG_ROW + d, MPI_COMM_WORLD);
        MPI_Wait(&rq, MPI_STATUS_IGNORE);
        for (long i = 0; i < n; i++) buf[i] += tmpbuf[i];
    }
}

/* w holds (A p) over row block R_prow; deliver q over column block C_pcol */
static void transpose_exchange(const double *w, double *q) {
    int pr, pcc;
    if (nprows == npcols) { pr = pcol; pcc = prow; }
    else                  { pr = pcol / 2; pcc = 2 * prow + (pcol % 2); }
    int partner = pr * npcols + pcc;            /* the mapping is an involution */
    long off = blk(na, npcols, pcc) - row_lo;
    long len = blk(na, npcols, pcc + 1) - blk(na, npcols, pcc);
    if (partner == me) { memcpy(q, w + off, (size_t)ncols * sizeof(double)); return; }
    MPI_Request rq;
    MPI_Irecv(q, (int)ncols, MPI_DOUBLE, partner, TAG_TR, MPI_COMM_WORLD, &rq);
    MPI_Send((void *)(w + off), (int)len, MPI_DOUBLE, partner, TAG_TR, MPI_COMM_WORLD);
    MPI_Wait(&rq, MPI_STATUS_IGNORE);
}

static void matvec(const double *p, double *q) {
    for (long li = 0; li < nrows; li++) {
        double s = 0.0;
        for (long k = rowptr[li]; k < rowptr[li + 1]; k++) s += val[k] * p[colidx[k]];
        wbuf[li] = s;
    }
    row_allreduce(wbuf, nrows);
    transpose_exchange(wbuf, q);
}

static double dot(const double *a, const double *b) {
    double s = 0.0;
    for (long i = 0; i < ncols; i++) s += a[i] * b[i];
    row_allreduce(&s, 1);
    return s;
}

static void build_matrix(void) {
    long *cnt = calloc((size_t)nrows + 1, sizeof(long));
    double *rsum = calloc((size_t)(nrows > ncols ? nrows : ncols) + 1, sizeof(double));
    long j, i; double v;
    /* pass 1: counts and partial |row| sums of this block */
    for (i = row_lo; i < row_hi; i++)
        for (int k = 0; k < m_per_row; k++) {
            gen(i, k, &j, &v);
            if (j != i && j >= col_lo && j < col_hi) { cnt[i - row_lo]++; rsum[i - row_lo] += v; }
        }
    for (j = col_lo; j < col_hi; j++)
        for (int k = 0; k < m_per_row; k++) {
            gen(j, k, &i, &v);
            if (i != j && i >= row_lo && i < row_hi) { cnt[i - row_lo]++; rsum[i - row_lo] += v; }
        }
    long dlo = row_lo > col_lo ? row_lo : col_lo, dhi = row_hi < col_hi ? row_hi : col_hi;
    for (i = dlo; i < dhi; i++) cnt[i - row_lo]++;
    row_allreduce(rsum, nrows);                 /* full off-diagonal row sums */

    rowptr = malloc(((size_t)nrows + 1) * sizeof(long));
    rowptr[0] = 0;
    for (long li = 0; li < nrows; li++) rowptr[li + 1] = rowptr[li] + cnt[li];
    long nnz = rowptr[nrows];
    colidx = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    val = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    for (long li = 0; li < nrows; li++) cnt[li] = rowptr[li];
    /* pass 2: fill */
    for (i = row_lo; i < row_hi; i++)
        for (int k = 0; k < m_per_row; k++) {
            gen(i, k, &j, &v);
            if (j != i && j >= col_lo && j < col_hi) { long p = cnt[i - row_lo]++; colidx[p] = (int)(j - col_lo); val[p] = -v; }
        }
    for (j = col_lo; j < col_hi; j++)
        for (int k = 0; k < m_per_row; k++) {
            gen(j, k, &i, &v);
            if (i != j && i >= row_lo && i < row_hi) { long p = cnt[i - row_lo]++; colidx[p] = (int)(j - col_lo); val[p] = -v; }
        }
    for (i = dlo; i < dhi; i++) { long p = cnt[i - row_lo]++; colidx[p] = (int)(i - col_lo); val[p] = rsum[i - row_lo] + 1.0; }

    long tot = 0;
    MPI_Reduce(&nnz, &tot, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (me == 0) printf(" nonzeros: %ld (%.1f per row)\n", tot, (double)tot / na);
    free(cnt); free(rsum);
}

static double conj_grad(double *x, double *z, double *p, double *q, double *r) {
    for (long i = 0; i < ncols; i++) { q[i] = 0; z[i] = 0; r[i] = x[i]; p[i] = r[i]; }
    double rho = dot(r, r);
    for (int it = 0; it < CGITMAX; it++) {
        matvec(p, q);
        double d = dot(p, q);
        double alpha = rho / d, rho0 = rho;
        for (long i = 0; i < ncols; i++) { z[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        rho = dot(r, r);
        double beta = rho / rho0;
        for (long i = 0; i < ncols; i++) p[i] = r[i] + beta * p[i];
    }
    matvec(z, r);
    double s = 0.0;
    for (long i = 0; i < ncols; i++) { double d = x[i] - r[i]; s += d * d; }
    row_allreduce(&s, 1);
    return sqrt(s);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    char cl = 'S'; int niter = -1, opt;
    while ((opt = getopt(argc, argv, "c:n:")) != -1) {
        if (opt == 'c') cl = optarg[0];
        else if (opt == 'n') niter = atoi(optarg);
    }
    const cls_t *C = NULL;
    for (size_t k = 0; k < sizeof classes / sizeof classes[0]; k++) if (classes[k].name[0] == cl) C = &classes[k];
    if (!C) { if (me == 0) fprintf(stderr, "unknown class %c\n", cl); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (P & (P - 1)) { if (me == 0) fprintf(stderr, "cg_mini needs a power-of-two number of ranks\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    na = C->na; if (niter < 0) niter = C->niter;
    m_per_row = C->nonzer * (C->nonzer + 1) / 2;

    int lg = 0; while ((1 << lg) < P) lg++;
    nprows = 1 << (lg / 2); npcols = P / nprows;
    prow = me / npcols; pcol = me % npcols;
    row_lo = blk(na, nprows, prow); row_hi = blk(na, nprows, prow + 1);
    col_lo = blk(na, npcols, pcol); col_hi = blk(na, npcols, pcol + 1);
    nrows = row_hi - row_lo; ncols = col_hi - col_lo;

    if (me == 0) printf(" CG-mini class %c: na=%ld nonzer=%d niter=%d shift=%.1f  P=%d grid=%dx%d  msg(row)=%ld B\n",
                        cl, na, C->nonzer, niter, C->shift, P, nprows, npcols, nrows * 8);
    long mx = nrows > ncols ? nrows : ncols;
    wbuf = malloc((size_t)mx * sizeof(double)); tmpbuf = malloc((size_t)mx * sizeof(double));
    double *x = malloc(ncols * sizeof(double)), *z = malloc(ncols * sizeof(double)),
           *p = malloc(ncols * sizeof(double)), *q = malloc(ncols * sizeof(double)), *r = malloc(ncols * sizeof(double));

    double t0 = MPI_Wtime();
    build_matrix();
    for (long i = 0; i < ncols; i++) x[i] = 1.0;
    conj_grad(x, z, p, q, r);                       /* untimed warm-up iteration, as NPB */
    for (long i = 0; i < ncols; i++) x[i] = 1.0;
    if (me == 0) printf(" setup %.3f s\n", MPI_Wtime() - t0);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1);                                 /* traced region starts */
    double t = MPI_Wtime(), zeta = 0;
    for (int it = 1; it <= niter; it++) {
        double rnorm = conj_grad(x, z, p, q, r);
        double n1 = dot(x, z), n2 = dot(z, z);
        n2 = 1.0 / sqrt(n2);
        zeta = C->shift + 1.0 / n1;
        for (long i = 0; i < ncols; i++) x[i] = n2 * z[i];
        if (me == 0 && (it == 1 || it % 10 == 0 || it == niter))
            printf("  it %4d  ||r|| = %.10e  zeta = %.12f\n", it, rnorm, zeta);
    }
    t = MPI_Wtime() - t;
    MPI_Pcontrol(0);                                 /* traced region ends */

    double tmax; MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (me == 0) printf(" RESULT app=cg class=%c P=%d time=%.6f zeta=%.12f\n", cl, P, tmax, zeta);
    MPI_Finalize();
    return 0;
}
