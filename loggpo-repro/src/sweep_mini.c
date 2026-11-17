/*
 * sweep_mini.c -- Sweep3D-style discrete-ordinates transport sweep
 * (paper section 3.2.2, "Sweep3D 50x50x50").
 *
 * KBA wavefront on a px x py process grid: for each of the 8 octants, each
 * angle block and each k-plane block, a rank receives the inflow faces from
 * its upstream x/y neighbours (MPI_Recv), sweeps the block with diamond
 * differencing + negative flux fix-up, and sends the outflow faces downstream
 * (MPI_Send) -- the same blocking pipelined pattern as Sweep3D.  A global
 * MPI_Allreduce checks convergence each iteration.
 *
 * Message size = ny_local * mk * mmi * 8 bytes (x faces); with the default
 * 50^3 grid, mk=5, mmi=3 on 2x2 ranks that is 3000 bytes (paper: 2-3 KB).
 *
 * usage: sweep_mini [-g nx,ny,nz] [-W] [-k mk] [-a mmi] [-m angles/octant] [-i iters]
 *   -W   weak scaling: -g is the per-rank sub-grid instead of the global grid
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#define TAGX 11
#define TAGY 12

static long blk(long n, int nb, int b) { return (long)((long long)n * b / nb); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int gx = 50, gy = 50, gz = 50, weak = 0, mk = 5, mmi = 3, nm = 6, iters = 20, opt;
    while ((opt = getopt(argc, argv, "g:Wk:a:m:i:")) != -1) {
        if (opt == 'g') sscanf(optarg, "%d,%d,%d", &gx, &gy, &gz);
        else if (opt == 'W') weak = 1;
        else if (opt == 'k') mk = atoi(optarg);
        else if (opt == 'a') mmi = atoi(optarg);
        else if (opt == 'm') nm = atoi(optarg);
        else if (opt == 'i') iters = atoi(optarg);
    }
    int px = 1;
    for (int d = 1; d * d <= P; d++) if (P % d == 0) px = d;
    int py = P / px;
    int ix = rank % px, iy = rank / px;
    long nx_g = weak ? (long)gx * px : gx, ny_g = weak ? (long)gy * py : gy;
    int nxl = (int)(blk(nx_g, px, ix + 1) - blk(nx_g, px, ix));
    int nyl = (int)(blk(ny_g, py, iy + 1) - blk(ny_g, py, iy));
    int nz = gz;
    if (mmi > nm) mmi = nm;

    if (rank == 0)
        printf(" Sweep-mini: global %ldx%ldx%d  P=%d grid=%dx%d  mk=%d mmi=%d angles/oct=%d iters=%d  msg(x)=%d B\n",
               nx_g, ny_g, nz, P, px, py, mk, mmi, nm, iters, nyl * mk * mmi * 8);

    long ncell = (long)nxl * nyl * nz;
    double *src = malloc(ncell * sizeof(double)), *flux = calloc(ncell, sizeof(double)),
           *fold = calloc(ncell, sizeof(double));
    double *xbuf = malloc((size_t)nyl * mk * mmi * sizeof(double));
    double *ybuf = malloc((size_t)nxl * mk * mmi * sizeof(double));
    double *zface = malloc((size_t)nxl * nyl * mmi * sizeof(double));
    double *mu = malloc(nm * sizeof(double)), *eta = malloc(nm * sizeof(double)),
           *xi = malloc(nm * sizeof(double)), *w = malloc(nm * sizeof(double));
    for (int m = 0; m < nm; m++) {
        double th = (m + 0.5) / nm * (M_PI / 2), ph = (m + 0.5) / nm;
        mu[m] = cos(th) * sqrt(1 - ph * ph * 0.5);
        eta[m] = sin(th) * sqrt(1 - ph * ph * 0.5);
        xi[m] = ph * sqrt(0.5);
        w[m] = 1.0 / (8.0 * nm);
    }
    const double sigt = 1.0, sigs = 0.5, qext = 1.0, dx = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1);
    double t = MPI_Wtime(), err = 0;
    int nkb = (nz + mk - 1) / mk, nab = (nm + mmi - 1) / mmi;
    for (int it = 0; it < iters; it++) {
        for (long c = 0; c < ncell; c++) { src[c] = qext + sigs * flux[c]; flux[c] = 0.0; }
        for (int oct = 0; oct < 8; oct++) {
            int sx = (oct & 1) ? -1 : 1, sy = (oct & 2) ? -1 : 1, sz = (oct & 4) ? -1 : 1;
            int upx = sx > 0 ? (ix > 0 ? rank - 1 : -1) : (ix < px - 1 ? rank + 1 : -1);
            int dnx = sx > 0 ? (ix < px - 1 ? rank + 1 : -1) : (ix > 0 ? rank - 1 : -1);
            int upy = sy > 0 ? (iy > 0 ? rank - px : -1) : (iy < py - 1 ? rank + px : -1);
            int dny = sy > 0 ? (iy < py - 1 ? rank + px : -1) : (iy > 0 ? rank - px : -1);
            for (int ab = 0; ab < nab; ab++) {
                int na_b = (ab + 1) * mmi <= nm ? mmi : nm - ab * mmi;
                memset(zface, 0, (size_t)nxl * nyl * mmi * sizeof(double));     /* vacuum z boundary */
                for (int kb = 0; kb < nkb; kb++) {
                    int k0, mkb;
                    if (sz > 0) { k0 = kb * mk; mkb = nz - k0 < mk ? nz - k0 : mk; }
                    else        { k0 = nz - 1 - kb * mk; mkb = k0 + 1 < mk ? k0 + 1 : mk; }
                    int nxm = nyl * mkb * mmi, nym = nxl * mkb * mmi;
                    if (upx >= 0) MPI_Recv(xbuf, nxm, MPI_DOUBLE, upx, TAGX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    else memset(xbuf, 0, nxm * sizeof(double));
                    if (upy >= 0) MPI_Recv(ybuf, nym, MPI_DOUBLE, upy, TAGY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    else memset(ybuf, 0, nym * sizeof(double));

                    for (int kk = 0; kk < mkb; kk++) {
                        int k = sz > 0 ? k0 + kk : k0 - kk;
                        for (int a = 0; a < na_b; a++) {
                            int m = ab * mmi + a;
                            double cx = 2 * mu[m] / dx, cy = 2 * eta[m] / dx, cz = 2 * xi[m] / dx;
                            double den = 1.0 / (sigt + cx + cy + cz);
                            for (int jj = 0; jj < nyl; jj++) {
                                int j = sy > 0 ? jj : nyl - 1 - jj;
                                double *pxp = &xbuf[((long)kk * nyl + j) * mmi + a];
                                double pxv = *pxp;
                                for (int ii = 0; ii < nxl; ii++) {
                                    int i = sx > 0 ? ii : nxl - 1 - ii;
                                    double *pyp = &ybuf[((long)kk * nxl + i) * mmi + a];
                                    double *pzp = &zface[((long)j * nxl + i) * mmi + a];
                                    long c = ((long)k * nyl + j) * nxl + i;
                                    double psi = (src[c] + cx * pxv + cy * *pyp + cz * *pzp) * den;
                                    if (psi < 0) psi = 0;
                                    pxv = 2 * psi - pxv;   if (pxv < 0) pxv = 0;
                                    *pyp = 2 * psi - *pyp; if (*pyp < 0) *pyp = 0;
                                    *pzp = 2 * psi - *pzp; if (*pzp < 0) *pzp = 0;
                                    flux[c] += w[m] * psi;
                                }
                                *pxp = pxv;
                            }
                        }
                    }
                    if (dnx >= 0) MPI_Send(xbuf, nxm, MPI_DOUBLE, dnx, TAGX, MPI_COMM_WORLD);
                    if (dny >= 0) MPI_Send(ybuf, nym, MPI_DOUBLE, dny, TAGY, MPI_COMM_WORLD);
                }
            }
        }
        double lerr = 0;
        for (long c = 0; c < ncell; c++) {
            double d = fabs(flux[c] - fold[c]) / (fabs(flux[c]) + 1e-30);
            if (d > lerr) lerr = d;
            fold[c] = flux[c];
        }
        MPI_Allreduce(&lerr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0 && (it < 3 || (it + 1) % 5 == 0 || it == iters - 1))
            printf("  iter %3d  max rel change = %.6e\n", it + 1, err);
    }
    t = MPI_Wtime() - t;
    MPI_Pcontrol(0);

    double lsum = 0, sum = 0, tmax;
    for (long c = 0; c < ncell; c++) lsum += flux[c];
    MPI_Reduce(&lsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf(" RESULT app=sweep P=%d time=%.6f flux_sum=%.10e\n", P, tmax, sum);
    MPI_Finalize();
    return 0;
}
