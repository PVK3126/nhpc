/*
 * pingpong.c -- IMB-3.0 style Ping-Pong (paper Fig. 5) plus an "overlap"
 * mode (paper Fig. 1): both ranks Isend/Irecv, compute, then Waitall.
 *
 * Every size is bracketed by MPI_Pcontrol(100+2i) / MPI_Pcontrol(101+2i)
 * markers so the simulator can predict the time of each size separately.
 *
 * usage: pingpong [-m pingpong|overlap] [-x maxbytes] [-c comp_us] [-o out.csv]
 * Run with 2 ranks (other ranks idle).
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void spin(double s) { double t = MPI_Wtime(); while (MPI_Wtime() - t < s) ; }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *mode = "pingpong", *out = "pingpong.csv";
    long maxk = 4L << 20; double comp_us = 0;
    int opt;
    while ((opt = getopt(argc, argv, "m:x:c:o:")) != -1) {
        if (opt == 'm') mode = optarg;
        else if (opt == 'x') maxk = atol(optarg);
        else if (opt == 'c') comp_us = atof(optarg);
        else if (opt == 'o') out = optarg;
    }
    int overlap = strcmp(mode, "overlap") == 0;
    if (size < 2) { if (rank == 0) fprintf(stderr, "need 2 ranks\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    char *sb = malloc(maxk + 1), *rb = malloc(maxk + 1);
    memset(sb, 1, maxk + 1); memset(rb, 0, maxk + 1);
    FILE *f = NULL;
    if (rank == 0) {
        f = fopen(out, "w");
        if (!f) { perror(out); MPI_Abort(MPI_COMM_WORLD, 1); }
        fprintf(f, "mode,bytes,reps,divisor,comp_us,t_usec,mark_start,mark_end\n");
        printf("# %s  comp=%.1f us\n#   bytes   reps     t[usec]\n", mode, comp_us);
    }

    int idx = 0;
    for (long K = 0; K <= maxk; K = (K == 0 ? 1 : 2 * K), idx++) {
        long reps = K == 0 ? 1000 : (40L << 20) / K;          /* IMB: 40 MB per size */
        if (reps > 1000) reps = 1000;
        if (reps < 10) reps = 10;
        int peer = 1 - rank;
        double t = 0;
        for (int phase = 0; phase < 2; phase++) {              /* phase 0: warm-up */
            long n = phase == 0 ? 2 : reps;
            MPI_Barrier(MPI_COMM_WORLD);
            if (phase == 1) { MPI_Pcontrol(100 + 2 * idx); t = MPI_Wtime(); }
            if (rank < 2) {
                for (long r = 0; r < n; r++) {
                    if (!overlap) {
                        if (rank == 0) { MPI_Send(sb, K, MPI_CHAR, 1, 1, MPI_COMM_WORLD); MPI_Recv(rb, K, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
                        else           { MPI_Recv(rb, K, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Send(sb, K, MPI_CHAR, 0, 1, MPI_COMM_WORLD); }
                    } else {
                        MPI_Request rq[2];
                        MPI_Irecv(rb, K, MPI_CHAR, peer, 2, MPI_COMM_WORLD, &rq[0]);
                        MPI_Isend(sb, K, MPI_CHAR, peer, 2, MPI_COMM_WORLD, &rq[1]);
                        spin(comp_us * 1e-6);
                        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                    }
                }
            }
            if (phase == 1) { t = MPI_Wtime() - t; MPI_Pcontrol(101 + 2 * idx); }
        }
        int div = overlap ? 1 : 2;
        if (rank == 0) {
            double us = t / reps / div * 1e6;
            printf("%9ld %6ld %11.3f\n", K, reps, us);
            fprintf(f, "%s,%ld,%ld,%d,%.3f,%.6f,%d,%d\n", mode, K, reps, div, comp_us, us, 2 * idx, 2 * idx + 1);
        }
    }
    if (rank == 0) fclose(f);
    free(sb); free(rb);
    MPI_Finalize();
    return 0;
}
