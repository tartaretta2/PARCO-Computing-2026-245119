#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "mpi_utils.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s matrix.mtx repeats [num_threads]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *matrix_file = argv[1];
    int repeats = atoi(argv[2]);
    
    //Default 1 thread; if OMP is active, read args
    int num_threads = 1;
    
#ifdef _OPENMP
    if (argc >= 4) {
        num_threads = atoi(argv[3]);
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
#else
    if (argc >= 4) {
        num_threads = atoi(argv[3]);
    }
#endif

    if (rank == 0) {
        printf("====================================\n");
        printf("SpMV Configuration\n");
        printf("====================================\n");
        printf("MPI processes:     %d\n", size);
        printf("OpenMP threads:    %d\n", num_threads);
#ifdef _OPENMP
        printf("OpenMP:            ENABLED\n");
#else
        printf("OpenMP:            DISABLED\n");
#endif
        printf("====================================\n\n");
    }

    //READ MATRIX (rank 0)
    Matrix global_mat;
    int global_M = 0, global_N = 0, global_nnz = 0;
    
    if (rank == 0) {
        readMatrixMarket(matrix_file, &global_mat);
        printf("Matrix %s: %d x %d  nnz=%d\n",
               matrix_file,
               global_mat.M,
               global_mat.N,
               global_mat.nnz);
        global_M = global_mat.M;
        global_N = global_mat.N;
        global_nnz = global_mat.nnz;
    }

    // Broadcast dimensions to all ranks
    MPI_Bcast(&global_M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //DISTRIBUTE MATRIX
    LocalCSR local_csr;
    distribute_matrix_1d_cyclic(
        rank, size,
        global_M,
        global_N,
        global_nnz,
        rank == 0 ? global_mat.I : NULL,
        rank == 0 ? global_mat.J : NULL,
        rank == 0 ? global_mat.val : NULL,
        &local_csr
    );

    if (rank == 0) {
        free(global_mat.I);
        free(global_mat.J);
        free(global_mat.val);
    }

    Matrix local_mat;
    localcsr_to_matrix(&local_csr, &local_mat);

    //LOAD BALANCE STATISTICS
    int local_nnz = local_mat.nnz;
    
    int *all_nnz = NULL;
    if (rank == 0) {
        all_nnz = malloc(size * sizeof(int));
    }
    
    MPI_Gather(&local_nnz, 1, MPI_INT, all_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int min_nnz = all_nnz[0];
        int max_nnz = all_nnz[0];
        long long sum_nnz = 0;
        
        for (int i = 0; i < size; i++) {
            if (all_nnz[i] < min_nnz) min_nnz = all_nnz[i];
            if (all_nnz[i] > max_nnz) max_nnz = all_nnz[i];
            sum_nnz += all_nnz[i];
        }
        
        double avg_nnz = (double)sum_nnz / size;
        double imbalance_pct = ((double)(max_nnz - min_nnz) / avg_nnz) * 100.0;
        
        const char *matrix_name = strrchr(matrix_file, '/');
        matrix_name = matrix_name ? matrix_name + 1 : matrix_file;
        char clean_name[256];
        strncpy(clean_name, matrix_name, sizeof(clean_name) - 1);
        clean_name[sizeof(clean_name) - 1] = '\0';
        char *dot = strrchr(clean_name, '.');
        if (dot) *dot = '\0';
        
#ifdef _OPENMP
        const char *csv_load = "../results/mpix_load_balance.csv";
#else
        const char *csv_load = "../results/mpi_load_balance.csv";
#endif
        
        FILE *fp_load = fopen(csv_load, "a");
        if (fp_load) {
            fprintf(fp_load, "%s,%d,%d,%d,%.2f,%d,%.2f\n",
                    clean_name, size, num_threads, min_nnz, avg_nnz, max_nnz, imbalance_pct);
            fclose(fp_load);
            printf("✓ Load balance stats written to CSV\n");
        }
        
        printf("\n=== LOAD BALANCE ===\n");
        printf("Min NNZ per rank: %d\n", min_nnz);
        printf("Avg NNZ per rank: %.2f\n", avg_nnz);
        printf("Max NNZ per rank: %d\n", max_nnz);
        printf("Imbalance: %.2f%%\n\n", imbalance_pct);
        
        free(all_nnz);
    }

    //CREATE AND DISTRIBUTE VECTOR x
    int local_n = 0;
    for (int i = 0; i < global_N; i++)
        if (i % size == rank)
            local_n++;

    double *local_x = malloc(local_n * sizeof(double));
    
    double *x_global = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    double *sendbuf = NULL;
    
    if (rank == 0) {
        x_global = malloc(global_N * sizeof(double));
        for (int i = 0; i < global_N; i++)
            x_global[i] = drand48();
        
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        
        for (int p = 0; p < size; p++) {
            sendcounts[p] = 0;
            for (int i = 0; i < global_N; i++)
                if (i % size == p)
                    sendcounts[p]++;
        }
        
        displs[0] = 0;
        for (int p = 1; p < size; p++)
            displs[p] = displs[p-1] + sendcounts[p-1];
        
        sendbuf = malloc(global_N * sizeof(double));
        int *pos = calloc(size, sizeof(int));
        
        for (int i = 0; i < global_N; i++) {
            int p = i % size;
            int idx = displs[p] + pos[p];
            sendbuf[idx] = x_global[i];
            pos[p]++;
        }
        
        free(pos);
        free(x_global);
    }
    
    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE,
                 local_x, local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        free(sendbuf);
        free(sendcounts);
        free(displs);
    }

    double *local_y = malloc(local_mat.M * sizeof(double));
    double *comm_times = malloc(repeats * sizeof(double));
    double *comp_times = malloc(repeats * sizeof(double));

    //WARM-UP
    double *full_x = NULL;
    communicate_vector_cyclic(rank, size, global_N, local_x, &full_x);
    spmv_csr_local_matrix(&local_mat, full_x, local_y);
    free(full_x);

    MPI_Barrier(MPI_COMM_WORLD);

    //TIMED LOOP
    for (int it = 0; it < repeats; it++) {
        double t0 = MPI_Wtime();
        communicate_vector_cyclic(rank, size, global_N, local_x, &full_x);
        double t1 = MPI_Wtime();

        spmv_csr_local_matrix(&local_mat, full_x, local_y);
        double t2 = MPI_Wtime();

        if (size > 1)
          comm_times[it] = t1 - t0;
        else 
          comm_times[it] = 0;
          
        comp_times[it] = t2 - t1;

        free(full_x);
    }

    //REDUCE (MAX over ranks)
    double *max_comm = NULL;
    double *max_comp = NULL;

    if (rank == 0) {
        max_comm = malloc(repeats * sizeof(double));
        max_comp = malloc(repeats * sizeof(double));
    }

    for (int i = 0; i < repeats; i++) {
        MPI_Reduce(&comm_times[i],
                   rank == 0 ? &max_comm[i] : NULL,
                   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&comp_times[i],
                   rank == 0 ? &max_comp[i] : NULL,
                   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    //WRITE TO CSV
    if (rank == 0) {
        const char *matrix_name = strrchr(matrix_file, '/');
        matrix_name = matrix_name ? matrix_name + 1 : matrix_file;
        
        char clean_name[256];
        strncpy(clean_name, matrix_name, sizeof(clean_name) - 1);
        clean_name[sizeof(clean_name) - 1] = '\0';
        char *dot = strrchr(clean_name, '.');
        if (dot) *dot = '\0';

#ifdef _OPENMP
        const char *csv_times = "../results/mpix_times.csv";
#else
        const char *csv_times = "../results/mpi_times.csv";
#endif

        FILE *fp = fopen(csv_times, "a");
        if (fp) {
            for (int i = 0; i < repeats; i++) {
                fprintf(fp, "%s,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e\n",
                        clean_name,
                        size,
                        num_threads,
                        global_M,
                        global_N,
                        global_nnz,
                        max_comm[i],
                        max_comp[i],
                        max_comm[i] + max_comp[i]);
            }
            fclose(fp);
            printf("✓ Written %d timing records to %s\n", repeats, csv_times);
        }

        double avg_comm = 0.0, avg_comp = 0.0;
        for (int i = 0; i < repeats; i++) {
            avg_comm += max_comm[i];
            avg_comp += max_comp[i];
        }
        avg_comm /= repeats;
        avg_comp /= repeats;

        printf("\n=== TIMING RESULTS ===\n");
        printf("Avg communication: %.6e s\n", avg_comm);
        printf("Avg computation:   %.6e s\n", avg_comp);
        printf("Avg total:         %.6e s\n", avg_comm + avg_comp);
        printf("======================\n\n");

        free(max_comm);
        free(max_comp);
    }

    //CLEANUP
    free(comm_times);
    free(comp_times);
    free(local_x);
    free(local_y);

    free(local_csr.row_ptr);
    free(local_csr.col_idx);
    free(local_csr.vals);
    free(local_csr.global_row_ids);

    MPI_Finalize();
    return 0;
}