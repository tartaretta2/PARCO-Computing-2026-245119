//
//  main.c
//  SparseMatrix
//
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timer.h"
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, const char *argv[])
{

    #ifdef _OPENMP
    if (argc != 5)
    {
        printf("Usage: %s <schedule_type> <chunk_size> <num_threads>\n", argv[0]);
        printf("schedule_type: static | dynamic | guided | auto\n");
        return 1;
    }

    // ---- PARSE ARGUMENTS ----
    const char *sched_str = argv[2];
    int chunk = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    // ---- MAP STRING TO OMP ENUM ----
    omp_sched_t sched_type;

    if (strcmp(sched_str, "static") == 0)
        sched_type = omp_sched_static;
    else if (strcmp(sched_str, "dynamic") == 0)
        sched_type = omp_sched_dynamic;
    else if (strcmp(sched_str, "guided") == 0)
        sched_type = omp_sched_guided;
    else
    {
        printf("Unknown schedule type '%s', default to 'auto'.\n", sched_str);
        return omp_sched_auto;
    }

    // ---- SET OMP PARAMETERS ----
    omp_set_num_threads(num_threads);
    omp_set_schedule(sched_type, chunk);

    // ---- DEBUG PRINT ----
    int nt = omp_get_num_threads();
    omp_sched_t st;
    int cs;
    omp_get_schedule(&st, &cs);
    printf("\nParallel run using schedule = %d , chunk size = %d, threads = %d\n", st, cs, num_threads);
#else
    printf("\nSequential run\n");
#endif

    srand(time(NULL));

    double start, finish, elapsed;

    // Matrix declaration
    Matrix mat;
    readMatrixMarket(argv[1], &mat);
    coo_to_csr(&mat);
    // Vector declaration and initialization
    double *vector = (double *)malloc(mat.N * sizeof(double));
    double *result = (double *)malloc(mat.M * sizeof(double));

    int i;
    for (i = 0; i < mat.N; i++)
    {
        vector[i] = 1.0; // Example initialization
    }

    // print matrix CSR vector
    //print_csr(&mat);

    // Perform CSR matrix-vector multiplication
    // clock_t start = clock();
    GET_TIME(start);
    csr_matvec(&mat, vector, result);
    GET_TIME(finish)

    // double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    elapsed = finish - start;
    printf("\nTime taken for CSR matrix-vector multiplication: %f seconds\n\n", elapsed);

    // Free allocated memory
    free(mat.I);
    free(mat.J);
    free(mat.val);
    free(mat.prefixSum);
    free(mat.sorted_J);
    free(mat.sorted_val);
    free(vector);
    free(result);

    return 0;
}
