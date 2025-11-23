//
//  main.c
//  SparseMatrix
//
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "timer.h"
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, const char *argv[])
{

    #ifdef _OPENMP
    if (argc != 6)
    {
        printf("Usage: %s <schedule_type> <chunk_size> <num_threads>\n", argv[0]);
        printf("schedule_type: static | dynamic | guided | auto\n");
        return 1;
    }

    // ---- PARSE ARGUMENTS ----
    const char *sched_str = argv[3];
    int chunk = atoi(argv[4]);
    int num_threads = atoi(argv[5]);

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
    
    omp_sched_t st;
    int cs;
    omp_get_schedule(&st, &cs);
    
    #endif

    srand(time(NULL));

    double start, finish, elapsed;
    int repeats = atoi(argv[2]);

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
        vector[i] = ((double)rand() / (double)RAND_MAX) * 20.0 - 10.0;
    }

    // print matrix CSR vector
    //print_csr(&mat);

    // Perform CSR matrix-vector multiplication
    const char *matrix_name = strrchr(argv[1], '/');
    if (matrix_name != NULL) {
      matrix_name++;  // salta lo slash
    } else {
        matrix_name = argv[1]; // non c’è lo slash
    }
    printf("\n=== MATRIX: %s ===\n", matrix_name);
    
    
    //WARM UP RUN
    #ifdef _OPENMP
      printf("\nSchedule = %d | Chunk size = %d | Threads = %d | WARM UP RUN", st, cs, num_threads);
    #else
     printf("\nSequential WARM UP RUN");
    #endif
    GET_TIME(start);
    csr_matvec(&mat, vector, result);
    GET_TIME(finish)

    elapsed = finish - start;
    
    //print_result(result, (&mat)->N);
    printf("\nWARMUP_TIME=%f\n\n", elapsed);
    
    
    
    for(i = 0; i < repeats; ++i){

      #ifdef _OPENMP
        printf("\nSchedule = %d | Chunk size = %d | Threads = %d | Run #%d", st, cs, num_threads, i);
      #else
       printf("\nSequential run #%d", i);
      #endif
      GET_TIME(start);
      csr_matvec(&mat, vector, result);
      GET_TIME(finish)
  
      elapsed = finish - start;
      
      //print_result(result, (&mat)->N);
      printf("\nSPMV_TIME=%f\n\n", elapsed);
    }
    

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
