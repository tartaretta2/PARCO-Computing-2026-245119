#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include "utils.h"
 
void distribute_matrix_1d_cyclic(
    int rank, int size,                    
    int M, int N, int nnz,                 
    int* row_ind, int* col_ind, double* values, 
    LocalCSR* local_csr
);

void communicate_vector_cyclic(
    int rank, int size,
    int N,
    double *local_x,
    double **full_x
);

#endif