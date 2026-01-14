#include "mpi_utils.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void distribute_matrix_1d_cyclic(
    int rank, int size,
    int M, int N, int nnz,
    int *row_ind, int *col_ind, double *values,
    LocalCSR *local_csr)
{


    //STEP 1: Determine rows owner
    int local_M = 0;
    for (int i = 0; i < M; i++)
        if (i % size == rank)
            local_M++;

    local_csr->local_M = local_M;
    local_csr->global_row_ids = malloc(local_M * sizeof(int));

    int idx = 0;
    for (int i = 0; i < M; i++)
        if (i % size == rank)
            local_csr->global_row_ids[idx++] = i;

    //STEP 2: Distribute COO triplets
    int local_nnz = 0;
    int *local_I = NULL;
    int *local_J = NULL;
    double *local_val = NULL;

    if (rank == 0)
    {
        int *nnz_per_rank = calloc(size, sizeof(int));

        for (int i = 0; i < nnz; i++)
            nnz_per_rank[row_ind[i] % size]++;

        int **send_I = malloc(size * sizeof(int *));
        int **send_J = malloc(size * sizeof(int *));
        double **send_val = malloc(size * sizeof(double *));
        int *offset = calloc(size, sizeof(int));

        for (int p = 0; p < size; p++)
        {
            send_I[p]   = malloc(nnz_per_rank[p] * sizeof(int));
            send_J[p]   = malloc(nnz_per_rank[p] * sizeof(int));
            send_val[p] = malloc(nnz_per_rank[p] * sizeof(double));
        }

        for (int i = 0; i < nnz; i++)
        {
            int p = row_ind[i] % size;
            int k = offset[p]++;
            send_I[p][k]   = row_ind[i];
            send_J[p][k]   = col_ind[i];
            send_val[p][k] = values[i];
        }

        for (int p = 0; p < size; p++)
        {
            if (p == 0)
            {
                local_nnz = nnz_per_rank[0];
                local_I   = send_I[0];
                local_J   = send_J[0];
                local_val = send_val[0];
            }
            else
            {
                MPI_Send(&nnz_per_rank[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(send_I[p], nnz_per_rank[p], MPI_INT, p, 1, MPI_COMM_WORLD);
                MPI_Send(send_J[p], nnz_per_rank[p], MPI_INT, p, 2, MPI_COMM_WORLD);
                MPI_Send(send_val[p], nnz_per_rank[p], MPI_DOUBLE, p, 3, MPI_COMM_WORLD);
                free(send_I[p]);
                free(send_J[p]);
                free(send_val[p]);
            }
        }

        free(send_I);
        free(send_J);
        free(send_val);
        free(nnz_per_rank);
        free(offset);
    }
    else
    {
        MPI_Status status;
        MPI_Recv(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        local_I   = malloc(local_nnz * sizeof(int));
        local_J   = malloc(local_nnz * sizeof(int));
        local_val = malloc(local_nnz * sizeof(double));

        MPI_Recv(local_I, local_nnz, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(local_J, local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(local_val, local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
    }

    local_csr->local_nnz = local_nnz;

    //STEP 3: Convert GLOBAL row indices to LOCAL
    for (int i = 0; i < local_nnz; i++)
        local_I[i] = local_I[i] / size;

    //STEP 4: Build LOCAL CSR via Matrix abstraction
    Matrix local_mat;
    local_mat.M   = local_M;
    local_mat.N   = N;
    local_mat.nnz = local_nnz;
    local_mat.I   = local_I;
    local_mat.J   = local_J;
    local_mat.val = local_val;

    local_mat.prefixSum  = NULL;
    local_mat.sorted_J   = NULL;
    local_mat.sorted_val = NULL;

    coo_to_csr(&local_mat);

    local_csr->row_ptr = local_mat.prefixSum;
    local_csr->col_idx = local_mat.sorted_J;
    local_csr->vals    = local_mat.sorted_val;

    //STEP 5: Cleanup COO buffers
    free(local_I);
    free(local_J);
    free(local_val);
}


void communicate_vector_cyclic(
    int rank, int size,
    int N,
    double *local_x,
    double **full_x)
{
    *full_x = calloc(N, sizeof(double));

    int k = 0;
    for (int i = 0; i < N; i++) {
        if (i % size == rank) {
            (*full_x)[i] = local_x[k++];
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE,
        *full_x,
        N,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );
}