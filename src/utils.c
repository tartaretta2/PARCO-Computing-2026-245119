#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mmio.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//READ MATRIX MARKET (COO) FILE
void readMatrixMarket(const char *filename, Matrix *mat)
{
    FILE *f;
    MM_typecode matcode;

    if ((f = fopen(filename, "r")) == NULL) {
        perror("fopen");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        fprintf(stderr, "Unsupported Matrix Market format.\n");
        exit(1);
    }

    if (mm_read_mtx_crd_size(f, &mat->M, &mat->N, &mat->nnz) != 0) {
        fprintf(stderr, "Failed reading matrix size.\n");
        exit(1);
    }

    mat->I   = malloc(mat->nnz * sizeof(int));
    mat->J   = malloc(mat->nnz * sizeof(int));
    mat->val = malloc(mat->nnz * sizeof(double));

    for (int i = 0; i < mat->nnz; i++) {
        fscanf(f, "%d %d %lf",
               &mat->I[i], &mat->J[i], &mat->val[i]);
        mat->I[i]--;
        mat->J[i]--;
    }

    mat->prefixSum  = NULL;
    mat->sorted_J   = NULL;
    mat->sorted_val = NULL;

    fclose(f);
}

void coo_to_csr(Matrix *mat)
{
    int *row_counts = calloc(mat->M, sizeof(int));

    for (int i = 0; i < mat->nnz; i++)
        row_counts[mat->I[i]]++;

    mat->prefixSum = malloc((mat->M + 1) * sizeof(int));
    mat->prefixSum[0] = 0;

    for (int i = 0; i < mat->M; i++)
        mat->prefixSum[i + 1] = mat->prefixSum[i] + row_counts[i];

    mat->sorted_J   = malloc(mat->nnz * sizeof(int));
    mat->sorted_val = malloc(mat->nnz * sizeof(double));

    int *next_pos = malloc(mat->M * sizeof(int));
    for (int i = 0; i < mat->M; i++)
        next_pos[i] = mat->prefixSum[i];

    for (int i = 0; i < mat->nnz; i++) {
        int row = mat->I[i];
        int pos = next_pos[row]++;
        mat->sorted_J[pos]   = mat->J[i];
        mat->sorted_val[pos] = mat->val[i];
    }

    free(row_counts);
    free(next_pos);
}

void localcsr_to_matrix(const LocalCSR *local, Matrix *mat)
{
    mat->M = local->local_M;
    mat->N = -1;
    mat->nnz = local->local_nnz;

    mat->I = NULL;
    mat->J = NULL;
    mat->val = NULL;

    mat->prefixSum  = local->row_ptr;
    mat->sorted_J   = local->col_idx;
    mat->sorted_val = local->vals;
}


//SpMV (GLOBAL / OPENMP)
void csr_matvec(const Matrix *mat, const double *x, double *y)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
    #endif
    for (int i = 0; i < mat->M; i++) {
        double sum = 0.0;
        for (int j = mat->prefixSum[i]; j < mat->prefixSum[i + 1]; j++)
            sum += mat->sorted_val[j] * x[mat->sorted_J[j]];
        y[i] = sum;
    }
}

//SpMV (LOCAL / MPI)
void spmv_csr_local_matrix(const Matrix *mat,
                           const double *x,
                           double *y)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < mat->M; i++) {
        double sum = 0.0;
        for (int j = mat->prefixSum[i]; j < mat->prefixSum[i + 1]; j++)
            sum += mat->sorted_val[j] * x[mat->sorted_J[j]];
        y[i] = sum;
    }
}