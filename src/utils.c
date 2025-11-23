#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mmio.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Function to read a matrix in Matrix Market format
void readMatrixMarket(const char *filename, Matrix *mat)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL)
        exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, &mat->M, &mat->N, &mat->nz)) != 0)
        exit(1);

    mat->I = (int *)malloc(mat->nz * sizeof(int));
    mat->J = (int *)malloc(mat->nz * sizeof(int));
    mat->val = (double *)malloc(mat->nz * sizeof(double));

    int i;
    for (i = 0; i < mat->nz; i++)
    {
        fscanf(f, "%d %d %lf\n", &mat->I[i], &mat->J[i], &mat->val[i]);
        mat->I[i]--; // adjust from 1-based to 0-based
        mat->J[i]--;
    }

    if (f != stdin)
        fclose(f);
}

// Function to convert COO format to CSR format
void coo_to_csr(Matrix *mat)
{
    int i;

    // Count non-zeros per row
    int *row_counts = (int *)calloc(mat->M, sizeof(int));
    for (i = 0; i < mat->nz; i++)
    {
        row_counts[mat->I[i]]++;
    }

    // Prefix sum creation
    mat->prefixSum = (int *)malloc((mat->M + 1) * sizeof(int));
    mat->prefixSum[0] = 0;
    for (i = 0; i < mat->M; i++)
    {
        mat->prefixSum[i + 1] = mat->prefixSum[i] + row_counts[i];
    }

    // CSR arrays allocation
    mat->sorted_J = (int *)malloc(mat->nz * sizeof(int));
    mat->sorted_val = (double *)malloc(mat->nz * sizeof(double));

    // Fill CSR arrays
    int *next_pos = (int *)malloc((mat->M + 1) * sizeof(int));

    for (i = 0; i < mat->M + 1; i++)
    {
        next_pos[i] = mat->prefixSum[i];
    }

    for (i = 0; i < mat->nz; i++)
    {
        int row = mat->I[i];
        int dest = next_pos[row];

        mat->sorted_val[dest] = mat->val[i];
        mat->sorted_J[dest] = mat->J[i];
        next_pos[row]++;
    }

    free(row_counts);
    free(next_pos);
}

// function to perform CSR matrix-vector multiplication
void csr_matvec(const Matrix *mat, const double *x, double *y)
{

    int i;
    #ifdef _OPENMP
        #pragma omp parallel for schedule(runtime)
    #endif
    for (i = 0; i < mat->M; i++)
    {         
        double sum = 0.0;
        int j;
        /* vectorization (works best with regular matrices)
          ifdef _OPENMP
            pragma omp simd
          endif
        */ 
        for (j = mat->prefixSum[i]; j < mat->prefixSum[i + 1]; j++)
        {
            /*  next iteration prefetch
              idef __GNUC__
                __builtin_prefetch(&x[col[j+8]], 0, 1);
                __builtin_prefetch(&val[j+8],    0, 1);
                __builtin_prefetch(&col[j+8],    0, 1);
              endif
            */
            sum += mat->sorted_val[j] * x[mat->sorted_J[j]];
        }
        y[i] = sum;
    }
}

void print_csr(const Matrix *mat)
{
    printf("\nSorted values: ");
    int i;
    for (i = 0; i < mat->nz; ++i)
    {
        printf("%lf | ", mat->sorted_val[i]);
    }

    printf("\nPrefix sum: ");
    for (i = 0; i <= mat->M; ++i)
    {
        printf("%d | ", mat->prefixSum[i]);
    }
    printf("\n");
}

void print_result(const double *y, int n)
{
    printf("\nResult vector: ");
    int i;
    for(i = 0; i < n; ++i){
      printf("%.3f", y[i]);
      if( i < n-1 ) printf(" | ");
    }
}