#ifndef UTILS_H
#define UTILS_H

typedef struct
{
    int M;              // Number of rows
    int N;              // Number of columns
    int nnz;            // Number of non-zero entries
    int *I;             // COO row indices
    int *J;             // COO column indices
    double *val;        // COO values
    int *prefixSum;     // CSR row pointer
    int *sorted_J;      // CSR column indices
    double *sorted_val; // CSR values
} Matrix;

// MPI-side local structure
typedef struct {
    int local_M;
    int local_nnz;
    int *row_ptr;
    int *col_idx;
    double *vals;
    int *global_row_ids;
} LocalCSR;

void readMatrixMarket(const char *filename, Matrix *mat);

void coo_to_csr(Matrix *mat);

void localcsr_to_matrix(const LocalCSR *local, Matrix *mat);

void csr_matvec(const Matrix *mat, const double *x, double *y); //Del.1 implementation
void spmv_csr_local_matrix(const Matrix *mat, const double *x, double *y); //Del.2 distributed implementation

#endif
