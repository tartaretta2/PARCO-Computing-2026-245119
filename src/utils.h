#ifndef UTILS_H
#define UTILS_H

typedef struct
{
    int M;              // Number of rows
    int N;              // Number of columns
    int nz;             // Number of non-zero entries
    int *I;             // Row indices
    int *J;             // Column indices
    double *val;        // Non-zero values
    int *prefixSum;     // CSR row pointer array
    int *sorted_J;      // CSR column indices
    double *sorted_val; // CSR non-zero values
} Matrix;

void readMatrixMarket(const char* filename, Matrix *mat);
void coo_to_csr(Matrix *mat);
void csr_matvec(const Matrix *mat, const double *x, double *y);
void print_csr(const Matrix *mat);
void print_result(const double *y, int n);

#endif