#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_random_sparse_matrix(const char* filename, int N, int avg_nnz_per_row) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: cannot create %s\n", filename);
        return;
    }
    
    // Header Matrix Market
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", N, N, N * avg_nnz_per_row);
    
    srand(time(NULL));
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < avg_nnz_per_row; k++) {
            int j = rand() % N;  
            double val = (double)rand() / RAND_MAX;  
            
            //Matrix Market Format: row col value
            fprintf(f, "%d %d %.10e\n", i+1, j+1, val);
        }
    }
    
    fclose(f);
    printf("Generated random matrix: %s (%dx%d, ~%d nnz)\n", 
           filename, N, N, N * avg_nnz_per_row);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <output.mtx> <N> <avg_nnz_per_row>\n", argv[0]);
        return 1;
    }
    
    const char* output = argv[1];
    int N = atoi(argv[2]);
    int avg_nnz_per_row = atoi(argv[3]);
    
    generate_random_sparse_matrix(output, N, avg_nnz_per_row);
    
    return 0;
}