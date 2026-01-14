# Makefile con supporto OpenMP condizionale
# Usage:
#   make              -> MPI puro
#   make USE_OPENMP=1 -> MPI+OpenMP

CC = mpicc
CFLAGS = -O3 -Wall -std=c99
LDFLAGS = -lm

# Se USE_OPENMP=1, aggiungi flag OpenMP
ifdef USE_OPENMP
    CFLAGS += -fopenmp -D_OPENMP
    LDFLAGS += -fopenmp
    $(info Building with OpenMP support)
else
    $(info Building MPI-only version)
endif

# Directories
SRC_DIR = src
INCLUDE_DIR = include
RESULTS_DIR = results
BIN_DIR = $(RESULTS_DIR)/bin

# Source files
UTILS_SRC = $(SRC_DIR)/utils.c
MPI_UTILS_SRC = $(SRC_DIR)/mpi_utils.c
MMIO_SRC = $(SRC_DIR)/mmio.c
MAIN_SRC = $(SRC_DIR)/main.c
GEN_MATRIX_SRC = $(SRC_DIR)/generate_random_matrix.c

# Object files
UTILS_OBJ = $(BIN_DIR)/utils.o
MPI_UTILS_OBJ = $(BIN_DIR)/mpi_utils.o
MMIO_OBJ = $(BIN_DIR)/mmio.o
MAIN_OBJ = $(BIN_DIR)/main.o

# Executables
SPMV_EXEC = $(BIN_DIR)/spmv_mpi
GEN_EXEC = $(BIN_DIR)/generate_matrix

# Include path
INCLUDES = -I$(INCLUDE_DIR)

.PHONY: all clean dirs test

all: dirs $(SPMV_EXEC) $(GEN_EXEC)

dirs:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RESULTS_DIR)/mpi_plots
	@mkdir -p $(RESULTS_DIR)/mpi_logs

# Compile object files
$(UTILS_OBJ): $(UTILS_SRC) $(INCLUDE_DIR)/utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(MPI_UTILS_OBJ): $(MPI_UTILS_SRC) $(INCLUDE_DIR)/mpi_utils.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(MMIO_OBJ): $(MMIO_SRC) $(INCLUDE_DIR)/mmio.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link executables
$(SPMV_EXEC): $(MAIN_OBJ) $(UTILS_OBJ) $(MPI_UTILS_OBJ) $(MMIO_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ Built: $(SPMV_EXEC)"

$(GEN_EXEC): $(GEN_MATRIX_SRC) $(UTILS_OBJ) $(MMIO_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)
	@echo "✓ Built: $(GEN_EXEC)"

clean:
	rm -rf $(BIN_DIR)/*.o $(SPMV_EXEC) $(GEN_EXEC)
	@echo "✓ Cleaned build artifacts"

clean-results:
	rm -f $(RESULTS_DIR)/*.csv
	rm -rf $(RESULTS_DIR)/mpi_plots/*
	rm -rf $(RESULTS_DIR)/mpi_logs/*
	@echo "✓ Cleaned results"

# Test targets
test:
	@echo "Testing with P=4..."
	mpirun -np 4 $(SPMV_EXEC) data_matrices/test.mtx 5

help:
	@echo "Available targets:"
	@echo "  make              - Build MPI-only version"
	@echo "  make USE_OPENMP=1 - Build MPI+OpenMP version"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-results- Remove CSV and plots"
	@echo "  make test         - Run quick test"
	@echo "  make help         - Show this help"