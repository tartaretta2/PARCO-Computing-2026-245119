#!/usr/bin/env bash
set -euo pipefail

# CONFIGURAZIONE
SOURCE="main.c"                 # C source file to be compiled
OUTDIR="results"                # results and logs folder
MATRIX_DIR="data_matrices"      # matrices folder
#FILENAME="market_matrix.mtx"    # Matrix Market file name
SEQ_BIN="seq_program"           # binary name for sequential version
PAR_BIN="par_program"           # binary name for parallel version
REPEATS=3
TIMEOUT_SECS=600

MATRIX_FILES=("$MATRIX_DIR"/*.mtx)

RUN_TYPES_PAR=("static" "dynamic" "guided")
CHUNK_SIZES=(1 10 100 )
NUM_THREADS=(1 2 4 8)

CC=${CC:-gcc}
PERF_CMD=$(command -v perf || true)

mkdir -p "$OUTDIR"/{bin,logs,perf}
LOGDIR="$OUTDIR/logs"
PERFDIR="$OUTDIR/perf"
BINDIR="$OUTDIR/bin"

CSV="$OUTDIR/summary.csv"
echo "run_type,mode,schedule,chunk,threads,repeat,metric,value,log_file" > "$CSV"

# Funzione per timeout
run_with_timeout() {
  local cmd=("$@")
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status ${TIMEOUT_SECS}s "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

###############################################
# COMPILAZIONI
###############################################

echo "Sequential compilation (-O3)"
$CC -O3 -std=c11 -o "${BINDIR}/${SEQ_BIN}" "*.c"

echo "Parallel compilation (no -O3)"
$CC -fopenmp -std=c11 -o "${BINDIR}/${PAR_BIN}" "*.c"

###############################################
# RUN SEQUENZIALI
###############################################
echo -e "\n=== SEQUENTIAL RUNS ===\n"
for MATRIX in "${MATRIX_FILES[@]}"; do
    FILENAME=$(basename "$MATRIX")
    echo -e "\n### USING MATRIX: $FILENAME ###\n"
    for ((i=1;i<=REPEATS;i++)); do
    stamp=$(date +"%Y%m%d_%H%M%S")
    logf="${LOGDIR}/seq_runs.log"
    perf_logf="${PERFDIR}/seq_perf.log"

    echo "Sequential run #$i"
    start=$(date +%s.%N)
    run_with_timeout "${BINDIR}/${SEQ_BIN}" "$FILENAME"> "$logf" 2>&1 || rc=$? || rc=$?
    end=$(date +%s.%N)
    elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN{print e - s}')
    echo "sequential,code,-,-,-,${i},time,${elapsed},${logf}" >> "$CSV"

    if [[ -n "$PERF_CMD" ]]; then
        echo "  → perf run #$i"
        run_with_timeout "$PERF_CMD" stat -e cache-misses -o "$perf_logf" "${BINDIR}/${SEQ_BIN}" > /dev/null 2>&1 || true
        cache_miss=$(grep "cache-misses" "$perf_logf" | awk '{print $1}' | tr -d ',')
        echo "sequential,perf,-,-,-,${i},cache-misses,${cache_miss:-NA},${perf_logf}" >> "$CSV"
    fi
    done
done
###############################################
# RUN PARALLELE
###############################################
echo -e "\n=== PARALLEL RUNS ===\n"
for MATRIX in "${MATRIX_FILES[@]}"; do
    FILENAME=$(basename "$MATRIX")
    echo -e "\n### USING MATRIX: $FILENAME ###\n"
    for schedule in "${RUN_TYPES_PAR[@]}"; do
    echo -e "\n--- ${schedule} scheduling ---\n"
        for chunk in "${CHUNK_SIZES[@]}"; do
            for th in "${NUM_THREADS[@]}"; do
                for ((i=1;i<=REPEATS;i++)); do
                    stamp=$(date +"%Y%m%d_%H%M%S")
                    logf="${LOGDIR}/${schedule}_c${chunk}_t${th}.log"
                    perf_logf="${PERFDIR}/${schedule}_c${chunk}_t${th}_perf.log"

                    echo "[$schedule] chunk=$chunk threads=$th run #$i"

                    start=$(date +%s.%N)
                    run_with_timeout "${BINDIR}/${PAR_BIN}" "$FILENAME" "$schedule" "$chunk" "$th" >> "$logf" 2>&1 || rc=$? || rc=$?
                    end=$(date +%s.%N)

                    elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN{print e - s}')

                    # Scrive un'intestazione solo se il file è nuovo
                    if [[ $i -eq 1 ]]; then
                    echo "=== [$schedule] chunk=$chunk threads=$th ===" >> "$logf"
                    fi

                    # Aggiunge la riga di tempo nel log
                    echo "Run #$i elapsed time: ${elapsed}s" >> "$logf"

                    # Salva anche nel CSV
                    echo "parallel,code,${schedule},${chunk},${th},${i},time,${elapsed},${logf}" >> "$CSV"

                    # Perf optional
                    if [[ "$PERF_CMD" ]]; then
                        run_with_timeout "$PERF_CMD" stat -e cache-misses -o "$perf_logf.tmp" "${BINDIR}/${PAR_BIN}" "$FILENAME" "$schedule" "$chunk" "$th" > /dev/null 2>&1 || true
                        cache_miss=$(grep "cache-misses" "$perf_logf.tmp" | awk '{print $1}' | tr -d ',')
                        echo "Run #$i cache-misses: ${cache_miss:-NA}" >> "$perf_logf"
                        echo "parallel,perf,${schedule},${chunk},${th},${i},cache-misses,${cache_miss:-NA},${perf_logf}" >> "$CSV"
                        rm -f "$perf_logf.tmp"
                    fi
                done
            done
        done
    done
done
echo "=== Runs completed ==="
echo "CSV summary: $CSV"
echo "Code logs: $LOGDIR"
echo "Perf logs: $PERFDIR"
