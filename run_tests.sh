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
$CC -O3 -fopenmp -std=c11 -o "${BINDIR}/${PAR_BIN}" "*.c"

for MATRIX in "${MATRIX_FILES[@]}"; do
    FILENAME=$(basename "$MATRIX")
    echo -e "\n### USING MATRIX: $FILENAME ###\n"
    
    ###############################################
    # RUN SEQUENZIALI
    ###############################################
    echo -e "\n=== SEQUENTIAL RUNS ===\n"
    for ((i=1;i<=REPEATS;i++)); do
        stamp=$(date +"%Y%m%d_%H%M%S")
        logf="${LOGDIR}/seq_runs.log"
        perf_logf="${PERFDIR}/seq_perf.log"

        echo "Sequential run #$i"
        start=$(date +%s.%N)
        run_with_timeout "${BINDIR}/${SEQ_BIN}" "data_matrices/${FILENAME}"> "$logf.tmp" 2>&1 || rc=$? || rc=$?
        end=$(date +%s.%N)
        
        elapsed=$(grep -Eo 'SPMV_TIME=[0-9.]+' "$logf.tmp" | cut -d= -f2 | tail -n1)
        if [[ -z "$elapsed" ]]; then
            elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN{print e - s}')
        fi

        cat "$logf.tmp" >> "$logf"
        rm -f "$logf.tmp"
        
        echo "Run #$i time_from_program: ${elapsed}s" >> "$logf"
        echo "sequential,code,-,-,-,${i},time,${elapsed},${logf}" >> "$CSV"

        if [[ -n "$PERF_CMD" ]]; then
            echo "  → perf run #$i"
            run_with_timeout "$PERF_CMD" stat -e cache-misses -o "$perf_logf" "${BINDIR}/${SEQ_BIN}" "data_matrices/${FILENAME}" > /dev/null 2>&1 || true
            cache_miss=$(grep "cache-misses" "$perf_logf" | awk '{print $1}' | tr -d ',')
            echo "sequential,perf,-,-,-,${i},cache-misses,${cache_miss:-NA},${perf_logf}" >> "$CSV"
        fi
    done

    ###############################################
    # RUN PARALLELE
    ###############################################
    echo -e "\n=== PARALLEL RUNS ===\n"
    for schedule in "${RUN_TYPES_PAR[@]}"; do
        echo -e "\n--- ${schedule} scheduling ---\n"
        for chunk in "${CHUNK_SIZES[@]}"; do
            for th in "${NUM_THREADS[@]}"; do
                for ((i=1;i<=REPEATS;i++)); do
                    stamp=$(date +"%Y%m%d_%H%M%S")
                    logf="${LOGDIR}/${schedule}_c${chunk}_t${th}.log"

                    echo "[$schedule] chunk=$chunk threads=$th run #$i"

                    run_with_timeout "${BINDIR}/${PAR_BIN}" "data_matrices/${FILENAME}" "$schedule" "$chunk" "$th" > "$logf.tmp" 2>&1 || rc=$? || rc=$?

                    elapsed=$(grep -Eo 'SPMV_TIME=[0-9.]+' "$logf.tmp" | cut -d= -f2 | tail -n1)

                    if [[ -z "$elapsed" ]]; then
                        start=$(date +%s.%N)
                        run_with_timeout "${BINDIR}/${PAR_BIN}" "data_matrices/${FILENAME}" "$schedule" "$chunk" "$th" > /dev/null 2>&1 || rc=$? || rc=$?
                        end=$(date +%s.%N)
                        elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN{print e - s}')
                    fi

                    cat "$logf.tmp" >> "$logf"
                    rm -f "$logf.tmp"

                    echo "Run #$i time_from_program: ${elapsed}s" >> "$logf"
                    echo "parallel,code,${schedule},${chunk},${th},${i},time,${elapsed},${logf}" >> "$CSV"

                    # Perf optional
                    if [[ "$PERF_CMD" ]]; then
                        run_with_timeout "$PERF_CMD" stat -e cache-misses -o "$perf_logf.tmp" "${BINDIR}/${PAR_BIN}" "data_matrices/${FILENAME}" "$schedule" "$chunk" "$th" > /dev/null 2>&1 || true
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

echo -e "\n=== Runs completed ===\n"
echo "CSV summary: $CSV"
echo "Code logs: $LOGDIR"
echo "Perf logs: $PERFDIR"