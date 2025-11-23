#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend senza display per cluster
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Configurazione
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "results" / "code_summary.csv"
OUTPUT_DIR = BASE_DIR / "results" / "code_plots"

print(f"[INFO] Base directory: {BASE_DIR}")
print(f"[INFO] CSV path: {CSV_PATH}")
print(f"[INFO] Output directory: {OUTPUT_DIR}")

# Crea directory output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verifica che il CSV esista
if not CSV_PATH.exists():
    print(f"[ERROR] CSV file not found: {CSV_PATH}")
    sys.exit(1)

# Leggi il CSV
try:
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Successfully loaded CSV with {len(df)} rows")
except Exception as e:
    print(f"[ERROR] Failed to read CSV: {e}") 

# Converti colonne numeriche
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['chunk'] = pd.to_numeric(df['chunk'], errors='coerce')
df['threads'] = pd.to_numeric(df['threads'], errors='coerce')

# Filtra solo le righe con metric='time'
df_time = df[df['metric'] == 'time'].copy()

print(f"[INFO] Total measurements: {len(df_time)}")
print(f"[INFO] Matrices: {df_time['matrix'].unique().tolist()}")
print(f"[INFO] Dataframe shape: {df_time.shape}\n")

# Crea identificatore configurazione
def create_config_id(row):
    if row['run_type'] == 'sequential':
        return 'seq'
    else:
        return f"{row['schedule']}_c{int(row['chunk'])}_t{int(row['threads'])}"

df_time['config'] = df_time.apply(create_config_id, axis=1)

###############################################
# 1. CALCOLA 90° PERCENTILE PER OGNI CONFIGURAZIONE
###############################################
print("[INFO] Computing 90th percentile for all configurations...")

p90_data = []

for matrix in df_time['matrix'].unique():
    df_matrix = df_time[df_time['matrix'] == matrix]
    
    # Sequential
    seq_data = df_matrix[df_matrix['run_type'] == 'sequential']
    if len(seq_data) > 0:
        p90_seq = np.percentile(seq_data['value'], 90)
        p90_data.append({
            'matrix': matrix,
            'run_type': 'sequential',
            'schedule': 'sequential',
            'chunk': '-',
            'threads': '-',
            'config': 'seq',
            'p90_time_ms': p90_seq * 1000,  # Converti in millisecondi
            'count': len(seq_data)
        })
    
    # Parallel
    df_parallel = df_matrix[df_matrix['run_type'] == 'parallel']
    
    for schedule in df_parallel['schedule'].unique():
        df_sched = df_parallel[df_parallel['schedule'] == schedule]
        
        for chunk in df_sched['chunk'].unique():
            df_chunk = df_sched[df_sched['chunk'] == chunk]
            
            for thread in df_chunk['threads'].unique():
                df_config = df_chunk[df_chunk['threads'] == thread]
                
                if len(df_config) > 0:
                    p90 = np.percentile(df_config['value'], 90)
                    p90_data.append({
                        'matrix': matrix,
                        'run_type': 'parallel',
                        'schedule': schedule,
                        'chunk': int(chunk),
                        'threads': int(thread),
                        'config': f"{schedule}_c{int(chunk)}_t{int(thread)}",
                        'p90_time_ms': p90 * 1000,
                        'count': len(df_config)
                    })

# Crea DataFrame con 90° percentili
df_p90 = pd.DataFrame(p90_data)

# Salva CSV con tutti i 90° percentili
p90_csv_path = OUTPUT_DIR / "90perc.csv"
df_p90.to_csv(p90_csv_path, index=False)
print(f"[INFO] Saved 90th percentiles to: {p90_csv_path}")
print(f"[INFO] Total configurations: {len(df_p90)}\n")

###############################################
# 2. TROVA MIGLIOR CHUNK PER OGNI SCHEDULE-THREAD
###############################################
print("[INFO] Finding best chunk size for each schedule-thread combination...")

best_configs_data = []

for matrix in df_p90['matrix'].unique():
    df_matrix = df_p90[df_p90['matrix'] == matrix]
    
    # Aggiungi sequential (non ha chunk/thread variations)
    seq_config = df_matrix[df_matrix['run_type'] == 'sequential']
    if len(seq_config) > 0:
        best_configs_data.append(seq_config.iloc[0].to_dict())
    
    # Per ogni schedule e thread, trova il miglior chunk
    df_parallel = df_matrix[df_matrix['run_type'] == 'parallel']
    
    for schedule in df_parallel['schedule'].unique():
        df_sched = df_parallel[df_parallel['schedule'] == schedule]
        
        for thread in df_sched['threads'].unique():
            df_thread = df_sched[df_sched['threads'] == thread]
            
            # Trova il chunk con il tempo minimo (90° percentile)
            best_idx = df_thread['p90_time_ms'].idxmin()
            best_config = df_thread.loc[best_idx].to_dict()
            best_configs_data.append(best_config)

# Crea DataFrame con le migliori configurazioni
df_best = pd.DataFrame(best_configs_data)

# Salva CSV con le migliori configurazioni
best_csv_path = OUTPUT_DIR / "best_configs.csv"
df_best.to_csv(best_csv_path, index=False)
print(f"[INFO] Saved best configurations to: {best_csv_path}")
print(f"[INFO] Total best configs: {len(df_best)}\n")

###############################################
# 3. CREA GRAFICI PER OGNI MATRICE
###############################################
print("[INFO] Generating performance plots...")

for matrix in df_best['matrix'].unique():
    df_matrix_best = df_best[df_best['matrix'] == matrix]
    
    print(f"[INFO] Creating plot for {matrix}...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colori per ogni tipo di scheduling
    colors = {
        'sequential': '#e74c3c',  # Rosso
        'static': '#3498db',       # Blu
        'dynamic': '#2ecc71',      # Verde
        'guided': '#f39c12'        # Arancione
    }
    
    markers = {
        'sequential': 'o',
        'static': 's',
        'dynamic': '^',
        'guided': 'D'
    }
    
    # Plot sequential (linea orizzontale)
    seq_data = df_matrix_best[df_matrix_best['run_type'] == 'sequential']
    if len(seq_data) > 0:
        seq_time = seq_data['p90_time_ms'].values[0]
        # Threads disponibili nel dataset parallelo
        df_parallel = df_matrix_best[df_matrix_best['run_type'] == 'parallel']
        if len(df_parallel) > 0:
            thread_range = sorted(df_parallel['threads'].unique())
            ax.plot(thread_range, [seq_time] * len(thread_range), 
                   color=colors['sequential'], linewidth=2.5, 
                   linestyle='--', marker=markers['sequential'], 
                   markersize=8, label='Sequential', zorder=3)
    
    # Plot parallel per ogni schedule
    df_parallel = df_matrix_best[df_matrix_best['run_type'] == 'parallel']
    
    for schedule in ['static', 'dynamic', 'guided']:
        df_sched = df_parallel[df_parallel['schedule'] == schedule]
        
        if len(df_sched) == 0:
            continue
        
        # Ordina per numero di thread
        df_sched_sorted = df_sched.sort_values('threads')
        
        threads = df_sched_sorted['threads'].values
        times = df_sched_sorted['p90_time_ms'].values
        chunks = df_sched_sorted['chunk'].values
        
        # Plot linea
        ax.plot(threads, times, 
               color=colors[schedule], linewidth=2.5, 
               marker=markers[schedule], markersize=8, 
               label=f'{schedule.capitalize()}', zorder=2)
        
        # Annota il chunk size migliore su ogni punto
        for i, (t, time, chunk) in enumerate(zip(threads, times, chunks)):
            ax.annotate(f'c={int(chunk)}', 
                       xy=(t, time), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', 
                       fontsize=8,
                       color=colors[schedule],
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor=colors[schedule], 
                                alpha=0.7))
    
    # Configurazione assi
    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Time (ms) - 90th Percentile', fontsize=13, fontweight='bold')
    ax.set_title(f'Performance Comparison - {matrix}', fontsize=15, fontweight='bold')
    
    # Scala logaritmica per l'asse x (threads sono potenze di 2)
    ax.set_xscale('log', base=2)
    
    # Imposta i tick sull'asse x per mostrare tutti i valori di thread
    if len(df_parallel) > 0:
        thread_values = sorted(df_parallel['threads'].unique())
        ax.set_xticks(thread_values)
        ax.set_xticklabels([str(int(t)) for t in thread_values])
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.grid(True, which='major', alpha=0.5, linestyle='-')
    
    # Legenda
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Layout
    plt.tight_layout()
    
    # Salva plot
    plot_path = OUTPUT_DIR / f"performance_{matrix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved: {plot_path.name}")

###############################################
# 4. STAMPA SUMMARY
###############################################
print("\n" + "=" * 80)
print("BEST CONFIGURATIONS SUMMARY (90th percentile)")
print("=" * 80)

for matrix in df_best['matrix'].unique():
    print(f"\n{matrix}:")
    print("-" * 80)
    
    df_matrix_best = df_best[df_best['matrix'] == matrix]
    
    # Sequential
    seq = df_matrix_best[df_matrix_best['run_type'] == 'sequential']
    if len(seq) > 0:
        seq_time = seq['p90_time_ms'].values[0]
        print(f"  Sequential: {seq_time:.4f} ms")
    
    # Parallel
    df_par = df_matrix_best[df_matrix_best['run_type'] == 'parallel']
    
    for schedule in ['static', 'dynamic', 'guided']:
        df_sched = df_par[df_par['schedule'] == schedule].sort_values('threads')
        
        if len(df_sched) == 0:
            continue
        
        print(f"\n  {schedule.capitalize()} scheduling:")
        for _, row in df_sched.iterrows():
            speedup = seq_time / row['p90_time_ms'] if len(seq) > 0 and seq_time > 0 else 0
            print(f"    threads={int(row['threads']):2d}, best_chunk={int(row['chunk']):5d}, "
                  f"time={row['p90_time_ms']:8.4f} ms, speedup={speedup:5.2f}x")

# Best overall configuration per matrice
print("\n" + "=" * 80)
print("BEST OVERALL PARALLEL CONFIGURATION PER MATRIX")
print("=" * 80)

for matrix in df_best['matrix'].unique():
    df_matrix_best = df_best[df_best['matrix'] == matrix]
    df_par = df_matrix_best[df_matrix_best['run_type'] == 'parallel']
    
    if len(df_par) > 0:
        best_idx = df_par['p90_time_ms'].idxmin()
        best = df_par.loc[best_idx]
        
        seq = df_matrix_best[df_matrix_best['run_type'] == 'sequential']
        if len(seq) > 0:
            seq_time = seq['p90_time_ms'].values[0]
            speedup = seq_time / best['p90_time_ms']
        else:
            speedup = 0
        
        print(f"\n{matrix}:")
        print(f"  Schedule: {best['schedule']}")
        print(f"  Chunk: {int(best['chunk'])}")
        print(f"  Threads: {int(best['threads'])}")
        print(f"  Time (90th perc): {best['p90_time_ms']:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")

print(f"\n\n[SUCCESS] Analysis completed!")
print(f"[INFO] Results saved in: {OUTPUT_DIR}/")
print(f"  - 90perc.csv: All configurations with 90th percentile")
print(f"  - best_configs.csv: Best chunk for each schedule-thread pair")
print(f"  - performance_*.png: Performance plots for each matrix")
print("=" * 80)