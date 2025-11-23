#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Configurazione
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "results3" / "perf_summary.csv"
OUTPUT_DIR = BASE_DIR / "results3" / "perf_plots"

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
    sys.exit(1)

print(f"[INFO] Total measurements: {len(df)}")
print(f"[INFO] Matrices: {df['matrix'].unique().tolist()}")
print(f"[INFO] Columns: {df.columns.tolist()}\n")

# Converti colonne numeriche
df['chunk'] = pd.to_numeric(df['chunk'], errors='coerce')
df['threads'] = pd.to_numeric(df['threads'], errors='coerce')

# Parsing dei valori dalle colonne
def parse_metric(col_str):
    try:
        if pd.isna(col_str) or col_str == 'NA':
            return np.nan
        if '=' in str(col_str):
            return float(str(col_str).split('=')[1])
        return float(col_str)
    except:
        return np.nan

# Parse delle colonne metriche
df['L1_misses_val'] = df['L1_misses'].apply(parse_metric)
df['L1_percent_val'] = df['L1_percent'].apply(parse_metric)
df['LLC_misses_val'] = df['LLC_misses'].apply(parse_metric)
df['LLC_percent_val'] = df['LLC_percent'].apply(parse_metric)

print("[INFO] Parsed metric values")
print(f"[INFO] L1 miss rate range: {df['L1_percent_val'].min():.2f}% - {df['L1_percent_val'].max():.2f}%")
print(f"[INFO] LLC miss rate range: {df['LLC_percent_val'].min():.2f}% - {df['LLC_percent_val'].max():.2f}%\n")

# Crea identificatore configurazione
def create_config_id(row):
    if row['run_type'] == 'sequential':
        return 'seq'
    else:
        schedule = row['schedule'] if pd.notna(row['schedule']) else 'unk'
        chunk = int(row['chunk']) if pd.notna(row['chunk']) else 0
        threads = int(row['threads']) if pd.notna(row['threads']) else 0
        return f"{schedule}_c{chunk}_t{threads}"

df['config'] = df.apply(create_config_id, axis=1)

# Calcola statistiche aggregate
print("[INFO] Computing statistics...")
stats = df.groupby(['matrix', 'config', 'run_type', 'schedule', 'chunk', 'threads']).agg({
    'L1_misses_val': ['mean', 'std', 'min', 'max'],
    'L1_percent_val': ['mean', 'std', 'min', 'max'],
    'LLC_misses_val': ['mean', 'std', 'min', 'max'],
    'LLC_percent_val': ['mean', 'std', 'min', 'max'],
    'repeat': 'count'
}).reset_index()

# Flatten column names
stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]

# Salva statistiche
stats_csv = OUTPUT_DIR / "perf_statistics_summary.csv"
stats.to_csv(stats_csv, index=False)
print(f"[INFO] Statistics saved to {stats_csv}\n")

# Ordine fisso per chunk e threads
CHUNK_ORDER = [1, 10, 100, 1000, 10000]
THREAD_ORDER = [1, 2, 4, 8, 16, 32, 64]

# === PLOT 1: Heatmap cache miss - TUTTI GLI SCHEDULING IN UNA FIGURA ===
print("[INFO] Generating combined cache miss heatmaps...")
for matrix in df['matrix'].unique():
    df_par = df[(df['matrix'] == matrix) & (df['run_type'] == 'parallel')]
    
    if len(df_par) == 0:
        continue
    
    # 3 scheduling x 2 metriche (L1 e LLC) = 3 colonne, 2 righe
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for idx, schedule in enumerate(['static', 'dynamic', 'guided']):
        df_sched = df_par[df_par['schedule'] == schedule]
        
        if len(df_sched) == 0:
            axes[0, idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, idx].transAxes)
            axes[1, idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, idx].transAxes)
            continue
        
        # L1 heatmap (riga 0)
        pivot_l1 = df_sched.groupby(['chunk', 'threads'])['L1_percent_val'].mean().reset_index()
        pivot_table_l1 = pivot_l1.pivot(index='chunk', columns='threads', values='L1_percent_val')
        
        available_chunks = [c for c in CHUNK_ORDER if c in pivot_table_l1.index]
        available_threads = [t for t in THREAD_ORDER if t in pivot_table_l1.columns]
        pivot_table_l1 = pivot_table_l1.reindex(index=available_chunks, columns=available_threads)
        
        vmin_l1 = pivot_table_l1.values.min()
        vmax_l1 = pivot_table_l1.values.max()
        
        im1 = axes[0, idx].imshow(pivot_table_l1.values, aspect='auto', cmap='RdYlGn_r',
                                  interpolation='nearest', vmin=vmin_l1, vmax=vmax_l1)
        
        axes[0, idx].set_xticks(np.arange(len(available_threads)))
        axes[0, idx].set_yticks(np.arange(len(available_chunks)))
        axes[0, idx].set_xticklabels(available_threads, fontsize=9)
        axes[0, idx].set_yticklabels(available_chunks, fontsize=9)
        axes[0, idx].set_xlabel('Threads', fontsize=10, fontweight='bold')
        axes[0, idx].set_ylabel('Chunk Size', fontsize=10, fontweight='bold')
        axes[0, idx].set_title(f'{schedule.capitalize()} - L1 Miss Rate (%)', fontsize=11, fontweight='bold')
        
        for i in range(len(available_chunks)):
            for j in range(len(available_threads)):
                val = pivot_table_l1.values[i, j]
                if not np.isnan(val):
                    norm_val = (val - vmin_l1) / (vmax_l1 - vmin_l1) if vmax_l1 != vmin_l1 else 0.5
                    text_color = 'white' if norm_val > 0.5 else 'black'
                    axes[0, idx].text(j, i, f'{val:.2f}', ha="center", va="center",
                                     color=text_color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im1, ax=axes[0, idx], label='Miss Rate (%)')
        
        # LLC heatmap (riga 1)
        pivot_llc = df_sched.groupby(['chunk', 'threads'])['LLC_percent_val'].mean().reset_index()
        pivot_table_llc = pivot_llc.pivot(index='chunk', columns='threads', values='LLC_percent_val')
        
        available_chunks_llc = [c for c in CHUNK_ORDER if c in pivot_table_llc.index]
        available_threads_llc = [t for t in THREAD_ORDER if t in pivot_table_llc.columns]
        pivot_table_llc = pivot_table_llc.reindex(index=available_chunks_llc, columns=available_threads_llc)
        
        vmin_llc = pivot_table_llc.values.min()
        vmax_llc = pivot_table_llc.values.max()
        
        im2 = axes[1, idx].imshow(pivot_table_llc.values, aspect='auto', cmap='RdYlGn_r',
                                  interpolation='nearest', vmin=vmin_llc, vmax=vmax_llc)
        
        axes[1, idx].set_xticks(np.arange(len(available_threads_llc)))
        axes[1, idx].set_yticks(np.arange(len(available_chunks_llc)))
        axes[1, idx].set_xticklabels(available_threads_llc, fontsize=9)
        axes[1, idx].set_yticklabels(available_chunks_llc, fontsize=9)
        axes[1, idx].set_xlabel('Threads', fontsize=10, fontweight='bold')
        axes[1, idx].set_ylabel('Chunk Size', fontsize=10, fontweight='bold')
        axes[1, idx].set_title(f'{schedule.capitalize()} - LLC Miss Rate (%)', fontsize=11, fontweight='bold')
        
        for i in range(len(available_chunks_llc)):
            for j in range(len(available_threads_llc)):
                val = pivot_table_llc.values[i, j]
                if not np.isnan(val):
                    norm_val = (val - vmin_llc) / (vmax_llc - vmin_llc) if vmax_llc != vmin_llc else 0.5
                    text_color = 'white' if norm_val > 0.5 else 'black'
                    axes[1, idx].text(j, i, f'{val:.2f}', ha="center", va="center",
                                     color=text_color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im2, ax=axes[1, idx], label='Miss Rate (%)')
    
    plt.suptitle(f'Cache Miss Rates Heatmaps - {matrix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"heatmap_cache_{matrix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {plot_path.name}")

# === PLOT 2: Miss rate vs numero di threads ===
print("[INFO] Generating miss rate vs threads plots...")
for matrix in df['matrix'].unique():
    df_par = df[(df['matrix'] == matrix) & (df['run_type'] == 'parallel')]
    
    if len(df_par) == 0:
        continue
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, schedule in enumerate(['static', 'dynamic', 'guided']):
        df_sched = df_par[df_par['schedule'] == schedule]
        
        if len(df_sched) == 0:
            continue
        
        # L1 miss rate
        ax1 = axes[0, idx]
        chunks = sorted(df_sched['chunk'].unique())
        
        for chunk in chunks:
            df_chunk = df_sched[df_sched['chunk'] == chunk].groupby('threads')['L1_percent_val'].mean()
            if len(df_chunk) > 0:
                threads_sorted = [t for t in THREAD_ORDER if t in df_chunk.index]
                values = [df_chunk[t] for t in threads_sorted]
                ax1.plot(range(len(threads_sorted)), values,
                        marker='o', linewidth=2, markersize=6, label=f'chunk={int(chunk)}')
        
        ax1.set_xlabel('Threads', fontsize=11, fontweight='bold')
        ax1.set_ylabel('L1 Miss Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{schedule.capitalize()} - L1', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(threads_sorted)))
        ax1.set_xticklabels([str(t) for t in threads_sorted])
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(alpha=0.3, linestyle='--')
        
        # LLC miss rate
        ax2 = axes[1, idx]
        for chunk in chunks:
            df_chunk = df_sched[df_sched['chunk'] == chunk].groupby('threads')['LLC_percent_val'].mean()
            if len(df_chunk) > 0:
                threads_sorted = [t for t in THREAD_ORDER if t in df_chunk.index]
                values = [df_chunk[t] for t in threads_sorted]
                ax2.plot(range(len(threads_sorted)), values,
                        marker='s', linewidth=2, markersize=6, label=f'chunk={int(chunk)}')
        
        ax2.set_xlabel('Threads', fontsize=11, fontweight='bold')
        ax2.set_ylabel('LLC Miss Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{schedule.capitalize()} - LLC', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(threads_sorted)))
        ax2.set_xticklabels([str(t) for t in threads_sorted])
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Cache Miss Rates vs Thread Count - {matrix}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"miss_rate_vs_threads_{matrix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {plot_path.name}")

# === PLOT 3: Istogramma best chunk - TUTTI GLI SCHEDULING IN UNA FIGURA ===
print("[INFO] Generating combined best chunk histograms...")
for matrix in df['matrix'].unique():
    df_par = df[(df['matrix'] == matrix) & (df['run_type'] == 'parallel')]
    
    if len(df_par) == 0:
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for idx, schedule in enumerate(['static', 'dynamic', 'guided']):
        df_sched = df_par[df_par['schedule'] == schedule]
        ax = axes[idx]
        
        if len(df_sched) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{schedule.capitalize()}', fontsize=12, fontweight='bold')
            continue
        
        # Per ogni thread, trova il chunk con il minor L1 miss rate medio
        best_data = []
        available_threads = [t for t in THREAD_ORDER if t in df_sched['threads'].unique()]
        
        for thread in available_threads:
            df_thread = df_sched[df_sched['threads'] == thread]
            
            chunk_stats = df_thread.groupby('chunk').agg({
                'L1_percent_val': 'mean',
                'LLC_percent_val': 'mean'
            }).reset_index()
            
            best_idx = chunk_stats['L1_percent_val'].idxmin()
            best_chunk = chunk_stats.loc[best_idx, 'chunk']
            best_l1 = chunk_stats.loc[best_idx, 'L1_percent_val']
            best_llc = chunk_stats.loc[best_idx, 'LLC_percent_val']
            
            best_data.append({
                'threads': thread,
                'best_chunk': int(best_chunk),
                'L1_percent': best_l1,
                'LLC_percent': best_llc
            })
        
        df_best = pd.DataFrame(best_data)
        
        x_pos = np.arange(len(df_best))
        width = 0.35
        
        bars_l1 = ax.bar(x_pos - width/2, df_best['L1_percent'], width, 
                        label='L1 Miss Rate', color='#3498db', alpha=0.8, edgecolor='black')
        bars_llc = ax.bar(x_pos + width/2, df_best['LLC_percent'], width,
                         label='LLC Miss Rate', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Annotazioni con il best chunk
        for i, row in df_best.iterrows():
            max_height = max(row['L1_percent'], row['LLC_percent'])
            ax.annotate(f'c={int(row["best_chunk"])}',
                       xy=(i, max_height + 0.3),
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cache Miss Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{schedule.capitalize()} Scheduling', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(int(t)) for t in df_best['threads']])
        ax.legend(fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Best Chunk Cache Miss Rates - {matrix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"best_chunk_histogram_{matrix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {plot_path.name}")

# === PLOT 4: Absolute cache misses ===
print("[INFO] Generating absolute cache miss plots...")
for matrix in df['matrix'].unique():
    stats_matrix = stats[stats['matrix'] == matrix].sort_values('config')
    
    if len(stats_matrix) == 0:
        continue
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    configs = stats_matrix['config'].values
    l1_misses = stats_matrix['L1_misses_val_mean'].values
    llc_misses = stats_matrix['LLC_misses_val_mean'].values
    
    x_pos = np.arange(len(configs))
    
    # L1 absolute misses
    colors = ['red' if c == 'seq' else 'blue' for c in configs]
    axes[0].bar(x_pos, l1_misses, color=colors, alpha=0.6, edgecolor='black')
    axes[0].set_ylabel('L1 Cache Misses (count)', fontsize=11, fontweight='bold')
    axes[0].set_title('L1 Cache Misses (Absolute)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(configs, rotation=90, ha='right', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # LLC absolute misses
    axes[1].bar(x_pos, llc_misses, color=colors, alpha=0.6, edgecolor='black')
    axes[1].set_xlabel('Configuration', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('LLC Misses (count)', fontsize=11, fontweight='bold')
    axes[1].set_title('Last Level Cache Misses (Absolute)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(configs, rotation=90, ha='right', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.suptitle(f'Absolute Cache Misses - {matrix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"absolute_misses_{matrix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {plot_path.name}")

# === SUMMARY FINALE ===
print("\n" + "=" * 80)
print("BEST CHUNK PER THREAD SUMMARY")
print("=" * 80)

for matrix in df['matrix'].unique():
    print(f"\n{matrix}:")
    print("-" * 80)
    
    df_par = df[(df['matrix'] == matrix) & (df['run_type'] == 'parallel')]
    
    for schedule in ['static', 'dynamic', 'guided']:
        df_sched = df_par[df_par['schedule'] == schedule]
        
        if len(df_sched) == 0:
            continue
        
        print(f"\n  {schedule.capitalize()} scheduling:")
        available_threads = sorted(df_sched['threads'].unique())
        
        for thread in available_threads:
            df_thread = df_sched[df_sched['threads'] == thread]
            chunk_stats = df_thread.groupby('chunk').agg({
                'L1_percent_val': 'mean',
                'LLC_percent_val': 'mean'
            }).reset_index()
            
            best_idx = chunk_stats['L1_percent_val'].idxmin()
            best_chunk = chunk_stats.loc[best_idx, 'chunk']
            best_l1 = chunk_stats.loc[best_idx, 'L1_percent_val']
            best_llc = chunk_stats.loc[best_idx, 'LLC_percent_val']
            
            print(f"    threads={int(thread):2d}: best_chunk={int(best_chunk):5d}, "
                  f"L1={best_l1:5.2f}%, LLC={best_llc:5.2f}%")

print(f"\n\n[SUCCESS] All perf plots saved to: {OUTPUT_DIR}/")
print("=" * 80)