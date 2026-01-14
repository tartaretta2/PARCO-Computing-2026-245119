#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
if SCRIPT_DIR.name == 'scripts':
    BASE_DIR = SCRIPT_DIR.parent
else:
    BASE_DIR = SCRIPT_DIR

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

CSV_MPI_TIMES = RESULTS_DIR / "mpi_times.csv"
CSV_MPI_LOAD = RESULTS_DIR / "mpi_load_balance.csv"
CSV_MPIX_TIMES = RESULTS_DIR / "mpix_times.csv"
CSV_MPIX_LOAD = RESULTS_DIR / "mpix_load_balance.csv"

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("=" * 100)
print("MPI SpMV Performance Analysis")
print("=" * 100)
print(f"[INFO] Working directory: {Path.cwd()}")
print(f"[INFO] Results directory: {RESULTS_DIR}")


def load_all_data():
    data = {}
    
    csv_files = {
        'mpi_times': CSV_MPI_TIMES,
        'mpi_load': CSV_MPI_LOAD,
        'mpix_times': CSV_MPIX_TIMES,
        'mpix_load': CSV_MPIX_LOAD
    }
    
    for name, path in csv_files.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                data[name] = df
                print(f"✓ Loaded {name}: {len(df)} records")
                if len(df) > 0:
                    print(f"  Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")
                data[name] = pd.DataFrame()
        else:
            print(f"[INFO] {name} not found (skipping)")
            data[name] = pd.DataFrame()
    
    return data


def aggregate_times(df_times):
    if len(df_times) == 0:
        return pd.DataFrame()
    
    agg = df_times.groupby(['matrix', 'num_procs']).agg({
        'comm_time': ['mean', 'std', 'min', 'max'],
        'comp_time': ['mean', 'std', 'min', 'max'],
        'total_time': ['mean', 'std', 'min', 'max'],
        'M': 'first',
        'N': 'first',
        'nnz': 'first'
    }).reset_index()
    
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    return agg


def calculate_flops(nnz):
    return 2 * nnz


def calculate_gflops(nnz, time_sec):
    if time_sec == 0:
        return 0
    return (calculate_flops(nnz) / time_sec) / 1e9


def plot_strong_scaling_comprehensive(df_agg):
    print("\n=== Strong Scaling Analysis (MPI-only) ===")
    
    if len(df_agg) == 0:
        print("[WARNING] No data available")
        return
    
    matrices = [m for m in df_agg['matrix'].unique() if not m.startswith('random_')]
    
    if not matrices:
        print("[WARNING] No real matrices found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Strong Scaling - MPI Only', fontsize=16, fontweight='bold')
    
    for matrix_name in matrices:
        data = df_agg[df_agg['matrix'] == matrix_name].sort_values('num_procs')
        
        if len(data) < 2:
            continue
        
        procs = data['num_procs'].values
        total_time = data['total_time_mean'].values
        total_time_std = data['total_time_std'].values
        nnz = int(data['nnz_first'].iloc[0])
        
        baseline_time = total_time[0]
        baseline_procs = procs[0]
        
        speedup = baseline_time / total_time
        ideal_speedup = procs / baseline_procs
        efficiency = speedup / (procs / baseline_procs) * 100
        gflops = [calculate_gflops(nnz, t) for t in total_time]
        
        # Plot 1: Execution Time
        ax = axes[0, 0]
        ax.errorbar(procs, total_time, yerr=total_time_std, fmt='o-', 
                   label=matrix_name, linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Execution Time')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend()
        
        # Plot 2: Speedup
        ax = axes[0, 1]
        ax.plot(procs, speedup, 'o-', label=matrix_name, linewidth=2.5, markersize=9)
        if matrix_name == matrices[0]:
            ax.plot(procs, ideal_speedup, 'k--', alpha=0.6, linewidth=2, label='Ideal')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup')
        ax.set_xscale('log', base=2)
        ax.legend()
        
        # Plot 3: Efficiency
        ax = axes[0, 2]
        ax.plot(procs, efficiency, 'o-', label=matrix_name, linewidth=2.5, markersize=9)
        if matrix_name == matrices[0]:
            ax.axhline(100, color='k', linestyle='--', alpha=0.5, label='Ideal')
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title('Parallel Efficiency')
        ax.set_xscale('log', base=2)
        ax.set_ylim([0, 110])
        ax.legend()
        
        # Plot 4: GFLOP/s
        ax = axes[1, 0]
        ax.plot(procs, gflops, 'o-', label=matrix_name, linewidth=2.5, markersize=9)
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('GFLOP/s')
        ax.set_title('Performance')
        ax.set_xscale('log', base=2)
        ax.legend()
        
        # Plot 5: Communication Overhead
        ax = axes[1, 1]
        comm_pct = data['comm_time_mean'].values / total_time * 100
        ax.plot(procs, comm_pct, 'o-', label=matrix_name, linewidth=2.5, markersize=9)
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Communication Overhead (%)')
        ax.set_title('Communication vs Total Time')
        ax.set_xscale('log', base=2)
        ax.legend()
        
        # Plot 6: Time Breakdown Stacked
        ax = axes[1, 2]
        comp_time = data['comp_time_mean'].values
        comm_time = data['comm_time_mean'].values
        x_pos = np.arange(len(procs))
        ax.bar(x_pos, comp_time, label='Computation', alpha=0.8)
        ax.bar(x_pos, comm_time, bottom=comp_time, label='Communication', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(procs, rotation=45)
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Time Breakdown - {matrix_name}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "strong_scaling.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'strong_scaling.png'}")
    plt.close()


def plot_weak_scaling(df_agg):
    print("\n=== Weak Scaling Analysis ===")
    
    if len(df_agg) == 0:
        return
    
    weak_data = df_agg[df_agg['matrix'].str.startswith('random_')].copy()
    
    if len(weak_data) == 0:
        print("[WARNING] No synthetic matrices found")
        return
    
    weak_data['matrix_size'] = weak_data['matrix'].str.extract(r'random_(\d+)x\d+')[0].astype(int)
    weak_data = weak_data.sort_values('num_procs')
    
    procs = weak_data['num_procs'].values
    total_time = weak_data['total_time_mean'].values
    comm_time = weak_data['comm_time_mean'].values
    comp_time = weak_data['comp_time_mean'].values
    nnz = weak_data['nnz_first'].values
    
    baseline_time = total_time[0]
    weak_efficiency = (baseline_time / total_time) * 100
    gflops = [calculate_gflops(n, t) for n, t in zip(nnz, total_time)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weak Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time
    ax = axes[0, 0]
    ax.plot(procs, total_time, 'o-', linewidth=2.5, markersize=9)
    ax.axhline(baseline_time, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Execution Time (Problem Size ∝ P)')
    ax.set_xscale('log', base=2)
    ax.legend()
    
    # Plot 2: Efficiency
    ax = axes[0, 1]
    ax.plot(procs, weak_efficiency, 'o-', linewidth=2.5, markersize=9)
    ax.axhline(100, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Weak Scaling Efficiency (%)')
    ax.set_title('Weak Scaling Efficiency')
    ax.set_xscale('log', base=2)
    ax.set_ylim([0, 110])
    ax.legend()
    
    # Plot 3: GFLOP/s
    ax = axes[1, 0]
    ax.plot(procs, gflops, 'o-', linewidth=2.5, markersize=9)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('GFLOP/s')
    ax.set_title('Performance')
    ax.set_xscale('log', base=2)
    
    # Plot 4: Communication %
    ax = axes[1, 1]
    comm_pct = comm_time / total_time * 100
    ax.plot(procs, comm_pct, 'o-', linewidth=2.5, markersize=9)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Communication (%)')
    ax.set_title('Communication Overhead')
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "weak_scaling.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'weak_scaling.png'}")
    plt.close()


def plot_load_balance(df_load):
    print("\n=== Load Balance Analysis ===")
    
    if len(df_load) == 0:
        print("[WARNING] No load balance data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Load Balance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Imbalance %
    ax = axes[0]
    real_matrices = df_load[~df_load['matrix'].str.startswith('random_')]
    for matrix in real_matrices['matrix'].unique()[:5]:
        data = real_matrices[real_matrices['matrix'] == matrix].sort_values('num_procs')
        ax.plot(data['num_procs'], data['imbalance_pct'], 'o-', 
               label=matrix, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Load Imbalance (%)')
    ax.set_title('Load Imbalance vs Process Count')
    ax.set_xscale('log', base=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.legend()
    
    # Plot 2: Box plot
    ax = axes[1]
    proc_counts = sorted(df_load['num_procs'].unique())
    data_to_plot = [df_load[df_load['num_procs'] == p]['imbalance_pct'].values 
                    for p in proc_counts]
    bp = ax.boxplot(data_to_plot, labels=proc_counts, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Load Imbalance (%)')
    ax.set_title('Imbalance Distribution')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "load_balance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'load_balance.png'}")
    plt.close()


def plot_hybrid_comparison(df_mpi, df_mpix):
    print("\n=== Hybrid MPI+OpenMP Comparison ===")

    if len(df_mpi) == 0 or len(df_mpix) == 0:
        print("[WARNING] Missing data for hybrid comparison")
        return

    agg_mpi = df_mpi.groupby(['matrix', 'num_procs']).agg({
        'total_time': 'mean',
        'comm_time': 'mean',
        'comp_time': 'mean'
    }).reset_index()
    agg_mpi['type'] = 'MPI-only'
    agg_mpi['num_threads'] = 1
    agg_mpi['total_cores'] = agg_mpi['num_procs']
    agg_mpi['config'] = agg_mpi.apply(
        lambda r: f"P={r['num_procs']}×T=1", axis=1
    )

    agg_mpix = df_mpix.groupby(['matrix', 'num_procs', 'num_threads']).agg({
        'total_time': 'mean',
        'comm_time': 'mean',
        'comp_time': 'mean'
    }).reset_index()
    agg_mpix['type'] = 'MPI+OpenMP'
    agg_mpix['total_cores'] = agg_mpix['num_procs'] * agg_mpix['num_threads']
    agg_mpix['config'] = agg_mpix.apply(
        lambda r: f"P={r['num_procs']}×T={r['num_threads']}", axis=1
    )

    target_cores = 16
    data_16 = pd.concat([agg_mpi, agg_mpix])
    data_16 = data_16[data_16['total_cores'] == target_cores]

    if len(data_16) == 0:
        print(f"[WARNING] No {target_cores}-core configs found")
        return

    matrices = [m for m in data_16['matrix'].unique()
                if not m.startswith('random_')]

    if not matrices:
        print("[WARNING] No real matrices in hybrid data")
        return

    for matrix in matrices:
        data = data_16[data_16['matrix'] == matrix] \
            .sort_values(['num_threads', 'num_procs'], ascending=[True, False])

        if len(data) < 2:
            continue

        configs = data['config'].values
        total_time = data['total_time'].values
        comm_time = data['comm_time'].values
        comp_time = data['comp_time'].values

        x_pos = np.arange(len(configs))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'MPI vs MPI+OpenMP – {matrix} ({target_cores} cores)',
                     fontsize=16, fontweight='bold')

        # --- Plot 1: Total Time ---
        ax = axes[0, 0]
        ax.bar(x_pos, total_time, color='steelblue', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel('Total Time (s)')
        ax.set_title('Execution Time')

        for i, t in enumerate(total_time):
            ax.text(i, t * 1.02, f'{t:.4f}s', ha='center', fontsize=9)

        # --- Plot 2: Time Breakdown ---
        ax = axes[0, 1]
        ax.bar(x_pos, comp_time, label='Computation', color='#2E86AB')
        ax.bar(x_pos, comm_time, bottom=comp_time,
               label='Communication', color='#A23B72')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel('Time (s)')
        ax.set_title('Time Breakdown')
        ax.legend()

        # --- Plot 3: Communication Overhead ---
        ax = axes[1, 0]
        comm_pct = comm_time / total_time * 100
        ax.bar(x_pos, comm_pct, color='coral', alpha=0.8)
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel('Communication (%)')
        ax.set_title('Communication Overhead')

        # --- Plot 4: Speedup vs Pure MPI ---
        ax = axes[1, 1]

        baseline_mask = (data['num_procs'] == target_cores) & (data['num_threads'] == 1)
        if not baseline_mask.any():
            baseline_time = total_time[0]
        else:
            baseline_time = total_time[baseline_mask.values.argmax()]

        speedup = baseline_time / total_time
        colors = ['green' if s >= 1.0 else 'red' for s in speedup]

        ax.bar(x_pos, speedup, color=colors, alpha=0.8)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.set_ylabel('Speedup vs Pure MPI')
        ax.set_title('Speedup Relative to Pure MPI')

        for i, s in enumerate(speedup):
            ax.text(i, s * 1.02, f'{s:.2f}×',
                    ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        out = PLOTS_DIR / f"hybrid_comparison_{matrix}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved {out}")



def generate_metrics_table(df_agg, df_load):
    print("\n=== Generating Metrics Table ===")
    
    if len(df_agg) == 0:
        return
    
    report_file = RESULTS_DIR / "mpi_metrics_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("MPI SpMV Performance Metrics\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("STRONG SCALING (MPI-only)\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Matrix':<20} {'Procs':>6} {'Time(s)':>10} {'Speedup':>10} {'Efficiency(%)':>12} "
               f"{'GFLOP/s':>10} {'Comm(%)':>10} {'Imbalance(%)':>12}\n")
        f.write("-" * 120 + "\n")
        
        real_matrices = [m for m in df_agg['matrix'].unique() if not m.startswith('random_')]
        
        for matrix in real_matrices:
            data = df_agg[df_agg['matrix'] == matrix].sort_values('num_procs')
            if len(data) < 2:
                continue
            
            baseline_time = data['total_time_mean'].iloc[0]
            baseline_procs = data['num_procs'].iloc[0]
            nnz = int(data['nnz_first'].iloc[0])
            
            for _, row in data.iterrows():
                procs = int(row['num_procs'])
                time = row['total_time_mean']
                speedup = baseline_time / time
                efficiency = speedup / (procs / baseline_procs) * 100
                gflops = calculate_gflops(nnz, time)
                comm_pct = row['comm_time_mean'] / time * 100
                
                load_row = df_load[(df_load['matrix'] == matrix) & (df_load['num_procs'] == procs)]
                imbalance = load_row['imbalance_pct'].values[0] if len(load_row) > 0 else 0.0
                
                f.write(f"{matrix:<20} {procs:>6} {time:>10.4f} {speedup:>10.2f} {efficiency:>12.1f} "
                       f"{gflops:>10.2f} {comm_pct:>10.1f} {imbalance:>12.2f}\n")
            f.write("\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"✓ Saved: {report_file}")


def generate_summary_report(df_agg, df_load):
    if len(df_agg) == 0:
        return
    
    report_file = RESULTS_DIR / "mpi_discussion_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("MPI SpMV Performance Analysis - Discussion\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("1. WHEN 1D PARTITIONING WORKS WELL\n")
        f.write("-" * 120 + "\n")
        f.write("1D cyclic partitioning is effective when:\n")
        f.write("  • Matrix has uniform row distribution\n")
        f.write("  • Communication-to-computation ratio is favorable\n")
        f.write("  • Number of processes is moderate (< 64)\n\n")
        
        real_matrices = [m for m in df_agg['matrix'].unique() if not m.startswith('random_')]
        for matrix in real_matrices[:3]:
            data = df_agg[df_agg['matrix'] == matrix].sort_values('num_procs')
            if len(data) >= 2:
                max_procs = data['num_procs'].max()
                max_data = data[data['num_procs'] == max_procs].iloc[0]
                eff = (data['total_time_mean'].iloc[0] / max_data['total_time_mean']) / (max_procs / data['num_procs'].iloc[0]) * 100
                f.write(f"  {matrix}: {eff:.1f}% efficiency at P={int(max_procs)}\n")
        
        f.write("\n2. WHEN 2D PARTITIONING IS ADVANTAGEOUS\n")
        f.write("-" * 120 + "\n")
        f.write("2D partitioning becomes beneficial when:\n")
        f.write("  • High communication overhead (>30%)\n")
        f.write("  • Large number of processes (>64)\n")
        f.write("  • Significant load imbalance\n\n")
        
        high_comm = []
        for matrix in real_matrices:
            data = df_agg[df_agg['matrix'] == matrix].sort_values('num_procs')
            if len(data) > 0:
                max_data = data[data['num_procs'] == data['num_procs'].max()].iloc[0]
                comm_pct = max_data['comm_time_mean'] / max_data['total_time_mean'] * 100
                if comm_pct > 30:
                    high_comm.append((matrix, comm_pct, max_data['num_procs']))
        
        if high_comm:
            f.write("  Matrices with high communication overhead:\n")
            for mat, comm, procs in high_comm:
                f.write(f"    {mat}: {comm:.1f}% at P={int(procs)}\n")
        
        f.write("\n3. INTERCONNECT BOUND ANALYSIS\n")
        f.write("-" * 120 + "\n")
        for matrix in real_matrices[:3]:
            data = df_agg[df_agg['matrix'] == matrix].sort_values('num_procs')
            if len(data) >= 2:
                f.write(f"  {matrix}:\n")
                for _, row in data.iterrows():
                    p = int(row['num_procs'])
                    comm_pct = row['comm_time_mean'] / row['total_time_mean'] * 100
                    speedup = data['total_time_mean'].iloc[0] / row['total_time_mean']
                    eff = speedup / (p / data['num_procs'].iloc[0]) * 100
                    status = "BOUND" if comm_pct > 50 or eff < 50 else "OK"
                    f.write(f"    P={p:3d}: Comm={comm_pct:5.1f}%, Eff={eff:5.1f}% [{status}]\n")
                f.write("\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"✓ Saved: {report_file}")

def plot_strong_scaling_per_matrix(df):
    matrices = [m for m in df['matrix'].unique() if not m.startswith("random_")]

    for matrix in matrices:
        data = df[df['matrix'] == matrix].sort_values('num_procs')
        if len(data) < 2:
            continue

        procs = data['num_procs'].values
        total_time = data['total_time_mean'].values
        total_std = data['total_time_std'].values
        comp_time = data['comp_time_mean'].values
        comm_time = data['comm_time_mean'].values
        nnz = int(data['nnz_first'].iloc[0])

        speedup = total_time[0] / total_time
        efficiency = speedup / (procs / procs[0]) * 100
        gflops = [calculate_gflops(nnz, t) for t in total_time]
        comm_pct = comm_time / total_time * 100

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Strong Scaling – {matrix}", fontsize=16, fontweight='bold')

        # Execution time
        axes[0,0].errorbar(procs, total_time, yerr=total_std, fmt='o-')
        axes[0,0].set_xscale('log', base=2)
        axes[0,0].set_yscale('log')
        axes[0,0].set_title("Execution Time [s]")
        axes[0,0].set_xlabel("Processes")

        # Speedup
        axes[0,1].plot(procs, speedup, 'o-', label="Measured")
        axes[0,1].plot(procs, procs / procs[0], 'k--', label="Ideal")
        axes[0,1].set_xscale('log', base=2)
        axes[0,1].set_title("Speedup")
        axes[0,1].legend()

        # Efficiency
        axes[0,2].plot(procs, efficiency, 'o-')
        axes[0,2].axhline(100, linestyle='--', alpha=0.5)
        axes[0,2].set_xscale('log', base=2)
        axes[0,2].set_title("Parallel Efficiency [%]")

        # GFLOP/s
        axes[1,0].plot(procs, gflops, 'o-')
        axes[1,0].set_xscale('log', base=2)
        axes[1,0].set_title("GFLOP/s")

        # Communication ratio
        axes[1,1].plot(procs, comm_pct, 'o-')
        axes[1,1].set_xscale('log', base=2)
        axes[1,1].set_title("Communication / Total [%]")

        # Time breakdown (FIX AXIS X)
        ax = axes[1, 2]
        ax.set_xscale('linear')

        x = np.arange(len(procs))
        width = 0.6

        ax.bar(x, comp_time, width, label='Computation')
        ax.bar(x, comm_time, width, bottom=comp_time, label='Communication')

        ax.set_xticks(x)
        ax.set_xticklabels(procs)

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Time [s]')
        ax.set_title('Time Breakdown')
        ax.legend()

        plt.tight_layout()
        out = PLOTS_DIR / f"strong_scaling_{matrix}.png"
        plt.savefig(out, dpi=300)
        plt.close()

        print(f"✓ Saved {out}")


def plot_load_balance_per_matrix(df):
    matrices = [m for m in df['matrix'].unique() if not m.startswith("random_")]

    for matrix in matrices:
        data = df[df['matrix'] == matrix].sort_values('num_procs')
        if len(data) == 0:
            continue

        plt.figure(figsize=(8,6))
        plt.plot(data['num_procs'], data['imbalance_pct'], 'o-')
        plt.xscale('log', base=2)
        plt.xlabel("Processes")
        plt.ylabel("Load Imbalance [%]")
        plt.title(f"Load Imbalance – {matrix}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)

        out = PLOTS_DIR / f"load_balance_{matrix}.png"
        plt.savefig(out, dpi=300)
        plt.close()

        print(f"✓ Saved {out}")
        
def plot_weak_scaling1(df):
    weak = df[df['matrix'].str.startswith("random_")].copy()
    if weak.empty:
        return

    def extract_size(name):
        return int(name.split("_")[1].split("x")[0])

    weak['problem_size'] = weak['matrix'].apply(extract_size)
    weak = weak.sort_values('num_procs')

    plt.figure(figsize=(8,6))
    plt.plot(weak['num_procs'], weak['total_time_mean'], 'o-')
    plt.xscale('log', base=2)
    plt.xlabel("Processes")
    plt.ylabel("Execution Time [s]")
    plt.title("Weak Scaling (random matrices)")
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    out = PLOTS_DIR / "weak_scaling_random.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"✓ Saved {out}")

def main():
    print("\n" + "=" * 100)
    print("Starting Analysis Pipeline")
    print("=" * 100)
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_all_data()
    
    df_mpi_times = data['mpi_times']
    df_mpi_load = data['mpi_load']
    df_mpix_times = data['mpix_times']
    df_mpix_load = data['mpix_load']
    
    if len(df_mpi_times) == 0:
        print("\n[ERROR] No MPI data found. Run experiments first.")
        sys.exit(1)
    
    df_agg = aggregate_times(df_mpi_times)
    
    print("\n" + "=" * 100)
    print("Generating Plots")
    print("=" * 100)
    
    plot_strong_scaling_per_matrix(df_agg)
    
    plot_weak_scaling1(df_agg)
    
    plot_load_balance_per_matrix(df_mpi_load)
    
    if len(df_mpix_times) > 0:
        plot_hybrid_comparison(df_mpi_times, df_mpix_times)
    else:
        print("\n[INFO] No MPI+OpenMP data - skipping hybrid comparison")
    
    print("\n" + "=" * 100)
    print("Generating Reports")
    print("=" * 100)
    
    generate_metrics_table(df_agg, df_mpi_load)
    generate_summary_report(df_agg, df_mpi_load)
    
    print("\n" + "=" * 100)
    print("Analysis Completed!")
    print("=" * 100)
    print(f"\nResults:")
    print(f"  • Plots:      {PLOTS_DIR}/")
    print(f"  • Metrics:    {RESULTS_DIR}/mpi_metrics_summary.txt")
    print(f"  • Discussion: {RESULTS_DIR}/mpi_discussion_report.txt")
    print("=" * 100)


if __name__ == "__main__":
    main()