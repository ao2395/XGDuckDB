#!/usr/bin/env python3
"""
Analyze TPC-DS benchmark results and generate visualizations.

Usage:
    python3 analyze_tpcds.py [--input tpcds_results.csv]
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
import sys

def load_data(filepath):
    """Load CSV data into list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both old and new CSV formats
            if 'rl_predicted' in row:
                # New format with both predictions
                data.append({
                    'query_num': int(row['query_num']),
                    'operator': row['operator'],
                    'actual': int(row['actual']),
                    'rl_predicted': int(row['rl_predicted']),
                    'duck_predicted': int(row['duck_predicted']),
                    'rl_q_error': float(row['rl_q_error']),
                    'duck_q_error': float(row['duck_q_error']),
                    'trees': int(row['trees']),
                    'timestamp': row['timestamp']
                })
            else:
                # Old format (backwards compatibility)
                data.append({
                    'query_num': int(row['query_num']),
                    'operator': row['operator'],
                    'actual': int(row['actual']),
                    'rl_predicted': int(row['predicted']),
                    'duck_predicted': 0,  # Not available
                    'rl_q_error': float(row['q_error']),
                    'duck_q_error': 0.0,  # Not available
                    'trees': int(row['trees']),
                    'timestamp': row['timestamp']
                })
    return data

def moving_average(data, window=100):
    """Calculate moving average."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    parser = argparse.ArgumentParser(description='Analyze TPC-DS benchmark results')
    parser.add_argument('--input', default='tpcds_results.csv', help='Input CSV file')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    data = load_data(args.input)

    if not data:
        print("No data found!")
        return 1

    print(f"Loaded {len(data)} data points")
    print(f"Queries: {min(d['query_num'] for d in data)} to {max(d['query_num'] for d in data)}")
    print(f"Trees: {min(d['trees'] for d in data)} to {max(d['trees'] for d in data)}")
    print()

    # Basic statistics - RL vs DuckDB
    rl_q_errors = [d['rl_q_error'] for d in data]
    duck_q_errors = [d['duck_q_error'] for d in data]

    print("=" * 80)
    print("Q-ERROR STATISTICS: RL MODEL vs DUCKDB BASELINE")
    print("=" * 80)
    print(f"{'Metric':<25} {'RL Model':<15} {'DuckDB':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Mean Q-error':<25} {np.mean(rl_q_errors):<15.2f} {np.mean(duck_q_errors):<15.2f} {((np.mean(duck_q_errors) - np.mean(rl_q_errors)) / np.mean(duck_q_errors) * 100):>+14.1f}%")
    print(f"{'Median Q-error':<25} {np.median(rl_q_errors):<15.2f} {np.median(duck_q_errors):<15.2f} {((np.median(duck_q_errors) - np.median(rl_q_errors)) / np.median(duck_q_errors) * 100):>+14.1f}%")
    print(f"{'Std Dev':<25} {np.std(rl_q_errors):<15.2f} {np.std(duck_q_errors):<15.2f}")
    print(f"{'Min Q-error':<25} {np.min(rl_q_errors):<15.2f} {np.min(duck_q_errors):<15.2f}")
    print(f"{'Max Q-error':<25} {np.max(rl_q_errors):<15.2f} {np.max(duck_q_errors):<15.2f}")
    print(f"{'95th percentile':<25} {np.percentile(rl_q_errors, 95):<15.2f} {np.percentile(duck_q_errors, 95):<15.2f}")
    print(f"{'99th percentile':<25} {np.percentile(rl_q_errors, 99):<15.2f} {np.percentile(duck_q_errors, 99):<15.2f}")
    print()

    # Distribution buckets for RL Model
    rl_buckets = {
        '1.0-2.0 (excellent)': sum(1 for q in rl_q_errors if 1.0 <= q < 2.0),
        '2.0-5.0 (good)': sum(1 for q in rl_q_errors if 2.0 <= q < 5.0),
        '5.0-10.0 (acceptable)': sum(1 for q in rl_q_errors if 5.0 <= q < 10.0),
        '10.0-100.0 (poor)': sum(1 for q in rl_q_errors if 10.0 <= q < 100.0),
        '100.0+ (terrible)': sum(1 for q in rl_q_errors if q >= 100.0),
    }
    duck_buckets = {
        '1.0-2.0 (excellent)': sum(1 for q in duck_q_errors if 1.0 <= q < 2.0),
        '2.0-5.0 (good)': sum(1 for q in duck_q_errors if 2.0 <= q < 5.0),
        '5.0-10.0 (acceptable)': sum(1 for q in duck_q_errors if 5.0 <= q < 10.0),
        '10.0-100.0 (poor)': sum(1 for q in duck_q_errors if 10.0 <= q < 100.0),
        '100.0+ (terrible)': sum(1 for q in duck_q_errors if q >= 100.0),
    }
    print("Q-ERROR DISTRIBUTION")
    print("=" * 80)
    print(f"{'Range':<30} {'RL Model':>12} {'DuckDB':>12}")
    print("-" * 80)
    for bucket in rl_buckets.keys():
        rl_count = rl_buckets[bucket]
        duck_count = duck_buckets[bucket]
        rl_pct = rl_count / len(rl_q_errors) * 100
        duck_pct = duck_count / len(duck_q_errors) * 100
        print(f"{bucket:30s} {rl_count:6d} ({rl_pct:4.1f}%)  {duck_count:6d} ({duck_pct:4.1f}%)")
    print()

    # Per-operator statistics
    rl_by_operator = defaultdict(list)
    duck_by_operator = defaultdict(list)
    for d in data:
        rl_by_operator[d['operator']].append(d['rl_q_error'])
        duck_by_operator[d['operator']].append(d['duck_q_error'])

    print("PER-OPERATOR Q-ERROR (RL Model)")
    print("=" * 80)
    print(f"{'Operator':<20} {'Count':>8} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)
    for op in sorted(rl_by_operator.keys()):
        qerrs = rl_by_operator[op]
        print(f"{op:<20} {len(qerrs):>8} {np.mean(qerrs):>12.2f} {np.median(qerrs):>12.2f} "
              f"{np.min(qerrs):>12.2f} {np.max(qerrs):>12.2f}")
    print()

    # Temporal analysis (beginning vs end) - RL Model
    third = len(data) // 3
    rl_beginning = [d['rl_q_error'] for d in data[:third]]
    rl_middle = [d['rl_q_error'] for d in data[third:2*third]]
    rl_end = [d['rl_q_error'] for d in data[2*third:]]

    print("TEMPORAL ANALYSIS - RL MODEL (Split into thirds)")
    print("=" * 80)
    print(f"{'Period':<20} {'Count':>8} {'Mean':>12} {'Median':>12}")
    print("-" * 80)
    print(f"{'Beginning':<20} {len(rl_beginning):>8} {np.mean(rl_beginning):>12.2f} {np.median(rl_beginning):>12.2f}")
    print(f"{'Middle':<20} {len(rl_middle):>8} {np.mean(rl_middle):>12.2f} {np.median(rl_middle):>12.2f}")
    print(f"{'End':<20} {len(rl_end):>8} {np.mean(rl_end):>12.2f} {np.median(rl_end):>12.2f}")

    rl_improvement = ((np.median(rl_beginning) - np.median(rl_end)) / np.median(rl_beginning)) * 100
    print(f"\nRL Model improvement from beginning to end: {rl_improvement:+.1f}%")
    print()

    # Create visualizations
    print("Generating visualizations...")

    # 1. Q-error over time (RL vs DuckDB comparison)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Raw scatter (log scale) - RL vs DuckDB
    indices = range(len(rl_q_errors))
    ax1.scatter(indices, rl_q_errors, alpha=0.3, s=5, c='blue', label='RL Model')
    ax1.scatter(indices, duck_q_errors, alpha=0.3, s=5, c='red', label='DuckDB Baseline')
    ax1.set_yscale('log')
    ax1.set_xlabel('Sample Index (Operator Number)', fontsize=12)
    ax1.set_ylabel('Q-error (log scale)', fontsize=12)
    ax1.set_title('TPC-DS Q-error Over Time: RL Model vs DuckDB Baseline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (Q-error = 1.0)')
    ax1.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Threshold (Q-error = 10.0)')
    ax1.legend()

    # Moving average (log scale)
    window = min(100, len(rl_q_errors) // 4)
    if len(rl_q_errors) > window:
        rl_ma = moving_average(rl_q_errors, window)
        duck_ma = moving_average(duck_q_errors, window)
        ax2.plot(range(window-1, len(rl_q_errors)), rl_ma, 'b-', linewidth=2, label=f'RL Model (MA window={window})')
        ax2.plot(range(window-1, len(duck_q_errors)), duck_ma, 'r-', linewidth=2, label=f'DuckDB (MA window={window})')
        ax2.set_yscale('log')
        ax2.set_xlabel('Sample Index (Operator Number)', fontsize=12)
        ax2.set_ylabel('Q-error (log scale)', fontsize=12)
        ax2.set_title('TPC-DS Q-error Moving Average Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax2.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        ax2.legend()

    plt.tight_layout()
    plt.savefig('tpcds_qerror_over_time.png', dpi=150)
    print("  ✓ Saved: tpcds_qerror_over_time.png")
    plt.close()

    # 2. Tree growth over time
    fig, ax = plt.subplots(figsize=(12, 6))
    trees = [d['trees'] for d in data]
    ax.plot(range(len(trees)), trees, 'g-', linewidth=2)
    ax.set_xlabel('Sample Index (Operator Number)', fontsize=12)
    ax.set_ylabel('Number of Trees', fontsize=12)
    ax.set_title('Model Size Growth (Tree Count Over Time)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tpcds_tree_growth.png', dpi=150)
    print("  ✓ Saved: tpcds_tree_growth.png")
    plt.close()

    # 3. Q-error vs Trees (correlation) - RL Model
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(trees, rl_q_errors, alpha=0.4, s=20, c='purple')
    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Q-error (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('RL Model Q-error vs Model Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(trees) > 10:
        # Use log of Q-error for trend
        log_qerr = np.log10(rl_q_errors)
        z = np.polyfit(trees, log_qerr, 1)
        p = np.poly1d(z)
        trend_x = np.linspace(min(trees), max(trees), 100)
        trend_y = 10 ** p(trend_x)
        ax.plot(trend_x, trend_y, "r--", linewidth=2, label=f'Trend (slope={z[0]:.6f})')
        ax.legend()

    plt.tight_layout()
    plt.savefig('tpcds_qerror_vs_trees.png', dpi=150)
    print("  ✓ Saved: tpcds_qerror_vs_trees.png")
    plt.close()

    # 4. Per-operator Q-error boxplot (RL Model)
    fig, ax = plt.subplots(figsize=(14, 8))

    operators = sorted(rl_by_operator.keys(), key=lambda x: np.median(rl_by_operator[x]))
    data_for_box = [rl_by_operator[op] for op in operators]

    bp = ax.boxplot(data_for_box, labels=operators, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_yscale('log')
    ax.set_xlabel('Operator Type', fontsize=12)
    ax.set_ylabel('Q-error (log scale)', fontsize=12)
    ax.set_title('RL Model Q-error Distribution by Operator Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('tpcds_qerror_by_operator.png', dpi=150)
    print("  ✓ Saved: tpcds_qerror_by_operator.png")
    plt.close()

    # 5. Per-query average Q-error (RL Model)
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_num']].append(d['rl_q_error'])

    query_nums = sorted(by_query.keys())
    query_avgs = [np.mean(by_query[q]) for q in query_nums]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(query_nums, query_avgs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Average Q-error (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('RL Model Average Q-error per Query', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('tpcds_qerror_by_query.png', dpi=150)
    print("  ✓ Saved: tpcds_qerror_by_query.png")
    plt.close()

    # 6. Prediction accuracy scatter (actual vs predicted) - RL Model
    fig, ax = plt.subplots(figsize=(10, 10))
    actual_vals = [d['actual'] for d in data]
    rl_pred_vals = [d['rl_predicted'] for d in data]

    ax.scatter(actual_vals, rl_pred_vals, alpha=0.4, s=20, c='blue')

    # Perfect prediction line
    min_val = min(min(actual_vals), min(rl_pred_vals))
    max_val = max(max(actual_vals), max(rl_pred_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # 2x and 10x error bands
    ax.plot([min_val, max_val], [min_val*2, max_val*2], 'orange', linestyle='--', alpha=0.5, label='2x overestimate')
    ax.plot([min_val, max_val], [min_val/2, max_val/2], 'orange', linestyle='--', alpha=0.5, label='2x underestimate')
    ax.plot([min_val, max_val], [min_val*10, max_val*10], 'red', linestyle='--', alpha=0.3, label='10x overestimate')
    ax.plot([min_val, max_val], [min_val/10, max_val/10], 'red', linestyle='--', alpha=0.3, label='10x underestimate')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Actual Cardinality', fontsize=12)
    ax.set_ylabel('Predicted Cardinality', fontsize=12)
    ax.set_title('Prediction Accuracy: Actual vs Predicted Cardinality', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tpcds_actual_vs_predicted.png', dpi=150)
    print("  ✓ Saved: tpcds_actual_vs_predicted.png")
    plt.close()

    # 7. Q-error by query number (simple scatter, no averaging) - RL Model
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Extract query numbers and Q-errors
    query_nums_for_plot = [d['query_num'] for d in data]
    rl_q_errs = [d['rl_q_error'] for d in data]

    # Linear scale
    ax1.scatter(query_nums_for_plot, rl_q_errs, alpha=0.5, s=15, c='blue')
    ax1.set_xlabel('Query Number', fontsize=12)
    ax1.set_ylabel('Q-error (linear scale)', fontsize=12)
    ax1.set_title('RL Model Q-error Over Query Execution (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Perfect (Q-error = 1.0)')
    ax1.axhline(y=10.0, color='orange', linestyle='--', linewidth=1, label='Threshold (Q-error = 10.0)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Log scale with trend line
    ax2.scatter(query_nums_for_plot, rl_q_errs, alpha=0.5, s=15, c='darkblue', label='Q-error values')
    ax2.set_xlabel('Query Number', fontsize=12)
    ax2.set_ylabel('Q-error (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('RL Model Q-error Over Query Execution (Log Scale) with Trend Line', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Perfect (Q-error = 1.0)')
    ax2.axhline(y=10.0, color='orange', linestyle='--', linewidth=1, label='Threshold (Q-error = 10.0)')

    # Add trend line (fit in log space)
    log_qerr = np.log10(rl_q_errs)
    z = np.polyfit(query_nums_for_plot, log_qerr, 1)  # Linear fit on log(Q-error)
    p = np.poly1d(z)

    # Generate trend line
    trend_x = np.linspace(min(query_nums_for_plot), max(query_nums_for_plot), 100)
    trend_y = 10 ** p(trend_x)  # Convert back from log space

    slope_percent = (z[0] / p(query_nums_for_plot[0])) * 100  # Slope as percentage
    trend_label = f'Trend (slope={z[0]:.6f}, {"↓ improving" if z[0] < 0 else "↑ worsening"})'
    ax2.plot(trend_x, trend_y, 'r-', linewidth=3, label=trend_label, alpha=0.8)

    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('tpcds_qerror_by_query_scatter.png', dpi=150)
    print("  ✓ Saved: tpcds_qerror_by_query_scatter.png")
    plt.close()

    # 8. Smooth curve comparison - CLEAN VERSION (using rolling median + clipping)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Use rolling MEDIAN (more robust to outliers than mean)
    window = min(500, len(rl_q_errors) // 5)
    if window < 50:
        window = 50

    def rolling_median(data, window):
        """Calculate rolling median."""
        result = []
        for i in range(len(data) - window + 1):
            result.append(np.median(data[i:i+window]))
        return np.array(result)

    # Calculate rolling medians
    rl_smooth = rolling_median(rl_q_errors, window)
    duck_smooth = rolling_median(duck_q_errors, window)

    # X-axis (operator index) - must match length of rl_smooth
    x_smooth = np.arange(len(rl_smooth))

    # Plot 1: Clipped view (focus on useful range)
    ax1.plot(x_smooth, rl_smooth, 'b-', linewidth=3, label='RL Model', alpha=0.9)
    ax1.plot(x_smooth, duck_smooth, 'r-', linewidth=3, label='DuckDB Baseline', alpha=0.9)

    # Add reference lines
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Perfect')
    ax1.axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='10x Error')

    # Formatting - CLIPPED Y-axis for readability
    ax1.set_xlabel('Operator Index', fontsize=12)
    ax1.set_ylabel('Q-error (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim([0.5, 1000])  # Clip to useful range
    ax1.set_title(f'Clean Comparison: Rolling Median Q-error (window={window}) - Clipped View',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')

    # Add statistics text box
    stats_text = f"Median Q-error:\nRL: {np.median(rl_q_errors):.2f}\nDuckDB: {np.median(duck_q_errors):.2f}\n\n"
    stats_text += f"RL is {((np.median(duck_q_errors) - np.median(rl_q_errors)) / np.median(duck_q_errors) * 100):.1f}% better"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Plot 2: Full range (log scale showing all data)
    ax2.plot(x_smooth, rl_smooth, 'b-', linewidth=3, label='RL Model', alpha=0.9)
    ax2.plot(x_smooth, duck_smooth, 'r-', linewidth=3, label='DuckDB Baseline', alpha=0.9)

    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)

    ax2.set_xlabel('Operator Index', fontsize=12)
    ax2.set_ylabel('Q-error (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Full Range View (Including Outliers)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('tpcds_smooth_comparison.png', dpi=150)
    print("  ✓ Saved: tpcds_smooth_comparison.png")
    plt.close()

    # 9. Direct RL vs DuckDB comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Top left: Median Q-error by query (RL vs DuckDB)
    by_query_rl = defaultdict(list)
    by_query_duck = defaultdict(list)
    for d in data:
        by_query_rl[d['query_num']].append(d['rl_q_error'])
        by_query_duck[d['query_num']].append(d['duck_q_error'])

    query_nums_comp = sorted(by_query_rl.keys())
    rl_medians = [np.median(by_query_rl[q]) for q in query_nums_comp]
    duck_medians = [np.median(by_query_duck[q]) for q in query_nums_comp]

    ax1.plot(query_nums_comp, rl_medians, 'b-', linewidth=2, label='RL Model', alpha=0.8)
    ax1.plot(query_nums_comp, duck_medians, 'r-', linewidth=2, label='DuckDB Baseline', alpha=0.8)
    ax1.set_xlabel('Query Number', fontsize=11)
    ax1.set_ylabel('Median Q-error (log scale)', fontsize=11)
    ax1.set_yscale('log')
    ax1.set_title('Median Q-error per Query: RL vs DuckDB', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top right: CDF comparison
    rl_sorted = np.sort(rl_q_errors)
    duck_sorted = np.sort(duck_q_errors)
    rl_cdf = np.arange(1, len(rl_sorted) + 1) / len(rl_sorted)
    duck_cdf = np.arange(1, len(duck_sorted) + 1) / len(duck_sorted)

    ax2.plot(rl_sorted, rl_cdf, 'b-', linewidth=2, label='RL Model', alpha=0.8)
    ax2.plot(duck_sorted, duck_cdf, 'r-', linewidth=2, label='DuckDB Baseline', alpha=0.8)
    ax2.set_xlabel('Q-error (log scale)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_xscale('log')
    ax2.set_title('CDF: RL Model vs DuckDB Baseline', fontsize=12, fontweight='bold')
    ax2.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax2.axvline(x=10.0, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom left: Per-operator median comparison
    operators_comp = sorted(rl_by_operator.keys())
    rl_op_medians = [np.median(rl_by_operator[op]) for op in operators_comp]
    duck_op_medians = [np.median(duck_by_operator[op]) for op in operators_comp]

    x_pos = np.arange(len(operators_comp))
    width = 0.35

    ax3.bar(x_pos - width/2, rl_op_medians, width, label='RL Model', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, duck_op_medians, width, label='DuckDB', color='red', alpha=0.7)
    ax3.set_xlabel('Operator Type', fontsize=11)
    ax3.set_ylabel('Median Q-error (log scale)', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_title('Median Q-error by Operator: RL vs DuckDB', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(operators_comp, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Bottom right: Improvement factor
    improvement_factors = []
    for op in operators_comp:
        rl_med = np.median(rl_by_operator[op])
        duck_med = np.median(duck_by_operator[op])
        if duck_med > 0:
            improvement = (duck_med - rl_med) / duck_med * 100
            improvement_factors.append(improvement)
        else:
            improvement_factors.append(0)

    colors = ['green' if x > 0 else 'red' for x in improvement_factors]
    ax4.barh(operators_comp, improvement_factors, color=colors, alpha=0.7)
    ax4.set_xlabel('Improvement (%)', fontsize=11)
    ax4.set_ylabel('Operator Type', fontsize=11)
    ax4.set_title('RL Model Improvement over DuckDB (%)', fontsize=12, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('tpcds_rl_vs_duckdb_comparison.png', dpi=150)
    print("  ✓ Saved: tpcds_rl_vs_duckdb_comparison.png")
    plt.close()

    print()
    print("=" * 80)
    print("Analysis complete! Generated 9 visualizations.")
    print("=" * 80)
    print("\nKey Files:")
    print("  1. tpcds_qerror_over_time.png - Raw scatter + moving average")
    print("  2. tpcds_tree_growth.png - Model size over time")
    print("  3. tpcds_qerror_vs_trees.png - Correlation plot")
    print("  4. tpcds_qerror_by_operator.png - Boxplots by operator")
    print("  5. tpcds_qerror_by_query.png - Bar chart per query")
    print("  6. tpcds_actual_vs_predicted.png - Scatter with error bands")
    print("  7. tpcds_qerror_by_query_scatter.png - Raw values with trend")
    print("  8. tpcds_smooth_comparison.png - Smooth curves (RL vs DuckDB)")
    print("  9. tpcds_rl_vs_duckdb_comparison.png - 4-panel comparison")
    print("\nKey insight:")
    overall_improvement = ((np.median(duck_q_errors) - np.median(rl_q_errors)) / np.median(duck_q_errors)) * 100
    print(f"  RL Model is {overall_improvement:.1f}% better than DuckDB baseline (median Q-error)")
    print(f"  RL: {np.median(rl_q_errors):.2f} vs DuckDB: {np.median(duck_q_errors):.2f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
