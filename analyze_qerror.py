#!/usr/bin/env python3
"""
Analyze Q-error results from benchmark CSV.

Usage:
    python3 analyze_qerror.py [--input qerror_results.csv]
"""

import csv
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

def analyze_qerror(csv_path):
    """Analyze Q-error metrics from CSV file."""

    # Read data
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'query_num': int(row['query_num']),
                'operator': row['operator'],
                'actual': int(row['actual']),
                'predicted': int(row['predicted']),
                'q_error': float(row['q_error']),
                'trees': int(row['trees']),
                'buffer_avg_qerror': float(row['buffer_avg_qerror'])
            })

    if not data:
        print("No data found in CSV file!")
        return

    print(f"{'='*70}")
    print(f"Q-ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Total operators measured: {len(data)}")

    # Overall statistics
    q_errors = [d['q_error'] for d in data]
    print(f"\nOVERALL Q-ERROR STATISTICS:")
    print(f"  Mean:     {statistics.mean(q_errors):.4f}")
    print(f"  Median:   {statistics.median(q_errors):.4f}")
    print(f"  Std Dev:  {statistics.stdev(q_errors) if len(q_errors) > 1 else 0:.4f}")
    print(f"  Min:      {min(q_errors):.4f}")
    print(f"  Max:      {max(q_errors):.4f}")

    # Percentiles
    q_errors_sorted = sorted(q_errors)
    p50 = q_errors_sorted[len(q_errors_sorted)//2]
    p90 = q_errors_sorted[int(len(q_errors_sorted)*0.9)]
    p95 = q_errors_sorted[int(len(q_errors_sorted)*0.95)]
    p99 = q_errors_sorted[int(len(q_errors_sorted)*0.99)]

    print(f"\nPERCENTILES:")
    print(f"  50th (Median): {p50:.4f}")
    print(f"  90th:          {p90:.4f}")
    print(f"  95th:          {p95:.4f}")
    print(f"  99th:          {p99:.4f}")

    # Q-error ranges
    excellent = sum(1 for q in q_errors if q <= 1.5)
    good = sum(1 for q in q_errors if 1.5 < q <= 3.0)
    fair = sum(1 for q in q_errors if 3.0 < q <= 10.0)
    poor = sum(1 for q in q_errors if q > 10.0)

    print(f"\nQ-ERROR DISTRIBUTION:")
    print(f"  Excellent (≤1.5):     {excellent:6d} ({100*excellent/len(q_errors):5.1f}%)")
    print(f"  Good (1.5-3.0):       {good:6d} ({100*good/len(q_errors):5.1f}%)")
    print(f"  Fair (3.0-10.0):      {fair:6d} ({100*fair/len(q_errors):5.1f}%)")
    print(f"  Poor (>10.0):         {poor:6d} ({100*poor/len(q_errors):5.1f}%)")

    # Per-operator statistics
    by_operator = defaultdict(list)
    for d in data:
        by_operator[d['operator']].append(d['q_error'])

    print(f"\nPER-OPERATOR Q-ERROR (Mean):")
    operator_stats = []
    for op, qerrs in by_operator.items():
        operator_stats.append((op, statistics.mean(qerrs), len(qerrs)))

    operator_stats.sort(key=lambda x: x[1])  # Sort by mean Q-error

    for op, mean_qerr, count in operator_stats:
        print(f"  {op:20s} {mean_qerr:8.4f}  (n={count})")

    # Model convergence over time (tree growth)
    unique_queries = sorted(set(d['query_num'] for d in data))
    print(f"\nMODEL CONVERGENCE:")
    print(f"  Number of unique queries: {len(unique_queries)}")
    print(f"  Total query executions: {len([d for d in data if d['operator'] == list(set([x['operator'] for x in data]))[0]])} (approximate)")

    # Tree growth and Q-error improvement over time
    if len(data) > 0:
        first_trees = data[0]['trees']
        last_trees = data[-1]['trees']

        print(f"  Starting trees: {first_trees}")
        print(f"  Ending trees:   {last_trees}")
        print(f"  Trees added:    {last_trees - first_trees}")

        # Analyze Q-error improvement per operator type over time
        print(f"\nPER-OPERATOR Q-ERROR OVER TIME:")

        # Split data into chunks (beginning, middle, end)
        chunk_size = len(data) // 3
        if chunk_size > 0:
            beginning = data[:chunk_size]
            middle = data[chunk_size:2*chunk_size]
            end = data[2*chunk_size:]

            # Group by operator for each time period
            for period_name, period_data in [("Beginning (first 33%)", beginning),
                                              ("Middle (middle 33%)", middle),
                                              ("End (last 33%)", end)]:
                if period_data:
                    period_by_op = defaultdict(list)
                    for d in period_data:
                        period_by_op[d['operator']].append(d['q_error'])

                    print(f"\n  {period_name}:")
                    for op, qerrs in sorted(period_by_op.items()):
                        mean_qerr = statistics.mean(qerrs)
                        print(f"    {op:20s} {mean_qerr:8.4f}  (n={len(qerrs)})")

    # Find worst predictions
    print(f"\nWORST 10 PREDICTIONS:")
    worst = sorted(data, key=lambda x: x['q_error'], reverse=True)[:10]
    for i, d in enumerate(worst, 1):
        print(f"  {i:2d}. Query {d['query_num']:4d}, {d['operator']:15s}: "
              f"Actual={d['actual']:8d}, Pred={d['predicted']:8d}, Q-err={d['q_error']:.2f}")

    # Find best predictions
    print(f"\nBEST 10 PREDICTIONS:")
    best = sorted(data, key=lambda x: x['q_error'])[:10]
    for i, d in enumerate(best, 1):
        print(f"  {i:2d}. Query {d['query_num']:4d}, {d['operator']:15s}: "
              f"Actual={d['actual']:8d}, Pred={d['predicted']:8d}, Q-err={d['q_error']:.4f}")

    print(f"\n{'='*70}")

    # Generate plots
    print(f"\nGenerating plots...")
    generate_plots(data, csv_path)

def generate_plots(data, csv_path):
    """Generate visualization plots for Q-error analysis."""
    output_dir = Path(csv_path).parent

    # 1a. Raw Q-error over time (individual values)
    fig, ax = plt.subplots(figsize=(12, 6))
    q_errors = [d['q_error'] for d in data]

    # Sample if too many points for visibility
    if len(q_errors) > 10000:
        import random
        indices = sorted(random.sample(range(len(q_errors)), 10000))
        sampled_qerrors = [q_errors[i] for i in indices]
        ax.scatter(indices, sampled_qerrors, alpha=0.3, s=10, c='blue')
        ax.set_title(f'Raw Q-error Values Over Time (sampled {len(indices)} of {len(q_errors)} points)',
                     fontsize=14, fontweight='bold')
    else:
        ax.scatter(range(len(q_errors)), q_errors, alpha=0.3, s=10, c='blue')
        ax.set_title('Raw Q-error Values Over Time', fontsize=14, fontweight='bold')

    ax.set_xlabel('Operator Index', fontsize=12)
    ax.set_ylabel('Q-error (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'qerror_raw.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 1b. Q-error over time (moving average)
    fig, ax = plt.subplots(figsize=(12, 6))
    window_size = 100

    # Calculate moving average
    moving_avg = []
    for i in range(len(q_errors)):
        start = max(0, i - window_size + 1)
        moving_avg.append(statistics.mean(q_errors[start:i+1]))

    ax.plot(range(len(moving_avg)), moving_avg, linewidth=2, color='blue', alpha=0.7)
    ax.set_xlabel('Operator Index', fontsize=12)
    ax.set_ylabel('Q-error (Moving Average)', fontsize=12)
    ax.set_title(f'Q-error Over Time (Window={window_size})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plot_path = output_dir / 'qerror_moving_avg.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 2. Tree growth over time
    fig, ax = plt.subplots(figsize=(12, 6))
    trees = [d['trees'] for d in data]
    ax.plot(range(len(trees)), trees, linewidth=2, color='green', alpha=0.7)
    ax.set_xlabel('Operator Index', fontsize=12)
    ax.set_ylabel('Number of Trees', fontsize=12)
    ax.set_title('Model Growth (Tree Count Over Time)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'tree_growth.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 3. Per-operator Q-error (box plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    by_operator = defaultdict(list)
    for d in data:
        by_operator[d['operator']].append(d['q_error'])

    # Sort by median Q-error
    operators = sorted(by_operator.keys(), key=lambda op: statistics.median(by_operator[op]))
    operator_data = [by_operator[op] for op in operators]

    bp = ax.boxplot(operator_data, labels=operators, patch_artist=True, vert=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Q-error (log scale)', fontsize=12)
    ax.set_ylabel('Operator Type', fontsize=12)
    ax.set_title('Q-error Distribution by Operator Type', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plot_path = output_dir / 'qerror_by_operator.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 4. Q-error improvement per operator over time
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by operator and calculate moving average for each
    for operator in sorted(by_operator.keys()):
        op_data = [(i, d['q_error']) for i, d in enumerate(data) if d['operator'] == operator]
        if len(op_data) < 10:
            continue

        indices, qerrors = zip(*op_data)

        # Calculate moving average
        window = min(50, len(qerrors) // 4)
        if window < 5:
            continue

        moving_avg = []
        moving_idx = []
        for i in range(len(qerrors)):
            start = max(0, i - window + 1)
            moving_avg.append(statistics.mean(qerrors[start:i+1]))
            moving_idx.append(indices[i])

        ax.plot(moving_idx, moving_avg, linewidth=2, label=operator, alpha=0.7)

    ax.set_xlabel('Operator Index', fontsize=12)
    ax.set_ylabel('Q-error (Moving Average)', fontsize=12)
    ax.set_title('Q-error Improvement by Operator Type Over Time', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'qerror_improvement_by_operator.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 5. Q-error vs Tree Count (scatter with trend)
    fig, ax = plt.subplots(figsize=(12, 6))
    trees = [d['trees'] for d in data]
    q_errors = [d['q_error'] for d in data]

    # Sample if too many points
    if len(data) > 5000:
        import random
        indices = random.sample(range(len(data)), 5000)
        trees = [trees[i] for i in indices]
        q_errors = [q_errors[i] for i in indices]

    ax.scatter(trees, q_errors, alpha=0.3, s=20, c='blue')
    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Q-error', fontsize=12)
    ax.set_title('Q-error vs Model Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'qerror_vs_trees.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    print(f"\nAll plots saved to: {output_dir}/")

def analyze_operator_trends(csv_path, operator_type):
    """Analyze Q-error trends for a specific operator type over time."""

    # Read data for specific operator
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['operator'] == operator_type:
                data.append({
                    'query_num': int(row['query_num']),
                    'actual': int(row['actual']),
                    'predicted': int(row['predicted']),
                    'q_error': float(row['q_error']),
                    'trees': int(row['trees'])
                })

    if not data:
        print(f"No data found for operator: {operator_type}")
        return

    print(f"\n{'='*70}")
    print(f"OPERATOR-SPECIFIC ANALYSIS: {operator_type}")
    print(f"{'='*70}")
    print(f"Total {operator_type} operators measured: {len(data)}")

    q_errors = [d['q_error'] for d in data]
    print(f"\nQ-ERROR STATISTICS:")
    print(f"  Mean:     {statistics.mean(q_errors):.4f}")
    print(f"  Median:   {statistics.median(q_errors):.4f}")
    print(f"  Min:      {min(q_errors):.4f}")
    print(f"  Max:      {max(q_errors):.4f}")

    # Analyze improvement over time (split into deciles)
    print(f"\nQ-ERROR IMPROVEMENT OVER TIME (by decile):")
    decile_size = len(data) // 10
    if decile_size > 0:
        for i in range(10):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < 9 else len(data)
            decile_data = data[start_idx:end_idx]
            decile_qerrs = [d['q_error'] for d in decile_data]
            avg_trees = statistics.mean([d['trees'] for d in decile_data])

            print(f"  Decile {i+1:2d} (rows {start_idx:5d}-{end_idx:5d}): "
                  f"Q-err={statistics.mean(decile_qerrs):7.4f}, "
                  f"Trees={int(avg_trees)}")

    print(f"\n{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Q-error benchmark results')
    parser.add_argument('--input', default='qerror_results.csv', help='Input CSV file')
    parser.add_argument('--operator', type=str, help='Analyze trends for specific operator type (e.g., HASH_JOIN, SEQ_SCAN)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print(f"\nRun the benchmark first:")
        print(f"  python3 run_benchmark.py")
        sys.exit(1)

    if args.operator:
        # Analyze specific operator trends
        analyze_operator_trends(args.input, args.operator)
    else:
        # General analysis
        analyze_qerror(args.input)

if __name__ == '__main__':
    main()
