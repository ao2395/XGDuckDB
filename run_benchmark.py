#!/usr/bin/env python3
"""
Benchmark script to run TPC-H queries and extract Q-error metrics.

Usage:
    python3 run_benchmark.py [--queries all_queries_1k.sql] [--output results.csv]
"""

import subprocess
import re
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path
import threading
import queue

def parse_rl_line(line):
    """Parse a single line for RL training metrics."""
    # Pattern: [RL TRAINING] OPERATOR_NAME: Actual=XXX, Pred=YYY, Q-err=Z.ZZZ
    training_pattern = r'\[RL TRAINING\] ([A-Z_]+(?:\s+[A-Z_]+)?)\s*:\s*Actual=(\d+),\s*Pred=(\d+),\s*Q-err=([\d.]+)'
    match = re.search(training_pattern, line)
    if match:
        return {
            'type': 'metric',
            'operator': match.group(1).strip(),
            'actual': int(match.group(2)),
            'predicted': int(match.group(3)),
            'q_error': float(match.group(4))
        }

    # Pattern: [RL BOOSTING] Incremental update #XXX: trained on YYY samples, total trees=ZZZ, avg Q-error=W.WWW
    update_pattern = r'\[RL BOOSTING\] Incremental update #(\d+): trained on (\d+) samples, total trees=(\d+), avg Q-error=([\d.]+)'
    match = re.search(update_pattern, line)
    if match:
        return {
            'type': 'update',
            'update_num': int(match.group(1)),
            'samples': int(match.group(2)),
            'trees': int(match.group(3)),
            'buffer_avg_q_error': float(match.group(4))
        }

    return None

def stream_reader(stream, result_queue, label):
    """Read stream line by line and put into queue."""
    try:
        for line in iter(stream.readline, ''):
            if line:
                result_queue.put((label, line))
        stream.close()
    except Exception as e:
        result_queue.put(('error', str(e)))

def parse_sql_file(filepath):
    """Parse SQL file into individual queries."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by query markers
    queries = []
    current_query = []
    query_num = 0
    in_query = False

    for line in content.split('\n'):
        # Check if this is a query marker
        if line.startswith('-- Query '):
            if current_query:
                queries.append({
                    'num': query_num,
                    'sql': '\n'.join(current_query)
                })
                current_query = []

            # Extract query number
            match = re.search(r'-- Query (\d+)', line)
            if match:
                query_num = int(match.group(1))
                in_query = True
        elif in_query and line.strip():
            current_query.append(line)

    # Add the last query
    if current_query:
        queries.append({
            'num': query_num,
            'sql': '\n'.join(current_query)
        })

    return queries

def main():
    parser = argparse.ArgumentParser(description='Run DuckDB RL benchmark and extract Q-error metrics')
    parser.add_argument('--queries', default='all_queries_1k.sql', help='Path to SQL queries file')
    parser.add_argument('--output', default='qerror_results.csv', help='Output CSV file for results')
    parser.add_argument('--duckdb', default='./build/release/duckdb', help='Path to DuckDB binary')
    parser.add_argument('--db', default='benchmark.duckdb', help='Path to DuckDB database file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of queries to run (for testing)')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the entire query set (for more training)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing database (skip dbgen)')
    parser.add_argument('--clean', action='store_true', help='Delete existing database before starting')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.queries).exists():
        print(f"Error: Query file not found: {args.queries}")
        sys.exit(1)

    if not Path(args.duckdb).exists():
        print(f"Error: DuckDB binary not found: {args.duckdb}")
        print("Please build DuckDB first: make release")
        sys.exit(1)

    # Clean database if requested
    if args.clean and Path(args.db).exists():
        print(f"Deleting existing database: {args.db}")
        Path(args.db).unlink()

    print(f"[{datetime.now()}] Starting benchmark...")
    print(f"  Query file: {args.queries}")
    print(f"  DuckDB: {args.duckdb}")
    print(f"  Database: {args.db}")
    print(f"  Output: {args.output}")

    # Parse queries
    print(f"\n[{datetime.now()}] Parsing queries...")
    queries = parse_sql_file(args.queries)

    # Filter out dbgen call and keep actual queries
    queries = [q for q in queries if 'dbgen' not in q['sql'].lower()]

    if args.limit:
        queries = queries[:args.limit]

    total_query_count = len(queries) * args.repeat
    print(f"  Found {len(queries)} queries to run")
    if args.repeat > 1:
        print(f"  Will repeat {args.repeat} times = {total_query_count} total query executions")

    # Run dbgen first if not resuming
    if not args.resume:
        print(f"\n[{datetime.now()}] Initializing TPC-H data (dbgen)...")
        print("  This will take a few minutes...")
        result = subprocess.run(
            [args.duckdb, args.db],
            input="CALL dbgen(sf=1);",
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            print(f"  ERROR: Failed to load TPC-H data")
            print(result.stderr)
            sys.exit(1)
        print("  TPC-H data loaded (SF=1)")
    else:
        print(f"\n[{datetime.now()}] Resuming from existing database: {args.db}")

    # Build a single SQL script with all queries (with repetition)
    print(f"\n[{datetime.now()}] Building combined query script...")
    all_queries_sql = []
    query_map = {}  # Track which query number for each position
    query_position = 0

    for repeat_num in range(1, args.repeat + 1):
        for query in queries:
            all_queries_sql.append(f"-- BENCHMARK_QUERY_START {query['num']}")
            all_queries_sql.append(query['sql'])
            all_queries_sql.append(f"-- BENCHMARK_QUERY_END {query['num']}")
            query_map[query_position] = query['num']
            query_position += 1

    combined_sql = '\n'.join(all_queries_sql)

    # Prepare output CSV
    csv_file = open(args.output, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['query_num', 'operator', 'actual', 'predicted', 'q_error', 'trees', 'buffer_avg_qerror'])

    # Run all queries in a SINGLE DuckDB session with streaming output
    print(f"[{datetime.now()}] Running all queries in single DuckDB session...")
    print("  (This preserves RL model training across queries)")
    print("  Writing results to CSV in real-time...")

    try:
        # Start DuckDB process with pipes
        process = subprocess.Popen(
            [args.duckdb, args.db],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        print(f"  Started DuckDB process (PID: {process.pid})")

        # Create queue for threaded output reading
        output_queue = queue.Queue()

        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, output_queue, 'stdout'))
        stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, output_queue, 'stderr'))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        print(f"  Sending {len(all_queries_sql)} lines of SQL...")
        # Send all queries to stdin
        process.stdin.write(combined_sql)
        process.stdin.close()
        print(f"  SQL sent, waiting for output...")

        # Process output in real-time
        current_trees = 0
        current_buffer_avg = 0
        current_query_idx = 0
        total_metrics = 0
        total_updates = 0
        last_output_time = datetime.now()
        timeout_seconds = 300  # 5 minute timeout for receiving first output

        print("")  # Newline for progress updates

        while True:
            try:
                label, line = output_queue.get(timeout=1)

                if label == 'error':
                    print(f"  ERROR: {line}")
                    break

                # Reset timeout on any output
                last_output_time = datetime.now()

                # Print all stderr for debugging (first 10 lines)
                if label == 'stderr' and total_metrics == 0 and total_updates < 10:
                    print(f"  [DEBUG] stderr: {line.strip()}")

                # Only process stderr (where RL logs go)
                if label == 'stderr':
                    parsed = parse_rl_line(line)

                    if parsed:
                        if parsed['type'] == 'metric':
                            # Write metric to CSV immediately
                            query_num = query_map.get(current_query_idx % len(queries), queries[0]['num'])
                            csv_writer.writerow([
                                query_num,
                                parsed['operator'],
                                parsed['actual'],
                                parsed['predicted'],
                                parsed['q_error'],
                                current_trees,
                                current_buffer_avg
                            ])
                            total_metrics += 1

                            # Progress update every 100 metrics
                            if total_metrics % 100 == 0:
                                print(f"  Processed {total_metrics} metrics, {current_trees} trees...", flush=True)
                                csv_file.flush()  # Flush to disk

                        elif parsed['type'] == 'update':
                            current_trees = parsed['trees']
                            current_buffer_avg = parsed['buffer_avg_q_error']
                            total_updates += 1
                            if total_updates == 1:
                                print(f"  First model update! Trees: {current_trees}", flush=True)

                # Check if process finished
                if process.poll() is not None:
                    print(f"  DuckDB process exited with code {process.returncode}")
                    break

            except queue.Empty:
                # Check timeout
                elapsed = (datetime.now() - last_output_time).total_seconds()
                if total_metrics == 0 and elapsed > timeout_seconds:
                    print(f"  WARNING: No output received for {timeout_seconds} seconds")
                    print(f"  Checking if process is still running...")
                    if process.poll() is not None:
                        print(f"  Process exited with code {process.returncode}")
                        break

                # Check if process finished
                if process.poll() is not None:
                    print(f"  DuckDB process exited with code {process.returncode}")
                    # Give threads a bit more time to drain queues
                    import time
                    time.sleep(1)
                    break

        # Wait for process to finish
        return_code = process.wait()
        print(f"  Process finished with return code: {return_code}")

        # Close CSV
        csv_file.close()

        print(f"\n[{datetime.now()}] Processing complete!")

        # Print summary
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Total queries planned: {len(queries)}")
        if args.repeat > 1:
            print(f"Number of repetitions: {args.repeat}")
            print(f"Total query executions planned: {len(queries) * args.repeat}")
        print(f"Total operators measured: {total_metrics}")
        print(f"Total model updates: {total_updates}")
        print(f"Final model size: {current_trees} trees")

        print(f"\nResults saved to: {args.output}")
        print(f"[{datetime.now()}] Done!")

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Cleaning up...")
        if 'process' in locals():
            process.kill()
        if 'csv_file' in locals():
            csv_file.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        if 'process' in locals():
            process.kill()
        if 'csv_file' in locals():
            csv_file.close()
        sys.exit(1)

if __name__ == '__main__':
    main()
