#!/usr/bin/env python3
"""
TPC-DS Benchmark script to run queries and extract Q-error metrics.

Usage:
    python3 run_tpcds_benchmark.py [--queries queries.sql] [--output tpcds_results.csv]
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
import time
import signal
import random

def percentile(values, p):
    """Compute percentile with linear interpolation (numpy-like) for a list of floats."""
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

def parse_rl_line(line):
    """Parse a single line for RL training metrics."""
    # NEW Pattern: [RL TRAINING] OPERATOR_NAME: Actual=XXX, RLPred=YYY, DuckPred=ZZZ, RLQerr=A.AA, DuckQerr=B.BB
    training_pattern = r'\[RL TRAINING\] ([A-Z_]+(?:\s+[A-Z_]+)?)\s*:\s*Actual=(\d+),\s*RLPred=(\d+),\s*DuckPred=(\d+),\s*RLQerr=([\d.]+),\s*DuckQerr=([\d.]+)'
    match = re.search(training_pattern, line)
    if match:
        return {
            'type': 'metric',
            'operator': match.group(1).strip(),
            'actual': int(match.group(2)),
            'rl_predicted': int(match.group(3)),
            'duck_predicted': int(match.group(4)),
            'rl_q_error': float(match.group(5)),
            'duck_q_error': float(match.group(6))
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

def convert_sqlserver_to_duckdb(sql):
    """Convert SQL Server syntax to DuckDB syntax."""
    # Remove "SELECT TOP N" -> "SELECT"
    sql_modified = re.sub(r'\bselect\s+top\s+\d+\s+', 'select ', sql, flags=re.IGNORECASE)

    # Remove any LIMIT clauses at the end
    sql_modified = re.sub(r'\blimit\s+\d+\s*;?\s*$', '', sql_modified, flags=re.IGNORECASE | re.MULTILINE)

    # Fix date arithmetic: "+ N days" -> "+ INTERVAL 'N days'"
    # Pattern: (expression) + N days  or  (expression) - N days
    sql_modified = re.sub(r'\+\s*(\d+)\s+days\b', r"+ INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'-\s*(\d+)\s+days\b', r"- INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)

    # Also handle: + N day (singular)
    sql_modified = re.sub(r'\+\s*(\d+)\s+day\b', r"+ INTERVAL '\1 day'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'-\s*(\d+)\s+day\b', r"- INTERVAL '\1 day'", sql_modified, flags=re.IGNORECASE)

    return sql_modified

def parse_queries_file(filepath):
    """Parse queries.sql file into individual queries."""
    queries = []
    current_query_lines = []
    current_query_num = None

    with open(filepath, 'r') as f:
        for line in f:
            # Check for query marker: -- Query X/TOTAL | Template: ...
            if line.startswith('-- Query '):
                # Save previous query if exists
                if current_query_num is not None and current_query_lines:
                    sql = '\n'.join(current_query_lines)
                    # Convert SQL Server syntax to DuckDB
                    sql = convert_sqlserver_to_duckdb(sql)
                    queries.append({
                        'num': current_query_num,
                        'sql': sql
                    })
                    current_query_lines = []

                # Extract query number
                match = re.search(r'-- Query (\d+)/', line)
                if match:
                    current_query_num = int(match.group(1))

            # Skip comment lines and empty lines when building query
            elif line.startswith('-- start query') or line.startswith('-- end query'):
                continue
            elif line.strip() == ';':
                # End of current query
                if current_query_num is not None and current_query_lines:
                    sql = '\n'.join(current_query_lines)
                    # Convert SQL Server syntax to DuckDB
                    sql = convert_sqlserver_to_duckdb(sql)
                    queries.append({
                        'num': current_query_num,
                        'sql': sql
                    })
                    current_query_lines = []
                    current_query_num = None
            elif current_query_num is not None:
                # Part of the current query
                current_query_lines.append(line.rstrip())

    # Save last query if exists
    if current_query_num is not None and current_query_lines:
        sql = '\n'.join(current_query_lines)
        # Convert SQL Server syntax to DuckDB
        sql = convert_sqlserver_to_duckdb(sql)
        queries.append({
            'num': current_query_num,
            'sql': sql
        })

    return queries

def main():
    parser = argparse.ArgumentParser(description='Run DuckDB TPC-DS benchmark and extract Q-error metrics')
    parser.add_argument('--queries', default='queries.sql', help='Path to TPC-DS queries file')
    parser.add_argument('--output', default='tpcds_results.csv', help='Output CSV file for results')
    parser.add_argument('--duckdb', default='./build/release/duckdb', help='Path to DuckDB binary')
    parser.add_argument('--db', default=':memory:', help='Path to DuckDB database file (default: in-memory)')
    parser.add_argument('--sf', type=float, default=1, help='Scale factor for TPC-DS data generation (default: 1)')
    parser.add_argument('--skip-load', action='store_true',
                        help='Skip TPC-DS data generation (assume --db already contains loaded TPC-DS tables)')
    parser.add_argument('--limit', type=int, help='Limit number of queries to run')
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV file')
    parser.add_argument('--clean', action='store_true', help='Start fresh (delete existing output)')
    parser.add_argument('--shuffle', action='store_true', help='Randomize query execution order')

    args = parser.parse_args()

    # Handle clean vs resume
    if args.clean and Path(args.output).exists():
        Path(args.output).unlink()
        print(f"Deleted existing {args.output}")

    # Parse queries
    print(f"Parsing queries from {args.queries}...")
    queries = parse_queries_file(args.queries)
    print(f"Found {len(queries)} queries")

    if args.limit:
        queries = queries[:args.limit]
        print(f"Limited to first {args.limit} queries")

    # Shuffle queries if requested
    if args.shuffle:
        random.shuffle(queries)
        print(f"Shuffled query order (randomized execution)")
        # Show first 10 query numbers to verify shuffle
        preview = [q['num'] for q in queries[:10]]
        print(f"First 10 queries after shuffle: {preview}")

    # Check if resuming
    processed_queries = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_queries.add(int(row['query_num']))
        print(f"Resuming: {len(processed_queries)} queries already processed")

    # Open CSV file for writing
    csv_file = open(args.output, 'a' if args.resume else 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'query_num', 'operator', 'actual', 'rl_predicted', 'duck_predicted',
        'rl_q_error', 'duck_q_error', 'trees', 'timestamp'
    ])

    if not args.resume or not Path(args.output).exists():
        csv_writer.writeheader()
        csv_file.flush()

    # Start DuckDB process with pipes
    print(f"\nStarting DuckDB process: {args.duckdb} {args.db}")
    process = subprocess.Popen(
        [args.duckdb, args.db],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    print(f"DuckDB process started (PID: {process.pid})")

    # Create queue for threaded output reading
    output_queue = queue.Queue()

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, output_queue, 'stdout'), daemon=True)
    stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, output_queue, 'stderr'), daemon=True)

    stdout_thread.start()
    stderr_thread.start()

    try:
        if not args.skip_load:
            # Generate TPC-DS data (extension is built-in)
            print(f"\nGenerating TPC-DS SF{args.sf} data...")
            print("NOTE: Make sure you rebuilt DuckDB with: make release")
            init_sql = f"""
CALL dsdgen(sf={args.sf});
SELECT 'TPC-DS data loaded successfully';
"""
            process.stdin.write(init_sql)
            process.stdin.flush()

            # Wait for initialization to complete (look for success message)
            init_timeout = time.time() + 300  # 5 minute timeout for data generation
            init_complete = False

            while time.time() < init_timeout:
                try:
                    label, line = output_queue.get(timeout=1)
                    if 'TPC-DS data loaded successfully' in line:
                        init_complete = True
                        print("TPC-DS initialization complete!")
                        break
                    elif 'Error' in line or 'error' in line:
                        print(f"[ERROR] {line.strip()}")
                except queue.Empty:
                    continue

            if not init_complete:
                print("ERROR: TPC-DS initialization timed out or failed")
                process.terminate()
                return 1
        else:
            print("\nSkipping TPC-DS data generation (--skip-load). Assuming database is already loaded.")

        # Run queries
        print(f"\nRunning {len(queries)} queries...")
        print("=" * 80)

        metrics_count = 0
        trees_count = 0
        last_progress_time = time.time()
        failed_queries = []  # Track failed queries
        rl_q_errors = []
        duck_q_errors = []

        for i, query in enumerate(queries, 1):
            query_num = query['num']

            # Skip if already processed (resume mode)
            if query_num in processed_queries:
                print(f"Skipping query {query_num} (already processed)")
                continue

            print(f"\n[{i}/{len(queries)}] Running Query {query_num}...")

            # Send query to DuckDB with completion marker
            query_marker = f"QUERY_COMPLETE_{query_num}"
            process.stdin.write(query['sql'] + ';\n')
            process.stdin.write(f"SELECT '{query_marker}';\n")
            process.stdin.flush()

            # Process output until query completes
            query_start = time.time()
            query_complete = False
            query_had_error = False
            last_activity = time.time()

            while True:
                try:
                    label, line = output_queue.get(timeout=1.0)
                    last_activity = time.time()

                    # Check for completion marker
                    if query_marker in line:
                        query_complete = True
                        break

                    # Check for errors
                    if 'Error:' in line or 'ERROR:' in line:
                        print(f"  ERROR in query: {line.strip()}")
                        query_complete = True
                        query_had_error = True
                        failed_queries.append({'query_num': query_num, 'error': line.strip()})
                        break

                    # Parse for RL metrics
                    parsed = parse_rl_line(line)

                    if parsed and parsed['type'] == 'metric':
                        # Write metric to CSV immediately
                        csv_writer.writerow({
                            'query_num': query_num,
                            'operator': parsed['operator'],
                            'actual': parsed['actual'],
                            'rl_predicted': parsed['rl_predicted'],
                            'duck_predicted': parsed['duck_predicted'],
                            'rl_q_error': parsed['rl_q_error'],
                            'duck_q_error': parsed['duck_q_error'],
                            'trees': trees_count,
                            'timestamp': datetime.now().isoformat()
                        })
                        csv_file.flush()
                        metrics_count += 1
                        rl_q_errors.append(parsed['rl_q_error'])
                        duck_q_errors.append(parsed['duck_q_error'])

                        # Progress update every 50 metrics
                        if metrics_count % 50 == 0:
                            elapsed = time.time() - last_progress_time
                            print(f"  Progress: {metrics_count} metrics, {trees_count} trees")
                            last_progress_time = time.time()

                    elif parsed and parsed['type'] == 'update':
                        trees_count = parsed['trees']
                        print(f"  [TRAINING UPDATE] Trees: {trees_count}, Samples: {parsed['samples']}, Avg Q-error: {parsed['buffer_avg_q_error']:.2f}")

                except queue.Empty:
                    # Check if process is still alive
                    if process.poll() is not None:
                        print("ERROR: DuckDB process terminated unexpectedly")
                        break

                    # Check for timeout (30 seconds of no activity)
                    if time.time() - last_activity > 30:
                        print(f"  WARNING: No activity for 30 seconds, assuming query stuck")
                        break

                    continue

            elapsed = time.time() - query_start
            print(f"  Completed in {elapsed:.2f}s (Metrics: {metrics_count}, Trees: {trees_count})")

        print("\n" + "=" * 80)
        print(f"Benchmark complete!")
        print(f"Total queries: {len(queries)}")
        print(f"Successful queries: {len(queries) - len(failed_queries)}")
        print(f"Failed queries: {len(failed_queries)}")
        print(f"Total metrics collected: {metrics_count}")
        print(f"Total trees: {trees_count}")
        print(f"Results written to: {args.output}")

        # Summary stats for Q-error distributions
        if metrics_count > 0:
            rl_med = percentile(rl_q_errors, 50)
            rl_p90 = percentile(rl_q_errors, 90)
            rl_p95 = percentile(rl_q_errors, 95)
            duck_med = percentile(duck_q_errors, 50)
            duck_p90 = percentile(duck_q_errors, 90)
            duck_p95 = percentile(duck_q_errors, 95)

            print("\nQ-error summary (per-operator metrics)")
            print("-" * 80)
            print(f"RL   : median={rl_med:.2f}  p90={rl_p90:.2f}  p95={rl_p95:.2f}")
            print(f"Duck : median={duck_med:.2f}  p90={duck_p90:.2f}  p95={duck_p95:.2f}")

        if failed_queries:
            print(f"\nFailed queries:")
            for failure in failed_queries[:10]:  # Show first 10
                print(f"  Query {failure['query_num']}: {failure['error'][:80]}")
            if len(failed_queries) > 10:
                print(f"  ... and {len(failed_queries) - 10} more")

            # Save failed queries to file
            failed_file = args.output.replace('.csv', '_failed.txt')
            with open(failed_file, 'w') as f:
                f.write("Failed Queries Summary\n")
                f.write("=" * 80 + "\n\n")
                for failure in failed_queries:
                    f.write(f"Query {failure['query_num']}\n")
                    f.write(f"Error: {failure['error']}\n\n")
            print(f"\nFailed queries details saved to: {failed_file}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")

    finally:
        # Clean up
        try:
            process.stdin.write('.quit\n')
            process.stdin.flush()
        except:
            pass

        process.terminate()
        process.wait(timeout=5)
        csv_file.close()

        print(f"CSV file saved: {args.output}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
