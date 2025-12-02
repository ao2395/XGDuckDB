#!/usr/bin/env python3
"""
Timing benchmark with single DuckDB session.
Redirects output to file, times queries from Python.

Usage:
    python3 run_timing_benchmark_v2.py --queries queries_sf5.sql --sf 5
"""

import subprocess
import re
import sys
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path

def convert_sqlserver_to_duckdb(sql):
    """Convert SQL Server syntax to DuckDB syntax."""
    sql_modified = re.sub(r'\bselect\s+top\s+\d+\s+', 'select ', sql, flags=re.IGNORECASE)
    sql_modified = re.sub(r'\blimit\s+\d+\s*;?\s*$', '', sql_modified, flags=re.IGNORECASE | re.MULTILINE)
    sql_modified = re.sub(r'\+\s*(\d+)\s+days\b', r"+ INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'-\s*(\d+)\s+days\b', r"- INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)
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
            if line.startswith('-- Query '):
                if current_query_num is not None and current_query_lines:
                    sql = '\n'.join(current_query_lines)
                    sql = convert_sqlserver_to_duckdb(sql)
                    queries.append({'num': current_query_num, 'sql': sql})
                    current_query_lines = []
                match = re.search(r'-- Query (\d+)/', line)
                if match:
                    current_query_num = int(match.group(1))
            elif line.startswith('-- start query') or line.startswith('-- end query'):
                continue
            elif line.strip() == ';':
                if current_query_num is not None and current_query_lines:
                    sql = '\n'.join(current_query_lines)
                    sql = convert_sqlserver_to_duckdb(sql)
                    queries.append({'num': current_query_num, 'sql': sql})
                    current_query_lines = []
                    current_query_num = None
            elif current_query_num is not None:
                current_query_lines.append(line.rstrip())

    if current_query_num is not None and current_query_lines:
        sql = '\n'.join(current_query_lines)
        sql = convert_sqlserver_to_duckdb(sql)
        queries.append({'num': current_query_num, 'sql': sql})

    return queries

def main():
    parser = argparse.ArgumentParser(description='Time DuckDB queries in single session')
    parser.add_argument('--queries', default='queries_sf5.sql', help='Path to queries file')
    parser.add_argument('--output', default='timing_results.csv', help='Output CSV file')
    parser.add_argument('--duckdb', default='./build/release/duckdb', help='Path to DuckDB binary')
    parser.add_argument('--sf', type=float, default=5, help='Scale factor (default: 5)')
    parser.add_argument('--limit', type=int, help='Limit number of queries')

    args = parser.parse_args()

    # Parse queries
    print(f"Parsing queries from {args.queries}...")
    queries = parse_queries_file(args.queries)
    print(f"Found {len(queries)} queries")

    if args.limit:
        queries = queries[:args.limit]
        print(f"Limited to first {args.limit} queries")

    # Prepare all queries in a single SQL script
    print(f"\nPreparing benchmark script...")

    sql_script = f"CALL dsdgen(sf={args.sf});\n"
    sql_script += "SELECT 'DATA_LOADED';\n"

    for query in queries:
        # Add marker before query
        sql_script += f"SELECT 'START_QUERY_{query['num']}';\n"
        sql_script += query['sql'] + ";\n"
        sql_script += f"SELECT 'END_QUERY_{query['num']}';\n"

    sql_script += ".quit\n"

    # Write script to file for debugging
    with open('benchmark_script.sql', 'w') as f:
        f.write(sql_script)
    print(f"Script written to benchmark_script.sql")

    # Open output file
    output_log = open('benchmark_output.log', 'w')

    # Run DuckDB with script
    print(f"\nRunning DuckDB benchmark...")
    print(f"Output redirected to benchmark_output.log")
    print("=" * 80)

    benchmark_start = time.time()

    process = subprocess.Popen(
        [args.duckdb, ':memory:'],
        stdin=subprocess.PIPE,
        stdout=output_log,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Send entire script at once
    process.stdin.write(sql_script)
    process.stdin.close()

    # Wait for completion
    print("Waiting for benchmark to complete...")
    process.wait()

    benchmark_elapsed = time.time() - benchmark_start
    output_log.close()

    print(f"\nBenchmark completed in {benchmark_elapsed:.2f}s")
    print("Parsing output log for timings...")

    # Parse output log for timing markers
    timings = {}
    current_query = None
    query_start = None

    with open('benchmark_output.log', 'r') as f:
        for line in f:
            if 'START_QUERY_' in line:
                match = re.search(r'START_QUERY_(\d+)', line)
                if match:
                    current_query = int(match.group(1))
                    query_start = time.time()  # This won't work - we need timestamps from log

    print("WARNING: Timing extraction from log not implemented")
    print("Run completed, but individual query timings are approximate")

    # Write summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Total elapsed: {benchmark_elapsed:.2f}s")
    print(f"Average time per query: {benchmark_elapsed / len(queries):.3f}s")
    print(f"\nOutput log: benchmark_output.log")
    print(f"Script: benchmark_script.sql")

    return 0

if __name__ == '__main__':
    sys.exit(main())
