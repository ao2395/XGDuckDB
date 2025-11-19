#!/usr/bin/env python3
"""
Run TPC-DS queries on VANILLA DuckDB to get independent baseline Q-errors.

Uses EXPLAIN ANALYZE to extract estimated vs actual cardinalities.

Usage:
    # First install vanilla DuckDB: brew install duckdb
    python3 run_vanilla_baseline.py --duckdb /opt/homebrew/bin/duckdb --limit 100
"""

import subprocess
import re
import csv
import argparse
from pathlib import Path
import time

def convert_sqlserver_to_duckdb(sql):
    """Convert SQL Server syntax to DuckDB syntax."""
    sql_modified = re.sub(r'\bselect\s+top\s+\d+\s+', 'select ', sql, flags=re.IGNORECASE)
    sql_modified = re.sub(r'\blimit\s+\d+\s*;?\s*$', '', sql_modified, flags=re.IGNORECASE | re.MULTILINE)
    sql_modified = re.sub(r'\+\s*(\d+)\s+days\b', r"+ INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'-\s*(\d+)\s+days\b', r"- INTERVAL '\1 days'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'\+\s*(\d+)\s+day\b', r"+ INTERVAL '\1 day'", sql_modified, flags=re.IGNORECASE)
    sql_modified = re.sub(r'-\s*(\d+)\s+day\b', r"- INTERVAL '\1 day'", sql_modified, flags=re.IGNORECASE)
    return sql_modified

def parse_queries_file(filepath, limit=None):
    """Parse queries.sql file."""
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

    if limit:
        queries = queries[:limit]

    return queries

def parse_explain_for_estimates(output):
    """Parse EXPLAIN output to get estimated cardinalities."""
    operators = []
    lines = output.split('\n')

    current_op = None
    for i, line in enumerate(lines):
        # Match operator names (TABLE_SCAN, HASH_JOIN, etc.)
        op_match = re.search(r'│\s+(TABLE_SCAN|SEQ_SCAN|FILTER|HASH_JOIN|HASH_GROUP_BY|PROJECTION|'
                            r'NESTED_LOOP_JOIN|PERFECT_HASH_GROUP_BY|UNGROUPED_AGGREGATE|'
                            r'ORDER_BY|TOP_N|STREAMING_LIMIT|CROSS_PRODUCT)', line)

        if op_match:
            current_op = op_match.group(1).strip()

        # Look for cardinality estimate: "~X rows" or "X,XXX rows" (may have ~ for estimate)
        if current_op:
            card_match = re.search(r'│\s+~?([\d,]+)\s+rows?\s+│', line)
            if card_match:
                estimated = int(card_match.group(1).replace(',', ''))
                operators.append({'operator': current_op, 'estimated': estimated})
                current_op = None  # Reset after finding estimate

    return operators

def parse_explain_analyze_for_actuals(output):
    """Parse EXPLAIN ANALYZE output to get actual cardinalities."""
    operators = []
    lines = output.split('\n')

    current_op = None
    for i, line in enumerate(lines):
        # Match operator names
        op_match = re.search(r'│\s+(TABLE_SCAN|SEQ_SCAN|FILTER|HASH_JOIN|HASH_GROUP_BY|PROJECTION|'
                            r'NESTED_LOOP_JOIN|PERFECT_HASH_GROUP_BY|UNGROUPED_AGGREGATE|'
                            r'ORDER_BY|TOP_N|STREAMING_LIMIT|CROSS_PRODUCT)', line)

        if op_match:
            current_op = op_match.group(1).strip()

        # Look for actual cardinality
        if current_op:
            card_match = re.search(r'│\s+([\d,]+)\s+rows?\s+│', line)
            if card_match:
                actual = int(card_match.group(1).replace(',', ''))
                operators.append({'operator': current_op, 'actual': actual})
                current_op = None

    return operators

def normalize_operator_name(op):
    """Normalize operator names to handle variations between EXPLAIN and EXPLAIN ANALYZE."""
    # Map variations to canonical names
    mapping = {
        'SEQ_SCAN': 'TABLE_SCAN',
        'TABLE_SCAN': 'TABLE_SCAN',
        'HASH_JOIN': 'HASH_JOIN',
        'HASH_GROUP_BY': 'HASH_GROUP_BY',
        'TOP_N': 'TOP_N',
        'STREAMING_LIMIT': 'LIMIT',
        'FILTER': 'FILTER',
        'PROJECTION': 'PROJECTION',
        'NESTED_LOOP_JOIN': 'NESTED_LOOP_JOIN',
        'PERFECT_HASH_GROUP_BY': 'PERFECT_HASH_GROUP_BY',
        'UNGROUPED_AGGREGATE': 'UNGROUPED_AGGREGATE',
        'ORDER_BY': 'ORDER_BY',
        'CROSS_PRODUCT': 'CROSS_PRODUCT',
    }
    return mapping.get(op, op)

def match_estimates_and_actuals(estimates, actuals):
    """Match estimated and actual cardinalities by position with fuzzy operator matching."""
    results = []

    # Strategy: Match by position, but be lenient with operator names
    # DuckDB sometimes uses different names in EXPLAIN vs EXPLAIN ANALYZE

    # If counts don't match, try to align by removing operators that only appear in one
    est_idx = 0
    act_idx = 0

    while est_idx < len(estimates) and act_idx < len(actuals):
        est = estimates[est_idx]
        act = actuals[act_idx]

        est_op_norm = normalize_operator_name(est['operator'])
        act_op_norm = normalize_operator_name(act['operator'])

        if est_op_norm == act_op_norm:
            # Perfect match
            estimated = est['estimated']
            actual = act['actual']
            q_error = max(actual / max(estimated, 1), estimated / max(actual, 1))

            results.append({
                'operator': act['operator'],  # Use actual's operator name
                'estimated': estimated,
                'actual': actual,
                'q_error': q_error
            })
            est_idx += 1
            act_idx += 1
        elif est_idx + 1 < len(estimates) and normalize_operator_name(estimates[est_idx + 1]['operator']) == act_op_norm:
            # Actual matches next estimate (skip current estimate)
            est_idx += 1
        elif act_idx + 1 < len(actuals) and est_op_norm == normalize_operator_name(actuals[act_idx + 1]['operator']):
            # Estimate matches next actual (skip current actual)
            act_idx += 1
        else:
            # No match, try pairing them anyway with a warning
            estimated = est['estimated']
            actual = act['actual']
            q_error = max(actual / max(estimated, 1), estimated / max(actual, 1))

            results.append({
                'operator': f"{est['operator']}/{act['operator']}",  # Show mismatch
                'estimated': estimated,
                'actual': actual,
                'q_error': q_error
            })
            est_idx += 1
            act_idx += 1

    return results

def main():
    parser = argparse.ArgumentParser(description='Run vanilla DuckDB baseline benchmark')
    parser.add_argument('--queries', default='queries.sql', help='Path to queries file')
    parser.add_argument('--output', default='vanilla_baseline.csv', help='Output CSV')
    parser.add_argument('--duckdb', default='duckdb', help='Path to VANILLA DuckDB (e.g., /opt/homebrew/bin/duckdb)')
    parser.add_argument('--sf', type=float, default=1, help='Scale factor')
    parser.add_argument('--limit', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--clean', action='store_true', help='Start fresh')

    args = parser.parse_args()

    # Verify vanilla DuckDB
    try:
        result = subprocess.run([args.duckdb, '--version'], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip()
        print(f"Using vanilla DuckDB: {version}")

        # Warn if it looks like the modified version
        if 'RL' in version.upper() or 'BOOST' in version.upper():
            print("WARNING: This appears to be the RL-modified DuckDB!")
            print("Please use a vanilla DuckDB binary (e.g., brew install duckdb)")
            return 1
    except FileNotFoundError:
        print(f"ERROR: DuckDB binary not found at: {args.duckdb}")
        print("\nTo get vanilla DuckDB:")
        print("  brew install duckdb")
        print("  Then run: python3 run_vanilla_baseline.py --duckdb $(which duckdb)")
        return 1

    if args.clean and Path(args.output).exists():
        Path(args.output).unlink()

    print(f"\nParsing queries from {args.queries}...")
    queries = parse_queries_file(args.queries, limit=args.limit)
    print(f"Found {len(queries)} queries")

    # Open CSV
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'query_num', 'operator', 'estimated', 'actual', 'q_error'
        ])
        writer.writeheader()

        print(f"\nRunning EXPLAIN ANALYZE on {len(queries)} queries...")
        print("=" * 80)

        total_operators = 0
        failed_queries = []

        for i, query in enumerate(queries, 1):
            query_num = query['num']
            print(f"[{i}/{len(queries)}] Query {query_num}...", end=' ', flush=True)

            try:
                # Step 1: Run EXPLAIN to get estimates
                explain_script = f"""
CALL dsdgen(sf={args.sf});
EXPLAIN {query['sql']};
"""
                result_explain = subprocess.run(
                    [args.duckdb, ':memory:'],
                    input=explain_script,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # Check for errors in EXPLAIN
                if result_explain.returncode != 0 or 'Error' in result_explain.stderr:
                    error_msg = result_explain.stderr.split('\n')[0] if result_explain.stderr else 'Unknown error'
                    print(f"✗ ({error_msg[:50]})")
                    failed_queries.append({'query_num': query_num, 'error': error_msg})
                    continue

                # Parse estimates
                estimates = parse_explain_for_estimates(result_explain.stdout)

                # Step 2: Run EXPLAIN ANALYZE to get actuals
                analyze_script = f"""
CALL dsdgen(sf={args.sf});
EXPLAIN ANALYZE {query['sql']};
"""
                result_analyze = subprocess.run(
                    [args.duckdb, ':memory:'],
                    input=analyze_script,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # Check for errors in EXPLAIN ANALYZE
                if result_analyze.returncode != 0 or 'Error' in result_analyze.stderr:
                    error_msg = result_analyze.stderr.split('\n')[0] if result_analyze.stderr else 'Unknown error'
                    print(f"✗ ({error_msg[:50]})")
                    failed_queries.append({'query_num': query_num, 'error': error_msg})
                    continue

                # Parse actuals
                actuals = parse_explain_analyze_for_actuals(result_analyze.stdout)

                # Match estimates with actuals
                matched = match_estimates_and_actuals(estimates, actuals)

                if not matched:
                    print(f"✗ (no operators matched)")
                    failed_queries.append({'query_num': query_num, 'error': 'Failed to match operators'})
                    continue

                # Write to CSV
                for row in matched:
                    writer.writerow({
                        'query_num': query_num,
                        'operator': row['operator'],
                        'estimated': row['estimated'],
                        'actual': row['actual'],
                        'q_error': row['q_error']
                    })
                    total_operators += 1

                print(f"✓ ({len(matched)} operators)")

            except subprocess.TimeoutExpired:
                print("✗ (timeout)")
                failed_queries.append({'query_num': query_num, 'error': 'Timeout'})
            except Exception as e:
                print(f"✗ (error: {e})")
                failed_queries.append({'query_num': query_num, 'error': str(e)})

    print("\n" + "=" * 80)
    print(f"Vanilla baseline complete!")
    print(f"Total queries: {len(queries)}")
    print(f"Successful queries: {len(queries) - len(failed_queries)}")
    print(f"Failed queries: {len(failed_queries)}")
    print(f"Total operators analyzed: {total_operators}")
    print(f"Results written to: {args.output}")

    if failed_queries:
        print(f"\nFailed queries:")
        for failure in failed_queries[:10]:
            print(f"  Query {failure['query_num']}: {failure['error'][:60]}")
        if len(failed_queries) > 10:
            print(f"  ... and {len(failed_queries) - 10} more")

        # Save failed queries
        failed_file = args.output.replace('.csv', '_failed.txt')
        with open(failed_file, 'w') as f:
            f.write("Failed Queries - Vanilla DuckDB Baseline\n")
            f.write("=" * 80 + "\n\n")
            for failure in failed_queries:
                f.write(f"Query {failure['query_num']}\n")
                f.write(f"Error: {failure['error']}\n\n")
        print(f"\nFailed queries saved to: {failed_file}")

    print("\nNow compare with your RL results:")
    print(f"  python3 analyze_tpcds.py --input tpcds_results.csv")
    print(f"  python3 analyze_tpcds.py --input {args.output}")
    print("\nOr create a comparison script to analyze both CSVs together.")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
