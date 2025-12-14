#!/bin/bash
set -e

DB_FILE="tpcds_sf5.db"
DUCKDB="./build/release/duckdb"

# Check if DB exists, if not generate data
if [ ! -f "$DB_FILE" ]; then
    echo "Generating TPC-DS SF=5 data..."
    $DUCKDB $DB_FILE "CALL dsdgen(sf=5);"
else
    echo "Database $DB_FILE already exists. Skipping generation."
fi

echo "Running queries_sf5.sql..."
# Use .timer on to get timing for each query, and time the whole process
/usr/bin/time -p $DUCKDB $DB_FILE <<EOF
.timer on
.read queries_sf5.sql
EOF
