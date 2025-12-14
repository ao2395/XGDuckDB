#!/usr/bin/env python3
"""
Quick script to check if the XGBoost model is actually varying predictions.
"""

import csv
from collections import defaultdict
import sys

# Read CSV
try:
    with open('qerror_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
except FileNotFoundError:
    print("Error: qerror_results.csv not found!")
    sys.exit(1)

if not data:
    print("Error: CSV file is empty!")
    sys.exit(1)

print(f"Loaded {len(data)} rows from CSV")
print(f"Columns: {list(data[0].keys())}")
print()

# Group by operator
by_operator = defaultdict(list)
for row in data:
    op = row['operator']
    try:
        pred = int(row['predicted'])
        actual = int(row['actual'])
        by_operator[op].append((pred, actual))
    except (ValueError, KeyError) as e:
        print(f"Error parsing row: {e}")
        continue

print("=== PREDICTION ANALYSIS ===\n")

for op in sorted(by_operator.keys()):
    preds = [p for p, a in by_operator[op]]
    actuals = [a for p, a in by_operator[op]]

    unique_preds = len(set(preds))
    unique_actuals = len(set(actuals))

    print(f"{op}:")
    print(f"  Total samples: {len(preds)}")
    print(f"  Unique predictions: {unique_preds}")
    print(f"  Unique actuals: {unique_actuals}")
    print(f"  Prediction variety: {unique_preds / len(preds) * 100:.1f}%")

    if unique_preds <= 5:
        pred_counts = {}
        for p in preds:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        print(f"  Most common predictions:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: -x[1])[:3]:
            print(f"    {pred}: {count} times ({count/len(preds)*100:.1f}%)")
    print()
