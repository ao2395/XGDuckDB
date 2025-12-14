# DuckDB Cardinality Estimation Features for RL Model

This document lists all features being logged for the reinforcement learning-based cardinality estimator.

All features are logged with the prefix `[RL FEATURE]` for easy parsing.

---

## 1. Table Scan Features

**Location**: `src/optimizer/join_order/relation_statistics_helper.cpp` - `ExtractGetStats()`

### Basic Table Statistics
- **Table Name**: Name of the table being scanned
  - Format: `[RL FEATURE] Table Name: <table_name>`

- **Base Table Cardinality**: Total number of rows in the table before filters
  - Format: `[RL FEATURE] Base Table Cardinality: <count>`

### Column Statistics (Per Column)
- **Column Name**: Name of the column
- **Distinct Count (HLL)**: Distinct value count from HyperLogLog sketch
  - Format: `[RL FEATURE] Column: <name> | Distinct Count (HLL): <count>`
  - If HLL unavailable: `[RL FEATURE] Column: <name> | Distinct Count (fallback to cardinality): <count>`

### Filter Statistics
- **Number of Table Filters**: Count of filters applied to the table
  - Format: `[RL FEATURE] Number of table filters: <count>`

- **Filter Type**: Type of filter (CONSTANT_COMPARISON, CONJUNCTION_AND, etc.)
  - Format: `[RL FEATURE] Filter Type: <type>`

- **Comparison Type**: Operator used in filter (=, <, >, <=, >=, etc.)
  - Format: `[RL FEATURE] Comparison Type: <operator>`

- **Column Distinct Count**: Distinct values in filtered column
  - Format: `[RL FEATURE] Column Distinct Count: <count>`

- **Cardinality After Filter**: Estimated rows after applying filter
  - Format: `[RL FEATURE] Filter on column <id> | Cardinality after filter: <count>`

- **Default Selectivity**: When used (0.2 = 20%)
  - Format: `[RL FEATURE] Using DEFAULT_SELECTIVITY: 0.200000`
  - Format: `[RL FEATURE] Cardinality after default selectivity: <count>`

- **Final Cardinality**: Rows after all filters
  - Format: `[RL FEATURE] Final Cardinality (after filters): <count>`

- **Filter Selectivity Ratio**: Ratio of output/input cardinality
  - Format: `[RL FEATURE] Filter Selectivity Ratio: <ratio>`
  - Value range: [0.0, 1.0]

---

## 2. Filter Statistics Update Features

**Location**: `src/optimizer/statistics/operator/propagate_filter.cpp` - `UpdateFilterStatistics()`

### Predicate Information
- **Comparison Type**: Type of comparison (=, <, >, <=, >=)
  - Format: `[RL FEATURE] Comparison Type: <type>`

- **Constant Value**: Value being compared against
  - Format: `[RL FEATURE] Constant Value: <value>`

- **Column Type**: Data type of the column (INTEGER, VARCHAR, DECIMAL, DATE, etc.)
  - Format: `[RL FEATURE] Column Type: <type>`

- **NULL Handling**: Whether column is set as NOT NULL
  - Format: `[RL FEATURE] Setting column as NOT NULL`

### Numeric Column Bounds
- **Current Min**: Minimum value before filter
  - Format: `[RL FEATURE] Current Min: <value>`

- **Current Max**: Maximum value before filter
  - Format: `[RL FEATURE] Current Max: <value>`

- **Updated Min**: New minimum after filter (for >= and > operations)
  - Format: `[RL FEATURE] Updated Min to: <value>`

- **Updated Max**: New maximum after filter (for <= and < operations)
  - Format: `[RL FEATURE] Updated Max to: <value>`

- **Equal Filter**: Both min and max set to constant (for = operations)
  - Format: `[RL FEATURE] Updated both Min and Max to: <value>`

---

## 3. Join Cardinality Estimation Features

**Location**: `src/optimizer/join_order/cardinality_estimator.cpp`

### Join Set Information
- **Join Relation Set**: String representation of relations being joined
  - Format: `[RL FEATURE] Join Relation Set: <set_string>`

- **Number of Relations**: Count of relations in the join
  - Format: `[RL FEATURE] Number of relations in join: <count>`

### Join Type and Structure
- **Join Type**: Type of join (INNER, LEFT, RIGHT, SEMI, ANTI, etc.)
  - Format: `[RL FEATURE] Join Type: <type>`

- **Left Relation Cardinality**: Number of relations on left side
  - Format: `[RL FEATURE] Left Relation Cardinality: <count>`

- **Right Relation Cardinality**: Number of relations on right side
  - Format: `[RL FEATURE] Right Relation Cardinality: <count>`

- **Left Denominator**: Denominator value for left subgraph
  - Format: `[RL FEATURE] Left Denominator: <value>`

- **Right Denominator**: Denominator value for right subgraph
  - Format: `[RL FEATURE] Right Denominator: <value>`

### Comparison and TDOM Features
- **Comparison Type**: Join predicate operator (EQUAL, LESSTHAN, GREATERTHAN, etc.)
  - Format: `[RL FEATURE] Comparison Type: <type>`

- **TDOM from HLL**: Whether TDOM (Total Domain) is from HyperLogLog
  - Format: `[RL FEATURE] TDOM from HLL: true/false`

- **TDOM Value**: Total domain (distinct count) value
  - Format: `[RL FEATURE] TDOM value: <count>`

### Join-Specific Ratios
- **Equality Join - Extra Ratio**: For equality joins, TDOM value used directly
  - Format: `[RL FEATURE] Equality Join - Extra Ratio: <value>`

- **Inequality Join - Extra Ratio**: For inequality joins, TDOM^(2/3)
  - Format: `[RL FEATURE] Inequality Join - Extra Ratio (tdom^2/3): <value>`

- **Semi/Anti Join Selectivity**: Default selectivity factor (5.0)
  - Format: `[RL FEATURE] Semi/Anti Join Selectivity: 5.000000`

### Cardinality Calculation
- **Numerator**: Product of all relation cardinalities
  - Format: `[RL FEATURE] Numerator (product of cardinalities): <value>`

- **Denominator**: TDOM-based denominator for join estimate
  - Format: `[RL FEATURE] Denominator (TDOM-based): <value>`

- **Estimated Cardinality**: Final estimated join output
  - Format: `[RL FEATURE] Estimated Cardinality: <value>`

---

## 4. Aggregate Features

**Location**: `src/optimizer/statistics/operator/propagate_aggregate.cpp` - `PropagateStatistics()`

### Aggregate Structure
- **Number of GROUP BY Columns**: Count of grouping columns
  - Format: `[RL FEATURE] Number of GROUP BY columns: <count>`

- **Number of Aggregate Functions**: Count of aggregate expressions (SUM, COUNT, etc.)
  - Format: `[RL FEATURE] Number of aggregate functions: <count>`

- **Number of Grouping Sets**: Count of grouping sets (for GROUPING SETS, CUBE, ROLLUP)
  - Format: `[RL FEATURE] Number of grouping sets: <count>`

### Cardinality Estimates
- **Input Cardinality**: Estimated rows coming into the aggregate
  - Format: `[RL FEATURE] Input Cardinality: <count>`

- **Estimated Output Cardinality**: Maximum possible output (worst case: all unique groups)
  - Format: `[RL FEATURE] Estimated Output Cardinality (max = input): <count>`

---

## 5. Actual Execution Features (Runtime)

**Location**: `src/parallel/pipeline_executor.cpp` - `EndOperator()`

### Actual Cardinality Tracking
- **Operator Name**: Name of the physical operator (HASH_JOIN, SEQ_SCAN, etc.)
- **Actual Output**: Real number of rows produced during execution
- **Estimated**: Estimated cardinality from optimizer
  - Format: `[RL FEATURE] *** ACTUAL CARDINALITY *** Operator: <name> | Actual Output: <count> | Estimated: <count>`

### Quality Metrics
- **Q-Error**: Ratio of estimated to actual (always >= 1.0)
  - Format: `[RL FEATURE] *** Q-ERROR *** <value>`
  - Calculation: `max(actual/estimated, estimated/actual)`
  - Lower is better (1.0 = perfect estimate)

---

## 6. Filter Inspection Features

**Location**: `src/optimizer/join_order/relation_statistics_helper.cpp` - `InspectTableFilter()`

### Detailed Filter Analysis
- **Column Index**: Column being filtered
  - Format: `[RL FEATURE] --- Filter Inspection on column <id> ---`

- **Filter Type**: CONSTANT_COMPARISON, CONJUNCTION_AND, etc.
  - Format: `[RL FEATURE] Filter Type: <type>`

- **Number of AND Child Filters**: For conjunction filters
  - Format: `[RL FEATURE] Number of AND child filters: <count>`

- **Equality Filter Selectivity**: Formula used for equality filters
  - Format: `[RL FEATURE] Equality Filter Selectivity: cardinality/distinct_count`
  - Format: `[RL FEATURE] Result: <card> / <distinct> = <result>`

---

## Feature Categories Summary

### **Categorical Features** (need encoding)
1. Table names
2. Column names
3. Operator types (HASH_JOIN, SEQ_SCAN, etc.)
4. Join types (INNER, LEFT, SEMI, ANTI, etc.)
5. Comparison types (EQUAL, LESSTHAN, GREATERTHAN, etc.)
6. Filter types (CONSTANT_COMPARISON, CONJUNCTION_AND, etc.)
7. Data types (INTEGER, VARCHAR, DECIMAL, DATE, etc.)

### **Numerical Features**
1. **Cardinalities**: Base, filtered, estimated, actual
2. **Distinct counts**: From HLL or fallback
3. **Selectivity ratios**: Filter output/input
4. **TDOM values**: Distinct counts for joins
5. **Denominators**: Left, right, combined
6. **Numerators**: Product of cardinalities
7. **Q-Errors**: Estimation quality metric
8. **Counts**: Number of filters, columns, grouping sets
9. **Min/Max bounds**: Numeric column ranges

### **Boolean Features**
1. TDOM from HLL (true/false)
2. NULL handling (can have null, cannot have null)
3. Filter types (equality vs inequality)

---

## Feature Engineering Recommendations

### Derived Features to Consider
1. **Uniqueness Ratio**: `distinct_count / total_rows` per column
2. **Join Selectivity**: `output_cardinality / (left_card * right_card)`
3. **Range Coverage**: For numeric filters: `(filter_max - filter_min) / (col_max - col_min)`
4. **TDOM Ratio**: `min(left_distinct, right_distinct) / max(left_distinct, right_distinct)`
5. **Aggregate Selectivity**: `output_groups / input_rows`
6. **Cumulative Selectivity**: Product of all selectivities in a pipeline

### Normalization Strategies
1. **Log-scale**: For cardinalities (can range from 1 to billions)
2. **Min-max scaling**: For selectivity ratios (already in [0,1])
3. **Standard scaling**: For Q-errors
4. **One-hot encoding**: For categorical features (join types, operators)
5. **Embedding layers**: For high-cardinality categoricals (table names, column names)

---

## Integration Points

### Where Features Are Captured

1. **Optimization Phase** (Planning):
   - Table scans: Statistics from catalog
   - Filters: Selectivity estimates
   - Joins: TDOM-based estimates
   - Aggregates: Group count estimates

2. **Execution Phase** (Runtime):
   - Actual cardinalities from DataChunks
   - Q-Errors computed on the fly
   - Per-operator tracking

### How to Parse Features

All features use the format:
```
[RL FEATURE] <description>: <value>
```

Recommended parsing strategy:
1. Capture all lines starting with `[RL FEATURE]`
2. Extract key-value pairs by splitting on `:`
3. Group features by their section markers:
   - `===== TABLE SCAN STATS =====`
   - `===== FILTER STATISTICS UPDATE =====`
   - `===== CARDINALITY ESTIMATION START =====`
   - `===== AGGREGATE STATISTICS =====`
   - `*** ACTUAL CARDINALITY ***`

---

## Example Feature Vector for a Join

For a query like: `SELECT * FROM customer c JOIN orders o ON c.id = o.customer_id WHERE c.age > 25`

**Features captured:**
- customer table: 150,000 rows
- customer.id distinct: 146,042 (HLL)
- customer.age distinct: 140,574 (HLL)
- Filter: age > 25, selectivity: 0.2 (default)
- customer filtered: 30,000 rows
- orders table: 1,500,000 rows
- orders.customer_id distinct: 107,255 (HLL)
- Join type: INNER
- Comparison: EQUAL
- TDOM: 146,042 (from HLL)
- Numerator: 45,000,000,000
- Denominator: 146,042
- Estimated join output: 308,130
- Actual join output: [captured at runtime]
- Q-Error: [computed at runtime]

---

## Files Modified

1. `src/optimizer/join_order/cardinality_estimator.cpp`
2. `src/optimizer/join_order/relation_statistics_helper.cpp`
3. `src/optimizer/statistics/operator/propagate_filter.cpp`
4. `src/optimizer/statistics/operator/propagate_aggregate.cpp`
5. `src/parallel/pipeline_executor.cpp`

---

## Next Steps for RL Model

1. **Feature Extraction**: Parse logged output to create feature vectors
2. **Training Data Collection**: Run workload and collect (features, actual_cardinality) pairs
3. **State Representation**: Encode features as RL state
4. **Reward Design**: Use Q-Error or other metrics as reward signal
5. **Model Architecture**: Consider GNN for tree-structured query plans
6. **Online Learning**: Update model based on query execution feedback

---

**Generated**: 2025-10-25
**DuckDB Version**: Development build based on main branch
**Purpose**: Feature documentation for RL-based cardinality estimation
