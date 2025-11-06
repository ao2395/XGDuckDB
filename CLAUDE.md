# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a modified version of DuckDB (an analytical database system) enhanced with **XGBoost Gradient Boosted Trees** for improved cardinality estimation. The system learns from query execution feedback using synchronous incremental tree building to improve query optimization over time.

## Build Commands

The project uses Make + CMake for building:

```bash
# Build release version (default)
make release

# Build debug version with sanitizers
make debug

# Clean build artifacts
make clean

# Run all tests
make unittest         # Debug build tests
make allunit          # All unit tests (uses release build)

# Build with benchmarks enabled
BUILD_BENCHMARK=1 BUILD_TPCH=1 make

# Run benchmarks
./build/release/benchmark/benchmark_runner
```

Build output is in `build/release/` or `build/debug/` directories.

### Common Build Options

```bash
# Build with specific extensions
BUILD_TPCH=1 BUILD_JSON=1 BUILD_PARQUET=1 make

# Build with all extensions
BUILD_ALL_IT_EXT=1 make

# Force assertions in release build (useful for debugging)
FORCE_ASSERT=1 make release
```

## Testing

```bash
# Run all unit tests
make unittest

# Run specific test group
./build/debug/test/unittest "[arrow]"

# Run tests one by one (for CI)
python3 scripts/run_tests_one_by_one.py build/debug/test/unittest --time_execution
```

## Architecture: XGBoost Cardinality Estimation Integration

### High-Level Data Flow

```
SQL Query Input
  ↓
Parser (src/parser/)
  ↓
Logical Planner (src/planner/)
  ├─→ RLFeatureCollector (collects statistics)
  └─→ RLModelInterface (extracts features) ← RLBoostingModel (XGBoost)
  ↓
Optimizer (src/optimizer/)
  ├─→ CardinalityEstimator (join_order/cardinality_estimator)
  └─→ XGBoost predictions used for join ordering
  ↓
Physical Planner (converts to physical operators)
  ├─→ Attach RLOperatorState to each operator
  └─→ Attach XGBoost predictions to physical operators
  ↓
Execution Engine (src/execution/)
  ├─→ Execute operators
  ├─→ Collect actual cardinalities (RLFeatureTracker)
  └─→ Accumulate rows atomically (RLOperatorState::AddRows)
  ↓
Query Profiler
  ↓
SYNCHRONOUS INCREMENTAL TRAINING (after EACH query)
  ├─→ Get recent 200 samples from RLTrainingBuffer (sliding window)
  ├─→ XGBoost adds 2 new trees to ensemble via UpdateIncremental()
  └─→ Model immediately available for next query
  ↓
Results Return to Client
```

### Core Components (src/main/)

**RLModelInterface** ([rl_model_interface.hpp](src/include/duckdb/main/rl_model_interface.hpp), [.cpp](src/main/rl_model_interface.cpp))
- Extracts features from LogicalOperator during planning
- Converts to fixed 64-dimensional vector:
  - Operator type (10 one-hot)
  - Table scan features (24): base cardinality, distinct counts, filter types
  - Join features (21): TDOM, left/right cardinalities, denominators
  - Aggregate features (4): group-by columns, aggregate functions
  - Filter features (2): comparison types, selectivity
  - Context features (3): estimated cardinality, relation set size
- Attaches RLOperatorState to physical operators for post-execution tracking

**RLBoostingModel** ([rl_boosting_model.hpp](src/include/duckdb/main/rl_boosting_model.hpp), [.cpp](src/main/rl_boosting_model.cpp))
- Singleton XGBoost gradient boosting model
- Thread-safe inference (concurrent reads) and training (exclusive write)
- Hyperparameters:
  - max_depth = 5 (moderate tree complexity)
  - learning_rate = 0.3 (higher for online learning)
  - trees_per_update = 2 (adds 2 trees after each query)
  - subsample = 0.8 (row sampling for regularization)
  - colsample_bytree = 0.8 (feature sampling)
  - min_child_weight = 5 (regularization)
- Predicts log(cardinality) for numerical stability
- Incremental tree building: maintains ensemble, adds trees synchronously

**RLFeatureCollector** ([rl_feature_collector.hpp](src/include/duckdb/optimizer/rl_feature_collector.hpp), [.cpp](src/optimizer/rl_feature_collector.cpp))
- Global singleton for statistics collection during optimization
- Stores features indexed by LogicalOperator pointer, relation set, and estimated cardinality
- Enables feature matching between logical planning and physical execution

**RLTrainingBuffer** ([rl_training_buffer.hpp](src/include/duckdb/main/rl_training_buffer.hpp), [.cpp](src/main/rl_training_buffer.cpp))
- Circular deque for experience replay (max 10,000 samples)
- Thread-safe storage of: feature vector, actual cardinality, predicted cardinality, Q-error
- **New method:** `GetRecentSamples(200)` - returns last 200 samples for sliding window training

**RLFeatureTracker** ([rl_feature_tracker.hpp](src/include/duckdb/main/rl_feature_tracker.hpp), [.cpp](src/main/rl_feature_tracker.cpp))
- Lightweight cardinality collector during execution
- Tracks actual vs estimated cardinalities per operator
- Uses atomic operations for thread-safe row counting

### Key Integration Points

1. **Logical Planning** ([src/planner/](src/planner/))
   - RLModelInterface extracts features from logical operators
   - RLFeatureCollector stores statistics for later matching

2. **Query Optimization** ([src/optimizer/](src/optimizer/))
   - Cardinality estimates influence join order selection
   - See [join_order/cardinality_estimator.cpp](src/optimizer/join_order/cardinality_estimator.cpp)

3. **Physical Execution** ([src/execution/](src/execution/))
   - RLOperatorState attached to each PhysicalOperator
   - Stores: feature_vector, rl_predicted_cardinality, duckdb_estimated_cardinality, actual_cardinality
   - Actual cardinalities collected via atomic `AddRows()` calls

4. **Synchronous Incremental Training** (after EACH query)
   - After query completes, `RLModelInterface::CollectActualCardinalities()` walks the operator tree
   - For each operator: adds sample to buffer
   - Gets recent 200 samples (sliding window)
   - Calls `RLBoostingModel::UpdateIncremental()` to add 2 new trees
   - Model immediately available for next query (no lag)

### Training Strategy

**Synchronous Incremental Tree Building:**
- After each query execution, model is updated immediately
- Uses sliding window of last 200 samples for training
- XGBoost adds 2 new trees per update (configurable via TREES_PER_UPDATE)
- No background thread needed - training happens synchronously
- Advantages:
  - Model always up-to-date with latest query patterns
  - No temporal mismatch between predictions and training data
  - Simpler architecture without async complexity

### Model Metrics

The model optimizes **Q-error**: `max(actual/predicted, predicted/actual)`
- Q-error = 1.0 means perfect prediction
- Q-error > 1.0 means under/over estimation
- XGBoost learns to minimize Q-error through boosting iterations

### XGBoost Integration

**Dependencies:**
- XGBoost library added as git submodule in `third_party/xgboost/`
- Linked via CMake in `src/CMakeLists.txt`
- Uses XGBoost C API for prediction and training

**Key XGBoost API Calls:**
- `XGBoosterCreate()` - Initialize booster
- `XGBoosterSetParam()` - Configure hyperparameters
- `XGBoosterPredict()` - Inference on feature vectors
- `XGBoosterUpdateOneIter()` - Add trees to ensemble
- `XGDMatrixCreateFromMat()` - Create training data matrix

## Important File Locations

```
src/main/                              # XGBoost integration (~400 lines)
├── rl_boosting_model.cpp             # XGBoost model implementation
└── rl_model_interface.cpp            # Feature extraction & training orchestration
src/optimizer/join_order/              # Join order optimization (uses cardinality estimates)
src/execution/                         # Physical operator execution
src/planner/                           # Logical query planning
third_party/xgboost/                   # XGBoost library (submodule)
build/release/                         # Release build output
build/debug/                           # Debug build output
test/                                  # Test suite
benchmark/                             # Benchmark queries (TPC-H, TPC-DS, etc.)
```

## Debugging Tips

- XGBoost predictions can be printed by examining RLOperatorState after execution
- Use `FORCE_ASSERT=1 make release` for assertions in optimized builds
- Query profiler output includes cardinality estimates vs actual values
- Training buffer size and tree count can be inspected via `RLBoostingModel::GetNumTrees()`
- Check `[RL BOOSTING]` log messages for model updates and predictions
