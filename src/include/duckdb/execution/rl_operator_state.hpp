//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/rl_operator_state.hpp
//
// RL operator state for tracking predictions and collecting actual cardinalities
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include <atomic>

namespace duckdb {

//! Stores RL prediction info for a physical operator
//! This is attached to each physical operator during planning
//! and used during execution to collect training data
struct RLOperatorState {
	//! The feature vector used for prediction
	vector<double> feature_vector;

	//! The RL model's prediction
	idx_t rl_predicted_cardinality = 0;

	//! The DuckDB estimate (for comparison)
	idx_t duckdb_estimated_cardinality = 0;

	//! Whether this operator has an RL prediction
	bool has_rl_prediction = false;

	//! Actual cardinality collected during execution (atomic for thread-safe updates)
	std::atomic<idx_t> actual_cardinality;

	//! Whether actual cardinality has been collected
	bool has_actual_cardinality = false;

	RLOperatorState() : actual_cardinality(0) {
	}

	RLOperatorState(vector<double> features, idx_t rl_pred, idx_t duckdb_est)
	    : feature_vector(std::move(features)), rl_predicted_cardinality(rl_pred),
	      duckdb_estimated_cardinality(duckdb_est), has_rl_prediction(true),
	      actual_cardinality(0), has_actual_cardinality(false) {
	}

	//! Add rows to the actual cardinality count (thread-safe)
	void AddRows(idx_t count) {
		actual_cardinality.fetch_add(count, std::memory_order_relaxed);
	}

	//! Get the final actual cardinality
	idx_t GetActualCardinality() const {
		return actual_cardinality.load(std::memory_order_relaxed);
	}
};

} // namespace duckdb
