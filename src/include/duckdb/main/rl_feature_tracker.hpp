//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_feature_tracker.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include <atomic>

namespace duckdb {

class ClientContext;

//! Lightweight tracker for RL cardinality estimation features
//! Tracks actual vs estimated cardinalities independent of query profiling
struct RLOperatorStats {
	string operator_name;
	idx_t estimated_cardinality = 0;
	std::atomic<idx_t> actual_cardinality{0};

	void AddActualRows(idx_t count) {
		actual_cardinality.fetch_add(count);
	}
};

//! The RLFeatureTracker is a lightweight system for tracking cardinality estimation features
//! for reinforcement learning, independent of the query profiler
class RLFeatureTracker {
public:
	explicit RLFeatureTracker(ClientContext &context);
	~RLFeatureTracker() = default;

	//! Check if RL feature tracking is enabled
	bool IsEnabled() const {
		return enabled;
	}

	//! Start tracking an operator
	void StartOperator(optional_ptr<const PhysicalOperator> phys_op);

	//! End tracking an operator and record actual cardinality
	void EndOperator(optional_ptr<const PhysicalOperator> phys_op, idx_t actual_rows);

	//! Finalize and log all aggregated actual vs estimated cardinalities
	void Finalize();

	//! Reset all tracked data
	void Reset();

private:
	ClientContext &context;
	bool enabled;

	//! Map of physical operators to their aggregated statistics
	reference_map_t<const PhysicalOperator, RLOperatorStats> operator_stats;

	//! Mutex for thread-safe access
	std::mutex lock;

	//! Unique ID for this tracker instance (to validate thread-local cache)
	uint64_t tracker_id;
	//! Generation counter (incremented on Reset)
	std::atomic<uint64_t> generation;
};

} // namespace duckdb
