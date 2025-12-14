//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_feature_tracker.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_feature_tracker.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/common/printer.hpp"
#include <mutex>
#include <vector>

namespace duckdb {

// Global counter for unique tracker IDs
static std::atomic<uint64_t> global_tracker_ids{1};

// Thread-local cache structure using a vector for fast linear scan
struct RLThreadCache {
	uint64_t tracker_id = 0;
	uint64_t generation = 0;
	// Cache entries: PhysicalOperator* -> RLOperatorStats*
	// Using vector because N is small (pipeline operators)
	std::vector<std::pair<const PhysicalOperator*, RLOperatorStats*>> entries;
};

static thread_local RLThreadCache local_cache;

RLFeatureTracker::RLFeatureTracker(ClientContext &context) : context(context), enabled(true) {
	// Assign unique ID
	tracker_id = global_tracker_ids++;
	generation = 1;
}

void RLFeatureTracker::StartOperator(optional_ptr<const PhysicalOperator> phys_op) {
	if (!enabled || !phys_op) {
		return;
	}

	// Validate cache
	if (local_cache.tracker_id != tracker_id || local_cache.generation != generation) {
		local_cache.tracker_id = tracker_id;
		local_cache.generation = generation;
		local_cache.entries.clear();
	}

	// Check thread-local cache first (Fast Path - Linear Scan)
	for (const auto &entry : local_cache.entries) {
		if (entry.first == phys_op.get()) {
			return;
		}
	}

	// Slow Path: Acquire lock
	std::lock_guard<std::mutex> guard(lock);
	
	auto &stats = operator_stats[*phys_op];
	if (stats.estimated_cardinality == 0) {
		stats.operator_name = phys_op->GetName();
		stats.estimated_cardinality = phys_op->estimated_cardinality;
	}

	// Cache the pointer
	if (local_cache.entries.size() < 64) { // Cap size to prevent degradation
		local_cache.entries.emplace_back(phys_op.get(), &stats);
	}
}

void RLFeatureTracker::EndOperator(optional_ptr<const PhysicalOperator> phys_op, idx_t actual_rows) {
	if (!enabled || !phys_op || actual_rows == 0) {
		return;
	}

	// Validate cache
	if (local_cache.tracker_id != tracker_id || local_cache.generation != generation) {
		local_cache.tracker_id = tracker_id;
		local_cache.generation = generation;
		local_cache.entries.clear();
	}

	// Check thread-local cache first (Fast Path - Linear Scan)
	for (const auto &entry : local_cache.entries) {
		if (entry.first == phys_op.get()) {
			entry.second->AddActualRows(actual_rows);
			return;
		}
	}

	// Slow Path: Acquire lock
	std::lock_guard<std::mutex> guard(lock);
	auto global_it = operator_stats.find(*phys_op);
	if (global_it != operator_stats.end()) {
		global_it->second.AddActualRows(actual_rows);
		// Cache it now
		if (local_cache.entries.size() < 64) {
			local_cache.entries.emplace_back(phys_op.get(), &global_it->second);
		}
	}
}

void RLFeatureTracker::Finalize() {
	if (!enabled) {
		return;
	}

	std::lock_guard<std::mutex> guard(lock);

	for (auto &entry : operator_stats) {
		auto &stats = entry.second;
		idx_t actual_count = stats.actual_cardinality.load();

		if (actual_count > 0) {
			// Printer::Print("\n[RL FEATURE] *** ACTUAL CARDINALITY *** Operator: " + stats.operator_name +
			//                " | Actual Output: " + std::to_string(actual_count) +
			//                " | Estimated: " + std::to_string(stats.estimated_cardinality));

			if (stats.estimated_cardinality > 0) {
				double error = static_cast<double>(actual_count) / static_cast<double>(stats.estimated_cardinality);
				if (error < 1.0) {
					error = 1.0 / error;
				}
				// Printer::Print("[RL FEATURE] *** Q-ERROR *** " + std::to_string(error));
			}
		}
	}
}

void RLFeatureTracker::Reset() {
	if (!enabled) {
		return;
	}

	std::lock_guard<std::mutex> guard(lock);
	// Increment generation to invalidate all thread-local caches
	generation++;
	operator_stats.clear();
}

} // namespace duckdb
