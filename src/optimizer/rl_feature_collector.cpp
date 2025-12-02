//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rl_feature_collector.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/rl_feature_collector.hpp"
#include <mutex>

namespace duckdb {

RLFeatureCollector &RLFeatureCollector::Get() {
	static RLFeatureCollector instance;
	return instance;
}

void RLFeatureCollector::AddTableScanFeatures(const LogicalOperator *op, const TableScanFeatures &features) {
	std::lock_guard<std::mutex> guard(lock);
	table_scan_features[op] = features;
}

void RLFeatureCollector::AddJoinFeatures(const LogicalOperator *op, const JoinFeatures &features) {
	std::lock_guard<std::mutex> guard(lock);
	join_features[op] = features;
}

void RLFeatureCollector::AddJoinFeaturesByRelationSet(const string &relation_set, const JoinFeatures &features) {
	std::lock_guard<std::mutex> guard(lock);
	
	// Prevent unbounded growth: if map gets too large, clear it
	// This is a simple eviction strategy to prevent memory explosion
	if (join_features_by_relation_set.size() > 5000) {
		join_features_by_relation_set.clear();
		join_features_by_estimate.clear();
	}
	
	join_features_by_relation_set[relation_set] = features;
	// Also store by estimated cardinality for lookup
	if (features.estimated_cardinality > 0) {
		join_features_by_estimate[(idx_t)features.estimated_cardinality] = features;
	}
}

void RLFeatureCollector::AddFilterFeatures(const LogicalOperator *op, const FilterFeatures &features) {
	std::lock_guard<std::mutex> guard(lock);
	filter_features[op] = features;
}

optional_ptr<TableScanFeatures> RLFeatureCollector::GetTableScanFeatures(const LogicalOperator *op) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = table_scan_features.find(op);
	if (it != table_scan_features.end()) {
		return &it->second;
	}
	return nullptr;
}

optional_ptr<JoinFeatures> RLFeatureCollector::GetJoinFeatures(const LogicalOperator *op) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = join_features.find(op);
	if (it != join_features.end()) {
		return &it->second;
	}
	return nullptr;
}

optional_ptr<JoinFeatures> RLFeatureCollector::GetJoinFeaturesByRelationSet(const string &relation_set) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = join_features_by_relation_set.find(relation_set);
	if (it != join_features_by_relation_set.end()) {
		return &it->second;
	}
	return nullptr;
}

optional_ptr<JoinFeatures> RLFeatureCollector::GetJoinFeaturesByEstimate(idx_t estimated_cardinality) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = join_features_by_estimate.find(estimated_cardinality);
	if (it != join_features_by_estimate.end()) {
		return &it->second;
	}
	return nullptr;
}

optional_ptr<FilterFeatures> RLFeatureCollector::GetFilterFeatures(const LogicalOperator *op) {
	std::lock_guard<std::mutex> guard(lock);
	auto it = filter_features.find(op);
	if (it != filter_features.end()) {
		return &it->second;
	}
	return nullptr;
}

void RLFeatureCollector::Clear() {
	std::lock_guard<std::mutex> guard(lock);
	table_scan_features.clear();
	join_features.clear();
	join_features_by_relation_set.clear();
	join_features_by_estimate.clear();
	filter_features.clear();
	// Also clear prediction cache to prevent memory growth between queries
	prediction_cache.clear();
}

void RLFeatureCollector::ClearPredictionCache() {
	std::lock_guard<std::mutex> guard(lock);
	prediction_cache.clear();
}

void RLFeatureCollector::RegisterPredictor(PredictorCallback callback) {
	std::lock_guard<std::mutex> guard(lock);
	predictor = callback;
	// Printer::Print("[RL DEBUG] Predictor registered in RLFeatureCollector\n");
}

double RLFeatureCollector::PredictCardinality(const JoinFeatures &features) {
	PredictorCallback local_predictor;
	{
		std::lock_guard<std::mutex> guard(lock);
		local_predictor = predictor;
	}

	if (local_predictor) {
		// Printer::Print("[RL DEBUG] Calling predictor for " + features.join_relation_set + "\n");
		return local_predictor(features);
	}
	// Printer::Print("[RL DEBUG] No predictor registered!\n");
	return 0.0;
}

} // namespace duckdb
