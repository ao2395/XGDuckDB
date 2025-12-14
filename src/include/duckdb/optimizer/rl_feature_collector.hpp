//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rl_feature_collector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Collected features for table scans
struct TableScanFeatures {
	string table_name;
	idx_t base_cardinality = 0;
	unordered_map<string, idx_t> column_distinct_counts;
	idx_t num_table_filters = 0;
	idx_t final_cardinality = 0;
	double filter_selectivity = 1.0;
	bool used_default_selectivity = false;
	idx_t cardinality_after_default_selectivity = 0;

	// Filter inspection details
	vector<string> filter_types;  // CONSTANT_COMPARISON, CONJUNCTION_AND, etc.
	vector<string> comparison_types;  // GREATERTHAN, EQUAL, etc.
	vector<idx_t> filter_column_ids;
	vector<bool> filter_has_selectivity;
};

//! Collected features for joins
struct JoinFeatures {
	string join_relation_set;  // e.g., "[0, 1]"
	idx_t num_relations = 0;
	string join_type;
	idx_t left_relation_card = 0;
	idx_t right_relation_card = 0;
	double left_denominator = 1.0;
	double right_denominator = 1.0;
	string comparison_type;  // "EQUAL", etc.
	bool tdom_from_hll = false;
	idx_t tdom_value = 0;
	double extra_ratio = 1.0;
	double numerator = 0;
	double denominator = 1.0;
	double estimated_cardinality = 0;
};

//! Collected features for filters
struct FilterFeatures {
	vector<string> comparison_types;
	vector<string> constant_values;
	vector<string> column_types;
};

//! Global feature collector that statistics propagation writes to
class RLFeatureCollector {
public:
	static RLFeatureCollector &Get();

	void AddTableScanFeatures(const LogicalOperator *op, const TableScanFeatures &features);
	void AddJoinFeatures(const LogicalOperator *op, const JoinFeatures &features);
	void AddJoinFeaturesByRelationSet(const string &relation_set, const JoinFeatures &features);
	void AddFilterFeatures(const LogicalOperator *op, const FilterFeatures &features);

	optional_ptr<TableScanFeatures> GetTableScanFeatures(const LogicalOperator *op);
	optional_ptr<JoinFeatures> GetJoinFeatures(const LogicalOperator *op);
	optional_ptr<JoinFeatures> GetJoinFeaturesByRelationSet(const string &relation_set);
	optional_ptr<JoinFeatures> GetJoinFeaturesByEstimate(idx_t estimated_cardinality);
	optional_ptr<FilterFeatures> GetFilterFeatures(const LogicalOperator *op);

	void Clear();
	
	using PredictorCallback = std::function<double(const JoinFeatures&)>;
	void RegisterPredictor(PredictorCallback callback);
	double PredictCardinality(const JoinFeatures &features);

	// Clear the prediction cache (called after each query to prevent memory growth)
	void ClearPredictionCache();

private:
	RLFeatureCollector() = default;

	unordered_map<const LogicalOperator*, TableScanFeatures> table_scan_features;
	unordered_map<const LogicalOperator*, JoinFeatures> join_features;
	unordered_map<string, JoinFeatures> join_features_by_relation_set;
	unordered_map<idx_t, JoinFeatures> join_features_by_estimate;
	unordered_map<const LogicalOperator*, FilterFeatures> filter_features;
	std::mutex lock;

	PredictorCallback predictor;

	// Prediction cache for join cardinality estimates (cleared after each query)
	unordered_map<string, double> prediction_cache;
};

} // namespace duckdb
