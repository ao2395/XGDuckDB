//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_model_interface.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class PhysicalOperator;
class ClientContext;
class QueryProfiler;
class RLTrainingBuffer;

//! Feature set for a single operator
struct OperatorFeatures {
	// Operator metadata
	string operator_type;
	string operator_name;
	idx_t estimated_cardinality;  // DuckDB's built-in estimate

	// Table scan features
	string table_name;
	idx_t base_table_cardinality = 0;
	unordered_map<string, idx_t> column_distinct_counts;
	idx_t num_table_filters = 0;
	idx_t final_cardinality = 0;
	double filter_selectivity = 1.0;
	bool used_default_selectivity = false;
	idx_t cardinality_after_default_selectivity = 0;

	// Filter features
	vector<string> filter_types;
	vector<string> comparison_types;
	vector<idx_t> filter_column_ids;
	vector<double> selectivity_ratios;
	idx_t child_cardinality = 0;  // For FILTER operators: cardinality of child operator
	// Filter constant/value summary (from RLFeatureCollector::FilterFeatures if available)
	idx_t filter_constant_count = 0;
	double filter_constant_numeric_log_mean = 0.0;
	double filter_constant_string_log_mean = 0.0;

	// Join features
	string join_type;
	// Join condition structure (helps distinguish single-predicate vs multi-predicate joins)
	// These are extracted from `LogicalComparisonJoin::conditions`.
	idx_t join_condition_count = 0;
	idx_t join_equality_condition_count = 0;
	// Join key identity summary (hashed type/signature of join predicates)
	double join_key_signature_hash = 0.0; // normalized to [0,1]
	double join_key_same_type_ratio = 0.0;
	double join_key_simple_ref_ratio = 0.0;
	idx_t left_cardinality = 0;
	idx_t right_cardinality = 0;
	idx_t tdom_value = 0;
	bool tdom_from_hll = false;
	string join_relation_set;
	idx_t num_relations = 0;
	idx_t left_relation_card = 0;
	idx_t right_relation_card = 0;
	double left_denominator = 1.0;
	double right_denominator = 1.0;
	string comparison_type_join;
	double extra_ratio = 1.0;
	double numerator = 0;
	double denominator = 1.0;

	// Aggregate features
	idx_t num_group_by_columns = 0;
	idx_t num_aggregate_functions = 0;
	idx_t num_grouping_sets = 0;

	// Convert to string for printing
	string ToString() const;
};

//! Interface for RL model cardinality estimation
class RLModelInterface {
public:
	explicit RLModelInterface(ClientContext &context);

	//! Extract features from a logical operator
	OperatorFeatures ExtractFeatures(LogicalOperator &op, ClientContext &context);

	//! Get a pure RL prediction (observe-only).
	//! Returns 0 if a prediction is not available.
	idx_t PredictCardinality(const OperatorFeatures &features);

	//! Get an RL prediction intended for planning/optimization (can be called from the optimizer).
	//! This uses a separate cache/cap from physical-plan prediction to avoid interference.
	//! Returns 0 if a prediction is not available.
	idx_t PredictPlanningCardinality(const OperatorFeatures &features);

	//! Reset per-thread prediction caches for the current connection.
	//! Call this at query boundaries to avoid cache growth across long sessions.
	static void ResetPredictionCachesForThread();

	//! Planning cardinality estimate to use for optimizer/execution decisions.
	//! If RL prediction is available, it is used; otherwise falls back to DuckDB's estimate.
	idx_t GetCardinalityEstimate(const OperatorFeatures &features);

	//! Create RL state and attach to physical operator
	//! This stores the feature vector and prediction for later training
	void AttachRLState(PhysicalOperator &physical_op, const OperatorFeatures &features, idx_t rl_prediction,
	                    idx_t duckdb_estimate);

	//! Convert features to numerical vector for ML model input
	//! Returns a fixed-size vector of doubles suitable for feeding to an ML model
	static vector<double> FeaturesToVector(const OperatorFeatures &features);

	//! Train the model with actual cardinality (to be implemented later)
	void TrainModel(const OperatorFeatures &features, idx_t actual_cardinality);

	//! Collect actual cardinalities from executed operators and add to training buffer
	//! This should be called after query execution completes
	void CollectActualCardinalities(PhysicalOperator &root_operator, class QueryProfiler &profiler,
	                                 RLTrainingBuffer &buffer);

private:
	ClientContext &context;
	bool enabled;

	//! Helper function to recursively collect cardinalities
	void CollectActualCardinalitiesRecursive(PhysicalOperator &op, class QueryProfiler &profiler,
	                                          RLTrainingBuffer &buffer);

	// Feature vector size:
	// - Operator type (10 one-hot)
	// - Table scan features (24)
	// - Join features (27) - expanded with 6 selectivity features
	// - Aggregate features (4)
	// - Filter features (2)
	// - Context+extra features (13)
	// Total: 80
	static constexpr idx_t FEATURE_VECTOR_SIZE = 80;
};

} // namespace duckdb
