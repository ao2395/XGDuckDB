//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_model_interface.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_model_interface.hpp"
#include "duckdb/main/rl_boosting_model.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/optimizer/rl_feature_collector.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/rl_training_buffer.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/main/profiling_node.hpp"
#include "duckdb/common/enums/metric_type.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/common/constants.hpp"

#include <unordered_map>
#include <mutex>
#include <cstring>
#include <atomic>

namespace duckdb {

static constexpr bool PHYSICAL_RL_ENABLED = true;

RLModelInterface::RLModelInterface(ClientContext &context) : context(context), enabled(true) {
	// Register the predictor callback with RLFeatureCollector
	// This allows the optimizer to request cardinality predictions for join sets
	// NOTE: Don't capture 'this' - RLModelInterface is per-context but RLFeatureCollector is singleton
	RLFeatureCollector::Get().RegisterPredictor([](const JoinFeatures &features) -> double {
		// ALWAYS use predictions - start after just 2 trees
		if (RLBoostingModel::Get().GetNumTrees() < 2) {
			return 0.0;
		}

		// CACHING: Use THREAD-LOCAL cache to avoid lock contention on HPC
		// Each thread has its own cache - NO MUTEX NEEDED for cache lookups!
		thread_local std::unordered_map<std::string, double> prediction_cache;
		
		std::string cache_key = features.join_relation_set;
		auto it = prediction_cache.find(cache_key);
		if (it != prediction_cache.end()) {
			return it->second;  // Cache hit - NO LOCK!
		}

		// Convert JoinFeatures to OperatorFeatures for the model
		OperatorFeatures op_features;
		op_features.operator_type = "LOGICAL_COMPARISON_JOIN"; // Assume join for optimizer predictions
		op_features.join_type = features.join_type;
		op_features.join_relation_set = features.join_relation_set;
		op_features.num_relations = features.num_relations;
		op_features.left_relation_card = features.left_relation_card;
		op_features.right_relation_card = features.right_relation_card;
		op_features.left_denominator = features.left_denominator;
		op_features.right_denominator = features.right_denominator;
		op_features.comparison_type_join = features.comparison_type;
		op_features.tdom_value = features.tdom_value;
		op_features.tdom_from_hll = features.tdom_from_hll;
		op_features.extra_ratio = features.extra_ratio;
		op_features.numerator = features.numerator;
		op_features.denominator = features.denominator;
		op_features.estimated_cardinality = features.estimated_cardinality; // DuckDB's estimate as context
		
		// FIX: left_relation_card and right_relation_card can be UINT64_MAX (invalid) for complex joins
		// Use numerator/denominator to derive reasonable estimates instead
		if (features.left_relation_card == std::numeric_limits<idx_t>::max() || 
		    features.left_relation_card == 0 ||
		    features.right_relation_card == std::numeric_limits<idx_t>::max() ||
		    features.right_relation_card == 0) {
			// Derive from numerator (which is product of all input cardinalities)
			// For a join, numerator ≈ left_card * right_card
			// Use sqrt as a rough estimate to split it
			if (features.numerator > 0) {
				double sqrt_num = std::sqrt(features.numerator);
				op_features.left_cardinality = static_cast<idx_t>(sqrt_num);
				op_features.right_cardinality = static_cast<idx_t>(sqrt_num);
			} else {
				op_features.left_cardinality = 1;
				op_features.right_cardinality = 1;
			}
		} else {
			// Use the provided values
			op_features.left_cardinality = features.left_relation_card;
			op_features.right_cardinality = features.right_relation_card;
		}

		// Convert to feature vector
		auto feature_vec = RLModelInterface::FeaturesToVector(op_features);

		// DEBUG: Log feature values to diagnose why predictions are identical
		// DISABLED: Too expensive - called for every prediction
		// Printer::Print("[RL FEATURE DEBUG] " + cache_key + 
		//                ": left_card=" + std::to_string(op_features.left_cardinality) +
		//                ", right_card=" + std::to_string(op_features.right_cardinality) +
		//                ", tdom=" + std::to_string(op_features.tdom_value) +
		//                ", num=" + std::to_string(op_features.numerator) +
		//                ", denom=" + std::to_string(op_features.denominator) + "\n");

		// Predict
		double prediction = RLBoostingModel::Get().Predict(feature_vec);

		// Cache the result - thread_local so no lock needed!
		// Large cache for better hit rate
		if (prediction_cache.size() > 5000) {
			prediction_cache.clear();  // Prevent unbounded growth
		}
		prediction_cache[cache_key] = prediction;

		return prediction;
	});
}

string OperatorFeatures::ToString() const {
	string result = "\n[RL MODEL] ========== OPERATOR FEATURES ==========\n";
	result += "[RL MODEL] Operator Type: " + operator_type + "\n";
	result += "[RL MODEL] Operator Name: " + operator_name + "\n";
	result += "[RL MODEL] DuckDB Estimated Cardinality: " + std::to_string(estimated_cardinality) + "\n";

	// TABLE SCAN STATS (matching original format)
	if (base_table_cardinality > 0) {
		result += "[RL MODEL] ===== TABLE SCAN STATS =====\n";
		if (!table_name.empty()) {
			result += "[RL MODEL] Table Name: " + table_name + "\n";
		}
		result += "[RL MODEL] Base Table Cardinality: " + std::to_string(base_table_cardinality) + "\n";

		if (!column_distinct_counts.empty()) {
			for (auto &entry : column_distinct_counts) {
				result += "[RL MODEL] Column: " + entry.first + " | Distinct Count (HLL): " + std::to_string(entry.second) + "\n";
			}
		}

		if (num_table_filters > 0) {
			result += "[RL MODEL] Number of table filters: " + std::to_string(num_table_filters) + "\n";

			// Filter inspection details with child count tracking
			idx_t child_count = 0;
			for (idx_t i = 0; i < filter_types.size(); i++) {
				if (i < filter_column_ids.size() && child_count == 0) {
					result += "[RL MODEL] --- Filter Inspection on column " + std::to_string(filter_column_ids[i]) + " ---\n";
				}
				result += "[RL MODEL] Filter Type: " + filter_types[i] + "\n";

				// Track CONJUNCTION_AND to count children
				if (filter_types[i] == "CONJUNCTION_AND") {
					// Count upcoming CONSTANT_COMPARISON children
					idx_t num_children = 0;
					for (idx_t j = i + 1; j < filter_types.size() && filter_types[j] != "CONJUNCTION_AND"; j++) {
						if (filter_types[j] == "CONSTANT_COMPARISON") {
							num_children++;
						}
					}
					if (num_children > 0) {
						result += "[RL MODEL] Number of AND child filters: " + std::to_string(num_children) + "\n";
						child_count = num_children;
					}
				} else if (child_count > 0) {
					child_count--;
					result += "[RL MODEL] --- Filter Inspection on column " + std::to_string(filter_column_ids[0]) + " ---\n";
				}

				if (i < comparison_types.size() && !comparison_types[i].empty()) {
					result += "[RL MODEL] Comparison Type: " + comparison_types[i] + "\n";
					if (comparison_types[i] != "EQUAL") {
						result += "[RL MODEL] Non-equality comparison - no selectivity applied\n";
					}
				}
			}

			if (used_default_selectivity) {
				result += "[RL MODEL] Using DEFAULT_SELECTIVITY: 0.200000\n";
				result += "[RL MODEL] Cardinality after default selectivity: " + std::to_string(cardinality_after_default_selectivity) + "\n";
			}
		}

		if (final_cardinality > 0) {
			result += "[RL MODEL] Final Cardinality (after filters): " + std::to_string(final_cardinality) + "\n";
			result += "[RL MODEL] Filter Selectivity Ratio: " + std::to_string(filter_selectivity) + "\n";
		}
		result += "[RL MODEL] ===== END TABLE SCAN STATS =====\n";
	}

	// JOIN FEATURES (matching original format)
	if (!join_type.empty()) {
		result += "[RL MODEL] ===== CARDINALITY ESTIMATION START =====\n";
		if (!join_relation_set.empty()) {
			result += "[RL MODEL] Join Relation Set: " + join_relation_set + "\n";
			result += "[RL MODEL] Number of relations in join: " + std::to_string(num_relations) + "\n";
		}
		result += "[RL MODEL] Join Type: " + join_type + "\n";
		if (left_relation_card > 0 && right_relation_card > 0) {
			result += "[RL MODEL] Left Relation Cardinality: " + std::to_string(left_relation_card) + "\n";
			result += "[RL MODEL] Right Relation Cardinality: " + std::to_string(right_relation_card) + "\n";
			result += "[RL MODEL] Left Denominator: " + std::to_string(left_denominator) + "\n";
			result += "[RL MODEL] Right Denominator: " + std::to_string(right_denominator) + "\n";
		} else {
			result += "[RL MODEL] Left Cardinality: " + std::to_string(left_cardinality) + "\n";
			result += "[RL MODEL] Right Cardinality: " + std::to_string(right_cardinality) + "\n";
		}
		if (!comparison_type_join.empty()) {
			result += "[RL MODEL] Comparison Type: " + comparison_type_join + "\n";
		}
		if (tdom_from_hll) {
			result += "[RL MODEL] TDOM from HLL: true\n";
		}
		if (tdom_value > 0) {
			result += "[RL MODEL] TDOM value: " + std::to_string(tdom_value) + "\n";
			if (extra_ratio > 1.0) {
				result += "[RL MODEL] Equality Join - Extra Ratio: " + std::to_string(extra_ratio) + "\n";
			}
		}
		if (numerator > 0 && denominator > 0) {
			result += "[RL MODEL] Numerator (product of cardinalities): " + std::to_string(numerator) + "\n";
			result += "[RL MODEL] Denominator (TDOM-based): " + std::to_string(denominator) + "\n";
			double calc_estimate = numerator / denominator;
			result += "[RL MODEL] Estimated Cardinality: " + std::to_string(calc_estimate) + "\n";
		}
		result += "[RL MODEL] ===== CARDINALITY ESTIMATION END =====\n";
	}

	// AGGREGATE STATS (matching original format)
	if (num_group_by_columns > 0 || num_aggregate_functions > 0) {
		result += "[RL MODEL] ===== AGGREGATE STATISTICS =====\n";
		result += "[RL MODEL] Number of GROUP BY columns: " + std::to_string(num_group_by_columns) + "\n";
		result += "[RL MODEL] Number of aggregate functions: " + std::to_string(num_aggregate_functions) + "\n";
		result += "[RL MODEL] Number of grouping sets: " + std::to_string(num_grouping_sets) + "\n";
		result += "[RL MODEL] ===== END AGGREGATE STATISTICS =====\n";
	}

	// FILTER FEATURES (for standalone filters)
	if (!filter_types.empty() && base_table_cardinality == 0) {
		result += "[RL MODEL] Filter Types: ";
		for (idx_t i = 0; i < filter_types.size(); i++) {
			result += filter_types[i];
			if (i < filter_types.size() - 1) result += ", ";
		}
		result += "\n";

		if (!comparison_types.empty()) {
			result += "[RL MODEL] Comparison Types: ";
			for (idx_t i = 0; i < comparison_types.size(); i++) {
				result += comparison_types[i];
				if (i < comparison_types.size() - 1) result += ", ";
			}
			result += "\n";
		}
	}

	result += "[RL MODEL] ============================================\n";
	return result;
}

OperatorFeatures RLModelInterface::ExtractFeatures(LogicalOperator &op, ClientContext &context) {
	OperatorFeatures features;

	// Basic operator info
	features.operator_type = LogicalOperatorToString(op.type);
	features.operator_name = op.GetName();
	features.estimated_cardinality = op.estimated_cardinality;

	// Try to get features from the collector (populated during statistics propagation)
	auto &collector = RLFeatureCollector::Get();

	// Extract operator-specific features
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.cardinality) {
			auto card_stats = get.function.cardinality(context, get.bind_data.get());
			if (card_stats) {
				features.base_table_cardinality = card_stats->estimated_cardinality;
			}
		}

		// Get detailed table scan features from collector
		auto table_features = collector.GetTableScanFeatures(&op);
		if (table_features) {
			features.table_name = table_features->table_name;
			features.base_table_cardinality = table_features->base_cardinality;
			features.column_distinct_counts = table_features->column_distinct_counts;
			features.num_table_filters = table_features->num_table_filters;
			features.final_cardinality = table_features->final_cardinality;
			features.filter_selectivity = table_features->filter_selectivity;
			features.used_default_selectivity = table_features->used_default_selectivity;
			features.cardinality_after_default_selectivity = table_features->cardinality_after_default_selectivity;
			features.filter_types = table_features->filter_types;
			features.comparison_types = table_features->comparison_types;
			features.filter_column_ids = table_features->filter_column_ids;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		auto &filter = op.Cast<LogicalFilter>();
		// Extract filter expression types
		for (auto &expr : filter.expressions) {
			features.filter_types.push_back(ExpressionTypeToString(expr->type));
		}

		// Get child cardinality as context
		if (!filter.children.empty()) {
			features.child_cardinality = filter.children[0]->estimated_cardinality;
		}

		// Get detailed filter features from collector
		auto filter_features = collector.GetFilterFeatures(&op);
		if (filter_features) {
			features.comparison_types = filter_features->comparison_types;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		features.join_type = JoinTypeToString(join.join_type);
		if (op.children.size() >= 2) {
			features.left_cardinality = op.children[0]->estimated_cardinality;
			features.right_cardinality = op.children[1]->estimated_cardinality;
		}

		// Try to get detailed join features from collector (by operator or by estimated cardinality)
		auto join_features = collector.GetJoinFeatures(&op);
		if (!join_features && op.estimated_cardinality > 0) {
			// Try matching by estimated cardinality
			join_features = collector.GetJoinFeaturesByEstimate(op.estimated_cardinality);
		}
		if (join_features) {
			features.tdom_value = join_features->tdom_value;
			features.tdom_from_hll = join_features->tdom_from_hll;
			features.join_relation_set = join_features->join_relation_set;
			features.num_relations = join_features->num_relations;
			features.left_relation_card = join_features->left_relation_card;
			features.right_relation_card = join_features->right_relation_card;
			features.left_denominator = join_features->left_denominator;
			features.right_denominator = join_features->right_denominator;
			features.comparison_type_join = join_features->comparison_type;
			features.extra_ratio = join_features->extra_ratio;
			features.numerator = join_features->numerator;
			features.denominator = join_features->denominator;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		features.num_group_by_columns = aggr.groups.size();
		features.num_aggregate_functions = aggr.expressions.size();
		features.num_grouping_sets = aggr.grouping_sets.size();
		break;
	}
	default:
		// For other operators, just use basic info
		break;
	}

	return features;
}

vector<double> RLModelInterface::FeaturesToVector(const OperatorFeatures &features) {
	vector<double> feature_vec(FEATURE_VECTOR_SIZE, 0.0);
	idx_t idx = 0;

	// Helper lambda for safe log (avoid log(0))
	auto safe_log = [](idx_t val) -> double {
		return val > 0 ? std::log(static_cast<double>(val)) : 0.0;
	};

	// 1. OPERATOR TYPE (One-hot encoding) - 10 features
	// GET, JOIN, FILTER, AGGREGATE, PROJECTION, TOP_N, ORDER_BY, LIMIT, UNION, OTHER
	if (!features.table_name.empty()) {
		feature_vec[idx] = 1.0; // GET
	} else if (!features.join_type.empty()) {
		feature_vec[idx + 1] = 1.0; // JOIN
	} else if (!features.filter_types.empty() && features.table_name.empty()) {
		feature_vec[idx + 2] = 1.0; // FILTER
	} else if (features.num_group_by_columns > 0 || features.num_aggregate_functions > 0) {
		feature_vec[idx + 3] = 1.0; // AGGREGATE
	} else {
		feature_vec[idx + 9] = 1.0; // OTHER (PROJECTION, TOP_N, etc.)
	}
	idx += 10;

	// 2. TABLE SCAN FEATURES - 24 features
	if (!features.table_name.empty()) {
		// Table identifier - use hash normalized to [0,1] for better neural network training
		std::hash<std::string> hasher;
		double table_hash = static_cast<double>(hasher(features.table_name) % 10000) / 10000.0;
		feature_vec[idx++] = table_hash;

		feature_vec[idx++] = safe_log(features.base_table_cardinality);
		feature_vec[idx++] = static_cast<double>(features.num_table_filters);
		feature_vec[idx++] = features.filter_selectivity;
		feature_vec[idx++] = features.used_default_selectivity ? 1.0 : 0.0;
		feature_vec[idx++] = static_cast<double>(features.filter_types.size());

		// Number of columns in the table
		feature_vec[idx++] = static_cast<double>(features.column_distinct_counts.size());

		// Column distinct count statistics
		if (!features.column_distinct_counts.empty() && features.base_table_cardinality > 0) {
			double sum = 0.0, min_ratio = 1.0, max_ratio = 0.0;
			double sum_log = 0.0;
			idx_t min_distinct = features.base_table_cardinality, max_distinct = 0;
			idx_t num_high_card_cols = 0;  // Columns with >50% distinct values
			idx_t num_low_card_cols = 0;   // Columns with <5% distinct values

			for (const auto &entry : features.column_distinct_counts) {
				double ratio = static_cast<double>(entry.second) / static_cast<double>(features.base_table_cardinality);
				sum += ratio;
				sum_log += std::log(std::max(1.0, static_cast<double>(entry.second)));
				min_ratio = std::min(min_ratio, ratio);
				max_ratio = std::max(max_ratio, ratio);
				min_distinct = std::min(min_distinct, entry.second);
				max_distinct = std::max(max_distinct, entry.second);
				if (ratio > 0.5) num_high_card_cols++;
				if (ratio < 0.05) num_low_card_cols++;
			}
			feature_vec[idx++] = sum / features.column_distinct_counts.size(); // avg ratio
			feature_vec[idx++] = max_ratio;
			feature_vec[idx++] = min_ratio;
			feature_vec[idx++] = sum_log / features.column_distinct_counts.size(); // avg log(distinct_count)
			feature_vec[idx++] = static_cast<double>(num_high_card_cols);
			feature_vec[idx++] = static_cast<double>(num_low_card_cols);
			feature_vec[idx++] = safe_log(min_distinct); // log of minimum distinct count - KEY DISTINGUISHER!
			feature_vec[idx++] = safe_log(max_distinct); // log of maximum distinct count
		} else {
			idx += 8;
		}

		// Filter comparison types one-hot (EQUAL, LT, GT, LTE, GTE, NEQ) - 6 features
		bool has_equal = false, has_lt = false, has_gt = false, has_lte = false, has_gte = false, has_neq = false;
		for (const auto &comp_type : features.comparison_types) {
			if (comp_type == "EQUAL") has_equal = true;
			else if (comp_type == "LESSTHAN") has_lt = true;
			else if (comp_type == "GREATERTHAN") has_gt = true;
			else if (comp_type == "LESSTHANOREQUALTO") has_lte = true;
			else if (comp_type == "GREATERTHANOREQUALTO") has_gte = true;
			else if (comp_type == "NOTEQUAL") has_neq = true;
		}
		feature_vec[idx++] = has_equal ? 1.0 : 0.0;
		feature_vec[idx++] = has_lt ? 1.0 : 0.0;
		feature_vec[idx++] = has_gt ? 1.0 : 0.0;
		feature_vec[idx++] = has_lte ? 1.0 : 0.0;
		feature_vec[idx++] = has_gte ? 1.0 : 0.0;
		feature_vec[idx++] = has_neq ? 1.0 : 0.0;
	} else {
		idx += 24;
	}

	// 3. JOIN FEATURES - 27 features (expanded from 21)
	if (!features.join_type.empty()) {
		feature_vec[idx++] = safe_log(features.left_cardinality);
		feature_vec[idx++] = safe_log(features.right_cardinality);
		feature_vec[idx++] = safe_log(features.tdom_value);
		feature_vec[idx++] = features.tdom_from_hll ? 1.0 : 0.0;

		// Join type one-hot (INNER, LEFT, RIGHT, SEMI, ANTI)
		if (features.join_type == "INNER") feature_vec[idx] = 1.0;
		else if (features.join_type == "LEFT") feature_vec[idx + 1] = 1.0;
		else if (features.join_type == "RIGHT") feature_vec[idx + 2] = 1.0;
		else if (features.join_type == "SEMI") feature_vec[idx + 3] = 1.0;
		else if (features.join_type == "ANTI") feature_vec[idx + 4] = 1.0;
		idx += 5;

		// Comparison type one-hot (EQUAL, LT, GT, LTE, GTE, NEQ)
		if (features.comparison_type_join == "EQUAL") feature_vec[idx] = 1.0;
		else if (features.comparison_type_join == "LESSTHAN") feature_vec[idx + 1] = 1.0;
		else if (features.comparison_type_join == "GREATERTHAN") feature_vec[idx + 2] = 1.0;
		else if (features.comparison_type_join == "LESSTHANOREQUALTO") feature_vec[idx + 3] = 1.0;
		else if (features.comparison_type_join == "GREATERTHANOREQUALTO") feature_vec[idx + 4] = 1.0;
		else if (features.comparison_type_join == "NOTEQUAL") feature_vec[idx + 5] = 1.0;
		idx += 6;

		feature_vec[idx++] = safe_log(static_cast<idx_t>(features.extra_ratio));
		feature_vec[idx++] = std::log(std::max(1.0, features.numerator));
		feature_vec[idx++] = std::log(std::max(1.0, features.denominator));
		feature_vec[idx++] = static_cast<double>(features.num_relations);
		feature_vec[idx++] = std::log(std::max(1.0, features.left_denominator));
		feature_vec[idx++] = std::log(std::max(1.0, features.right_denominator));

		// NEW FEATURES FOR LOW-CARDINALITY JOIN DETECTION (6 additional features)
		// These help distinguish high-selectivity joins (low cardinality) from cross-product-like joins

		// 1. Selectivity factor: ratio of expected output to cross product
		// Low values (<< 1.0) indicate high selectivity → low cardinality result
		double cross_product = features.left_cardinality * features.right_cardinality;
		double selectivity_factor = features.denominator > 0 ? cross_product / features.denominator : 1.0;
		feature_vec[idx++] = std::log(std::max(1.0, selectivity_factor));

		// 2. TDOM ratio: how selective is the join key?
		// Small TDOM relative to input sizes → many rows filtered out
		double tdom_ratio = 0.0;
		if (features.left_cardinality > 0 && features.right_cardinality > 0 && features.tdom_value > 0) {
			double avg_input_card = (features.left_cardinality + features.right_cardinality) / 2.0;
			tdom_ratio = features.tdom_value / avg_input_card;
		}
		feature_vec[idx++] = tdom_ratio;  // Small values → high selectivity

		// 3. Denominator/numerator ratio: directly captures selectivity
		double selectivity_ratio = features.numerator > 0 ? features.denominator / features.numerator : 1.0;
		feature_vec[idx++] = std::log(std::max(1.0, selectivity_ratio));

		// 4. Input size imbalance: large difference in input sizes affects join behavior
		double size_imbalance = 1.0;
		if (features.left_cardinality > 0 && features.right_cardinality > 0) {
			double larger = std::max(features.left_cardinality, features.right_cardinality);
			double smaller = std::min(features.left_cardinality, features.right_cardinality);
			size_imbalance = larger / smaller;
		}
		feature_vec[idx++] = std::log(std::max(1.0, size_imbalance));

		// 5. Low-cardinality indicator: flag if TDOM is very small (<1000)
		feature_vec[idx++] = (features.tdom_value > 0 && features.tdom_value < 1000) ? 1.0 : 0.0;

		// 6. Expected output size magnitude (helps model learn scale)
		// This is what DuckDB estimates - provides a baseline
		double expected_output = features.numerator > 0 && features.denominator > 0
		                         ? features.numerator / features.denominator : 0.0;
		feature_vec[idx++] = std::log(std::max(1.0, expected_output));
	} else {
		idx += 27;  // Updated from 21 to 27
	}

	// 4. AGGREGATE FEATURES - 4 features
	if (features.num_group_by_columns > 0 || features.num_aggregate_functions > 0) {
		feature_vec[idx++] = safe_log(features.estimated_cardinality); // Input from child
		feature_vec[idx++] = static_cast<double>(features.num_group_by_columns);
		feature_vec[idx++] = static_cast<double>(features.num_aggregate_functions);
		feature_vec[idx++] = static_cast<double>(features.num_grouping_sets);
	} else {
		idx += 4;
	}

	// 5. FILTER FEATURES - 2 features
	if (!features.filter_types.empty() && features.table_name.empty()) {
		feature_vec[idx++] = safe_log(features.child_cardinality); // Input from child operator
		feature_vec[idx++] = static_cast<double>(features.filter_types.size());
	} else {
		idx += 2;
	}

	// 6. CONTEXT FEATURES - 1 feature
	feature_vec[idx++] = safe_log(features.estimated_cardinality); // DuckDB's estimate

	// Remaining features are padding (already initialized to 0.0)
	D_ASSERT(idx <= FEATURE_VECTOR_SIZE);

	return feature_vec;
}

idx_t RLModelInterface::GetCardinalityEstimate(const OperatorFeatures &features) {
	if (!enabled || !PHYSICAL_RL_ENABLED) {
		return 0; // Don't override
	}

	// DIAGNOSTIC + CACHE: track predictions and cache results per query
	static constexpr idx_t MAX_PHYSICAL_PREDICTIONS = 300;
	static thread_local std::unordered_map<std::string, idx_t> physical_prediction_cache;
	static thread_local idx_t cached_query_id = DConstants::INVALID_INDEX;
	static thread_local idx_t physical_prediction_count = 0;
	static thread_local bool physical_cap_logged = false;

	idx_t query_id = DConstants::INVALID_INDEX;
	try {
		query_id = context.transaction.GetActiveQuery();
	} catch (...) {
		// No active query yet
	}

	if (cached_query_id != query_id) {
		physical_prediction_cache.clear();
		physical_prediction_count = 0;
		physical_cap_logged = false;
		cached_query_id = query_id;
	}

	// Only allow RL overrides on join operators (high impact). Everything else uses DuckDB estimates.
	bool is_join = !features.join_type.empty();
	if (!is_join) {
		return features.estimated_cardinality;
	}

	// REMOVED: Print all features (too expensive - 33k+ calls with multi-line output)
	// Printer::Print(features.ToString());

	// Convert features to vector
	if (physical_prediction_count >= MAX_PHYSICAL_PREDICTIONS) {
		if (!physical_cap_logged) {
			// Printer::Print("[RL PHYSICAL PREDICTION] cap reached (" + std::to_string(MAX_PHYSICAL_PREDICTIONS) +
			//                "), falling back to DuckDB estimates.\n");
			physical_cap_logged = true;
		}
		return features.estimated_cardinality;
	}

	auto feature_vec = FeaturesToVector(features);

	// Build operator-specific cache key
	std::string cache_key;
	cache_key.reserve(128);
	cache_key.append(features.operator_type);
	cache_key.push_back('|');

	if (!features.table_name.empty()) {
		cache_key.append(features.table_name);
		cache_key.push_back('|');
		cache_key.append(std::to_string(features.filter_types.size()));
		cache_key.push_back('|');
		for (const auto &cmp : features.comparison_types) {
			cache_key.append(cmp);
			cache_key.push_back(',');
		}
	} else if (!features.join_type.empty()) {
		cache_key.append(features.join_type);
		cache_key.push_back('|');
		cache_key.append(features.join_relation_set);
		cache_key.push_back('|');
		cache_key.append(features.comparison_type_join);
	} else if (!features.filter_types.empty()) {
		cache_key.append(std::to_string(features.filter_types.size()));
		cache_key.push_back('|');
		for (const auto &cmp : features.comparison_types) {
			cache_key.append(cmp);
			cache_key.push_back(',');
		}
	} else if (features.num_group_by_columns > 0 || features.num_aggregate_functions > 0) {
		cache_key.append(std::to_string(features.num_group_by_columns));
		cache_key.push_back('|');
		cache_key.append(std::to_string(features.num_aggregate_functions));
		cache_key.push_back('|');
		cache_key.append(std::to_string(features.num_grouping_sets));
	}

	auto cached = physical_prediction_cache.find(cache_key);
	if (cached != physical_prediction_cache.end()) {
		return cached->second;
	}

	// Call the singleton model's Predict method (XGBoost)
	double predicted_cardinality = RLBoostingModel::Get().Predict(feature_vec);

	// If model returns 0, use DuckDB's estimate
	if (predicted_cardinality <= 0.0) {
		// Printer::Print("[RL MODEL] Returning DuckDB estimate: " + std::to_string(features.estimated_cardinality) + "\n");
		return features.estimated_cardinality;
	}

	// Otherwise, use the model's prediction
	idx_t result = static_cast<idx_t>(predicted_cardinality);
	physical_prediction_cache[cache_key] = result;
	physical_prediction_count++;
	if (physical_prediction_count % 100 == 0) {
		// Printer::Print("[RL PHYSICAL PREDICTION] count=" + std::to_string(physical_prediction_count) +
		//                ", operator=" + features.operator_type + ", join_relations=" + features.join_relation_set + "\n");
	}
	// Printer::Print("[RL MODEL] Returning model prediction: " + std::to_string(result) + "\n");
	return result;
}

void RLModelInterface::TrainModel(const OperatorFeatures &features, idx_t actual_cardinality) {
// legacy code ignore, not used
}

void RLModelInterface::AttachRLState(PhysicalOperator &physical_op, const OperatorFeatures &features,
                                      idx_t rl_prediction, idx_t duckdb_estimate) {
	if (!enabled || !PHYSICAL_RL_ENABLED) {
		return;
	}

	// Convert features to vector and attach RL state for training
	auto feature_vec = FeaturesToVector(features);
	physical_op.rl_state = make_uniq<RLOperatorState>(std::move(feature_vec), rl_prediction, duckdb_estimate);
}

void RLModelInterface::CollectActualCardinalities(PhysicalOperator &root_operator,
                                                   QueryProfiler &profiler,
                                                   RLTrainingBuffer &buffer) {
	if (!enabled || !PHYSICAL_RL_ENABLED) {
		return;
	}

	// Printer::Print("[RL TRAINING] Starting collection of actual cardinalities...\n");

	// If root is a RESULT_COLLECTOR, we need to get the actual plan from it
	PhysicalOperator *actual_root = &root_operator;
	if (root_operator.type == PhysicalOperatorType::RESULT_COLLECTOR) {
		auto &result_collector = root_operator.Cast<PhysicalResultCollector>();
		actual_root = &result_collector.plan;
		// Printer::Print("[RL TRAINING] Root is RESULT_COLLECTOR, traversing actual plan\n");
	}

	// Recursively traverse the physical operator tree
	CollectActualCardinalitiesRecursive(*actual_root, profiler, buffer);

	// Train EVERY query for fastest model building
	static std::atomic<idx_t> query_counter{0};
	idx_t current_count = ++query_counter;

	// Train every query with moderate batch for fast learning
	auto recent_samples = buffer.GetRecentSamples(500);
	if (recent_samples.size() >= 10) {
		auto &model = RLBoostingModel::Get();
		model.UpdateIncremental(recent_samples);
	}

	// Printer::Print("[RL TRAINING] Finished collecting actual cardinalities\n");
}

void RLModelInterface::CollectActualCardinalitiesRecursive(PhysicalOperator &op,
                                                             QueryProfiler &profiler,
                                                             RLTrainingBuffer &buffer) {
	// Check if this operator has RL state attached
	if (op.rl_state && op.rl_state->has_rl_prediction) {
		// Get the actual cardinality that was tracked during execution
		idx_t actual_cardinality = op.rl_state->GetActualCardinality();

		// Only train if we actually have data
		if (actual_cardinality > 0 || op.rl_state->rl_predicted_cardinality > 0) {
			// Mark as collected
			op.rl_state->has_actual_cardinality = true;

			// Add to buffer first
			buffer.AddSample(op.rl_state->feature_vector,
			                 actual_cardinality,
			                 op.rl_state->rl_predicted_cardinality);

			// SYNCHRONOUS INCREMENTAL TRAINING: Train immediately using sliding window
			// Get recent samples (sliding window: use more samples for better learning)
			// Using 2000 samples gives the model more context to learn patterns
			
			// OPTIMIZATION: Removed per-operator training call.
			// Training is now done in batch at the end of CollectActualCardinalities
			// to reduce lock contention and overhead.
			
			/*
			auto recent_samples = buffer.GetRecentSamples(2000);

			if (recent_samples.size() >= 50) {  // Need minimum samples for meaningful training
				auto &model = RLBoostingModel::Get();
				model.UpdateIncremental(recent_samples);
			}
			*/

			// Calculate Q-error for RL model
			double rl_q_error = std::max(
				actual_cardinality / (double)std::max(op.rl_state->rl_predicted_cardinality, (idx_t)1),
				op.rl_state->rl_predicted_cardinality / (double)std::max(actual_cardinality, (idx_t)1)
			);

			// Calculate Q-error for DuckDB's native estimate (for comparison)
			double duckdb_q_error = std::max(
				actual_cardinality / (double)std::max(op.rl_state->duckdb_estimated_cardinality, (idx_t)1),
				op.rl_state->duckdb_estimated_cardinality / (double)std::max(actual_cardinality, (idx_t)1)
			);

			// Logging with BOTH predictions for comparison
			auto &model = RLBoostingModel::Get();
			// Printer::Print("[RL TRAINING] " + op.GetName() + ": Actual=" +
			//                std::to_string(actual_cardinality) +
			//                ", RLPred=" + std::to_string(op.rl_state->rl_predicted_cardinality) +
			//                ", DuckPred=" + std::to_string(op.rl_state->duckdb_estimated_cardinality) +
			//                ", RLQerr=" + std::to_string(rl_q_error) +
			//                ", DuckQerr=" + std::to_string(duckdb_q_error) + 
			//                ", Trees=" + std::to_string(model.GetNumTrees()) + "\n");
		}
	}

	// Recursively process children
	for (auto &child : op.children) {
		CollectActualCardinalitiesRecursive(child.get(), profiler, buffer);
	}
}

} // namespace duckdb
