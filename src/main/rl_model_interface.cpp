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

namespace duckdb {

RLModelInterface::RLModelInterface(ClientContext &context) : context(context), enabled(true) {
	// Model is a singleton - no need to create it here
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

	// 3. JOIN FEATURES - 21 features
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
	} else {
		idx += 21;
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
	if (!enabled) {
		return 0; // Don't override
	}

	// Print all features received by the model
	Printer::Print(features.ToString());

	// Convert features to vector
	auto feature_vec = FeaturesToVector(features);

	// Call the singleton model's Predict method (XGBoost)
	double predicted_cardinality = RLBoostingModel::Get().Predict(feature_vec);

	// If model returns 0, use DuckDB's estimate
	if (predicted_cardinality <= 0.0) {
		Printer::Print("[RL MODEL] Returning DuckDB estimate: " + std::to_string(features.estimated_cardinality) + "\n");
		return features.estimated_cardinality;
	}

	// Otherwise, use the model's prediction
	idx_t result = static_cast<idx_t>(predicted_cardinality);
	Printer::Print("[RL MODEL] Returning model prediction: " + std::to_string(result) + "\n");
	return result;
}

void RLModelInterface::TrainModel(const OperatorFeatures &features, idx_t actual_cardinality) {
	// To be implemented later for training
	// This will be called after each operator executes with the actual cardinality
}

void RLModelInterface::AttachRLState(PhysicalOperator &physical_op, const OperatorFeatures &features,
                                      idx_t rl_prediction, idx_t duckdb_estimate) {
	if (!enabled) {
		return;
	}

	// Convert features to vector
	auto feature_vec = FeaturesToVector(features);

	// Create and attach RL state
	physical_op.rl_state = make_uniq<RLOperatorState>(std::move(feature_vec), rl_prediction, duckdb_estimate);

	Printer::Print("[RL MODEL] Attached RL state to operator: RL=" + std::to_string(rl_prediction) + ", DuckDB=" +
	               std::to_string(duckdb_estimate) + "\n");
}

void RLModelInterface::CollectActualCardinalities(PhysicalOperator &root_operator,
                                                   QueryProfiler &profiler,
                                                   RLTrainingBuffer &buffer) {
	if (!enabled) {
		return;
	}

	Printer::Print("[RL TRAINING] Starting collection of actual cardinalities...\n");

	// If root is a RESULT_COLLECTOR, we need to get the actual plan from it
	PhysicalOperator *actual_root = &root_operator;
	if (root_operator.type == PhysicalOperatorType::RESULT_COLLECTOR) {
		auto &result_collector = root_operator.Cast<PhysicalResultCollector>();
		actual_root = &result_collector.plan;
		Printer::Print("[RL TRAINING] Root is RESULT_COLLECTOR, traversing actual plan\n");
	}

	// Recursively traverse the physical operator tree
	CollectActualCardinalitiesRecursive(*actual_root, profiler, buffer);
	Printer::Print("[RL TRAINING] Finished collecting actual cardinalities\n");
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
			auto recent_samples = buffer.GetRecentSamples(2000);

			if (recent_samples.size() >= 50) {  // Need minimum samples for meaningful training
				auto &model = RLBoostingModel::Get();
				model.UpdateIncremental(recent_samples);
			}

			// Calculate Q-error for logging
			double q_error = std::max(
				actual_cardinality / (double)std::max(op.rl_state->rl_predicted_cardinality, (idx_t)1),
				op.rl_state->rl_predicted_cardinality / (double)std::max(actual_cardinality, (idx_t)1)
			);

			// Simplified logging
			Printer::Print("[RL TRAINING] " + op.GetName() + ": Actual=" +
			               std::to_string(actual_cardinality) + ", Pred=" +
			               std::to_string(op.rl_state->rl_predicted_cardinality) + ", Q-err=" +
			               std::to_string(q_error) + "\n");
		}
	}

	// Recursively process children
	for (auto &child : op.children) {
		CollectActualCardinalitiesRecursive(child.get(), profiler, buffer);
	}
}

} // namespace duckdb
