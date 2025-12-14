#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/main/rl_model_interface.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalTopN &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);

	// RL MODEL INFERENCE (observe-only): compute a prediction for Q-error/training.
	// IMPORTANT: Do NOT override `op.estimated_cardinality` - planning must not depend on RL estimates.
	RLModelInterface rl_model(context);
	auto features = rl_model.ExtractFeatures(op, context);
	const idx_t original_duckdb_estimate =
	    op.has_duckdb_estimated_cardinality ? op.duckdb_estimated_cardinality : op.estimated_cardinality;
	const idx_t rl_raw_prediction = rl_model.PredictCardinality(features);
	const idx_t rl_prediction = rl_raw_prediction > 0 ? rl_raw_prediction : original_duckdb_estimate;

	auto &top_n =
	    Make<PhysicalTopN>(op.types, std::move(op.orders), NumericCast<idx_t>(op.limit), NumericCast<idx_t>(op.offset),
	                       std::move(op.dynamic_filter), op.estimated_cardinality);
	top_n.children.push_back(plan);
	rl_model.AttachRLState(top_n, features, rl_prediction, original_duckdb_estimate);
	return top_n;
}

} // namespace duckdb
