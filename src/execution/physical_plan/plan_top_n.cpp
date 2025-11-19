#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/main/rl_model_interface.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalTopN &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);

	// RL MODEL INFERENCE: After child is created, extract features and get estimate
	RLModelInterface rl_model(context);
	auto features = rl_model.ExtractFeatures(op, context);
	idx_t original_duckdb_estimate = op.estimated_cardinality;
	auto rl_estimate = rl_model.GetCardinalityEstimate(features);
	if (rl_estimate > 0) {
		op.estimated_cardinality = rl_estimate;
	}
	const idx_t rl_prediction = rl_estimate > 0 ? rl_estimate : original_duckdb_estimate;

	auto &top_n =
	    Make<PhysicalTopN>(op.types, std::move(op.orders), NumericCast<idx_t>(op.limit), NumericCast<idx_t>(op.offset),
	                       std::move(op.dynamic_filter), op.estimated_cardinality);
	top_n.children.push_back(plan);
	rl_model.AttachRLState(top_n, features, rl_prediction, original_duckdb_estimate);
	return top_n;
}

} // namespace duckdb
