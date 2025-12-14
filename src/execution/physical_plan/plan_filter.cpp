#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/main/rl_model_interface.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalFilter &op) {
	D_ASSERT(op.children.size() == 1);
	reference<PhysicalOperator> plan = CreatePlan(*op.children[0]);

	// RL MODEL INFERENCE (observe-only): After child is created, extract features and compute a prediction.
	// IMPORTANT: Do NOT override `op.estimated_cardinality` - planning must not depend on RL estimates.
	RLModelInterface rl_model(context);
	auto features = rl_model.ExtractFeatures(op, context);
	// Use the physical child's cardinality as context.
	features.child_cardinality = plan.get().estimated_cardinality;
	const idx_t original_duckdb_estimate =
	    op.has_duckdb_estimated_cardinality ? op.duckdb_estimated_cardinality : op.estimated_cardinality;
	const idx_t rl_raw_prediction = rl_model.PredictCardinality(features);
	const idx_t rl_prediction = rl_raw_prediction > 0 ? rl_raw_prediction : original_duckdb_estimate;

	if (!op.expressions.empty()) {
		D_ASSERT(!plan.get().GetTypes().empty());
		// create a filter if there is anything to filter
		auto &filter = Make<PhysicalFilter>(plan.get().GetTypes(), std::move(op.expressions), op.estimated_cardinality);
		filter.children.push_back(plan);

		// Attach RL state to track prediction for training
		rl_model.AttachRLState(filter, features, rl_prediction, original_duckdb_estimate);

		plan = filter;
	}
	if (op.HasProjectionMap()) {
		// there is a projection map, generate a physical projection
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < op.projection_map.size(); i++) {
			select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], op.projection_map[i]));
		}
		auto &proj = Make<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
		proj.children.push_back(plan);
		plan = proj;
	}
	return plan;
}

} // namespace duckdb
