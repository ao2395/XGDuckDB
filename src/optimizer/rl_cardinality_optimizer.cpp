//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rl_cardinality_optimizer.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/rl_cardinality_optimizer.hpp"
#include "duckdb/common/helper.hpp"

namespace duckdb {

RLCardinalityOptimizer::RLCardinalityOptimizer(ClientContext &context_p) : context(context_p), rl_model(context_p) {
}

void RLCardinalityOptimizer::VisitOperator(LogicalOperator &op) {
	// Post-order: first update children, then compute estimate for current operator using updated child context.
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
	ApplyToOperator(op);
}

void RLCardinalityOptimizer::ApplyToOperator(LogicalOperator &op) {
	// Preserve DuckDB baseline estimate the first time we overwrite it
	if (!op.has_duckdb_estimated_cardinality && op.has_estimated_cardinality) {
		op.duckdb_estimated_cardinality = op.estimated_cardinality;
		op.has_duckdb_estimated_cardinality = true;
	}

	auto features = rl_model.ExtractFeatures(op, context);

	// Ensure child cardinality context is set for operators whose feature vector expects it
	if (features.child_cardinality == 0 && !op.children.empty()) {
		features.child_cardinality = op.children[0]->estimated_cardinality;
	}

	const auto rl_pred = rl_model.PredictPlanningCardinality(features);
	const auto fallback = features.estimated_cardinality;
	const auto effective = rl_pred > 0 ? rl_pred : fallback;

	op.estimated_cardinality = MaxValue<idx_t>(effective, 1);
	op.has_estimated_cardinality = true;
}

} // namespace duckdb


