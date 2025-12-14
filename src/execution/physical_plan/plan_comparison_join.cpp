#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_iejoin.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_piecewise_merge_join.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/main/rl_model_interface.hpp"

namespace duckdb {

static void RewriteJoinCondition(unique_ptr<Expression> &root_expr, idx_t offset) {
	ExpressionIterator::VisitExpressionMutable<BoundReferenceExpression>(
	    root_expr, [&](BoundReferenceExpression &ref, unique_ptr<Expression> &expr) { ref.index += offset; });
}

PhysicalOperator &PhysicalPlanGenerator::PlanComparisonJoin(LogicalComparisonJoin &op) {
	// now visit the children
	D_ASSERT(op.children.size() == 2);
	auto &left = CreatePlan(*op.children[0]);
	auto &right = CreatePlan(*op.children[1]);
	// We intentionally do not overwrite child operators' cardinalities here.
	// (They reflect DuckDB's planning estimates; RL is observe-only.)

	// RL MODEL INFERENCE (observe-only): After children are created, extract features and compute a prediction.
	// IMPORTANT: Do NOT override `op.estimated_cardinality` - planning must not depend on RL estimates.
	RLModelInterface rl_model(context);
	auto features = rl_model.ExtractFeatures(op, context);
	const idx_t original_duckdb_estimate =
	    op.has_duckdb_estimated_cardinality ? op.duckdb_estimated_cardinality : op.estimated_cardinality;
	const idx_t rl_raw_prediction = rl_model.PredictCardinality(features);
	const idx_t rl_prediction = rl_raw_prediction > 0 ? rl_raw_prediction : original_duckdb_estimate;

	if (op.conditions.empty()) {
		// no conditions: insert a cross product
		auto &cross_product = Make<PhysicalCrossProduct>(op.types, left, right, op.estimated_cardinality);
		rl_model.AttachRLState(cross_product, features, rl_prediction, original_duckdb_estimate);
		return cross_product;
	}

	idx_t has_range = 0;
	bool has_equality = op.HasEquality(has_range);
	bool can_merge = has_range > 0;
	bool can_iejoin = has_range >= 2 && recursive_cte_tables.empty();
	switch (op.join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::RIGHT_ANTI:
	case JoinType::RIGHT_SEMI:
	case JoinType::MARK:
		can_merge = can_merge && op.conditions.size() == 1;
		can_iejoin = false;
		break;
	default:
		break;
	}
	//	TODO: Extend PWMJ to handle all comparisons and projection maps
	bool prefer_range_joins = DBConfig::GetSetting<PreferRangeJoinsSetting>(context);
	prefer_range_joins = prefer_range_joins && can_iejoin;
	if (has_equality && !prefer_range_joins) {
		// Equality join with small number of keys : possible perfect join optimization
		auto &join = Make<PhysicalHashJoin>(op, left, right, std::move(op.conditions), op.join_type,
		                                    op.left_projection_map, op.right_projection_map, std::move(op.mark_types),
		                                    op.estimated_cardinality, std::move(op.filter_pushdown));
		join.Cast<PhysicalHashJoin>().join_stats = std::move(op.join_stats);
		rl_model.AttachRLState(join, features, rl_prediction, original_duckdb_estimate);
		return join;
	}

	D_ASSERT(op.left_projection_map.empty());
	idx_t nested_loop_join_threshold = DBConfig::GetSetting<NestedLoopJoinThresholdSetting>(context);
	if (left.estimated_cardinality < nested_loop_join_threshold ||
	    right.estimated_cardinality < nested_loop_join_threshold) {
		can_iejoin = false;
		can_merge = false;
	}

	if (can_merge && can_iejoin) {
		idx_t merge_join_threshold = DBConfig::GetSetting<MergeJoinThresholdSetting>(context);
		if (left.estimated_cardinality < merge_join_threshold || right.estimated_cardinality < merge_join_threshold) {
			can_iejoin = false;
		}
	}

	if (can_iejoin) {
		auto &iejoin = Make<PhysicalIEJoin>(op, left, right, std::move(op.conditions), op.join_type, op.estimated_cardinality,
		                                    std::move(op.filter_pushdown));
		rl_model.AttachRLState(iejoin, features, rl_prediction, original_duckdb_estimate);
		return iejoin;
	}
	if (can_merge) {
		// range join: use piecewise merge join
		auto &merge_join = Make<PhysicalPiecewiseMergeJoin>(op, left, right, std::move(op.conditions), op.join_type,
		                                                    op.estimated_cardinality, std::move(op.filter_pushdown));
		rl_model.AttachRLState(merge_join, features, rl_prediction, original_duckdb_estimate);
		return merge_join;
	}
	if (PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type)) {
		// inequality join: use nested loop
		auto &nl_join = Make<PhysicalNestedLoopJoin>(op, left, right, std::move(op.conditions), op.join_type,
		                                             op.estimated_cardinality, std::move(op.filter_pushdown));
		rl_model.AttachRLState(nl_join, features, rl_prediction, original_duckdb_estimate);
		return nl_join;
	}

	for (auto &cond : op.conditions) {
		RewriteJoinCondition(cond.right, left.types.size());
	}
	auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
	auto &blockwise_join = Make<PhysicalBlockwiseNLJoin>(op, left, right, std::move(condition), op.join_type, op.estimated_cardinality);
	rl_model.AttachRLState(blockwise_join, features, rl_prediction, original_duckdb_estimate);
	return blockwise_join;
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalComparisonJoin &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		return PlanAsOfJoin(op);
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		return PlanComparisonJoin(op);
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PlanDelimJoin(op);
	default:
		throw InternalException("Unrecognized operator type for LogicalComparisonJoin");
	}
}

} // namespace duckdb
