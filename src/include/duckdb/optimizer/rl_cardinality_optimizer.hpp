//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/rl_cardinality_optimizer.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/main/rl_model_interface.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

namespace duckdb {

//! Replaces logical operator estimated cardinalities with RL predictions (fallback to DuckDB estimate).
//! This is intended to run inside the optimizer pipeline so subsequent decisions (join algo choice,
//! build/probe side, TopN, etc.) consume RL estimates via `op.estimated_cardinality`.
class RLCardinalityOptimizer : public LogicalOperatorVisitor {
public:
	explicit RLCardinalityOptimizer(ClientContext &context);

	void VisitOperator(LogicalOperator &op) override;

private:
	ClientContext &context;
	RLModelInterface rl_model;

	void ApplyToOperator(LogicalOperator &op);
};

} // namespace duckdb


