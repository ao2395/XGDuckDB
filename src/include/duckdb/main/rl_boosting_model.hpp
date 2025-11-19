//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_boosting_model.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/main/rl_training_buffer.hpp"
#include "xgboost/c_api.h"

namespace duckdb {

//! Gradient Boosted Trees model for online reinforcement learning cardinality estimation
//! Uses XGBoost library for efficient gradient boosting
//! Singleton pattern - one model instance shared across all queries
//! Implements incremental tree building: adds new trees after each query execution
class RLBoostingModel {
public:
	//! Get the singleton instance
	static RLBoostingModel &Get();

	// Delete copy constructor and assignment operator
	RLBoostingModel(const RLBoostingModel &) = delete;
	RLBoostingModel &operator=(const RLBoostingModel &) = delete;

private:
	RLBoostingModel();
	~RLBoostingModel();

public:
	//! Perform inference: takes feature vector and returns estimated cardinality
	//! Input: 64-dimensional feature vector
	//! Output: predicted cardinality (NOT log - we convert internally)
	//! Thread-safe for concurrent reads
	double Predict(const vector<double> &features);

	//! Train incrementally: adds trees based on recent samples from sliding window
	//! Uses synchronous training after each query
	//! Thread-safe with exclusive write lock
	void UpdateIncremental(const vector<RLTrainingSample> &recent_samples);

	//! Check if model is ready for inference
	//! Model must be initialized AND have real training (> 1 tree, since 1st is dummy)
	bool IsReady() const {
		return initialized && num_trees > 1;
	}

	//! Get current number of trees in the ensemble
	idx_t GetNumTrees() const {
		return num_trees;
	}

	//! Get total number of training updates performed
	idx_t GetTotalUpdates() const {
		return total_updates;
	}

	//! Reset model to initial state (emergency recovery)
	void ResetModel();

private:
	void InitializeBooster();
	DMatrixHandle CreateDMatrix(const vector<RLTrainingSample> &samples);
	void FreeDMatrix(DMatrixHandle dmat);

	bool initialized;
	mutable mutex model_lock;

	// XGBoost handles
	BoosterHandle booster;

	// Model state
	idx_t num_trees;
	idx_t total_updates;

	// Hyperparameters - chosen for online learning cardinality estimation
	static constexpr int MAX_DEPTH = 5;                // Moderate complexity
	static constexpr float LEARNING_RATE = 0.3f;       // Higher for online setting
	static constexpr int TREES_PER_UPDATE = 1;         // Add 1 tree per operator update
	static constexpr float SUBSAMPLE = 0.8f;           // Row sampling for regularization
	static constexpr float COLSAMPLE_BYTREE = 0.8f;    // Feature sampling
	static constexpr int MIN_CHILD_WEIGHT = 5;         // Regularization
	static constexpr int FEATURE_VECTOR_SIZE = 64;     // Must match RLModelInterface
	static constexpr idx_t MAX_TOTAL_TREES = 10000;    // Prevent runaway ensemble growth

	// Sliding window size for training
	static constexpr idx_t DEFAULT_WINDOW_SIZE = 200;
};

} // namespace duckdb
