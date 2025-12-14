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
#include <atomic>

namespace duckdb {

typedef void *DMatrixHandle;
typedef void *BoosterHandle;

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
	//! Input: 70-dimensional feature vector (expanded with selectivity features)
	//! Output: predicted cardinality (NOT log - we convert internally)
	//! Thread-safe for concurrent reads
	double Predict(const vector<double> &features);
	void PredictBatch(const vector<vector<double>> &feature_matrix, vector<double> &output);

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
	BoosterHandle CloneBooster(BoosterHandle source);
	void EnsureTrainingBooster();
	DMatrixHandle CreateDMatrix(const vector<RLTrainingSample> &samples);
	void FreeDMatrix(DMatrixHandle dmat);

	bool initialized;
	//! Serialize predictions (XGBoost booster is not guaranteed thread-safe for concurrent Predict calls)
	mutable mutex predict_lock;
	//! Serialize training updates (but do NOT block prediction)
	mutable mutex train_lock;

	// XGBoost handles
	std::atomic<BoosterHandle> active_booster;
	BoosterHandle training_booster = nullptr;

	// Model state
	std::atomic<idx_t> num_trees;
	std::atomic<idx_t> total_updates;
	idx_t training_num_trees = 0;
	idx_t training_total_updates = 0;
	idx_t training_update_calls = 0;

public:
	// Hyperparameters - chosen for online learning cardinality estimation
	// Defaults are "best known" median-first settings.
	// You can override these at runtime using environment variables (see rl_boosting_model.cpp).
	static constexpr int DEFAULT_MAX_DEPTH = 6;
	static constexpr float DEFAULT_LEARNING_RATE = 0.1f;
	static constexpr int DEFAULT_TREES_PER_UPDATE = 10;
	static constexpr float DEFAULT_SUBSAMPLE = 0.8f;
	static constexpr float DEFAULT_COLSAMPLE_BYTREE = 0.8f;
	static constexpr int DEFAULT_MIN_CHILD_WEIGHT = 3;
	static constexpr int FEATURE_VECTOR_SIZE = 80;     // Must match RLModelInterface
	static constexpr idx_t DEFAULT_MAX_TOTAL_TREES = 2000;

private:
	// Sliding window size for training
	static constexpr idx_t DEFAULT_WINDOW_SIZE = 200;

private:
	// Runtime-tunable hyperparameters (loaded from env in constructor, then fixed)
	int max_depth;
	float learning_rate;
	int trees_per_update;
	float subsample;
	float colsample_bytree;
	int min_child_weight;
	idx_t max_total_trees;
	string objective;
	float lambda_l2;
	float alpha_l1;
	float gamma;
};

} // namespace duckdb
