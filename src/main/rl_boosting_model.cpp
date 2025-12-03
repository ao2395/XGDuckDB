#include "duckdb/main/rl_boosting_model.hpp"
#include "xgboost/c_api.h"
#include "duckdb/common/printer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

namespace duckdb {

RLBoostingModel &RLBoostingModel::Get() {
	static RLBoostingModel instance;
	return instance;
}

RLBoostingModel::RLBoostingModel()
	: initialized(false), booster(nullptr), num_trees(0), total_updates(0) {
	// Printer::Print("[RL BOOSTING] Initializing XGBoost model for online learning...\n");
	InitializeBooster();
	initialized = true;
	// Printer::Print("[RL BOOSTING] XGBoost initialized with hyperparameters:\n");
	// Printer::Print("  max_depth=" + std::to_string(MAX_DEPTH) + "\n");
	// Printer::Print("  learning_rate=" + std::to_string(LEARNING_RATE) + "\n");
	// Printer::Print("  trees_per_update=" + std::to_string(TREES_PER_UPDATE) + "\n");
	// Printer::Print("  subsample=" + std::to_string(SUBSAMPLE) + "\n");
	// Printer::Print("  colsample_bytree=" + std::to_string(COLSAMPLE_BYTREE) + "\n");
}

RLBoostingModel::~RLBoostingModel() {
	// Signal shutdown to prevent any further access
	initialized = false;
	
	if (booster) {
		XGBoosterFree(booster);
		booster = nullptr;
	}
}

void RLBoostingModel::InitializeBooster() {
	// Create an initial empty DMatrix for booster initialization
	vector<float> init_data(FEATURE_VECTOR_SIZE, 0.0f);
	vector<float> init_labels = {1.0f}; // log(1) = 0, but we'll use 1.0

	DMatrixHandle dtrain;
	int ret = XGDMatrixCreateFromMat(init_data.data(), 1, FEATURE_VECTOR_SIZE, -1.0f, &dtrain);
	if (ret != 0) {
		// Printer::Print("[RL BOOSTING ERROR] Failed to create initial DMatrix: " +
		//                std::string(XGBGetLastError()) + "\n");
		return;
	}

	// Set labels
	ret = XGDMatrixSetFloatInfo(dtrain, "label", init_labels.data(), 1);
	if (ret != 0) {
		// Printer::Print("[RL BOOSTING ERROR] Failed to set labels: " +
		//                std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dtrain);
		return;
	}

	// Create booster
	ret = XGBoosterCreate(&dtrain, 1, &booster);
	if (ret != 0) {
		// Printer::Print("[RL BOOSTING ERROR] Failed to create booster: " +
		//                std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dtrain);
		return;
	}

	// Set hyperparameters
	XGBoosterSetParam(booster, "max_depth", std::to_string(MAX_DEPTH).c_str());
	XGBoosterSetParam(booster, "eta", std::to_string(LEARNING_RATE).c_str());
	XGBoosterSetParam(booster, "objective", "reg:squarederror");
	XGBoosterSetParam(booster, "subsample", std::to_string(SUBSAMPLE).c_str());
	XGBoosterSetParam(booster, "colsample_bytree", std::to_string(COLSAMPLE_BYTREE).c_str());
	XGBoosterSetParam(booster, "min_child_weight", std::to_string(MIN_CHILD_WEIGHT).c_str());
	XGBoosterSetParam(booster, "tree_method", "exact"); // Use exact tree method for small datasets

	// Set verbosity to silent
	XGBoosterSetParam(booster, "verbosity", "0");

	// Train on dummy data to configure num_features in XGBoost
	// This is necessary so subsequent training/prediction knows the feature count
	ret = XGBoosterUpdateOneIter(booster, 0, dtrain);
	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to train on initial data: " +
		               std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dtrain);
		return;
	}
	num_trees = 1;  // We now have one dummy tree

	// Clean up initial DMatrix
	XGDMatrixFree(dtrain);
}

DMatrixHandle RLBoostingModel::CreateDMatrix(const vector<RLTrainingSample> &samples) {
	if (samples.empty()) {
		return nullptr;
	}

	idx_t num_samples = samples.size();

	// Flatten features into row-major array
	vector<float> data;
	data.reserve(num_samples * FEATURE_VECTOR_SIZE);

	vector<float> labels;
	labels.reserve(num_samples);

	for (const auto &sample : samples) {
		// Add features
		for (const auto &feature : sample.features) {
			data.push_back(static_cast<float>(feature));
		}

		// Add label (log of actual cardinality for numerical stability)
		double actual_log = std::log(std::max(1.0, static_cast<double>(sample.actual_cardinality)));
		labels.push_back(static_cast<float>(actual_log));
	}

	// Create DMatrix
	DMatrixHandle dmat;
	int ret = XGDMatrixCreateFromMat(
		data.data(),
		num_samples,
		FEATURE_VECTOR_SIZE,
		-1.0f,  // missing value indicator
		&dmat
	);

	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to create DMatrix: " +
		               std::string(XGBGetLastError()) + "\n");
		return nullptr;
	}

	// Set labels
	ret = XGDMatrixSetFloatInfo(dmat, "label", labels.data(), num_samples);
	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to set labels: " +
		               std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dmat);
		return nullptr;
	}

	return dmat;
}

void RLBoostingModel::FreeDMatrix(DMatrixHandle dmat) {
	if (dmat) {
		XGDMatrixFree(dmat);
	}
}

double RLBoostingModel::Predict(const vector<double> &features) {
	// Safety check for shutdown - prevent access during/after destruction
	static std::atomic<bool> shutting_down{false};
	if (shutting_down.load(std::memory_order_relaxed)) {
		return 0.0;
	}

	// Validate input size
	if (features.size() != FEATURE_VECTOR_SIZE) {
		return 0.0;  // Silent fail for speed
	}

	// Check if model is ready for prediction (must have real training, not just dummy tree)
	if (!initialized || !booster || num_trees <= 1) {
		return 0.0;
	}

	// OPTIMIZATION: Use thread-local storage to create a JSON array interface once per thread
	thread_local vector<float> features_float;
	thread_local char array_interface_buffer[256];
	static const char PREDICT_CONFIG[] =
	    "{\"type\": 0, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": true, \"missing\": NaN}";

	if (features_float.size() != FEATURE_VECTOR_SIZE) {
		features_float.assign(FEATURE_VECTOR_SIZE, 0.0f);
	}

	// Convert features to float array (reusing the thread-local buffer)
	for (idx_t i = 0; i < FEATURE_VECTOR_SIZE; i++) {
		features_float[i] = static_cast<float>(features[i]);
	}

	// Build __array_interface__ JSON for the single-row dense input
	int written = snprintf(array_interface_buffer, sizeof(array_interface_buffer),
	                       "{\"data\": [%zu, true], \"shape\": [1, %zu], \"typestr\": \"<f4\", \"version\": 3}",
	                       reinterpret_cast<size_t>(features_float.data()), static_cast<size_t>(FEATURE_VECTOR_SIZE));
	if (written < 0 || written >= static_cast<int>(sizeof(array_interface_buffer))) {
		Printer::Print("[RL BOOSTING ERROR] Failed to encode array interface for prediction\n");
		return 0.0;
	}

	const bst_ulong *out_shape = nullptr;
	bst_ulong out_dim = 0;
	const float *out_result = nullptr;
	double log_cardinality;

	{
		lock_guard<mutex> lock(model_lock);

		int ret = XGBoosterPredictFromDense(booster, array_interface_buffer, PREDICT_CONFIG, nullptr, &out_shape,
		                                    &out_dim, &out_result);
		if (ret != 0 || !out_result || !out_shape || out_dim == 0) {
			Printer::Print("[RL BOOSTING ERROR] Inplace prediction failed: " +
			               std::string(XGBGetLastError()) + "\n");
			return 0.0;
		}

		// Expecting shape (1, 1) for a single regression output
		if (out_dim < 1 || out_shape[0] != 1) {
			Printer::Print("[RL BOOSTING ERROR] Unexpected prediction shape\n");
			return 0.0;
		}

		log_cardinality = static_cast<double>(out_result[0]);
	}

	// Clamp log prediction to reasonable range
	const double MIN_LOG_CARD = 0.0;   // exp(0) = 1
	log_cardinality = std::max(MIN_LOG_CARD, log_cardinality);

	// Convert from log(cardinality) to cardinality
	double cardinality = std::exp(log_cardinality);

	// Final safety clamp
	if (cardinality < 1.0) cardinality = 1.0;

	// REMOVED: Per-prediction logging (too expensive - 33k+ calls per benchmark)
	// Printer::Print("[RL BOOSTING] Prediction: log(card)=" + std::to_string(log_cardinality) +
	//                " -> card=" + std::to_string(cardinality) +
	//                " (trees=" + std::to_string(num_trees) + ")\n");

	return cardinality;
}

void RLBoostingModel::PredictBatch(const vector<vector<double>> &feature_matrix, vector<double> &output) {
	output.clear();
	if (feature_matrix.empty()) {
		return;
	}
	if (!initialized || !booster || num_trees <= 1) {
		return;
	}

	idx_t rows = feature_matrix.size();
	vector<float> dense(rows * FEATURE_VECTOR_SIZE, 0.0f);
	for (idx_t row = 0; row < rows; row++) {
		if (feature_matrix[row].size() != FEATURE_VECTOR_SIZE) {
			return;
		}
		for (idx_t col = 0; col < FEATURE_VECTOR_SIZE; col++) {
			dense[row * FEATURE_VECTOR_SIZE + col] = static_cast<float>(feature_matrix[row][col]);
		}
	}

	DMatrixHandle dmat;
	int ret = XGDMatrixCreateFromMat(dense.data(), rows, FEATURE_VECTOR_SIZE, -1.0f, &dmat);
	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to create batch prediction DMatrix: " +
		               std::string(XGBGetLastError()) + "\n");
		return;
	}

	bst_ulong out_len;
	const float *out_result;
	{
		lock_guard<mutex> lock(model_lock);
		ret = XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result);
	}
	if (ret != 0 || out_len < rows) {
		Printer::Print("[RL BOOSTING ERROR] Batch prediction failed: " + std::string(XGBGetLastError()) + "\n");
		FreeDMatrix(dmat);
		return;
	}

	output.reserve(rows);
	for (idx_t i = 0; i < rows; i++) {
		double log_cardinality = std::max(0.0, static_cast<double>(out_result[i]));
		double cardinality = std::exp(log_cardinality);
		if (cardinality < 1.0) {
			cardinality = 1.0;
		}
		output.push_back(cardinality);
	}

	FreeDMatrix(dmat);
}

void RLBoostingModel::UpdateIncremental(const vector<RLTrainingSample> &recent_samples) {
	if (!initialized || !booster) {
		return;
	}

	if (recent_samples.size() < 10) {
		// Need minimum samples for meaningful training
		return;
	}

	// Create DMatrix from recent samples
	DMatrixHandle dtrain = CreateDMatrix(recent_samples);
	if (!dtrain) {
		return;
	}

	// Perform incremental training (exclusive write lock)
	idx_t trees_added = 0;
	bool reached_tree_budget = false;
	{
		lock_guard<mutex> lock(model_lock);

		if (num_trees >= MAX_TOTAL_TREES) {
			reached_tree_budget = true;
		} else {
			idx_t remaining_capacity = MAX_TOTAL_TREES - num_trees;
			idx_t trees_to_add = std::min<idx_t>(remaining_capacity, static_cast<idx_t>(TREES_PER_UPDATE));
			if (trees_to_add > 0) {
				// Train for multiple iterations to add TREES_PER_UPDATE trees
				// IMPORTANT: XGBoosterUpdateOneIter's iteration parameter should be the CURRENT iteration
				// (starting from 0), NOT the total number of trees. Each call adds 1 tree.
				for (idx_t i = 0; i < trees_to_add; i++) {
					// Use total_updates * TREES_PER_UPDATE + i as the iteration number
					// This ensures each tree gets a unique iteration ID
					int iteration = total_updates * TREES_PER_UPDATE + static_cast<int>(i);
					int ret = XGBoosterUpdateOneIter(booster, iteration, dtrain);
					if (ret != 0) {
						Printer::Print("[RL BOOSTING ERROR] Training iteration failed: " +
						               std::string(XGBGetLastError()) + "\n");
						break;
					}
					num_trees++;
					trees_added++;
				}

				if (trees_added > 0) {
					total_updates++;
				}
			}
		}
	}

	if (reached_tree_budget) {
		// Printer::Print("[RL BOOSTING] Skipping update: reached max tree budget (" +
		//                std::to_string(MAX_TOTAL_TREES) + ")\n");
	}

	// Clean up
	FreeDMatrix(dtrain);

	// Calculate average Q-error for logging
	double total_q_error = 0.0;
	for (const auto &sample : recent_samples) {
		total_q_error += sample.q_error;
	}
	double avg_q_error = total_q_error / recent_samples.size();

	// Printer::Print("[RL BOOSTING] Incremental update #" + std::to_string(total_updates) +
	//                ": trained on " + std::to_string(recent_samples.size()) + " samples, " +
	//                "total trees=" + std::to_string(num_trees) +
	//                ", avg Q-error=" + std::to_string(avg_q_error) + "\n");
}

void RLBoostingModel::ResetModel() {
	lock_guard<mutex> lock(model_lock);

	if (booster) {
		XGBoosterFree(booster);
		booster = nullptr;
	}

	num_trees = 0;
	total_updates = 0;
	initialized = false;

	// Printer::Print("[RL BOOSTING] Model reset\n");

	InitializeBooster();
	initialized = true;

	// Printer::Print("[RL BOOSTING] Model reinitialized\n");
}

} // namespace duckdb
