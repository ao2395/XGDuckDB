#include "duckdb/main/rl_boosting_model.hpp"
#include "duckdb/common/printer.hpp"
#include <cmath>
#include <string>
#include <sstream>

namespace duckdb {

RLBoostingModel &RLBoostingModel::Get() {
	static RLBoostingModel instance;
	return instance;
}

RLBoostingModel::RLBoostingModel()
	: initialized(false), booster(nullptr), num_trees(0), total_updates(0) {
	Printer::Print("[RL BOOSTING] Initializing XGBoost model for online learning...\n");
	InitializeBooster();
	initialized = true;
	Printer::Print("[RL BOOSTING] XGBoost initialized with hyperparameters:\n");
	Printer::Print("  max_depth=" + std::to_string(MAX_DEPTH) + "\n");
	Printer::Print("  learning_rate=" + std::to_string(LEARNING_RATE) + "\n");
	Printer::Print("  trees_per_update=" + std::to_string(TREES_PER_UPDATE) + "\n");
	Printer::Print("  subsample=" + std::to_string(SUBSAMPLE) + "\n");
	Printer::Print("  colsample_bytree=" + std::to_string(COLSAMPLE_BYTREE) + "\n");
}

RLBoostingModel::~RLBoostingModel() {
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
		Printer::Print("[RL BOOSTING ERROR] Failed to create initial DMatrix: " +
		               std::string(XGBGetLastError()) + "\n");
		return;
	}

	// Set labels
	ret = XGDMatrixSetFloatInfo(dtrain, "label", init_labels.data(), 1);
	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to set labels: " +
		               std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dtrain);
		return;
	}

	// Create booster
	ret = XGBoosterCreate(&dtrain, 1, &booster);
	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to create booster: " +
		               std::string(XGBGetLastError()) + "\n");
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
	// Validate input size
	if (features.size() != FEATURE_VECTOR_SIZE) {
		Printer::Print("[RL BOOSTING ERROR] Invalid feature vector size: " +
		               std::to_string(features.size()) +
		               " (expected " + std::to_string(FEATURE_VECTOR_SIZE) + ")\n");
		return 0.0;
	}

	// Check if model is ready for prediction (must have real training, not just dummy tree)
	if (!initialized || !booster || num_trees <= 1) {
		// Model not ready yet - caller will use DuckDB estimate
		return 0.0;
	}

	// Convert features to float array
	vector<float> features_float(FEATURE_VECTOR_SIZE);
	for (idx_t i = 0; i < FEATURE_VECTOR_SIZE; i++) {
		features_float[i] = static_cast<float>(features[i]);
	}

	// Create DMatrix for prediction (single row)
	DMatrixHandle dmat;
	int ret = XGDMatrixCreateFromMat(
		features_float.data(),
		1,  // Single sample
		FEATURE_VECTOR_SIZE,
		-1.0f,
		&dmat
	);

	if (ret != 0) {
		Printer::Print("[RL BOOSTING ERROR] Failed to create prediction DMatrix: " +
		               std::string(XGBGetLastError()) + "\n");
		return 0.0;
	}

	// Perform prediction (thread-safe read)
	bst_ulong out_len;
	const float *out_result;
	double log_cardinality;

	{
		lock_guard<mutex> lock(model_lock);

		// Use XGBoosterPredict for inference
		ret = XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out_result);

		if (ret != 0 || out_len == 0) {
			Printer::Print("[RL BOOSTING ERROR] Prediction failed: " +
			               std::string(XGBGetLastError()) + "\n");
			FreeDMatrix(dmat);
			return 0.0;
		}

		log_cardinality = static_cast<double>(out_result[0]);
	}

	// Clean up
	FreeDMatrix(dmat);

	// Clamp log prediction to reasonable range
	const double MAX_LOG_CARD = 15.0;  // exp(15) ~= 3.3M
	const double MIN_LOG_CARD = 0.0;   // exp(0) = 1
	log_cardinality = std::max(MIN_LOG_CARD, std::min(MAX_LOG_CARD, log_cardinality));

	// Convert from log(cardinality) to cardinality
	double cardinality = std::exp(log_cardinality);

	// Final safety clamp
	if (cardinality < 1.0) cardinality = 1.0;

	Printer::Print("[RL BOOSTING] Prediction: log(card)=" + std::to_string(log_cardinality) +
	               " -> card=" + std::to_string(cardinality) +
	               " (trees=" + std::to_string(num_trees) + ")\n");

	return cardinality;
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
	{
		lock_guard<mutex> lock(model_lock);

		// Train for multiple iterations to add TREES_PER_UPDATE trees
		// IMPORTANT: XGBoosterUpdateOneIter's iteration parameter should be the CURRENT iteration
		// (starting from 0), NOT the total number of trees. Each call adds 1 tree.
		for (int i = 0; i < TREES_PER_UPDATE; i++) {
			// Use total_updates * TREES_PER_UPDATE + i as the iteration number
			// This ensures each tree gets a unique iteration ID
			int iteration = total_updates * TREES_PER_UPDATE + i;
			int ret = XGBoosterUpdateOneIter(booster, iteration, dtrain);
			if (ret != 0) {
				Printer::Print("[RL BOOSTING ERROR] Training iteration failed: " +
				               std::string(XGBGetLastError()) + "\n");
				break;
			}
			num_trees++;
		}

		total_updates++;
	}

	// Clean up
	FreeDMatrix(dtrain);

	// Calculate average Q-error for logging
	double total_q_error = 0.0;
	for (const auto &sample : recent_samples) {
		total_q_error += sample.q_error;
	}
	double avg_q_error = total_q_error / recent_samples.size();

	Printer::Print("[RL BOOSTING] Incremental update #" + std::to_string(total_updates) +
	               ": trained on " + std::to_string(recent_samples.size()) + " samples, " +
	               "total trees=" + std::to_string(num_trees) +
	               ", avg Q-error=" + std::to_string(avg_q_error) + "\n");
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

	Printer::Print("[RL BOOSTING] Model reset\n");

	InitializeBooster();
	initialized = true;

	Printer::Print("[RL BOOSTING] Model reinitialized\n");
}

} // namespace duckdb
