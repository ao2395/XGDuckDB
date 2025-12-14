#include "duckdb/main/rl_boosting_model.hpp"
#include "xgboost/c_api.h"
#include "duckdb/common/printer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <cstring>

namespace duckdb {

static bool EnvBool(const char *name, bool default_value) {
	auto val = std::getenv(name);
	if (!val) {
		return default_value;
	}
	string s(val);
	for (auto &c : s) {
		c = char(std::tolower(c));
	}
	if (s == "1" || s == "true" || s == "yes" || s == "on") {
		return true;
	}
	if (s == "0" || s == "false" || s == "no" || s == "off") {
		return false;
	}
	return default_value;
}

static int EnvInt(const char *name, int default_value) {
	auto val = std::getenv(name);
	if (!val) {
		return default_value;
	}
	try {
		return std::stoi(val);
	} catch (...) {
		return default_value;
	}
}

static double EnvDouble(const char *name, double default_value) {
	auto val = std::getenv(name);
	if (!val) {
		return default_value;
	}
	try {
		return std::stod(val);
	} catch (...) {
		return default_value;
	}
}

static string EnvString(const char *name, const string &default_value) {
	auto val = std::getenv(name);
	if (!val) {
		return default_value;
	}
	return string(val);
}

RLBoostingModel &RLBoostingModel::Get() {
	static RLBoostingModel instance;
	return instance;
}

RLBoostingModel::RLBoostingModel()
	: initialized(false), active_booster(nullptr), num_trees(0), total_updates(0),
	  max_depth(DEFAULT_MAX_DEPTH), learning_rate(DEFAULT_LEARNING_RATE), trees_per_update(DEFAULT_TREES_PER_UPDATE),
	  subsample(DEFAULT_SUBSAMPLE), colsample_bytree(DEFAULT_COLSAMPLE_BYTREE),
	  min_child_weight(DEFAULT_MIN_CHILD_WEIGHT), max_total_trees(DEFAULT_MAX_TOTAL_TREES),
	  objective("reg:absoluteerror"), lambda_l2(1.0f), alpha_l1(0.0f), gamma(0.0f) {
	// Runtime overrides (no recompile needed):
	// - RL_MAX_DEPTH (int)
	// - RL_ETA (float)
	// - RL_TREES_PER_UPDATE (int)
	// - RL_SUBSAMPLE (float)
	// - RL_COLSAMPLE_BYTREE (float)
	// - RL_MIN_CHILD_WEIGHT (int)
	// - RL_MAX_TOTAL_TREES (int)
	// - RL_OBJECTIVE (string, e.g. reg:absoluteerror)
	// - RL_LAMBDA, RL_ALPHA, RL_GAMMA (float)
	max_depth = EnvInt("RL_MAX_DEPTH", max_depth);
	learning_rate = (float)EnvDouble("RL_ETA", learning_rate);
	trees_per_update = EnvInt("RL_TREES_PER_UPDATE", trees_per_update);
	subsample = (float)EnvDouble("RL_SUBSAMPLE", subsample);
	colsample_bytree = (float)EnvDouble("RL_COLSAMPLE_BYTREE", colsample_bytree);
	min_child_weight = EnvInt("RL_MIN_CHILD_WEIGHT", min_child_weight);
	max_total_trees = (idx_t)EnvInt("RL_MAX_TOTAL_TREES", (int)max_total_trees);
	objective = EnvString("RL_OBJECTIVE", objective);
	lambda_l2 = (float)EnvDouble("RL_LAMBDA", lambda_l2);
	alpha_l1 = (float)EnvDouble("RL_ALPHA", alpha_l1);
	gamma = (float)EnvDouble("RL_GAMMA", gamma);

	// Printer::Print("[RL BOOSTING] Initializing XGBoost model for online learning...\n");
	InitializeBooster();
	initialized = true;
	// Printer::Print("[RL BOOSTING] XGBoost initialized with hyperparameters:\n");
	// Printer::Print("  max_depth=" + std::to_string(max_depth) + "\n");
	// Printer::Print("  learning_rate=" + std::to_string(learning_rate) + "\n");
	// Printer::Print("  trees_per_update=" + std::to_string(trees_per_update) + "\n");
	// Printer::Print("  subsample=" + std::to_string(subsample) + "\n");
	// Printer::Print("  colsample_bytree=" + std::to_string(colsample_bytree) + "\n");
}

RLBoostingModel::~RLBoostingModel() {
	// Signal shutdown to prevent any further access
	initialized = false;

	auto booster = active_booster.load(std::memory_order_acquire);
	if (booster) {
		XGBoosterFree(booster);
		active_booster.store(nullptr, std::memory_order_release);
	}
	if (training_booster) {
		XGBoosterFree(training_booster);
		training_booster = nullptr;
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
	BoosterHandle booster = nullptr;
	ret = XGBoosterCreate(&dtrain, 1, &booster);
	if (ret != 0) {
		// Printer::Print("[RL BOOSTING ERROR] Failed to create booster: " +
		//                std::string(XGBGetLastError()) + "\n");
		XGDMatrixFree(dtrain);
		return;
	}

	// Set hyperparameters
	XGBoosterSetParam(booster, "max_depth", std::to_string(max_depth).c_str());
	XGBoosterSetParam(booster, "eta", std::to_string(learning_rate).c_str());
	XGBoosterSetParam(booster, "objective", objective.c_str());
	XGBoosterSetParam(booster, "subsample", std::to_string(subsample).c_str());
	XGBoosterSetParam(booster, "colsample_bytree", std::to_string(colsample_bytree).c_str());
	XGBoosterSetParam(booster, "min_child_weight", std::to_string(min_child_weight).c_str());
	XGBoosterSetParam(booster, "tree_method", "exact"); // Use exact tree method for small datasets
	// Regularization for better generalization (helps median and tail)
	XGBoosterSetParam(booster, "lambda", std::to_string(lambda_l2).c_str());  // L2
	XGBoosterSetParam(booster, "alpha", std::to_string(alpha_l1).c_str());   // L1
	XGBoosterSetParam(booster, "gamma", std::to_string(gamma).c_str());      // min split loss
	XGBoosterSetParam(booster, "max_delta_step", "0");

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
	num_trees.store(1, std::memory_order_release);  // We now have one dummy tree
	total_updates.store(0, std::memory_order_release);
	active_booster.store(booster, std::memory_order_release);
	if (training_booster) {
		XGBoosterFree(training_booster);
		training_booster = nullptr;
	}
	training_num_trees = 0;
	training_total_updates = 0;
	training_update_calls = 0;

	// Clean up initial DMatrix
	XGDMatrixFree(dtrain);
}

BoosterHandle RLBoostingModel::CloneBooster(BoosterHandle source) {
	if (!source) {
		return nullptr;
	}
	// Serialize model to a buffer
	bst_ulong len = 0;
	const char *buf = nullptr;
	int ret = XGBoosterSerializeToBuffer(source, &len, &buf);
	if (ret != 0 || !buf || len == 0) {
		return nullptr;
	}

	// Create a new booster and unserialize into it
	BoosterHandle cloned = nullptr;
	ret = XGBoosterCreate(nullptr, 0, &cloned);
	if (ret != 0 || !cloned) {
		return nullptr;
	}
	ret = XGBoosterUnserializeFromBuffer(cloned, buf, len);
	if (ret != 0) {
		XGBoosterFree(cloned);
		return nullptr;
	}
	return cloned;
}

void RLBoostingModel::EnsureTrainingBooster() {
	if (training_booster) {
		return;
	}
	// Initialize training booster as a clone of the current active booster
	BoosterHandle snapshot = nullptr;
	{
		lock_guard<mutex> pred_guard(predict_lock);
		snapshot = active_booster.load(std::memory_order_acquire);
	}
	training_booster = CloneBooster(snapshot);
	training_num_trees = num_trees.load(std::memory_order_acquire);
	training_total_updates = total_updates.load(std::memory_order_acquire);
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
	auto booster = active_booster.load(std::memory_order_acquire);
	auto trees = num_trees.load(std::memory_order_acquire);
	if (!initialized || !booster || trees <= 1) {
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
		lock_guard<mutex> lock(predict_lock);

		int ret = XGBoosterPredictFromDense(booster, array_interface_buffer, PREDICT_CONFIG, nullptr, &out_shape,
		                                    &out_dim, &out_result);
		if (ret != 0 || !out_result || !out_shape || out_dim == 0) {
			return 0.0;  // Silent fail for speed
		}

		if (out_dim < 1 || out_shape[0] != 1) {
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
	auto booster = active_booster.load(std::memory_order_acquire);
	auto trees = num_trees.load(std::memory_order_acquire);
	if (!initialized || !booster || trees <= 1) {
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
		lock_guard<mutex> lock(predict_lock);
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
	auto current = active_booster.load(std::memory_order_acquire);
	if (!initialized || !current) {
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

	// Train on a persistent shadow booster, then swap occasionally.
	idx_t trees_added = 0;
	bool reached_tree_budget = false;
	idx_t new_total_updates = 0;
	idx_t new_num_trees = 0;
	{
		lock_guard<mutex> train_guard(train_lock);
		EnsureTrainingBooster();
		if (!training_booster) {
			FreeDMatrix(dtrain);
			return;
		}

		new_num_trees = training_num_trees;
		new_total_updates = training_total_updates;

		if (new_num_trees >= max_total_trees) {
			reached_tree_budget = true;
		} else {
			idx_t remaining_capacity = max_total_trees - new_num_trees;
			idx_t trees_to_add = std::min<idx_t>(remaining_capacity, static_cast<idx_t>(trees_per_update));
			if (trees_to_add > 0) {
				for (idx_t i = 0; i < trees_to_add; i++) {
					int iteration = (int)(new_total_updates * (idx_t)trees_per_update + i);
					int ret = XGBoosterUpdateOneIter(training_booster, iteration, dtrain);
					if (ret != 0) {
						Printer::Print("[RL BOOSTING ERROR] Training iteration failed: " +
						               std::string(XGBGetLastError()) + "\n");
						break;
					}
					new_num_trees++;
					trees_added++;
				}
				if (trees_added > 0) {
					new_total_updates++;
				}
			}
		}

		training_num_trees = new_num_trees;
		training_total_updates = new_total_updates;
		training_update_calls++;

		// Swap policy: default every 5 training updates (env override)
		const idx_t swap_every = (idx_t)EnvInt("RL_SWAP_EVERY_N_UPDATES", 5);
		if (swap_every > 0 && (training_update_calls % swap_every) == 0) {
			lock_guard<mutex> pred_guard(predict_lock);
			auto old_active = active_booster.load(std::memory_order_acquire);
			active_booster.store(training_booster, std::memory_order_release);
			num_trees.store(training_num_trees, std::memory_order_release);
			total_updates.store(training_total_updates, std::memory_order_release);
			// Start a fresh training booster cloned from the new active model for the next segment
			training_booster = nullptr;
			if (old_active) {
				XGBoosterFree(old_active);
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

	// NOTE: `run_tpcds_benchmark.py` parses this exact line format.
	Printer::Print("[RL BOOSTING] Incremental update #" + std::to_string(new_total_updates) +
	               ": trained on " + std::to_string(recent_samples.size()) + " samples, " +
	               "total trees=" + std::to_string(new_num_trees) +
	               ", avg Q-error=" + std::to_string(avg_q_error) + "\n");
}

void RLBoostingModel::ResetModel() {
	lock_guard<mutex> train_guard(train_lock);
	lock_guard<mutex> pred_guard(predict_lock);
	auto booster = active_booster.load(std::memory_order_acquire);
	if (booster) {
		XGBoosterFree(booster);
		active_booster.store(nullptr, std::memory_order_release);
	}
	if (training_booster) {
		XGBoosterFree(training_booster);
		training_booster = nullptr;
	}

	num_trees.store(0, std::memory_order_release);
	total_updates.store(0, std::memory_order_release);
	training_num_trees = 0;
	training_total_updates = 0;
	training_update_calls = 0;
	initialized = false;

	// Printer::Print("[RL BOOSTING] Model reset\n");

	InitializeBooster();
	initialized = true;

	// Printer::Print("[RL BOOSTING] Model reinitialized\n");
}

} // namespace duckdb

