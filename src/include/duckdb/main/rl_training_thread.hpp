//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_training_thread.hpp
//
// Background training thread for RL cardinality model
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/main/rl_training_buffer.hpp"
#include <thread>
#include <atomic>
#include <condition_variable>

namespace duckdb {

class RLCardinalityModel;

//! Configuration for background training
struct RLTrainingConfig {
	idx_t batch_size = 32;              // Number of samples per training batch
	idx_t min_buffer_size = 50;         // Minimum samples before training starts
	idx_t training_interval_ms = 1000;  // Training frequency (milliseconds)
	idx_t max_iterations_per_cycle = 1; // Process each sample once (online learning)
	double learning_rate = 0.0003;       // Learning rate for gradient descent

	RLTrainingConfig() = default;
};

//! Background thread manager for model training
//! Runs asynchronously and periodically trains the model on buffered samples
class RLTrainingThread {
public:
	RLTrainingThread(RLCardinalityModel &model, RLTrainingBuffer &buffer);
	~RLTrainingThread();

	//! Start the background training thread
	void Start(const RLTrainingConfig &config = RLTrainingConfig());

	//! Stop the background training thread
	void Stop();

	//! Check if thread is running
	bool IsRunning() const;

	//! Get training statistics
	idx_t GetTotalUpdates() const;
	double GetAverageTrainingLoss() const;

private:
	//! Main training loop (runs in background thread)
	void TrainingLoop();

	//! Perform one training cycle (batch updates)
	void TrainBatch();

	// References to model and buffer
	RLCardinalityModel &model;
	RLTrainingBuffer &buffer;

	// Thread management
	std::thread training_thread;
	std::atomic<bool> should_stop;
	std::atomic<bool> is_running;
	mutable mutex training_mutex;
	std::condition_variable training_cv;

	// Configuration
	RLTrainingConfig config;

	// Statistics
	std::atomic<idx_t> total_updates;
	std::atomic<double> running_loss_sum;
	std::atomic<idx_t> loss_count;
};

} // namespace duckdb
