//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_training_buffer.hpp
//
// Experience replay buffer for RL model training
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/mutex.hpp"
#include <deque>

namespace duckdb {

//! Training sample for the RL model
struct RLTrainingSample {
	vector<double> features;           // Feature vector (must match RLBoostingModel input size)
	idx_t actual_cardinality;          // Ground truth from execution
	idx_t predicted_cardinality;       // Model's prediction
	double q_error;                    // Quality metric
	uint64_t timestamp_ms;             // For prioritization/aging

	RLTrainingSample(vector<double> feat, idx_t actual, idx_t predicted)
	    : features(std::move(feat)), actual_cardinality(actual),
	      predicted_cardinality(predicted), timestamp_ms(0) {
		// Compute Q-error
		q_error = std::max(
			static_cast<double>(actual) / std::max(static_cast<double>(predicted), 1.0),
			static_cast<double>(predicted) / std::max(static_cast<double>(actual), 1.0)
		);
	}
};

//! Thread-safe circular buffer for experience replay
class RLTrainingBuffer {
public:
	RLTrainingBuffer(idx_t max_size = 10000);
	~RLTrainingBuffer();

	//! Add a training sample (called from query execution thread)
	void AddSample(const vector<double> &features, idx_t actual_cardinality, idx_t predicted_cardinality);

	//! Get a batch of samples for training (called from background thread)
	//! Returns empty vector if buffer is empty
	vector<RLTrainingSample> GetBatch(idx_t batch_size);

	//! Get the most recent N samples from the buffer (for sliding window training)
	//! Returns fewer than count if buffer has fewer samples
	vector<RLTrainingSample> GetRecentSamples(idx_t count);

	//! Get buffer statistics
	idx_t Size() const;
	idx_t Capacity() const;
	bool IsEmpty() const;
	double AverageQError() const;

private:
	mutable mutex buffer_lock;
	std::deque<RLTrainingSample> buffer;
	idx_t max_size;

	// Statistics
	double running_q_error_sum;
	idx_t sample_count;
};

} // namespace duckdb
