//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_training_buffer.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_training_buffer.hpp"
#include "duckdb/common/printer.hpp"
#include <chrono>
#include <algorithm>

namespace duckdb {

RLTrainingBuffer::RLTrainingBuffer(idx_t max_size)
    : max_size(max_size), running_q_error_sum(0.0), sample_count(0) {
	// Printer::Print("[RL TRAINING BUFFER] Initialized with capacity: " + std::to_string(max_size) + "\n");
}

RLTrainingBuffer::~RLTrainingBuffer() {
}

void RLTrainingBuffer::AddSample(const vector<double> &features, idx_t actual_cardinality,
                                   idx_t predicted_cardinality) {
	lock_guard<mutex> lock(buffer_lock);

	// Create training sample
	RLTrainingSample sample(features, actual_cardinality, predicted_cardinality);

	// Add timestamp
	auto now = std::chrono::system_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
	sample.timestamp_ms = ms.count();

	// Add to buffer (circular buffer behavior)
	if (buffer.size() >= max_size) {
		// Remove oldest sample
		auto &oldest = buffer.front();
		running_q_error_sum -= oldest.q_error;
		buffer.pop_front();
		sample_count--;
	}

	// Add new sample
	buffer.push_back(std::move(sample));
	running_q_error_sum += sample.q_error;
	sample_count++;
}

vector<RLTrainingSample> RLTrainingBuffer::GetBatch(idx_t batch_size) {
	lock_guard<mutex> lock(buffer_lock);

	if (buffer.empty()) {
		return {};
	}

	// Determine actual batch size (min of requested and available)
	idx_t actual_batch_size = MinValue<idx_t>(batch_size, buffer.size());

	vector<RLTrainingSample> batch;
	batch.reserve(actual_batch_size);

	// Uniform random sampling
	// For now, we'll just take the most recent samples
	// TODO: Implement prioritized experience replay based on Q-error
	for (idx_t i = 0; i < actual_batch_size; i++) {
		batch.push_back(buffer[buffer.size() - actual_batch_size + i]);
	}

	return batch;
}

vector<RLTrainingSample> RLTrainingBuffer::GetRecentSamples(idx_t count) {
	lock_guard<mutex> lock(buffer_lock);

	if (buffer.empty()) {
		return {};
	}

	// Determine how many samples to return (min of requested and available)
	idx_t actual_count = MinValue<idx_t>(count, buffer.size());

	vector<RLTrainingSample> samples;
	samples.reserve(actual_count);

	// Get the most recent samples (from the end of the deque)
	for (idx_t i = buffer.size() - actual_count; i < buffer.size(); i++) {
		samples.push_back(buffer[i]);
	}

	return samples;
}

idx_t RLTrainingBuffer::Size() const {
	lock_guard<mutex> lock(buffer_lock);
	return buffer.size();
}

idx_t RLTrainingBuffer::Capacity() const {
	return max_size;
}

bool RLTrainingBuffer::IsEmpty() const {
	lock_guard<mutex> lock(buffer_lock);
	return buffer.empty();
}

double RLTrainingBuffer::AverageQError() const {
	lock_guard<mutex> lock(buffer_lock);
	if (sample_count == 0) {
		return 0.0;
	}
	return running_q_error_sum / sample_count;
}

} // namespace duckdb
