// Copyright 2026.
// Author: Mooncake Team.
//
// Concurrent, exponentially decayed quantile estimation.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "decaying_quantile/decaying_ddsketch.hpp"

namespace mooncake::tool {
namespace {

constexpr std::size_t MAX_BUCKET_COUNT = 1'000'000;
constexpr std::size_t CACHE_LINE_SIZE = 64;
constexpr std::uint32_t SPIN_BEFORE_YIELD = 64;

bool IsPowerOfTwo(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

class SpinMutex {
   public:
    SpinMutex() = default;

    SpinMutex(const SpinMutex&) = delete;
    SpinMutex& operator=(const SpinMutex&) = delete;

    void Lock() {
        std::uint32_t spin_count = 0;
        for (;;) {
            if (!is_locked_.test_and_set(std::memory_order_acquire)) {
                return;
            }

            while (is_locked_.test(std::memory_order_relaxed)) {
                ++spin_count;
                if (spin_count >= SPIN_BEFORE_YIELD) {
                    spin_count = 0;
                    std::this_thread::yield();
                }
            }
        }
    }

    void Unlock() { is_locked_.clear(std::memory_order_release); }

   private:
    std::atomic_flag is_locked_ = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
   public:
    explicit SpinLockGuard(SpinMutex& mutex) : mutex_(mutex) { mutex_.Lock(); }

    ~SpinLockGuard() { mutex_.Unlock(); }

    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;

   private:
    SpinMutex& mutex_;
};

// weight is stored as float: upstream heat values are bounded by decay and
// clamping, so double mantissa precision is unnecessary. The Bucket layout
// (alignment/padding) is intentionally left unchanged.
struct alignas(CACHE_LINE_SIZE) Bucket {
    mutable SpinMutex mutex;
    float weight = 0.0f;
    DecayingDdSketch::Timestamp reference_time = 0;
};

struct Shard {
    explicit Shard(std::size_t bucket_count)
        : buckets(std::make_unique<Bucket[]>(bucket_count)) {}

    std::unique_ptr<Bucket[]> buckets;
};

}  // namespace

class DecayingDdSketch::Impl {
   public:
    explicit Impl(const Config& config)
        : config_(ValidateAndCopyConfig(config)),
          gamma_((1.0 + config_.relative_accuracy) /
                 (1.0 - config_.relative_accuracy)),
          log_gamma_(std::log(gamma_)),
          inverse_half_life_(1.0 / static_cast<double>(config_.half_life)),
          min_key_(CalculateKey(config_.min_indexed_value)),
          max_key_(CalculateKey(config_.max_indexed_value)),
          bucket_count_(CalculateBucketCount()),
          shard_mask_(config_.shard_count - 1) {
        if (bucket_count_ < 2 || bucket_count_ > MAX_BUCKET_COUNT) {
            throw std::invalid_argument("configured bucket count is invalid");
        }

        representatives_.reserve(bucket_count_);
        representatives_.push_back(0.0);
        for (std::int64_t key = min_key_; key <= max_key_; ++key) {
            representatives_.push_back(2.0 * std::pow(gamma_, key) /
                                       (gamma_ + 1.0));
        }

        shards_.reserve(config_.shard_count);
        for (std::size_t i = 0; i < config_.shard_count; ++i) {
            shards_.emplace_back(bucket_count_);
        }
    }

    bool Add(std::uint64_t key_hash, double value, Timestamp inserted_at) {
        const auto bucket_index = MapValue(value);
        if (!bucket_index.has_value()) {
            return false;
        }

        UpdateMaxTimestamp(inserted_at);
        ApplyDelta(GetShard(key_hash), *bucket_index, 1.0, inserted_at);
        return true;
    }

    bool Replace(std::uint64_t key_hash, double old_value,
                 Timestamp old_inserted_at, double new_value,
                 Timestamp new_inserted_at) {
        const auto old_bucket_index = MapValue(old_value);
        const auto new_bucket_index = MapValue(new_value);
        if (!old_bucket_index.has_value() || !new_bucket_index.has_value() ||
            new_inserted_at < old_inserted_at) {
            return false;
        }

        UpdateMaxTimestamp(new_inserted_at);
        Shard& shard = GetShard(key_hash);
        if (*old_bucket_index == *new_bucket_index) {
            ApplyReplacement(shard, *old_bucket_index, old_inserted_at,
                             new_inserted_at);
        } else {
            ApplyDelta(shard, *old_bucket_index, -1.0, old_inserted_at);
            ApplyDelta(shard, *new_bucket_index, 1.0, new_inserted_at);
        }

        return true;
    }

    bool Remove(std::uint64_t key_hash, double value, Timestamp inserted_at) {
        const auto bucket_index = MapValue(value);
        if (!bucket_index.has_value()) {
            return false;
        }

        ApplyDelta(GetShard(key_hash), *bucket_index, -1.0, inserted_at);
        return true;
    }

    std::vector<std::optional<double>> GetQuantiles(
        const std::vector<double>& quantiles, Timestamp query_time) const {
        std::vector<std::optional<double>> results(quantiles.size());
        if (query_time < max_timestamp_.load(std::memory_order_acquire)) {
            return results;
        }

        std::vector<std::pair<double, std::size_t>> requests;
        requests.reserve(quantiles.size());
        for (std::size_t i = 0; i < quantiles.size(); ++i) {
            if (std::isfinite(quantiles[i]) && quantiles[i] >= 0.0 &&
                quantiles[i] <= 1.0) {
                requests.emplace_back(quantiles[i], i);
            }
        }
        if (requests.empty()) {
            return results;
        }

        std::sort(requests.begin(), requests.end());
        const std::vector<double> weights = SnapshotWeights(query_time);
        double total_weight = 0.0;
        for (double weight : weights) {
            total_weight += weight;
        }
        if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
            return results;
        }

        std::size_t request_index = 0;
        double cumulative_weight = 0.0;
        for (std::size_t bucket_index = 0;
             bucket_index < bucket_count_ && request_index < requests.size();
             ++bucket_index) {
            if (!(weights[bucket_index] > 0.0)) {
                continue;
            }
            cumulative_weight += weights[bucket_index];
            while (request_index < requests.size() &&
                   cumulative_weight >=
                       requests[request_index].first * total_weight) {
                results[requests[request_index].second] =
                    representatives_[bucket_index];
                ++request_index;
            }
        }

        while (request_index < requests.size()) {
            results[requests[request_index].second] = representatives_.back();
            ++request_index;
        }
        return results;
    }

    std::optional<double> GetTotalWeight(Timestamp query_time) const {
        if (query_time < max_timestamp_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        const std::vector<double> weights = SnapshotWeights(query_time);
        double total_weight = 0.0;
        for (double weight : weights) {
            total_weight += weight;
        }
        return std::max(0.0, total_weight);
    }

    std::size_t GetBucketCount() const { return bucket_count_; }

    std::size_t GetShardCount() const { return shards_.size(); }

   private:
    static Config ValidateAndCopyConfig(const Config& config) {
        if (!std::isfinite(config.relative_accuracy) ||
            config.relative_accuracy <= 0.0 ||
            config.relative_accuracy >= 1.0) {
            throw std::invalid_argument(
                "relative_accuracy must be finite and in (0, 1)");
        }
        if (!std::isfinite(config.min_indexed_value) ||
            !std::isfinite(config.max_indexed_value) ||
            config.min_indexed_value <= 0.0 ||
            config.max_indexed_value < config.min_indexed_value) {
            throw std::invalid_argument(
                "indexed value range must be finite, positive, and ordered");
        }
        if (config.half_life == 0) {
            throw std::invalid_argument("half_life must be positive");
        }
        if (!IsPowerOfTwo(config.shard_count)) {
            throw std::invalid_argument(
                "shard_count must be a non-zero power of two");
        }

        const double gamma =
            (1.0 + config.relative_accuracy) / (1.0 - config.relative_accuracy);
        if (!std::isfinite(gamma) || !(gamma > 1.0) ||
            !(std::log(gamma) > 0.0)) {
            throw std::invalid_argument(
                "relative_accuracy is too small for double precision");
        }
        return config;
    }

    std::int64_t CalculateKey(double value) const {
        return static_cast<std::int64_t>(
            std::ceil(std::log(value) / log_gamma_));
    }

    std::size_t CalculateBucketCount() const {
        if (max_key_ < min_key_) {
            return 0;
        }

        const std::uint64_t positive_bucket_count =
            static_cast<std::uint64_t>(max_key_ - min_key_) + 1;
        if (positive_bucket_count >= MAX_BUCKET_COUNT) {
            return MAX_BUCKET_COUNT + 1;
        }
        return static_cast<std::size_t>(positive_bucket_count + 1);
    }

    std::optional<std::size_t> MapValue(double value) const {
        if (!std::isfinite(value) || value < 0.0 ||
            value > config_.max_indexed_value) {
            return std::nullopt;
        }
        if (value < config_.min_indexed_value) {
            return 0;
        }

        const std::int64_t key = CalculateKey(value);
        if (key < min_key_ || key > max_key_) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(key - min_key_) + 1;
    }

    double Decay(Timestamp from, Timestamp to) const {
        if (to <= from) {
            return 1.0;
        }
        const double half_lives =
            static_cast<double>(to - from) * inverse_half_life_;
        return std::exp2(-half_lives);
    }

    Shard& GetShard(std::uint64_t key_hash) {
        return shards_[static_cast<std::size_t>(key_hash) & shard_mask_];
    }

    void ApplyDelta(Shard& shard, std::size_t bucket_index, double sign,
                    Timestamp inserted_at) {
        Bucket& bucket = shard.buckets[bucket_index];
        SpinLockGuard lock(bucket.mutex);

        const Timestamp reference_time =
            std::max(bucket.reference_time, inserted_at);
        // Keep the bucket arithmetic in float semantics: each operand is
        // rounded to float before the update so the stored weight reflects
        // float precision end to end.
        const float decay_factor =
            static_cast<float>(Decay(bucket.reference_time, reference_time));
        const float contribution = static_cast<float>(
            sign * Decay(inserted_at, reference_time));
        bucket.weight = bucket.weight * decay_factor + contribution;
        bucket.reference_time = reference_time;
    }

    void ApplyReplacement(Shard& shard, std::size_t bucket_index,
                          Timestamp old_inserted_at,
                          Timestamp new_inserted_at) {
        Bucket& bucket = shard.buckets[bucket_index];
        SpinLockGuard lock(bucket.mutex);

        const Timestamp reference_time =
            std::max({bucket.reference_time, old_inserted_at, new_inserted_at});
        // Float-rounded operands; see ApplyDelta for the precision rationale.
        const float decay_factor =
            static_cast<float>(Decay(bucket.reference_time, reference_time));
        const float removed_contribution = static_cast<float>(
            Decay(old_inserted_at, reference_time));
        const float added_contribution =
            static_cast<float>(Decay(new_inserted_at, reference_time));
        bucket.weight = bucket.weight * decay_factor -
                        removed_contribution + added_contribution;
        bucket.reference_time = reference_time;
    }

    std::vector<double> SnapshotWeights(Timestamp query_time) const {
        std::vector<double> weights(bucket_count_, 0.0);
        for (const Shard& shard : shards_) {
            for (std::size_t bucket_index = 0; bucket_index < bucket_count_;
                 ++bucket_index) {
                const Bucket& bucket = shard.buckets[bucket_index];
                double weight = 0.0;
                Timestamp reference_time = 0;
                {
                    SpinLockGuard lock(bucket.mutex);
                    weight = bucket.weight;
                    reference_time = bucket.reference_time;
                }

                if (weight > 0.0) {
                    weight *= Decay(reference_time, query_time);
                    weights[bucket_index] += weight;
                }
            }
        }
        return weights;
    }

    void UpdateMaxTimestamp(Timestamp timestamp) {
        Timestamp current = max_timestamp_.load(std::memory_order_relaxed);
        while (current < timestamp &&
               !max_timestamp_.compare_exchange_weak(
                   current, timestamp, std::memory_order_release,
                   std::memory_order_relaxed)) {
        }
    }

    Config config_;
    double gamma_;
    double log_gamma_;
    double inverse_half_life_;
    std::int64_t min_key_;
    std::int64_t max_key_;
    std::size_t bucket_count_;
    std::size_t shard_mask_;
    std::vector<double> representatives_;
    std::vector<Shard> shards_;
    std::atomic<Timestamp> max_timestamp_ = 0;
};

DecayingDdSketch::DecayingDdSketch(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

DecayingDdSketch::~DecayingDdSketch() = default;

bool DecayingDdSketch::Add(std::uint64_t key_hash, double value,
                           Timestamp inserted_at) {
    return impl_->Add(key_hash, value, inserted_at);
}

bool DecayingDdSketch::Replace(std::uint64_t key_hash, double old_value,
                               Timestamp old_inserted_at, double new_value,
                               Timestamp new_inserted_at) {
    return impl_->Replace(key_hash, old_value, old_inserted_at, new_value,
                          new_inserted_at);
}

bool DecayingDdSketch::Remove(std::uint64_t key_hash, double value,
                              Timestamp inserted_at) {
    return impl_->Remove(key_hash, value, inserted_at);
}

std::optional<double> DecayingDdSketch::GetQuantile(
    double quantile, Timestamp query_time) const {
    return GetQuantiles({quantile}, query_time).front();
}

std::vector<std::optional<double>> DecayingDdSketch::GetQuantiles(
    const std::vector<double>& quantiles, Timestamp query_time) const {
    return impl_->GetQuantiles(quantiles, query_time);
}

std::optional<double> DecayingDdSketch::GetTotalWeight(
    Timestamp query_time) const {
    return impl_->GetTotalWeight(query_time);
}

std::size_t DecayingDdSketch::GetBucketCount() const {
    return impl_->GetBucketCount();
}

std::size_t DecayingDdSketch::GetShardCount() const {
    return impl_->GetShardCount();
}

}  // namespace mooncake::tool
