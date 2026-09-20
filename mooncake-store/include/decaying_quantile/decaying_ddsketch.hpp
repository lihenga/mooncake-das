// Copyright 2026.
// Author: Mooncake Team.
//
// Concurrent, exponentially decayed quantile estimation.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace mooncake::tool {

/**
 * @brief Estimates quantiles over exponentially decayed, replaceable samples.
 *
 * Values use a fixed DDSketch-style logarithmic mapping. Each sample's weight
 * decays exponentially from its insertion timestamp. The caller supplies the
 * exact value and timestamp previously inserted when replacing or removing a
 * sample, so this class does not retain per-key state.
 *
 * @note Timestamp and Config::half_life must use the same caller-defined unit.
 * @note Concurrent mutations are safe and order-independent for valid sample
 *       transitions. Quantile reads are weakly consistent while writes run.
 */
class DecayingDdSketch {
   public:
    using Timestamp = std::uint64_t;

    /** @brief Construction parameters. */
    struct Config {
        double relative_accuracy = 0.01;
        double min_indexed_value = 1e-6;
        double max_indexed_value = 1e6;
        Timestamp half_life = 60'000;
        std::size_t shard_count = 64;
    };

    /**
     * @brief Constructs a decaying quantile estimator.
     * @param config Mapping, decay, and concurrency parameters.
     * @throws std::invalid_argument if the configuration is invalid.
     */
    explicit DecayingDdSketch(const Config& config);

    ~DecayingDdSketch();

    DecayingDdSketch(const DecayingDdSketch&) = delete;
    DecayingDdSketch& operator=(const DecayingDdSketch&) = delete;
    DecayingDdSketch(DecayingDdSketch&&) = delete;
    DecayingDdSketch& operator=(DecayingDdSketch&&) = delete;

    /**
     * @brief Adds a new sample with initial weight one.
     * @param key_hash Stable hash used only to select a concurrency shard.
     * @param value Non-negative sample value.
     * @param inserted_at Sample insertion timestamp.
     * @return true when the value is within the configured mapping range.
     */
    bool Add(std::uint64_t key_hash, double value, Timestamp inserted_at);

    /**
     * @brief Replaces one previously inserted sample.
     * @param key_hash Stable hash used only to select a concurrency shard.
     * @param old_value Exact value previously supplied to Add or Replace.
     * @param old_inserted_at Timestamp associated with old_value.
     * @param new_value New non-negative sample value.
     * @param new_inserted_at Timestamp associated with new_value.
     * @return true when both values and timestamps are valid.
     *
     * @note A successful old transition must be replaced exactly once. Calls
     *       for a transition chain may complete out of order.
     */
    bool Replace(std::uint64_t key_hash, double old_value,
                 Timestamp old_inserted_at, double new_value,
                 Timestamp new_inserted_at);

    /**
     * @brief Removes one previously inserted sample.
     * @param key_hash Stable hash used only to select a concurrency shard.
     * @param value Exact value previously supplied to Add or Replace.
     * @param inserted_at Timestamp associated with value.
     * @return true when the value is within the configured mapping range.
     */
    bool Remove(std::uint64_t key_hash, double value, Timestamp inserted_at);

    /**
     * @brief Returns one estimated quantile at query_time.
     * @param quantile Quantile in the closed interval [0, 1].
     * @param query_time Timestamp at which sample weights are evaluated.
     * @return Estimated value, or std::nullopt for invalid/empty queries.
     */
    std::optional<double> GetQuantile(double quantile,
                                      Timestamp query_time) const;

    /**
     * @brief Returns several quantiles using one bucket scan.
     * @param quantiles Quantiles in the closed interval [0, 1].
     * @param query_time Timestamp at which sample weights are evaluated.
     * @return Results in input order. Invalid quantiles contain std::nullopt.
     */
    std::vector<std::optional<double>> GetQuantiles(
        const std::vector<double>& quantiles, Timestamp query_time) const;

    /**
     * @brief Returns the effective decayed sample weight.
     * @param query_time Timestamp at which sample weights are evaluated.
     * @return Effective weight, or std::nullopt if query_time is too old.
     */
    std::optional<double> GetTotalWeight(Timestamp query_time) const;

    /** @return Number of logical value buckets, including the zero bucket. */
    std::size_t GetBucketCount() const;

    /** @return Number of independent concurrency shards. */
    std::size_t GetShardCount() const;

   private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace mooncake::tool
