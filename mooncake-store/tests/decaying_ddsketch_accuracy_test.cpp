// Copyright 2026.
// Author: Mooncake Team.
//
// Large-scale accuracy tests for exponentially decayed quantiles.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "decaying_quantile/decaying_ddsketch.hpp"

namespace mooncake::tool {
namespace {

constexpr std::size_t KEY_COUNT = 250'000;
constexpr std::size_t REPLACEMENT_COUNT = 1'000'000;
constexpr DecayingDdSketch::Timestamp HALF_LIFE = 120'000;
constexpr std::uint64_t HASH_MULTIPLIER = 0x9e3779b97f4a7c15ULL;

struct ExactSample {
    double value = 0.0;
    DecayingDdSketch::Timestamp inserted_at = 0;
};

struct WeightedSample {
    double value = 0.0;
    double weight = 0.0;
};

std::uint64_t MakeStableHash(std::uint64_t key) {
    key ^= key >> 30;
    key *= HASH_MULTIPLIER;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    return key ^ (key >> 31);
}

double ClampValue(double value) { return std::clamp(value, 1e-3, 1e6); }

double CalculateExactWeight(DecayingDdSketch::Timestamp inserted_at,
                            DecayingDdSketch::Timestamp query_time) {
    const double half_lives = static_cast<double>(query_time - inserted_at) /
                              static_cast<double>(HALF_LIFE);
    return std::exp2(-half_lives);
}

double CalculateExactQuantile(const std::vector<WeightedSample>& samples,
                              double total_weight, double quantile) {
    const double target_weight = quantile * total_weight;
    double cumulative_weight = 0.0;
    for (const WeightedSample& sample : samples) {
        cumulative_weight += sample.weight;
        if (cumulative_weight >= target_weight) {
            return sample.value;
        }
    }
    return samples.back().value;
}

double CalculateExactRank(const std::vector<WeightedSample>& samples,
                          double total_weight, double value) {
    double cumulative_weight = 0.0;
    for (const WeightedSample& sample : samples) {
        if (sample.value > value) {
            break;
        }
        cumulative_weight += sample.weight;
    }
    return cumulative_weight / total_weight;
}

TEST(DecayingDdSketchAccuracyTest,
     TracksMillionReplacementStreamWithinConfiguredError) {
    DecayingDdSketch::Config config;
    config.relative_accuracy = 0.01;
    config.min_indexed_value = 1e-3;
    config.max_indexed_value = 1e6;
    config.half_life = HALF_LIFE;
    config.shard_count = 64;
    DecayingDdSketch sketch(config);

    std::mt19937_64 random_engine(0x4d4f4f4e43414b45ULL);
    std::lognormal_distribution<double> value_distribution(std::log(100.0),
                                                           1.5);
    std::uniform_int_distribution<std::size_t> key_distribution(0,
                                                                KEY_COUNT - 1);
    std::vector<ExactSample> current_samples(KEY_COUNT);

    for (std::size_t key = 0; key < KEY_COUNT; ++key) {
        const double value = ClampValue(value_distribution(random_engine));
        current_samples[key] = ExactSample{value, 0};
        ASSERT_TRUE(sketch.Add(MakeStableHash(key), value, 0));
    }

    for (std::size_t update = 1; update <= REPLACEMENT_COUNT; ++update) {
        const std::size_t key = key_distribution(random_engine);
        const double new_value = ClampValue(value_distribution(random_engine));
        const auto timestamp = static_cast<DecayingDdSketch::Timestamp>(update);
        const ExactSample old_sample = current_samples[key];

        ASSERT_TRUE(sketch.Replace(MakeStableHash(key), old_sample.value,
                                   old_sample.inserted_at, new_value,
                                   timestamp));
        current_samples[key] = ExactSample{new_value, timestamp};
    }

    const auto query_time =
        static_cast<DecayingDdSketch::Timestamp>(REPLACEMENT_COUNT);
    std::vector<WeightedSample> exact_samples;
    exact_samples.reserve(KEY_COUNT);
    double exact_total_weight = 0.0;
    for (const ExactSample& sample : current_samples) {
        const double weight =
            CalculateExactWeight(sample.inserted_at, query_time);
        exact_samples.push_back(WeightedSample{sample.value, weight});
        exact_total_weight += weight;
    }
    std::sort(exact_samples.begin(), exact_samples.end(),
              [](const WeightedSample& left, const WeightedSample& right) {
                  return left.value < right.value;
              });

    const std::vector<double> requested_quantiles = {0.5, 0.9, 0.99, 0.999};
    const auto estimated_quantiles =
        sketch.GetQuantiles(requested_quantiles, query_time);
    const auto estimated_total_weight = sketch.GetTotalWeight(query_time);
    ASSERT_TRUE(estimated_total_weight.has_value());

    // Bucket weights are stored as float (24-bit mantissa), so each decayed
    // accumulation rounds at ~2^-24 per operation. Over ~1M updates spread
    // across ~1k logical buckets the relative total-weight error is expected
    // around 1e-6; 1e-4 is a conservative bound that still flags regressions.
    const double total_weight_error =
        std::abs(*estimated_total_weight - exact_total_weight) /
        exact_total_weight;
    EXPECT_LE(total_weight_error, 1e-4);

    std::cout << std::fixed << std::setprecision(8)
              << "\nLarge accuracy workload, keys: " << KEY_COUNT
              << ", replacements: " << REPLACEMENT_COUNT
              << ", total_weight_relative_error: " << total_weight_error
              << '\n';
    std::cout << "quantile,exact,estimated,value_relative_error,rank_error\n";

    for (std::size_t i = 0; i < requested_quantiles.size(); ++i) {
        ASSERT_TRUE(estimated_quantiles[i].has_value());
        const double exact_quantile = CalculateExactQuantile(
            exact_samples, exact_total_weight, requested_quantiles[i]);
        const double estimated_quantile = *estimated_quantiles[i];
        const double value_error =
            std::abs(estimated_quantile - exact_quantile) / exact_quantile;
        const double estimated_rank = CalculateExactRank(
            exact_samples, exact_total_weight, estimated_quantile);
        const double rank_error =
            std::abs(estimated_rank - requested_quantiles[i]);

        std::cout << requested_quantiles[i] << ',' << exact_quantile << ','
                  << estimated_quantile << ',' << value_error << ','
                  << rank_error << '\n';

        EXPECT_LE(value_error, config.relative_accuracy + 1e-12);
        EXPECT_LE(rank_error, 0.005);
    }
}

}  // namespace
}  // namespace mooncake::tool
