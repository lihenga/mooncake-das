// Copyright 2026.
// Author: Mooncake Team.
//
// Unit tests for concurrent, exponentially decayed quantile estimation.

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "decaying_quantile/decaying_ddsketch.hpp"

namespace mooncake::tool {
namespace {

DecayingDdSketch::Config MakeConfig() {
    DecayingDdSketch::Config config;
    config.relative_accuracy = 0.01;
    config.min_indexed_value = 1e-3;
    config.max_indexed_value = 1e6;
    config.half_life = 100;
    config.shard_count = 8;
    return config;
}

void ExpectRelativeNear(double actual, double expected, double tolerance) {
    ASSERT_GT(expected, 0.0);
    EXPECT_LE(std::abs(actual - expected) / expected, tolerance);
}

TEST(DecayingDdSketchTest, EstimatesBasicQuantiles) {
    DecayingDdSketch sketch(MakeConfig());
    for (std::uint64_t value = 1; value <= 100; ++value) {
        ASSERT_TRUE(sketch.Add(value, static_cast<double>(value), 0));
    }

    const auto quantiles = sketch.GetQuantiles({0.0, 0.5, 0.9, 1.0}, 0);
    ASSERT_EQ(quantiles.size(), 4);
    for (const auto& quantile : quantiles) {
        ASSERT_TRUE(quantile.has_value());
    }

    ExpectRelativeNear(*quantiles[0], 1.0, 0.02);
    ExpectRelativeNear(*quantiles[1], 50.0, 0.02);
    ExpectRelativeNear(*quantiles[2], 90.0, 0.02);
    ExpectRelativeNear(*quantiles[3], 100.0, 0.02);
}

TEST(DecayingDdSketchTest, RecentSampleDominatesAfterDecay) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    ASSERT_TRUE(sketch.Add(2, 100.0, 100));

    const auto median = sketch.GetQuantile(0.5, 100);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 100.0, 0.02);

    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, 1.5, 1e-12);
}

TEST(DecayingDdSketchTest, ReplaceRemovesDecayedOldContribution) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    ASSERT_TRUE(sketch.Add(2, 100.0, 0));
    ASSERT_TRUE(sketch.Replace(1, 10.0, 0, 1000.0, 100));

    const auto quantiles = sketch.GetQuantiles({0.2, 0.5}, 100);
    ASSERT_TRUE(quantiles[0].has_value());
    ASSERT_TRUE(quantiles[1].has_value());
    ExpectRelativeNear(*quantiles[0], 100.0, 0.02);
    ExpectRelativeNear(*quantiles[1], 1000.0, 0.02);

    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, 1.5, 1e-12);
}

TEST(DecayingDdSketchTest, ReplacesWithinSameBucket) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    ASSERT_TRUE(sketch.Replace(1, 10.0, 0, 10.01, 100));

    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, 1.0, 1e-12);

    const auto median = sketch.GetQuantile(0.5, 100);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 10.01, 0.02);
}

TEST(DecayingDdSketchTest, SupportsZeroAndNearZeroValues) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 0.0, 0));
    ASSERT_TRUE(sketch.Add(2, 1e-4, 0));
    ASSERT_TRUE(sketch.Add(3, 1.0, 0));

    const auto quantiles = sketch.GetQuantiles({0.0, 0.5, 1.0}, 0);
    ASSERT_TRUE(quantiles[0].has_value());
    ASSERT_TRUE(quantiles[1].has_value());
    ASSERT_TRUE(quantiles[2].has_value());
    EXPECT_DOUBLE_EQ(*quantiles[0], 0.0);
    EXPECT_DOUBLE_EQ(*quantiles[1], 0.0);
    ExpectRelativeNear(*quantiles[2], 1.0, 0.02);
}

TEST(DecayingDdSketchTest, MutationOrderDoesNotChangeFinalDistribution) {
    DecayingDdSketch sketch(MakeConfig());

    ASSERT_TRUE(sketch.Replace(1, 10.0, 0, 1000.0, 100));
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));

    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, 1.0, 1e-12);

    const auto median = sketch.GetQuantile(0.5, 100);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 1000.0, 0.02);
}

TEST(DecayingDdSketchTest, RemoveEliminatesSample) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    ASSERT_TRUE(sketch.Add(2, 100.0, 0));
    ASSERT_TRUE(sketch.Remove(1, 10.0, 0));

    const auto total_weight = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, 1.0, 1e-12);

    const auto median = sketch.GetQuantile(0.5, 0);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 100.0, 0.02);
}

TEST(DecayingDdSketchTest, RejectsInvalidInputsAndQueries) {
    DecayingDdSketch sketch(MakeConfig());
    EXPECT_FALSE(sketch.Add(1, -1.0, 0));
    EXPECT_FALSE(sketch.Add(1, std::numeric_limits<double>::infinity(), 0));
    EXPECT_FALSE(sketch.Add(1, 1e7, 0));
    EXPECT_FALSE(sketch.Replace(1, 1.0, 10, 2.0, 9));

    EXPECT_FALSE(sketch.GetQuantile(-0.1, 0).has_value());
    EXPECT_FALSE(sketch.GetQuantile(1.1, 0).has_value());
    EXPECT_FALSE(sketch.GetQuantile(0.5, 0).has_value());

    ASSERT_TRUE(sketch.Add(1, 1.0, 10));
    EXPECT_FALSE(sketch.GetQuantile(0.5, 9).has_value());
}

TEST(DecayingDdSketchTest, HandlesConcurrentAddsAndReplacements) {
    DecayingDdSketch::Config config = MakeConfig();
    config.shard_count = 32;
    DecayingDdSketch sketch(config);

    constexpr std::size_t THREAD_COUNT = 16;
    constexpr std::size_t KEY_COUNT_PER_THREAD = 2'000;
    std::atomic<bool> start = false;
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);

    for (std::size_t thread_index = 0; thread_index < THREAD_COUNT;
         ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (std::size_t i = 0; i < KEY_COUNT_PER_THREAD; ++i) {
                const std::uint64_t key =
                    thread_index * KEY_COUNT_PER_THREAD + i;
                const double new_value = 10.0 + static_cast<double>(i % 100);
                EXPECT_TRUE(sketch.Add(key, 1.0, 0));
                EXPECT_TRUE(sketch.Replace(key, 1.0, 0, new_value, 100));
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight,
                static_cast<double>(THREAD_COUNT * KEY_COUNT_PER_THREAD), 1e-6);

    const auto median = sketch.GetQuantile(0.5, 100);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 59.0, 0.03);
}

TEST(DecayingDdSketchTest, AllowsQuantileReadsDuringConcurrentWrites) {
    DecayingDdSketch sketch(MakeConfig());
    constexpr std::size_t KEY_COUNT = 8'000;
    constexpr std::size_t WRITER_COUNT = 8;

    for (std::size_t key = 0; key < KEY_COUNT; ++key) {
        ASSERT_TRUE(sketch.Add(key, 1.0, 0));
    }

    std::atomic<bool> start = false;
    std::atomic<std::size_t> finished_writers = 0;
    std::atomic<bool> observed_invalid_result = false;
    std::vector<std::thread> writers;
    writers.reserve(WRITER_COUNT);

    for (std::size_t writer = 0; writer < WRITER_COUNT; ++writer) {
        writers.emplace_back([&, writer]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (std::size_t key = writer; key < KEY_COUNT;
                 key += WRITER_COUNT) {
                const double new_value = 10.0 + static_cast<double>(key % 100);
                sketch.Replace(key, 1.0, 0, new_value, 100);
            }
            finished_writers.fetch_add(1, std::memory_order_release);
        });
    }

    std::thread reader([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        do {
            const auto quantile = sketch.GetQuantile(0.9, 100);
            if (!quantile.has_value() || !std::isfinite(*quantile) ||
                *quantile < 0.0 || *quantile > 1e6) {
                observed_invalid_result.store(true, std::memory_order_relaxed);
            }
        } while (finished_writers.load(std::memory_order_acquire) <
                 WRITER_COUNT);
    });

    start.store(true, std::memory_order_release);
    for (std::thread& writer : writers) {
        writer.join();
    }
    reader.join();

    EXPECT_FALSE(observed_invalid_result.load(std::memory_order_relaxed));
    const auto total_weight = sketch.GetTotalWeight(100);
    ASSERT_TRUE(total_weight.has_value());
    EXPECT_NEAR(*total_weight, static_cast<double>(KEY_COUNT), 1e-6);
}

TEST(DecayingDdSketchTest, ValidatesConfiguration) {
    DecayingDdSketch::Config config = MakeConfig();
    config.relative_accuracy = 0.0;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);

    config = MakeConfig();
    config.half_life = 0;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);

    config = MakeConfig();
    config.shard_count = 3;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);
}

// A configuration whose dynamic range would need more buckets than the sketch
// is willing to materialize must be rejected up front, instead of allocating an
// unbounded representatives_ vector on construction.
TEST(DecayingDdSketchTest, RejectsConfigurationWithExcessiveBucketCount) {
    DecayingDdSketch::Config config = MakeConfig();
    config.relative_accuracy = 1e-6;
    config.min_indexed_value = 1e-3;
    config.max_indexed_value = 1e9;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);
}

// Config validation boundaries beyond the three cases covered above: the whole
// legal domain of each field must be rejected/accepted on the correct side.
TEST(DecayingDdSketchTest, ValidatesConfigurationDomainBoundaries) {
    for (const double accuracy :
         {0.0, -0.01, 1.0, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()}) {
        DecayingDdSketch::Config config = MakeConfig();
        config.relative_accuracy = accuracy;
        EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument)
            << "relative_accuracy=" << accuracy;
    }

    DecayingDdSketch::Config config = MakeConfig();
    config.min_indexed_value = 0.0;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);

    config = MakeConfig();
    config.max_indexed_value = config.min_indexed_value / 2.0;
    EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument);

    for (const std::size_t shards :
         {std::size_t{0}, std::size_t{3}, std::size_t{6}}) {
        config = MakeConfig();
        config.shard_count = shards;
        EXPECT_THROW(DecayingDdSketch sketch(config), std::invalid_argument)
            << "shard_count=" << shards;
    }

    // Smallest legal values: half_life == 1 and shard_count == 1.
    config = MakeConfig();
    config.half_life = 1;
    config.shard_count = 1;
    DecayingDdSketch minimal(config);
    EXPECT_EQ(minimal.GetShardCount(), 1u);
    ASSERT_TRUE(minimal.Add(0, 1.0, 0));
    ASSERT_TRUE(minimal.Add(7, 1.0, 1));  // one half life after the first sample
    const auto total = minimal.GetTotalWeight(1);
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, 1.5, 1e-6);
}

// The mapping guarantee is a relative error of at most `relative_accuracy`, and
// it has to hold at both ends of the configured range, where the logarithmic
// buckets are coarsest relative to the magnitude. Each sample is queried alone
// so the assertion measures the bucket representative rather than a ranking.
TEST(DecayingDdSketchTest, HoldsRelativeAccuracyAcrossTheIndexedRange) {
    const DecayingDdSketch::Config config = MakeConfig();
    for (int magnitude = -3; magnitude <= 6; ++magnitude) {
        const double value = std::pow(10.0, magnitude);
        DecayingDdSketch sketch(config);
        ASSERT_TRUE(sketch.Add(1, value, 0)) << "value=" << value;

        const auto estimate = sketch.GetQuantile(1.0, 0);
        ASSERT_TRUE(estimate.has_value()) << "value=" << value;
        // Every representative is within (1 - alpha, 1 + alpha) of the values
        // it covers; the extra 1% absorbs double rounding in gamma**key.
        EXPECT_LE(std::abs(*estimate - value) / value,
                  config.relative_accuracy * 1.01)
            << "value=" << value << " estimate=" << *estimate;
    }
}

// Both ends of the indexed range are usable and values above the range are
// rejected; values below it collapse into the zero bucket, which is the only
// honest answer when the sketch has no resolution down there.
TEST(DecayingDdSketchTest, TreatsIndexedRangeBoundariesConsistently) {
    const DecayingDdSketch::Config config = MakeConfig();

    DecayingDdSketch at_max(config);
    ASSERT_TRUE(at_max.Add(1, config.max_indexed_value, 0));
    const auto max_quantile = at_max.GetQuantile(1.0, 0);
    ASSERT_TRUE(max_quantile.has_value());
    ExpectRelativeNear(*max_quantile, config.max_indexed_value, 0.02);
    EXPECT_FALSE(
        at_max.Add(2, std::nextafter(config.max_indexed_value, 1e300), 0))
        << "values above max_indexed_value must be rejected";

    DecayingDdSketch at_min(config);
    ASSERT_TRUE(at_min.Add(1, config.min_indexed_value, 0));
    const auto min_quantile = at_min.GetQuantile(0.5, 0);
    ASSERT_TRUE(min_quantile.has_value());
    EXPECT_GT(*min_quantile, 0.0)
        << "min_indexed_value is the smallest resolvable value";
    ExpectRelativeNear(*min_quantile, config.min_indexed_value, 0.02);

    DecayingDdSketch below_min(config);
    ASSERT_TRUE(below_min.Add(1, config.min_indexed_value / 2.0, 0));
    const auto below_quantile = below_min.GetQuantile(0.5, 0);
    ASSERT_TRUE(below_quantile.has_value());
    EXPECT_DOUBLE_EQ(*below_quantile, 0.0);
}

// A degenerate single-value range is legal and yields exactly two buckets: the
// zero bucket plus the one mapped bucket.
TEST(DecayingDdSketchTest, SupportsDegenerateSingleValueRange) {
    DecayingDdSketch::Config config = MakeConfig();
    config.min_indexed_value = 1.0;
    config.max_indexed_value = 1.0;
    DecayingDdSketch sketch(config);
    EXPECT_EQ(sketch.GetBucketCount(), 2u);

    ASSERT_TRUE(sketch.Add(1, 1.0, 0));
    EXPECT_FALSE(sketch.Add(2, 1.0 + config.relative_accuracy, 0))
        << "anything above the single mapped value is out of range";

    const auto median = sketch.GetQuantile(0.5, 0);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 1.0, config.relative_accuracy * 1.01);
    const auto total = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total.has_value());
    EXPECT_DOUBLE_EQ(*total, 1.0);
}

// Replace allows new_inserted_at == old_inserted_at; only a backwards timestamp
// is invalid.
TEST(DecayingDdSketchTest, ReplaceAcceptsEqualTimestamps) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    EXPECT_TRUE(sketch.Replace(1, 10.0, 0, 20.0, 0));

    const auto total = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, 1.0, 1e-12);

    const auto median = sketch.GetQuantile(0.5, 0);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 20.0, 0.02);
}

// Quantiles are answered in input order and invalid entries are reported as
// nullopt without disturbing their valid neighbours.
TEST(DecayingDdSketchTest, PreservesRequestOrderForMixedQuantiles) {
    DecayingDdSketch sketch(MakeConfig());
    for (std::uint64_t value = 1; value <= 100; ++value) {
        ASSERT_TRUE(sketch.Add(value, static_cast<double>(value), 0));
    }

    const auto quantiles =
        sketch.GetQuantiles({0.5, -0.1, 1.0, 1.1,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()},
                            0);
    ASSERT_EQ(quantiles.size(), 6u);
    ASSERT_TRUE(quantiles[0].has_value());
    ASSERT_TRUE(quantiles[2].has_value());
    ExpectRelativeNear(*quantiles[0], 50.0, 0.02);
    ExpectRelativeNear(*quantiles[2], 100.0, 0.02);
    EXPECT_FALSE(quantiles[1].has_value());
    EXPECT_FALSE(quantiles[3].has_value());
    EXPECT_FALSE(quantiles[4].has_value());
    EXPECT_FALSE(quantiles[5].has_value());
}

// An empty request is answered with an empty result, and an empty sketch
// reports "no observation" instead of fabricating a value.
TEST(DecayingDdSketchTest, EmptyRequestAndEmptySketchAreWellDefined) {
    DecayingDdSketch empty_sketch(MakeConfig());
    EXPECT_TRUE(empty_sketch.GetQuantiles({}, 0).empty());
    const auto none = empty_sketch.GetQuantiles({0.5}, 0);
    ASSERT_EQ(none.size(), 1u);
    EXPECT_FALSE(none[0].has_value());

    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    EXPECT_TRUE(sketch.GetQuantiles({}, 0).empty());
    const auto total = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total.has_value());
    EXPECT_DOUBLE_EQ(*total, 1.0);
}

// Once every sample is older than ~1074 half lives its weight underflows to
// zero: the sketch must report an empty distribution, not NaN or a stale
// value. Timestamps up to the unsigned 64-bit limit must not overflow.
TEST(DecayingDdSketchTest, FullyDecayedSamplesReportNoObservation) {
    DecayingDdSketch::Config config = MakeConfig();
    config.half_life = 100;
    DecayingDdSketch sketch(config);
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));

    constexpr std::uint64_t kFarFuture = 100 * 1'200;  // 1200 half lives
    const auto total = sketch.GetTotalWeight(kFarFuture);
    ASSERT_TRUE(total.has_value());
    EXPECT_DOUBLE_EQ(*total, 0.0);
    EXPECT_FALSE(sketch.GetQuantile(0.5, kFarFuture).has_value());

    constexpr std::uint64_t kMaxTimestamp =
        std::numeric_limits<std::uint64_t>::max();
    const auto max_total = sketch.GetTotalWeight(kMaxTimestamp);
    ASSERT_TRUE(max_total.has_value());
    EXPECT_TRUE(std::isfinite(*max_total));
    EXPECT_DOUBLE_EQ(*max_total, 0.0);
    EXPECT_FALSE(sketch.GetQuantile(0.5, kMaxTimestamp).has_value());
}

// Remove is not guarded by an "is this sample still present" check, so a
// repeated removal drives the bucket negative. That residue must be invisible:
// no negative weight, no negative estimate, and the shard must stay usable.
TEST(DecayingDdSketchTest, RepeatedRemoveLeavesNoNegativeResidue) {
    DecayingDdSketch sketch(MakeConfig());
    ASSERT_TRUE(sketch.Add(1, 10.0, 0));
    ASSERT_TRUE(sketch.Remove(1, 10.0, 0));
    sketch.Remove(1, 10.0, 0);  // second removal: sample is already gone

    const auto total = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total.has_value());
    EXPECT_DOUBLE_EQ(*total, 0.0);
    EXPECT_FALSE(sketch.GetQuantile(0.5, 0).has_value());

    ASSERT_TRUE(sketch.Add(1, 20.0, 0));
    const auto median = sketch.GetQuantile(0.5, 0);
    ASSERT_TRUE(median.has_value());
    ExpectRelativeNear(*median, 20.0, 0.02);
    const auto total_after = sketch.GetTotalWeight(0);
    ASSERT_TRUE(total_after.has_value());
    EXPECT_NEAR(*total_after, 1.0, 1e-12);
}

}  // namespace
}  // namespace mooncake::tool
