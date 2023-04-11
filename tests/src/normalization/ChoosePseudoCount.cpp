#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"
#include "../utils/compare_almost_equal.h"
#include "scran/normalization/ChoosePseudoCount.hpp"

#include <cmath>
#include <random>

TEST(ChoosePseudoCount, FindQuantile) {
    size_t n = 101;
    std::vector<double> contents(n);
    for (size_t r = 0; r < n; ++r) {
        contents[r] = static_cast<double>(r) / 100;
    }

    {
        auto copy = contents;
        compare_almost_equal(scran::ChoosePseudoCount::find_quantile(0.1, n, copy.data()), 0.1);
        compare_almost_equal(scran::ChoosePseudoCount::find_quantile(0.1111, n, copy.data()), 0.1111);
        compare_almost_equal(scran::ChoosePseudoCount::find_quantile(0.9, n, copy.data()), 0.9);
        compare_almost_equal(scran::ChoosePseudoCount::find_quantile(0.995, n, copy.data()), 0.995);
    }

    // Works as expected.
    scran::ChoosePseudoCount runner;
    auto left = scran::ChoosePseudoCount::find_quantile(0.05, n - 1, contents.data() + 1); // ignore the zero at the start.
    auto right = scran::ChoosePseudoCount::find_quantile(0.95, n - 1, contents.data() + 1);
    compare_almost_equal(runner.run(n, contents.data()), (1/left - 1 /right) / (8 * 0.1));

    runner.set_quantile(0);
    compare_almost_equal(runner.run(n, contents.data()), (1/0.01 - 1/1) / (8 * 0.1));
}

TEST(ChoosePseudoCount, MoreInteresting) {
    std::mt19937_64 rng(1293876123);
    std::uniform_real_distribution dist;
    size_t n = 99;
    std::vector<double> contents(n);
    for (size_t r = 0; r < n; ++r) {
        contents[r] = 0.2 + dist(rng);
    }

    scran::ChoosePseudoCount runner;
    auto out = runner.run(n, contents.data());
    EXPECT_TRUE(out > 1);
    EXPECT_TRUE(out < 5);

    runner.set_min_value(10);
    {
        auto out = runner.run(n, contents.data());
        EXPECT_EQ(out, 10);
    }
    runner.set_min_value();

    runner.set_max_bias(1);
    {
        auto out = runner.run(n, contents.data());
        EXPECT_EQ(out, 1);
    }
}

TEST(ChoosePseudoCount, EdgeCases) {
    std::vector<double> contents;

    scran::ChoosePseudoCount runner;
    EXPECT_EQ(runner.run(0, contents.data()), 1);

    contents.push_back(0);
    EXPECT_EQ(runner.run(1, contents.data()), 1);

    contents.push_back(1);
    EXPECT_EQ(runner.run(2, contents.data()), 1);

    contents.push_back(0.1);
    auto out = runner.run(3, contents.data());
    EXPECT_NE(out, 1);
    auto out2 = runner.run(2, contents.data() + 1);
    EXPECT_EQ(out, out2);
}

