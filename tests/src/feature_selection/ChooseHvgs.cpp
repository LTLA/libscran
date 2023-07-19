#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/feature_selection/ChooseHvgs.hpp"
#include "aarand/aarand.hpp"
#include <random>

class ChooseHvgsTest : public ::testing::TestWithParam<std::tuple<int, int> > {};

TEST_P(ChooseHvgsTest, Basic) {
    auto p = GetParam();
    size_t ngenes = std::get<0>(p);
    size_t ntop = std::get<1>(p);

    // Setting up the test.
    std::mt19937_64 rng(42);
    std::vector<double> x (ngenes);
    for (auto& x0 : x) { x0 = aarand::standard_uniform(rng); }

    scran::ChooseHvgs chooser;
    chooser.set_top(ntop);
    auto output = chooser.run(ngenes, x.data());

    // Checking that everything is in order.
    EXPECT_EQ(std::min(ngenes, ntop), std::accumulate(output.begin(), output.end(), 0)); 

    double min_has = 100, max_lost = 0;
    for (size_t o = 0; o < ngenes; ++o) {
        if (output[o]) {
            min_has = std::min(min_has, x[o]);
        } else {
            max_lost = std::max(max_lost, x[o]);
        }
    }
    EXPECT_TRUE(min_has > max_lost);
}

INSTANTIATE_TEST_SUITE_P(
    ChooseHvgs,
    ChooseHvgsTest,
    ::testing::Combine(
        ::testing::Values(11, 111, 1111), // number of values
        ::testing::Values(5, 50, 500) // number of tops
    )
);
