#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/aggregation/DownsampleByNeighbors.hpp"

#include <set>
#include <random>
#include <vector>

class DownsampleByNeighborsTest : public ::testing::Test {
protected:
    std::vector<double> data;

    void fill(int ndim, int nobs) {
        data.resize(ndim * nobs); 
        std::mt19937_64 rng(91872631);
        std::normal_distribution ndist;
        for (auto& d : data) {
            d = ndist(rng);
        }
    }

    void fill_with_crap(int ndim, size_t i, double crap) {
        auto ptr = data.data() + ndim * i;
        std::fill(ptr, ptr + ndim, crap);
    }
};

TEST_F(DownsampleByNeighborsTest, Sanity) {
    int ndim = 5;
    int nobs = 101;

    fill(ndim, nobs);
    fill_with_crap(ndim, 7, 10.1);
    fill_with_crap(ndim, 13, -5.2);

    scran::DownsampleByNeighbors down;
    auto res = down.run(ndim, nobs, data.data());
    EXPECT_TRUE(res.size() < 50); // should see some downsampling.

    // Ensure that our special points are also collected.
    std::set<int> collected(res.begin(), res.end());
    EXPECT_TRUE(collected.find(7) != collected.end());
    EXPECT_TRUE(collected.find(13) != collected.end());

    // Same results in parallel.
    down.set_num_threads(3);
    auto pres = down.run(ndim, nobs, data.data());
    EXPECT_EQ(pres, res);
}

TEST_F(DownsampleByNeighborsTest, Approximate) {
    int ndim = 5;
    int nobs = 1001;

    fill(ndim, nobs);
    fill_with_crap(ndim, 17, 10.1);
    fill_with_crap(ndim, 235, -5.2);

    scran::DownsampleByNeighbors down;
    down.set_approximate(true).set_num_neighbors(50);
    auto res = down.run(ndim, nobs, data.data());
    EXPECT_TRUE(res.size() < 200);

    std::set<int> collected(res.begin(), res.end());
    EXPECT_TRUE(collected.find(17) != collected.end());
    EXPECT_TRUE(collected.find(235) != collected.end());
}
