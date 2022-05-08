#include <gtest/gtest.h>
#include "scran/dimensionality_reduction/ScaleByNeighbors.hpp"
#include <vector>
#include <random>

class ScaleByNeighborsTest : public ::testing::Test {
protected:
    std::vector<double> simulate_dense_array(int ndim, size_t nobs, int seed) const {
        std::normal_distribution<> dist;
        std::mt19937_64 rng(seed);
        std::vector<double> vec(nobs * ndim);
        for (auto& v : vec) {
            v = dist(rng);
        }
        return vec;
    }

    double run(const scran::ScaleByNeighbors& runner, size_t nobs, int ndim1, const double* ptr1, int ndim2, const double* ptr2) const {
        auto scale1 = runner.compute_distance(ndim1, nobs, ptr1);
        auto scale2 = runner.compute_distance(ndim2, nobs, ptr2);
        return runner.compute_scale(scale1, scale2);
    }
};

TEST_F(ScaleByNeighborsTest, Basic) {
    size_t nobs = 1234;
    int ndim = 5;
    auto first = simulate_dense_array(ndim, nobs, 1000);
    auto second = first;
    for (auto& s : second) {
        s *= 2;
    }

    scran::ScaleByNeighbors runner;
    auto out = run(runner, nobs, ndim, first.data(), ndim, second.data());
    EXPECT_FLOAT_EQ(out, 0.5);

    // Works for other options.
    runner.set_neighbors(10).set_approximate(false);
    out = run(runner, nobs, ndim, first.data(), ndim, second.data());
    EXPECT_FLOAT_EQ(out, 0.5);
}

TEST_F(ScaleByNeighborsTest, DifferentlyDimensioned) {
    size_t nobs = 1234;
    int ndim = 5;
    auto first = simulate_dense_array(ndim, nobs, 1000);

    std::vector<double> second(ndim*2*nobs);
    auto fIt = first.begin();
    auto sIt = second.begin();
    for (size_t o = 0; o < nobs; ++o) {
        std::copy(fIt, fIt + ndim, sIt);
        sIt += ndim;
        std::copy(fIt, fIt + ndim, sIt);
        fIt += ndim;
        sIt += ndim;
    }

    scran::ScaleByNeighbors runner;
    auto out = run(runner, nobs, ndim, first.data(), ndim * 2, second.data());
    EXPECT_FLOAT_EQ(out, 1 / std::sqrt(2));
}

TEST_F(ScaleByNeighborsTest, Zeros) {
    size_t nobs = 1234;
    int ndim = 5;
    auto first = simulate_dense_array(ndim, nobs, 1000);
    std::vector<double> second(ndim * nobs);

    scran::ScaleByNeighbors runner;

    // Switches to the RMSD.
    {
        std::vector<double> second(ndim * nobs);
        second[0] = 1;
        auto out = run(runner, nobs, ndim, first.data(), ndim, second.data());
        EXPECT_FALSE(std::isinf(out));
        EXPECT_TRUE(out > 0);
    }

    // Falls back to the edge cases.
    {
        std::vector<double> second(ndim * nobs);
        auto out = run(runner, nobs, ndim, first.data(), ndim, second.data());
        EXPECT_TRUE(std::isinf(out));

        auto out2 = run(runner, nobs, ndim, second.data(), ndim, first.data());
        EXPECT_EQ(out2, 0);
    }
}
