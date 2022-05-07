#include <gtest/gtest.h>
#include "scran/dimensionality_reduction/ScaleByNeighbors.hpp"
#include <vector>
#include <random>

class ScaleByNeighborsTest : public ::testing::Test {
protected:
    std::vector<double> simulate_dense_array(int ndim, size_t nobs, int seed) {
        std::normal_distribution<> dist;
        std::mt19937_64 rng(seed);
        std::vector<double> vec(nobs * ndim);
        for (auto& v : vec) {
            v = dist(rng);
        }
        return vec;
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
    auto out = runner.run(nobs, ndim, first.data(), ndim, second.data());
    EXPECT_FLOAT_EQ(out, 0.5);

    // Also works for index input.
    knncolle::VpTreeEuclidean<> s1(ndim, nobs, first.data());
    knncolle::VpTreeEuclidean<> s2(ndim, nobs, second.data());
    auto out2 = runner.run(&s1, &s2);
    EXPECT_EQ(out, out2);
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
    auto out = runner.run(nobs, ndim, first.data(), ndim * 2, second.data());
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
        auto out = runner.run(nobs, ndim, first.data(), ndim, second.data());
        EXPECT_FALSE(std::isinf(out));
        EXPECT_TRUE(out > 0);
    }

    // Falls back to the edge cases.
    {
        std::vector<double> second(ndim * nobs);
        auto out = runner.run(nobs, ndim, first.data(), ndim, second.data());
        EXPECT_TRUE(std::isinf(out));

        auto out2 = runner.run(nobs, ndim, second.data(), ndim, first.data());
        EXPECT_EQ(out2, 0);
    }
}
