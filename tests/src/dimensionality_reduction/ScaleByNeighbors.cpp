#include <gtest/gtest.h>
#include "../utils/macros.h"
#include "scran/dimensionality_reduction/ScaleByNeighbors.hpp"
#include <vector>
#include <random>

class ScaleByNeighborsTestCore {
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
        return scran::ScaleByNeighbors::compute_scale(scale1, scale2);
    }
};

/*********************************************/

class ScaleByNeighborsTest : public ::testing::TestWithParam<bool>, public ScaleByNeighborsTestCore {};

TEST_P(ScaleByNeighborsTest, Basic) {
    size_t nobs = 1234;
    int ndim = 5;
    auto first = simulate_dense_array(ndim, nobs, 1000);
    auto second = first;
    for (auto& s : second) {
        s *= 2;
    }

    scran::ScaleByNeighbors runner;
    runner.set_num_threads(GetParam());
    auto out = run(runner, nobs, ndim, first.data(), ndim, second.data());
    EXPECT_FLOAT_EQ(out, 0.5);

    // Works for other options.
    runner.set_neighbors(10).set_approximate(false);
    out = run(runner, nobs, ndim, first.data(), ndim, second.data());
    EXPECT_FLOAT_EQ(out, 0.5);
}

TEST_P(ScaleByNeighborsTest, DifferentlyDimensioned) {
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
    runner.set_num_threads(GetParam());
    auto out = run(runner, nobs, ndim, first.data(), ndim * 2, second.data());
    EXPECT_FLOAT_EQ(out, 1 / std::sqrt(2));
}

TEST_P(ScaleByNeighborsTest, Zeros) {
    size_t nobs = 1234;
    int ndim = 5;
    auto first = simulate_dense_array(ndim, nobs, 1000);
    std::vector<double> second(ndim * nobs);

    scran::ScaleByNeighbors runner;
    runner.set_num_threads(GetParam());

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

INSTANTIATE_TEST_SUITE_P(
    ScaleByNeighbors,
    ScaleByNeighborsTest,
    ::testing::Values(1, 3) // number of threads
);

/*********************************************/

class ScaleByNeighborsUtilsTest : public ::testing::Test, public ScaleByNeighborsTestCore {};

TEST_F(ScaleByNeighborsUtilsTest, ComputeDistances) {
    {
        std::vector<std::pair<double, double> > distances{ {3, 3}, { 2, 2 }, { 1, 1 } };
        auto output = scran::ScaleByNeighbors::compute_scale(distances);
        std::vector<double> expected { 1, 1.5, 3 };
        EXPECT_EQ(output, expected);
    }

    // Skips the first.
    {
        std::vector<std::pair<double, double> > distances{ { 0, 0 }, { 10, 10 }, { 1, 1 } };
        auto output = scran::ScaleByNeighbors::compute_scale(distances);
        EXPECT_TRUE(std::isinf(output[0]));
        EXPECT_EQ(output[1], 1);
        EXPECT_EQ(output[2], 10);
    }

    // Skips all of them.
    {
        std::vector<std::pair<double, double> > distances(3);
        auto output = scran::ScaleByNeighbors::compute_scale(distances);
        std::vector<double> expected(3);
        EXPECT_EQ(output, expected);

    }
}

TEST_F(ScaleByNeighborsUtilsTest, CombineEmbeddings) {
    size_t nobs = 123;
    auto first = simulate_dense_array(20, nobs, 1000);
    auto second = simulate_dense_array(5, nobs, 2000);

    {
        std::vector<double> output(25 * nobs);
        scran::ScaleByNeighbors::combine_scaled_embeddings({ 20, 5 }, nobs, std::vector<const double*>{ first.data(), second.data() }, { 0.5, 1.2 }, output.data());

        // Interleaving is done correctly.
        EXPECT_EQ(output[0], first[0] * 0.5);
        EXPECT_EQ(output[19], first[19] * 0.5);
        EXPECT_EQ(output[25], first[20] * 0.5);
        EXPECT_EQ(output[25 * (nobs - 1)], first[20 * (nobs - 1)] * 0.5);
        EXPECT_EQ(output[25 * (nobs - 1) + 19], first[20 * nobs - 1] * 0.5);

        EXPECT_EQ(output[20], second[0] * 1.2);
        EXPECT_EQ(output[24], second[4] * 1.2);
        EXPECT_EQ(output[45], second[5] * 1.2);
        EXPECT_EQ(output[25 * (nobs - 1) + 20], second[5 * (nobs - 1)] * 1.2);
        EXPECT_EQ(output[25 * nobs - 1], second[5 * nobs - 1] * 1.2);
    }

    // Handles the infinite special case.
    {
        std::vector<double> output(25 * nobs);
        scran::ScaleByNeighbors::combine_scaled_embeddings({ 20, 5 }, nobs, std::vector<const double*>{ first.data(), second.data() }, { 0.5, std::numeric_limits<double>::infinity() }, output.data());

        // Interleaving is done correctly.
        EXPECT_EQ(output[0], first[0] * 0.5);
        EXPECT_EQ(output[19], first[19] * 0.5);
        EXPECT_EQ(output[25], first[20] * 0.5);
        EXPECT_EQ(output[25 * (nobs - 1)], first[20 * (nobs - 1)] * 0.5);
        EXPECT_EQ(output[25 * (nobs - 1) + 19], first[20 * nobs - 1] * 0.5);

        EXPECT_EQ(output[20], 0);
        EXPECT_EQ(output[24], 0);
        EXPECT_EQ(output[45], 0);
        EXPECT_EQ(output[25 * (nobs - 1) + 20], 0);
        EXPECT_EQ(output[25 * nobs - 1], 0);
    }
}
