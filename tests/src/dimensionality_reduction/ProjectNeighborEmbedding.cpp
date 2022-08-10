#include <gtest/gtest.h>
#include "../utils/macros.h"
#include "scran/dimensionality_reduction/ProjectNeighborEmbedding.hpp"
#include <vector>
#include <random>

class ProjectNeighborEmbeddingTest : public ::testing::Test {
protected:
    std::vector<double> simulate_dense_array(int ndim, size_t nobs, int seed) const {
        return vec;
    }
};

TEST(ProjectNeighborEmbeddingTest, Basic) {
    std::vector<std::vector<std::pair<int, double> > > neighbors(1); // empty first neighbor.
    neighbors.emplace_back(std::vector<std::pair<int, double> >{ { 0, 1.5 } }); // 1 neighbor
    neighbors.emplace_back(std::vector<std::pair<int, double> >{ { 2, 0.5 }, { 1, 3 } }); // 2 neighbor
    neighbors.emplace_back(std::vector<std::pair<int, double> >{ { 1, 0.5 }, { 2, 0.5 } }); // all neighbors equi-distant.

    int ndim = 2;
    std::vector<double> destination { -1, -1, 1, 1, 2, 2 };
    std::vector<double> output(neighbors.size() * 2);

    scran::ProjectNeighborEmbedding runner;
    runner.run(neighbors, ndim, destination.data(), output.data());

    // Checking special cases.
    for (int d = 0; d < ndim; ++d) {
        EXPECT_TRUE(std::isnan(output[d]));
        EXPECT_EQ(output[2 + d], -1);
        EXPECT_TRUE(1.5 < output[4 + d] && output[4 + d] < 2); // between 1 and 2, but leaning towards 2.
        EXPECT_EQ(output[6 + d], 1.5);
    }
}

TEST(ProjectNeighborEmbeddingTest, Outliers) {
    std::vector<std::vector<std::pair<int, double> > > neighbors;
    neighbors.emplace_back(std::vector<std::pair<int, double> >{ { 1, 0.5 }, { 0, 0.6 }, { 2, 0.65 }, { 3, 10 }}); // obvious outlier here.

    int ndim = 2;
    std::vector<double> destination { -1, -1, 1, 1, 2, 2, -2, -2 };
    std::vector<double> output(neighbors.size() * 2);

    scran::ProjectNeighborEmbedding runner;
    runner.run(neighbors, ndim, destination.data(), output.data());

    // Around about the average of -1, 1 and 2; point 3 is excluded here.
    for (int d = 0; d < ndim; ++d) {
        EXPECT_TRUE(0.6 < output[d] && output[d] < 0.8);
    }
}

TEST(ProjectNeighborEmbeddingTest, OtherInputs) {
    // Creating the source.
    int ndim = 7;
    size_t nref = 510;

    std::normal_distribution<> dist;
    std::mt19937_64 rng(9999);
    std::vector<double> src_ref(nref * ndim);
    for (auto& v : vec) {
        v = dist(rng);
    }

    size_t ntest = 12;
    std::vector<double> src_test(ntest * ndim);
    for (auto& v : vec) {
        v = dist(rng);
    }

    int nembed = 2;
    std::vector<double> dest_ref(nref * nembed);
    for (auto& v : vec) {
        v = dist(rng);
    }
}

