#include <gtest/gtest.h>

#include "../utils/macros.h"
#include "scran/dimensionality_reduction/ProjectNeighborEmbedding.hpp"
#include "compare_pcs.h"

#include <vector>
#include <random>

class ProjectNeighborEmbeddingTest : public ::testing::Test {
protected:
    void assemble(int ndim, size_t nref, size_t ntest, int nembed, int seed) { 
        std::normal_distribution<> dist;
        std::mt19937_64 rng(seed);

        src_ref.resize(nref * ndim);
        for (auto& v : src_ref) {
            v = dist(rng);
        }

        src_test.resize(ntest * ndim);
        for (auto& v : src_test) {
            v = dist(rng);
        }

        dest_ref.resize(nref * nembed);
        for (auto& v : dest_ref) {
            v = dist(rng);
        }
    }

    std::vector<double> src_ref, src_test, dest_ref;
};

TEST_F(ProjectNeighborEmbeddingTest, Basic) {
    int ndim = 8;
    size_t nref = 511;
    size_t ntest = 132;
    int nembed = 2;
    assemble(ndim, nref, ntest, nembed, 9999);

    scran::ProjectNeighborEmbedding runner;
    auto output = runner.run(ndim, nref, src_ref.data(), ntest, src_test.data(), nembed, dest_ref.data());

    // Comparing to supplying a neighbor list.
    knncolle::VpTreeEuclidean<int, double> index(ndim, nref, src_ref.data());
    std::vector<std::vector<std::pair<int, double> > > neighbors(ntest);
    for (size_t t = 0; t < ntest; ++t) {
        neighbors[t] = index.find_nearest_neighbors(src_test.data() + t * ndim, scran::ProjectNeighborEmbedding::Defaults::num_neighbors);
    }

    std::vector<double> expected(nembed * ntest);
    runner.run(ndim, nref, src_ref.data(), ntest, src_test.data(), nembed, dest_ref.data(), neighbors, expected.data());
    EXPECT_EQ(output, expected);

    // Same results in parallel.
    runner.set_num_threads(3);
    auto poutput = runner.run(ndim, nref, src_ref.data(), ntest, src_test.data(), nembed, dest_ref.data());
    EXPECT_EQ(output, poutput);
}

TEST_F(ProjectNeighborEmbeddingTest, SanityCheck) {
    int ndim = 7;
    size_t nref = 135;
    size_t ntest = 99;
    int nembed = 2;
    assemble(ndim, nref, ntest, nembed, 9999);

    // Duplicating the reference for easier testing.
    int duplicator = 5;
    std::vector<double> src_ref2, dest_ref2;
    for (int d = 0; d < duplicator; ++d) {
        src_ref2.insert(src_ref2.end(), src_ref.begin(), src_ref.end());
        dest_ref2.insert(dest_ref2.end(), dest_ref.begin(), dest_ref.end());
    }

    scran::ProjectNeighborEmbedding runner;
    auto output = runner.run(ndim, nref * duplicator, src_ref2.data(), ntest, src_test.data(), nembed, dest_ref2.data());

    runner.set_num_neighbors(duplicator);
    auto exact = runner.run(ndim, nref * duplicator, src_ref2.data(), ntest, src_test.data(), nembed, dest_ref2.data());

    // 'exact' should be mapped exactly onto the closest neighbor.
    // Otherwise we should get something that is similar but not exact.
    knncolle::VpTreeEuclidean<int, double> index(ndim, nref, src_ref.data());

    for (size_t t = 0; t < ntest; ++t) {
        auto best = index.find_nearest_neighbors(src_test.data() + t * ndim, 1)[0].first;
        auto expected = dest_ref.data() + best * nembed;
        auto observed = exact.data() + t * nembed;
        auto other = output.data() + t * nembed;

        for (int e = 0; e < nembed; ++e) {
            constexpr double tol = 1e-8;
            EXPECT_TRUE(same_same(expected[e], observed[e], tol));
            EXPECT_TRUE(!same_same(expected[e], other[e], tol)); 
        }
    }
}

TEST_F(ProjectNeighborEmbeddingTest, EdgeCases) {
    int ndim = 7;
    size_t nref = 135;
    size_t ntest = 99;
    int nembed = 2;
    assemble(ndim, nref, ntest, nembed, 9999);

    // No neighbors => NaNs.
    std::vector<std::vector<std::pair<int, double> > > neighbors(ntest);
    std::vector<double> all_nans(nembed * ntest);

    scran::ProjectNeighborEmbedding runner;
    runner.run(ndim, nref, src_ref.data(), ntest, src_test.data(), nembed, dest_ref.data(), neighbors, all_nans.data());
    for (auto x : all_nans) {
        EXPECT_TRUE(std::isnan(x));
    }

    // Exact matches => exact results.
    auto output = runner.run(ndim, nref, src_ref.data(), nref, src_ref.data(), nembed, dest_ref.data());
    for (size_t i = 0; i < nref; ++i) {
        for (int e = 0; e < nembed; ++e) {
            constexpr double tol = 1e-8;
            EXPECT_TRUE(same_same(dest_ref[i * nembed + e], output[i * nembed + e], tol));
        }
    }
}
