#include <gtest/gtest.h>

#ifdef TEST_SCRAN_CUSTOM_PARALLEL
#include "../utils/custom_parallel.h"
#endif

#include "scran/clustering/BuildSNNGraph.hpp"

#include <random>
#include <cmath>
#include <map>

template<class PARAM>
class BuildSNNGraphTestCore : public ::testing::TestWithParam<PARAM> {
protected:
    size_t nobs, ndim;
    std::vector<double> data;

    void assemble(const PARAM& param) {
        std::mt19937_64 rng(42);
        std::normal_distribution distr;

        ndim = std::get<0>(param);
        nobs = std::get<1>(param);
        data.resize(nobs * ndim);
        for (auto& d : data) {
            d = distr(rng);
        }
    }
};

/*****************************************
 *****************************************/

class BuildSNNGraphRefTest : public BuildSNNGraphTestCore<std::tuple<int, int, int, scran::BuildSNNGraph::Scheme, int> > {
protected:
    std::deque<scran::BuildSNNGraph::WeightedEdge> reference(size_t ndims, size_t ncells, const double* mat, int k, scran::BuildSNNGraph::Scheme scheme) {
        std::deque<scran::BuildSNNGraph::WeightedEdge> output;
        knncolle::VpTreeEuclidean<> vp(ndims, ncells, mat);

        for (size_t i = 0; i < ncells; ++i) {
            auto neighbors = vp.find_nearest_neighbors(i, k);
            std::map<int, double> rankings;
            rankings[i] = 0;
            for (size_t r = 1; r <= neighbors.size(); ++r) {
                rankings[neighbors[r-1].first] = r;
            }

            for (size_t j = 0; j < i; ++j) {
                auto neighbors_of_neighbors = vp.find_nearest_neighbors(j, k);
                double weight = 0;
                bool found = false;

                for (size_t r = 0; r <= neighbors_of_neighbors.size(); ++r) {
                    size_t candidate = (r == 0 ? j : static_cast<size_t>(neighbors_of_neighbors[r-1].first));
                    auto rIt = rankings.find(candidate);
                    if (rIt != rankings.end()) {
                        if (scheme == scran::BuildSNNGraph::RANKED) {
                            double otherrank = rIt->second + r;
                            if (!found || otherrank < weight) {
                                weight = otherrank;
                            }
                        } else {
                            ++weight;
                            found = true;
                        }
                        found = true;
                    }

                }

                if (found) {
                    if (scheme == scran::BuildSNNGraph::RANKED) {
                        weight = static_cast<double>(neighbors.size()) - 0.5 * weight;
                    } else if (scheme == scran::BuildSNNGraph::JACCARD) {
                        weight /= (2 * (neighbors.size() + 1) - weight);
                    }
                    output.push_back(scran::BuildSNNGraph::WeightedEdge(i, j, std::max(weight, 1e-6)));
                }
            }
        }

        return output;
    }
};

TEST_P(BuildSNNGraphRefTest, Reference) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    auto scheme = std::get<3>(param);
    int nthreads = std::get<4>(param);
    
    scran::BuildSNNGraph builder;
    builder.set_neighbors(k).set_weighting_scheme(scheme).set_num_threads(nthreads);
    auto edges = builder.run(ndim, nobs, data.data());

    auto ref = reference(ndim, nobs, data.data(), k, scheme);
    EXPECT_EQ(edges.size(), ref.size());

    std::sort(edges.begin(), edges.end());
    std::sort(ref.begin(), ref.end());
    EXPECT_EQ(edges, ref);
}

INSTANTIATE_TEST_CASE_P(
    BuildSNNGraph,
    BuildSNNGraphRefTest,
    ::testing::Combine(
        ::testing::Values(10), // number of dimensions
        ::testing::Values(200), // number of observations
        ::testing::Values(2, 5, 10), // number of neighbors
        ::testing::Values(scran::BuildSNNGraph::RANKED, scran::BuildSNNGraph::NUMBER, scran::BuildSNNGraph::JACCARD), // weighting scheme
        ::testing::Values(1, 3) // number of threads
    )
);

/*****************************************
 *****************************************/

class BuildSNNGraphSymmetryTest : public BuildSNNGraphTestCore<std::tuple<int, int, int, scran::BuildSNNGraph::Scheme> > {};

TEST_P(BuildSNNGraphSymmetryTest, Symmetry) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    auto scheme = std::get<3>(param);

    scran::BuildSNNGraph builder;
    builder.set_neighbors(k).set_weighting_scheme(scheme);

    // Checking for symmetry by flipping the matrix. The idea is that the
    // identities of the first and second nodes are flipped, but if symmetry of
    // the SNN calculations hold, then the original results should be recoverable
    // by just flipping the indices of the edge.
    std::vector<double> revdata(data.rbegin(), data.rend());
    auto revgraph = builder.run(ndim, nobs, revdata.data());
    for (auto& e : revgraph) {
        std::get<0>(e) = nobs - std::get<0>(e) - 1;
        std::get<1>(e) = nobs - std::get<1>(e) - 1;
        std::swap(std::get<0>(e), std::get<1>(e));
    }
    std::sort(revgraph.begin(), revgraph.end());

    auto refgraph = builder.run(ndim, nobs, data.data());
    std::sort(refgraph.begin(), refgraph.end());
    EXPECT_EQ(revgraph, refgraph);
}

INSTANTIATE_TEST_CASE_P(
    BuildSNNGraph,
    BuildSNNGraphSymmetryTest,
    ::testing::Combine(
        ::testing::Values(10), // number of dimensions
        ::testing::Values(1000), // number of observations
        ::testing::Values(4, 9), // number of neighbors
        ::testing::Values(scran::BuildSNNGraph::RANKED, scran::BuildSNNGraph::NUMBER, scran::BuildSNNGraph::JACCARD) // weighting scheme
    )
);

/*****************************************
 *****************************************/

class BuildSNNGraphAnnoyTest : public BuildSNNGraphTestCore<std::tuple<int, int, int> > {};

TEST_P(BuildSNNGraphAnnoyTest, Annoy) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);

    scran::BuildSNNGraph builder;
    builder.set_neighbors(k).set_approximate(true);

    auto output = builder.run(ndim, nobs, data.data());
    EXPECT_TRUE(output.size() > 1); // well, it gives _something_, at least.
}

INSTANTIATE_TEST_CASE_P(
    BuildSNNGraph,
    BuildSNNGraphAnnoyTest,
    ::testing::Combine(
        ::testing::Values(5, 10), // number of dimensions
        ::testing::Values(100, 1000), // number of observations
        ::testing::Values(17, 32) // number of neighbors
    )
);
