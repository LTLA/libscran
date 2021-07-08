#include <gtest/gtest.h>

#include "scran/clustering/BuildSNNGraph.hpp"

#include <random>
#include <cmath>

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

class BuildSNNGraphRefTest : public BuildSNNGraphTestCore<std::tuple<int, int, int, scran::BuildSNNGraph::Scheme> > {
protected:
    std::deque<scran::BuildSNNGraph::WeightedEdge> reference(size_t ndims, size_t ncells, const double* mat, int k, scran::BuildSNNGraph::Scheme scheme) {
        std::deque<scran::BuildSNNGraph::WeightedEdge> output;
        knncolle::VpTreeEuclidean<> vp(ncells, ndims, mat);
        std::vector<knncolle::CellIndex_t> index1, index2;
        std::vector<double> dist1, dist2;

        for (size_t i = 0; i < ncells; ++i) {
            vp.find_nearest_neighbors(i, k, index1, dist1);
            std::map<int, double> rankings;
            rankings[i] = 0;
            for (size_t r = 1; r <= index1.size(); ++r) {
                rankings[index1[r-1]] = r;
            }

            for (size_t j = 0; j < i; ++j) {
                vp.find_nearest_neighbors(j, k, index2, dist2);
                double weight = 0;
                bool found = false;

                for (size_t r = 0; r <= index2.size(); ++r) {
                    size_t candidate = (r == 0 ? j : index2[r-1]);
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
                        weight = static_cast<double>(index1.size()) - 0.5 * weight;
                    } else if (scheme == scran::BuildSNNGraph::JACCARD) {
                        weight /= (2 * (index1.size() + 1) - weight);
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
    
    scran::BuildSNNGraph builder;
    builder.set_neighbors(k).set_weighting_scheme(scheme);
    auto edges = builder.run(ndim, nobs, data.data());

    auto ref = reference(ndim, nobs, data.data(), k, scheme);
    EXPECT_EQ(edges.size(), ref.size());

    // Flattening.
    std::sort(edges.begin(), edges.end());
    std::sort(ref.begin(), ref.end());
    EXPECT_EQ(edges, ref);
}

INSTANTIATE_TEST_CASE_P(
    BuildSNNGraph,
    BuildSNNGraphRefTest,
    ::testing::Combine(
        ::testing::Values(20), // number of dimensions
        ::testing::Values(100), // number of observations
        ::testing::Values(2, 5, 10), // number of neighbors
        ::testing::Values(scran::BuildSNNGraph::RANKED, scran::BuildSNNGraph::NUMBER, scran::BuildSNNGraph::JACCARD) // weighting scheme
    )
);
