#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/clustering/BuildSnnGraph.hpp"

#include <random>
#include <cmath>
#include <map>

template<class PARAM>
class BuildSnnGraphTestCore : public ::testing::TestWithParam<PARAM> {
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

    typedef std::tuple<int, int, double> WeightedEdge;

    std::deque<WeightedEdge> convert_to_deque(const scran::BuildSnnGraph::Results& res) {
        std::deque<WeightedEdge> expected;
        for (size_t e = 0; e < res.weights.size(); ++e) {
            expected.emplace_back(res.edges[e*2], res.edges[e*2+1], res.weights[e]);
        }
        return expected;
    }
};

/*****************************************
 *****************************************/

class BuildSnnGraphRefTest : public BuildSnnGraphTestCore<std::tuple<int, int, int, scran::BuildSnnGraph::Scheme, int> > {
protected:
    std::deque<WeightedEdge> reference(size_t ndims, size_t ncells, const double* mat, int k, scran::BuildSnnGraph::Scheme scheme) {
        std::deque<WeightedEdge> output;
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
                        if (scheme == scran::BuildSnnGraph::RANKED) {
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
                    if (scheme == scran::BuildSnnGraph::RANKED) {
                        weight = static_cast<double>(neighbors.size()) - 0.5 * weight;
                    } else if (scheme == scran::BuildSnnGraph::JACCARD) {
                        weight /= (2 * (neighbors.size() + 1) - weight);
                    }
                    output.emplace_back(i, j, std::max(weight, 1e-6));
                }
            }
        }

        return output;
    }
};

TEST_P(BuildSnnGraphRefTest, Reference) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    auto scheme = std::get<3>(param);
    int nthreads = std::get<4>(param);

    scran::BuildSnnGraph builder;
    builder.set_neighbors(k).set_weighting_scheme(scheme).set_num_threads(nthreads);
    auto res = builder.run(ndim, nobs, data.data());

    auto ref = reference(ndim, nobs, data.data(), k, scheme);
    EXPECT_EQ(res.ncells, nobs);
    EXPECT_EQ(res.edges.size(), 2 * ref.size());
    EXPECT_EQ(res.weights.size(), ref.size());

    auto expected = convert_to_deque(res);
    std::sort(expected.begin(), expected.end());
    std::sort(ref.begin(), ref.end());
    EXPECT_EQ(expected, ref);
}

INSTANTIATE_TEST_SUITE_P(
    BuildSnnGraph,
    BuildSnnGraphRefTest,
    ::testing::Combine(
        ::testing::Values(10), // number of dimensions
        ::testing::Values(200), // number of observations
        ::testing::Values(2, 5, 10), // number of neighbors
        ::testing::Values(scran::BuildSnnGraph::RANKED, scran::BuildSnnGraph::NUMBER, scran::BuildSnnGraph::JACCARD), // weighting scheme
        ::testing::Values(1, 3) // number of threads
    )
);

/*****************************************
 *****************************************/

class BuildSnnGraphSymmetryTest : public BuildSnnGraphTestCore<std::tuple<int, int, int, scran::BuildSnnGraph::Scheme> > {};

TEST_P(BuildSnnGraphSymmetryTest, Symmetry) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    auto scheme = std::get<3>(param);

    scran::BuildSnnGraph builder;
    builder.set_neighbors(k).set_weighting_scheme(scheme);

    // Checking for symmetry by flipping the matrix. The idea is that the
    // identities of the first and second nodes are flipped, but if symmetry of
    // the SNN calculations hold, then the original results should be recoverable
    // by just flipping the indices of the edge.
    std::vector<double> revdata(data.rbegin(), data.rend());
    auto revres = builder.run(ndim, nobs, revdata.data());
    auto revd = convert_to_deque(revres);

    for (auto& e : revd) {
        std::get<0>(e) = nobs - std::get<0>(e) - 1;
        std::get<1>(e) = nobs - std::get<1>(e) - 1;
        std::swap(std::get<0>(e), std::get<1>(e));
    }
    std::sort(revd.begin(), revd.end());

    auto refres = builder.run(ndim, nobs, data.data());
    auto refd = convert_to_deque(refres);
    std::sort(refd.begin(), refd.end());
    EXPECT_EQ(refd, revd);
}

INSTANTIATE_TEST_SUITE_P(
    BuildSnnGraph,
    BuildSnnGraphSymmetryTest,
    ::testing::Combine(
        ::testing::Values(10), // number of dimensions
        ::testing::Values(1000), // number of observations
        ::testing::Values(4, 9), // number of neighbors
        ::testing::Values(scran::BuildSnnGraph::RANKED, scran::BuildSnnGraph::NUMBER, scran::BuildSnnGraph::JACCARD) // weighting scheme
    )
);

/*****************************************
 *****************************************/

class BuildSnnGraphAnnoyTest : public BuildSnnGraphTestCore<std::tuple<int, int, int> > {};

TEST_P(BuildSnnGraphAnnoyTest, Annoy) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);

    scran::BuildSnnGraph builder;
    builder.set_neighbors(k).set_approximate(true);

    auto output = builder.run(ndim, nobs, data.data());
    EXPECT_EQ(output.ncells, nobs);
    EXPECT_TRUE(output.weights.size() > 1); // well, it gives _something_, at least.
}

INSTANTIATE_TEST_SUITE_P(
    BuildSnnGraph,
    BuildSnnGraphAnnoyTest,
    ::testing::Combine(
        ::testing::Values(5, 10), // number of dimensions
        ::testing::Values(100, 1000), // number of observations
        ::testing::Values(17, 32) // number of neighbors
    )
);

/*****************************************
 *****************************************/

TEST(IgraphUtils, GraphChecks) {
    std::mt19937_64 rng(42);
    std::normal_distribution distr;
    std::uniform_real_distribution distu;

    scran::BuildSnnGraph::Results res;
    size_t nobs = 1001;
    res.ncells = nobs;

    for (size_t o = 0; o < nobs; ++o) {
        for (size_t o2 = 0; o2 < o; ++o2) {
            if (distu(rng) < 0.2) {
                res.edges.push_back(o);
                res.edges.push_back(o2);
                res.weights.push_back(std::abs(distr(rng)));
            }
        }
    }

    auto g = res.to_igraph();

    // Checking that the graph is valid and all that.
    EXPECT_EQ(igraph_vcount(g.get_graph()), nobs);
    auto nedges = igraph_ecount(g.get_graph());
    EXPECT_EQ(nedges, res.weights.size());
    EXPECT_EQ(nedges, res.edges.size() / 2);

    // Trying copy construction/assignment to get some coverage.
    {
        scran::igraph::Graph g2(g);
        EXPECT_EQ(igraph_vcount(g2.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g2.get_graph()), nedges);

        auto rescopy = res;
        rescopy.edges.resize(res.edges.size() - 2);
        rescopy.weights.resize(res.weights.size() - 1);

        auto g3 = rescopy.to_igraph();
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges - 1);

        g3 = g2;
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges);

        g3 = g3; // self-assign is a no-op.
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges);
    }

    // Trying move construction/assignment to get some coverage.
    {
        scran::igraph::Graph g2(std::move(g));
        EXPECT_EQ(igraph_vcount(g2.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g2.get_graph()), nedges);

        auto rescopy = res;
        rescopy.edges.resize(res.edges.size() - 2);
        rescopy.weights.resize(res.weights.size() - 1);

        auto g3 = rescopy.to_igraph();
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges - 1);

        g3 = std::move(g2);
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges);

        g3 = std::move(g3); // self-move is a no-op.
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_ecount(g3.get_graph()), nedges);
    }
}
