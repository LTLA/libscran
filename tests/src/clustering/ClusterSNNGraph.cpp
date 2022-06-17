#include <gtest/gtest.h>

#include "scran/clustering/ClusterSNNGraph.hpp"

#include <random>
#include <cmath>
#include <map>

template<class PARAM>
class ClusterSNNGraphTestCore : public ::testing::TestWithParam<PARAM> {
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

/*************************************************************
 *************************************************************/

using ClusterSNNGraphMultiLevelTest = ClusterSNNGraphTestCore<std::tuple<int, int, int, double> >;

TEST_P(ClusterSNNGraphMultiLevelTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    double res = std::get<3>(param);
    
    scran::ClusterSNNGraphMultiLevel cluster;
    cluster.set_neighbors(k).set_resolution(res);
    auto output = cluster.run(ndim, nobs, data.data());

    EXPECT_EQ(output.membership.size(), output.modularity.size());
    EXPECT_TRUE(output.max >= 0 && output.max < output.membership.size());

    double max_mod = output.modularity[output.max];
    for (auto m : output.modularity) {
        EXPECT_TRUE(max_mod >= m);
    }

    for (const auto& mem : output.membership) {
        EXPECT_EQ(mem.size(), nobs);
    }
}

INSTANTIATE_TEST_CASE_P(
    ClusterSNNGraph,
    ClusterSNNGraphMultiLevelTest,
    ::testing::Combine(
        ::testing::Values(20), // number of dimensions
        ::testing::Values(200), // number of observations
        ::testing::Values(3, 5, 7), // number of neighbors
        ::testing::Values(0.1, 0.5, 1) // resolution 
    )
);

/*************************************************************
 *************************************************************/

using ClusterSNNGraphLeidenTest = ClusterSNNGraphTestCore<std::tuple<int, int, int, double, bool> >;

TEST_P(ClusterSNNGraphLeidenTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    double res = std::get<3>(param);
    bool mod = std::get<4>(param);
    
    scran::ClusterSNNGraphLeiden cluster;
    cluster.set_neighbors(k).set_modularity(mod);
    cluster.set_resolution(mod ? res * 10 : res); // ramping up scale of resolution when optimizing modularity.
    auto output = cluster.run(ndim, nobs, data.data());

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(output.quality > 0);
}

INSTANTIATE_TEST_CASE_P(
    ClusterSNNGraph,
    ClusterSNNGraphLeidenTest,
    ::testing::Combine(
        ::testing::Values(20), // number of dimensions
        ::testing::Values(200), // number of observations
        ::testing::Values(3, 5, 7), // number of neighbors
        ::testing::Values(0.05, 0.1), // resolution 
        ::testing::Values(false, true) // whether to use modularity
    )
);

/*************************************************************
 *************************************************************/

using ClusterSNNGraphWalktrapTest = ClusterSNNGraphTestCore<std::tuple<int, int, int, double> >;

TEST_P(ClusterSNNGraphWalktrapTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);
    double steps = std::get<3>(param);
    
    scran::ClusterSNNGraphWalktrap cluster;
    cluster.set_neighbors(k).set_steps(steps);
    auto output = cluster.run(ndim, nobs, data.data());

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(output.merges.size() > 0);
    EXPECT_EQ(output.merges.size() + 1, output.modularity.size());
}

INSTANTIATE_TEST_CASE_P(
    ClusterSNNGraph,
    ClusterSNNGraphWalktrapTest,
    ::testing::Combine(
        ::testing::Values(20), // number of dimensions
        ::testing::Values(200), // number of observations
        ::testing::Values(3, 5, 7), // number of neighbors
        ::testing::Values(3, 4) // steps
    )
);

/*************************************************************
 *************************************************************/

class ClusterSNNGraphSanityTest : public ::testing::Test {
protected:
    size_t nobs = 100, ndim = 5, nclusters =4;
    std::vector<double> data;

    void SetUp() {
        std::mt19937_64 rng(420);
        std::normal_distribution distr;
        data.resize(nobs * ndim);

        // Creating 4 gaussian populations of the same size,
        // separated by 10 units in each dimension.
        size_t counter = 0;
        for (auto& d : data) {
            d = distr(rng) + (counter % nclusters) * 10;
            ++counter;
        }
    }

    void validate(const std::vector<int>& clustering) {
        // Each cluster should be of length > 1 and should only contain cells from the same modulo.
        std::map<int, std::vector<int> > by_clusters;
        for (size_t c = 0; c < clustering.size(); ++c) {
            by_clusters[clustering[c]].push_back(c);
        }

        ASSERT_TRUE(by_clusters.size() >= nclusters);
        for (const auto& clust : by_clusters) {
            ASSERT_TRUE(clust.second.size() > 1);

            int first_modulo = clust.second[0] % nclusters;
            for (auto c : clust.second) {
                ASSERT_EQ(c % nclusters, first_modulo);
            }
        }
    }
};

TEST_F(ClusterSNNGraphSanityTest, MultiLevel) {
    scran::ClusterSNNGraphMultiLevel cluster;
    cluster.set_resolution(0.5);
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership[output.max]);
}

TEST_F(ClusterSNNGraphSanityTest, Leiden) {
    scran::ClusterSNNGraphLeiden cluster;
    cluster.set_resolution(0.01);
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership);
}

TEST_F(ClusterSNNGraphSanityTest, Walktrap) {
    scran::ClusterSNNGraphWalktrap cluster;
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership);
}

