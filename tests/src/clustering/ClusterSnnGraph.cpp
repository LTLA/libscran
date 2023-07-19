#include <gtest/gtest.h>

#include "scran/clustering/ClusterSnnGraph.hpp"
#include "aarand/aarand.hpp"

#include <random>
#include <cmath>
#include <map>

class ClusterSnnGraphTestCore {
protected:
    size_t nobs, ndim;
    std::vector<double> data;
    scran::BuildSnnGraph::Results snngraph;

    void assemble(int nd, int no, int nclusters, double scale, int seed) {
        std::mt19937_64 rng(seed);

        ndim = nd;
        nobs = no;
        data.resize(nobs * ndim);

        // Creating gaussian populations with a shift.
        size_t counter = 0;
        for (auto& d : data) {
            d = aarand::standard_normal(rng).first + (counter % nclusters) * scale;
            ++counter;
        }

        scran::BuildSnnGraph builder;
        snngraph = builder.run(ndim, nobs, data.data());
    }

    void assemble(int seed) {
        assemble(7, 505, 4, 1, seed); // arbitrary parameters.
    }
};

/*************************************************************
 *************************************************************/

class ClusterSnnGraphMultiLevelTest : public ClusterSnnGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSnnGraphMultiLevelTest, Basic) {
    assemble(42);

    scran::ClusterSnnGraphMultiLevel cluster;
    auto output = cluster.run(snngraph);

    EXPECT_EQ(output.membership.size(), output.modularity.size());
    EXPECT_TRUE(output.membership.size() > 1);
    EXPECT_TRUE(output.max >= 0 && output.max < output.membership.size());

    double max_mod = output.modularity[output.max];
    for (auto m : output.modularity) {
        EXPECT_TRUE(max_mod >= m);
    }

    for (const auto& mem : output.membership) {
        EXPECT_EQ(mem.size(), nobs);
    }

    // Actually generates multiple clusters.
    bool has_multiple = false;
    for (const auto& mem : output.membership) {
        if (*std::max_element(mem.begin(), mem.end()) > 0) {
            has_multiple = true;
        }
    }
    EXPECT_TRUE(has_multiple);
}

TEST_F(ClusterSnnGraphMultiLevelTest, Parameters) {
    assemble(43);

    scran::ClusterSnnGraphMultiLevel cluster;
    auto ref = cluster.run(snngraph);

    // Checking that the resolution has an effect.
    {
        cluster.set_resolution(2);
        auto out = cluster.run(snngraph);
        cluster.set_resolution();
        EXPECT_NE(ref.membership[ref.max], out.membership[out.max]);
    }
}

TEST_F(ClusterSnnGraphMultiLevelTest, Seeding) {
    assemble(44);

    // We need to drop the number of neighbors to get more ambiguity so
    // that the seed has an effect.
    scran::BuildSnnGraph builder;
    builder.set_neighbors(5);
    auto snngraph2 = builder.run(ndim, nobs, data.data());

    // Using the same seed... 
    scran::ClusterSnnGraphMultiLevel cluster;
    auto ref1 = cluster.run(snngraph2);
    auto ref2 = cluster.run(snngraph2);
    EXPECT_EQ(ref1.modularity, ref2.modularity);
    EXPECT_EQ(ref1.membership, ref2.membership);

    // Using a different seed...
    cluster.set_seed(100);
    auto output2 = cluster.run(snngraph2);
    EXPECT_NE(ref1.membership[ref1.max], output2.membership[output2.max]);
}

/*************************************************************
 *************************************************************/

class ClusterSnnGraphLeidenTest : public ClusterSnnGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSnnGraphLeidenTest, Basic) {
    assemble(66);

    scran::ClusterSnnGraphLeiden cluster;
    auto output = cluster.run(snngraph);

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(*std::max_element(output.membership.begin(), output.membership.end()) > 0); // at least two clusters
    EXPECT_TRUE(output.quality > 0);
}

TEST_F(ClusterSnnGraphLeidenTest, Parameters) {
    assemble(67);

    scran::ClusterSnnGraphLeiden cluster;
    auto ref = cluster.run(snngraph);

    {
        cluster.set_resolution(2);
        auto output2 = cluster.run(snngraph);
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_resolution();
    }

    {
        cluster.set_modularity(true);
        auto output2 = cluster.run(snngraph);
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_modularity();
    }

    // Need to drop this lower so that there's more ambiguity,
    // such that the differences in the parameters may manifest.
    {
        scran::BuildSnnGraph builder;
        builder.set_neighbors(5);
        auto snngraph2 = builder.run(ndim, nobs, data.data());
        auto fine_ref = cluster.run(snngraph2);

        {
            cluster.set_iterations(10);
            auto output2 = cluster.run(snngraph2);
            EXPECT_NE(fine_ref.membership, output2.membership);
            cluster.set_iterations();
        }

        {
            cluster.set_beta(0.5);
            auto output2 = cluster.run(snngraph2);
            EXPECT_NE(fine_ref.membership, output2.membership);
            cluster.set_beta();
        }
    }
}

TEST_F(ClusterSnnGraphLeidenTest, Seeding) {
    assemble(67);

    scran::BuildSnnGraph builder;
    builder.set_neighbors(5);
    auto snngraph2 = builder.run(ndim, nobs, data.data());

    // Using the same seed... 
    scran::ClusterSnnGraphLeiden cluster;
    auto ref1 = cluster.run(snngraph2);
    auto ref2 = cluster.run(snngraph2);
    EXPECT_EQ(ref1.membership, ref2.membership);

    // Using a different seed...
    cluster.set_seed(1000000);
    auto output2 = cluster.run(snngraph2);
    EXPECT_NE(ref1.membership, output2.membership);
}

/*************************************************************
 *************************************************************/

class ClusterSnnGraphWalktrapTest : public ClusterSnnGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSnnGraphWalktrapTest, Basic) {
    assemble(72);

    scran::ClusterSnnGraphWalktrap cluster;
    auto output = cluster.run(snngraph);

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(output.merges.size() > 0);
    EXPECT_EQ(output.merges.size() + 1, output.modularity.size());
    EXPECT_TRUE(*std::max_element(output.membership.begin(), output.membership.end()) > 0); // at least two clusters
}

TEST_F(ClusterSnnGraphWalktrapTest, Parameters) {
    assemble(73);

    scran::ClusterSnnGraphWalktrap cluster;
    auto ref = cluster.run(snngraph);

    {
        cluster.set_steps(8);
        auto output2 = cluster.run(snngraph);
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_steps();
    }

    // Make sure it's reproducible for a range of steps.
    std::vector<int> steps{3, 5, 8};
    for (auto s : steps) {
        cluster.set_steps(s);
        auto ref1 = cluster.run(snngraph);
        auto ref2 = cluster.run(snngraph);
        EXPECT_EQ(ref1.membership, ref2.membership);
    }
}

/*************************************************************
 *************************************************************/

class ClusterSnnGraphSanityTest : public ClusterSnnGraphTestCore, public ::testing::Test {
protected:
    int nclusters =4;

    void SetUp() {
        nobs = 100;
        ndim = 5;

        // Creating 4 gaussian populations of the same size,
        // separated by 10 units in each dimension.
        assemble(ndim, nobs, nclusters, /* scale = */ 10, /* seed = */ 1234567);
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

TEST_F(ClusterSnnGraphSanityTest, MultiLevel) {
    scran::ClusterSnnGraphMultiLevel cluster;
    cluster.set_resolution(0.5);
    auto output = cluster.run(snngraph);
    validate(output.membership[output.max]);
}

TEST_F(ClusterSnnGraphSanityTest, Leiden) {
    scran::ClusterSnnGraphLeiden cluster;
    cluster.set_resolution(0.5);
    auto output = cluster.run(snngraph);
    validate(output.membership);
}

TEST_F(ClusterSnnGraphSanityTest, Walktrap) {
    scran::ClusterSnnGraphWalktrap cluster;
    auto output = cluster.run(snngraph);
    validate(output.membership);
}

