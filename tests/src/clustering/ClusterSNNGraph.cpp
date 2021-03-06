#include <gtest/gtest.h>

#include "scran/clustering/ClusterSNNGraph.hpp"
#include "aarand/aarand.hpp"

#include <random>
#include <cmath>
#include <map>

class ClusterSNNGraphTestCore {
protected:
    size_t nobs, ndim;
    std::vector<double> data;

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
    }

    void assemble(int seed) {
        assemble(7, 505, 4, 1, seed); // arbitrary parameters.
    }
};

/*************************************************************
 *************************************************************/

class ClusterSNNGraphMultiLevelTest : public ClusterSNNGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSNNGraphMultiLevelTest, Basic) {
    assemble(42);

    scran::ClusterSNNGraphMultiLevel cluster;
    auto output = cluster.run(ndim, nobs, data.data());

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

TEST_F(ClusterSNNGraphMultiLevelTest, Parameters) {
    assemble(43);

    scran::ClusterSNNGraphMultiLevel cluster;
    auto ref = cluster.run(ndim, nobs, data.data());

    {
        cluster.set_neighbors(5);
        auto out = cluster.run(ndim, nobs, data.data());
        cluster.set_neighbors();
        EXPECT_NE(ref.membership[ref.max], out.membership[out.max]);
    }

    {
        cluster.set_resolution(2);
        auto out = cluster.run(ndim, nobs, data.data());
        cluster.set_resolution();
        EXPECT_NE(ref.membership[ref.max], out.membership[out.max]);
    }
}

TEST_F(ClusterSNNGraphMultiLevelTest, Seeding) {
    assemble(44);

    // We need to drop the number of neighbors to get more ambiguity so
    // that the seed has an effect.
    scran::ClusterSNNGraphMultiLevel cluster;
    cluster.set_neighbors(5);

    // Using the same seed... 
    auto ref1 = cluster.run(ndim, nobs, data.data());
    auto ref2 = cluster.run(ndim, nobs, data.data());
    EXPECT_EQ(ref1.modularity, ref2.modularity);
    EXPECT_EQ(ref1.membership, ref2.membership);

    // Using a different seed...
    cluster.set_seed(100);
    auto output2 = cluster.run(ndim, nobs, data.data());
    EXPECT_NE(ref1.membership[ref1.max], output2.membership[output2.max]);
}

TEST_F(ClusterSNNGraphMultiLevelTest, GraphChecks) {
    assemble(45);

    // Checking we can build from an existing index.
    scran::ClusterSNNGraphMultiLevel cluster;
    cluster.set_neighbors(5);
    knncolle::VpTreeEuclidean<int, double> vpindex(ndim, nobs, data.data());
    auto g = cluster.build(&vpindex);

    // Checking that the graph is valid and all that.
    EXPECT_EQ(igraph_vcount(g.get_graph()), nobs);
    auto nedges = igraph_ecount(g.get_graph());
    EXPECT_EQ(igraph_vector_size(g.get_weights()), nedges);

    // Trying copy construction to get some coverage.
    {
        scran::ClusterSNNGraph::Graph g2(g);
        EXPECT_EQ(igraph_vcount(g2.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g2.get_weights()), nedges);
    }

    // Trying copy assignment to get some coverage.
    {
        auto g2 = g;
        EXPECT_EQ(igraph_vcount(g2.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g2.get_weights()), nedges);

        auto g3 = cluster.build(ndim, nobs - 1, data.data());
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs - 1);
        g3 = g2;
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g3.get_weights()), nedges);

        g3 = g3; // self-assign is a no-op.
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g3.get_weights()), nedges);
    }

    // Trying move assignment to get some coverage.
    {
        auto g2 = std::move(g);
        EXPECT_EQ(igraph_vcount(g2.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g2.get_weights()), nedges);

        auto g3 = cluster.build(ndim, nobs - 1, data.data());
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs - 1);
        g3 = std::move(g2);
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g3.get_weights()), nedges);

        g3 = std::move(g3); // self-move is a no-op.
        EXPECT_EQ(igraph_vcount(g3.get_graph()), nobs);
        EXPECT_EQ(igraph_vector_size(g3.get_weights()), nedges);
    }
}

/*************************************************************
 *************************************************************/

class ClusterSNNGraphLeidenTest : public ClusterSNNGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSNNGraphLeidenTest, Basic) {
    assemble(66);

    scran::ClusterSNNGraphLeiden cluster;
    auto output = cluster.run(ndim, nobs, data.data());

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(*std::max_element(output.membership.begin(), output.membership.end()) > 0); // at least two clusters
    EXPECT_TRUE(output.quality > 0);
}

TEST_F(ClusterSNNGraphLeidenTest, Parameters) {
    assemble(67);

    scran::ClusterSNNGraphLeiden cluster;
    auto ref = cluster.run(ndim, nobs, data.data());

    {
        cluster.set_neighbors(5);
        auto output = cluster.run(ndim, nobs, data.data());
        EXPECT_NE(ref.membership, output.membership);
        cluster.set_neighbors();
    }

    {
        cluster.set_resolution(2);
        auto output2 = cluster.run(ndim, nobs, data.data());
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_resolution();
    }

    {
        cluster.set_modularity(true);
        auto output2 = cluster.run(ndim, nobs, data.data());
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_modularity();
    }

    // Need to drop this lower so that there's more ambiguity,
    // such that the differences in the parameters may manifest.
    {
        cluster.set_neighbors(5);
        auto fine_ref = cluster.run(ndim, nobs, data.data());

        {
            cluster.set_iterations(10);
            auto output2 = cluster.run(ndim, nobs, data.data());
            EXPECT_NE(fine_ref.membership, output2.membership);
            cluster.set_iterations();
        }

        {
            cluster.set_beta(0.5);
            auto output2 = cluster.run(ndim, nobs, data.data());
            EXPECT_NE(fine_ref.membership, output2.membership);
            cluster.set_iterations();
        }
    }
}

TEST_F(ClusterSNNGraphLeidenTest, Seeding) {
    assemble(67);
    scran::ClusterSNNGraphLeiden cluster;
    cluster.set_neighbors(5);

    // Using the same seed... 
    auto ref1 = cluster.run(ndim, nobs, data.data());
    auto ref2 = cluster.run(ndim, nobs, data.data());
    EXPECT_EQ(ref1.membership, ref2.membership);

    // Using a different seed...
    cluster.set_seed(1000000);
    auto output2 = cluster.run(ndim, nobs, data.data());
    EXPECT_NE(ref1.membership, output2.membership);
}

/*************************************************************
 *************************************************************/

class ClusterSNNGraphWalktrapTest : public ClusterSNNGraphTestCore, public ::testing::Test {};

TEST_F(ClusterSNNGraphWalktrapTest, Basic) {
    assemble(72);

    scran::ClusterSNNGraphWalktrap cluster;
    auto output = cluster.run(ndim, nobs, data.data());

    EXPECT_EQ(output.membership.size(), nobs);
    EXPECT_TRUE(output.merges.size() > 0);
    EXPECT_EQ(output.merges.size() + 1, output.modularity.size());
    EXPECT_TRUE(*std::max_element(output.membership.begin(), output.membership.end()) > 0); // at least two clusters
}

TEST_F(ClusterSNNGraphWalktrapTest, Parameters) {
    assemble(73);

    scran::ClusterSNNGraphWalktrap cluster;
    auto ref = cluster.run(ndim, nobs, data.data());

    {
        cluster.set_steps(8);
        auto output2 = cluster.run(ndim, nobs, data.data());
        EXPECT_NE(ref.membership, output2.membership);
        cluster.set_steps();
    }

    // Make sure it's reproducible for a range of steps.
    std::vector<int> steps{3, 5, 8};
    for (auto s : steps) {
        cluster.set_steps(s);
        auto ref1 = cluster.run(ndim, nobs, data.data());
        auto ref2 = cluster.run(ndim, nobs, data.data());
        EXPECT_EQ(ref1.membership, ref2.membership);
    }
}

/*************************************************************
 *************************************************************/

class ClusterSNNGraphSanityTest : public ClusterSNNGraphTestCore, public ::testing::Test {
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

TEST_F(ClusterSNNGraphSanityTest, MultiLevel) {
    scran::ClusterSNNGraphMultiLevel cluster;
    cluster.set_resolution(0.5);
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership[output.max]);
}

TEST_F(ClusterSNNGraphSanityTest, Leiden) {
    scran::ClusterSNNGraphLeiden cluster;
    cluster.set_resolution(0.5);
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership);
}

TEST_F(ClusterSNNGraphSanityTest, Walktrap) {
    scran::ClusterSNNGraphWalktrap cluster;
    auto output = cluster.run(ndim, nobs, data.data());
    validate(output.membership);
}

