#include <gtest/gtest.h>

#include "scran/clustering/AssignReferenceClusters.hpp"

#include <random>
#include <cmath>
#include <vector>

class AssignReferenceClustersTest : public ::testing::TestWithParam<double> {
protected:
    static std::vector<double> create(int ndim, int nobs, int seed) {
        std::vector<double> data(ndim * nobs); 
        std::mt19937_64 rng(seed);
        std::normal_distribution ndist;
        for (auto& d : data) {
            d = ndist(rng);
        }
        return data;
    }
};

TEST_P(AssignReferenceClustersTest, Basic) {
    int ndim = 8;
    int nref = 501;
    auto ref = create(ndim, nref, ndim * nref / 3);

    // Mocking up the clusters.
    std::mt19937_64 rng(123123);
    std::vector<int> clusters(nref);
    int nclusters = 5;
    for (auto& c : clusters) {
        c = rng() % nclusters;
    }

    // Mocking up the test dataset.
    int ntest = 21;
    auto test = create(ndim, ntest, ndim * ntest / 2);

    // Running the damn thing.
    double k = GetParam();
    scran::AssignReferenceClusters runner;
    runner.set_num_neighbors(k);
    auto output = runner.run(ndim, nref, ref.data(), clusters.data(), ntest, test.data());

    EXPECT_EQ(output.assigned.size(), ntest);
    EXPECT_EQ(output.best_prop.size(), ntest);
    EXPECT_EQ(output.second_prop.size(), ntest);

    for (size_t t = 0; t < ntest; ++t) {
        EXPECT_TRUE(output.assigned[t] >= 0);
        EXPECT_TRUE(output.assigned[t] < nclusters);
        EXPECT_FALSE(output.best_prop[t] < output.second_prop[t]);
        EXPECT_TRUE(output.best_prop[t] > 0);
    }

    // Comparing to the other method.
    knncolle::VpTreeEuclidean<> index(ndim, nref, ref.data());
    std::vector<std::vector<std::pair<int, double> > > collected;
    for (size_t t = 0; t < ntest; ++t) {
        collected.push_back(index.find_nearest_neighbors(test.data() + t * ndim, k));
    }
    auto output2 = runner.run(collected, nref, clusters.data());
    EXPECT_EQ(output.assigned, output2.assigned);
    EXPECT_EQ(output.best_prop, output2.best_prop);
    EXPECT_EQ(output.second_prop, output2.second_prop);

    // Comparing to multiple threads.
    runner.set_num_threads(3);
    auto poutput = runner.run(ndim, nref, ref.data(), clusters.data(), ntest, test.data());
    EXPECT_EQ(output.assigned, poutput.assigned);
    EXPECT_EQ(output.best_prop, poutput.best_prop);
    EXPECT_EQ(output.second_prop, poutput.second_prop);
}

TEST_P(AssignReferenceClustersTest, SanityCheck) {
    int ndim = 7;

    int nref1 = 11;
    auto ref1 = create(ndim, nref1, ndim * nref1 / 3);

    int nref2 = 23;
    auto ref2 = create(ndim, nref2, ndim * nref2 / 3);
    for (auto& r : ref2) {
        r += 5;
    }

    int nref3 = 57;
    auto ref3 = create(ndim, nref3, ndim * nref3 / 3);
    for (auto& r : ref3) {
        r += 10;
    }

    auto ref = ref1;
    int nref = nref1 + nref2 + nref3;
    ref.insert(ref.end(), ref2.begin(), ref2.end());
    ref.insert(ref.end(), ref3.begin(), ref3.end());

    std::vector<int> clusters;
    clusters.insert(clusters.end(), nref1, 0);
    clusters.insert(clusters.end(), nref2, 1);
    clusters.insert(clusters.end(), nref3, 2);

    std::vector<double> test(ndim);
    test.insert(test.end(), ndim, 5);
    test.insert(test.end(), ndim, 10);

    // Actually running it.
    int k = GetParam();
    scran::AssignReferenceClusters runner;
    runner.set_num_neighbors(k).set_approximate(true);

    auto output = runner.run(ndim, nref, ref.data(), clusters.data(), 3, test.data());
    std::vector<int> expected { 0, 1, 2 };
    EXPECT_EQ(output.assigned, expected);

    for (auto p : output.best_prop) {
        EXPECT_TRUE(p > 0.9);
    }
    for (auto p : output.second_prop) {
        EXPECT_TRUE(p < 0.1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AssignReferenceClusters,
    AssignReferenceClustersTest,
    ::testing::Values(1, 5, 10) 
);
