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

TEST_P(AssignReferenceClustersTest, ReferenceCheck) {
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
    double quantile = GetParam();
    scran::AssignReferenceClusters runner;
    runner.set_quantile(quantile);
    auto output = runner.run(ndim, nref, ref.data(), clusters.data(), ntest, test.data());

    // Comparing to the reference calculation.
    std::vector<int> expected(ntest);
    for (size_t t = 0; t < ntest; ++t) {
        std::vector<std::vector<double> > distances(nclusters);
        auto tptr = test.data() + t * ndim;

        for (size_t r = 0; r < nref; ++r) {
            auto rptr = ref.data() + r * ndim;
            double dval = 0;
            for (int d = 0; d < ndim; ++d) {
                dval += (tptr[d] - rptr[d]) * (tptr[d] - rptr[d]);
            }
            distances[clusters[r]].push_back(std::sqrt(dval));
        }

        double best = -1;
        int& keep = expected[t];

        for (int c = 0; c < nclusters; ++c) {
            auto& current = distances[c];
            std::sort(current.begin(), current.end());

            double pos = (current.size() - 1) * quantile;
            int left = pos;
            int right = left + 1;
            double frac_left = (right - pos);
            double frac_right = (pos - left);

            double score = (right == current.size() ? current[left] : current[left] * frac_left + current[right] * frac_right);
            if (best < 0 || score < best) {
                best = score;
                keep = c;
            }
        }
    }

    EXPECT_EQ(expected, output);

    // Comparing to multiple threads.
    runner.set_num_threads(3);
    auto poutput = runner.run(ndim, nref, ref.data(), clusters.data(), ntest, test.data());
    EXPECT_EQ(output, poutput);
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
    double quantile = GetParam();
    scran::AssignReferenceClusters runner;
    runner.set_quantile(quantile).set_approximate(true);

    auto output = runner.run(ndim, nref, ref.data(), clusters.data(), 3, test.data());
    std::vector<int> expected { 0, 1, 2 };
    EXPECT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    AssignReferenceClusters,
    AssignReferenceClustersTest,
    ::testing::Values(0, 0.1, 0.2, 0.25) // zero looks for the closest only.
);
