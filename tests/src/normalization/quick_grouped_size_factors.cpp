#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "../utils/compare_almost_equal.h"

#include "tatami/tatami.hpp"
#include "scran/normalization/quick_grouped_size_factors.hpp"

#include <cmath>
#include <random>

class quick_grouped_size_factors_Test : public ::testing::Test {
protected:
    std::shared_ptr<tatami::Matrix<double, int> > ptr;
    size_t NR = 100, NC = 200;
    void simple_assemble() {
        Simulator sim;
        sim.lower = 1;
        sim.upper = 10;
        auto vec = sim.vector(NR * NC);
        ptr.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(vec)));
    }

    size_t num_groups = 10;
    std::vector<double> true_factors;
    void sanity_assemble() {
        Simulator sim;
        sim.lower = 1;
        sim.upper = 10;
        sim.density = 1;
        true_factors = sim.vector(NC);

        sim.seed = 12345;
        auto gene_abundances = sim.vector(NR);
        size_t group_size = NC / num_groups;
        size_t de_per_group = NR / num_groups;

        std::vector<double> vec(NR * NC);
        for (size_t c = 0; c < NC; ++c) {
            size_t offset = c * NR;
            for (size_t r = 0; r < NR; ++r) {
                vec[offset + r] = gene_abundances[r] * true_factors[c];
            }

            size_t group = c / group_size;
            size_t de_offset = group * de_per_group; 
            for (size_t r = 0; r < de_per_group; ++r) {
                vec[offset + de_offset + r] = 10000 * true_factors[c];
            }
        }

        ptr.reset(new tatami::DenseColumnMatrix<double, int>(NR, NC, std::move(vec)));
    }
};

TEST_F(quick_grouped_size_factors_Test, Simple) {
    simple_assemble();

    // Just check that the damn thing runs and gives somewhat sensible output.
    auto out = scran::quick_grouped_size_factors::run(ptr.get());
    EXPECT_EQ(out.size(), NC);

    bool is_positive = true;
    for (auto x : out) {
        if (x <= 0) {
            is_positive = false;
        }
    }
    EXPECT_TRUE(is_positive);

    // Same results with some initial size factors equal to the column sums.
    // We pass an output buffer this time to get some coverage.
    auto sums = tatami::column_sums(ptr.get());
    scran::quick_grouped_size_factors::Options opt;
    opt.initial_factors = sums.data();
    std::vector<double> out2(NC);
    scran::quick_grouped_size_factors::run(ptr.get(), out2.data());
    EXPECT_EQ(out, out2);
}

TEST_F(quick_grouped_size_factors_Test, Blocked) {
    simple_assemble();

    std::vector<int> block;
    block.insert(block.end(), ptr->ncol(), 0);
    block.insert(block.end(), ptr->ncol(), 1);
    auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{ ptr, ptr });

    scran::quick_grouped_size_factors::Options opt;
    opt.block = block.data();
    auto out = scran::quick_grouped_size_factors::run(combined.get(), opt);
    EXPECT_EQ(out.size(), NC * 2);

    // Both halves should be equal to the reference, as the clustering
    // should be the same for each half.
    auto ref = scran::quick_grouped_size_factors::run(ptr.get());
    compare_almost_equal(ref, std::vector<double>(out.begin(), out.begin() + NC));
    compare_almost_equal(ref, std::vector<double>(out.begin() + NC, out.end()));

    // Same results with some initial size factors.
    auto sums = tatami::column_sums(combined.get());
    opt.initial_factors = sums.data();
    auto init = scran::quick_grouped_size_factors::run(ptr.get());
    EXPECT_EQ(ref, init);
}

TEST_F(quick_grouped_size_factors_Test, Sanity) {
    sanity_assemble();

    scran::quick_grouped_size_factors::Options opt;
    opt.clusters = num_groups; // setting it exactly for simplicity in an exact test.
    auto out = scran::quick_grouped_size_factors::run(ptr.get(), opt);

    // Checking that we get a consistent ratio with the true size factor.
    std::vector<double> ratios(NC);
    for (size_t c = 0; c < NC; ++c) {
        ratios[c] = out[c] / true_factors[c];
    }
    auto min_ratio = *std::min_element(ratios.begin(), ratios.end());
    auto max_ratio = *std::max_element(ratios.begin(), ratios.end());
    EXPECT_TRUE(min_ratio > 0);
    EXPECT_TRUE(max_ratio > 0);
    EXPECT_TRUE(std::abs(min_ratio - max_ratio) < 1e-8);

    // Works when blocked.
    std::vector<int> block;
    block.insert(block.end(), ptr->ncol() / 2, 0);
    block.insert(block.end(), ptr->ncol() / 2, 1);
    opt.clusters = num_groups/2; // setting it exactly for simplicity.
    opt.block = block.data();
    auto blocked = scran::quick_grouped_size_factors::run(ptr.get(), opt);
    EXPECT_EQ(out, blocked);
} 
