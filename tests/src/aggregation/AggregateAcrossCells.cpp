#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/aggregation/AggregateAcrossCells.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"
#include <map>
#include <random>

std::vector<int> create_groupings(size_t n, int ngroups) {
    std::vector<int> groupings(n);
    for (size_t g = 0; g < groupings.size(); ++g) {
        groupings[g] = g % ngroups;
    }
    return groupings;
}

/*********************************************/

class AggregateAcrossCellsTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }
};

TEST_P(AggregateAcrossCellsTest, Basics) {
    auto param = GetParam();
    auto ngroups = std::get<0>(param);
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::AggregateAcrossCells chd;
    auto res = chd.run(dense_row.get(), groupings.data());

    auto nthreads = std::get<1>(param);
    chd.set_num_threads(nthreads);        

    if (nthreads == 1) {
        // Running cursory checks on the metrics.
        size_t ngenes = dense_row->nrow();
        auto ext = dense_row->dense_row();

        for (size_t g = 0; g < ngenes; ++g) {
            std::vector<double> buf(dense_row->ncol());
            auto ptr = ext->fetch(g, buf.data());

            for (int l = 0; l < ngroups; ++l) {
                double cursum = 0, curdetected = 0;
                for (int i = l; i < dense_row->ncol(); i += ngroups) { // repeats in a pattern, see create_groupings.
                    cursum += ptr[i];
                    curdetected += (ptr[i] > 0);
                }

                EXPECT_EQ(cursum, res.sums[l][g]);
                EXPECT_EQ(curdetected, res.detected[l][g]);
            }
        }
    } else {
        // Comparing to the same call, but parallelized.
        auto res1 = chd.run(dense_row.get(), groupings.data());
        for (int l = 0; l < ngroups; ++l) {
            EXPECT_EQ(res.sums[l], res1.sums[l]);
            EXPECT_EQ(res.detected[l], res1.detected[l]);
        }
    }

    // Comparing to other implementations. 
    auto compare = [&](const auto& other) -> void {
        for (int l = 0; l < ngroups; ++l) {
            compare_almost_equal(res.sums[l], other.sums[l]);
            compare_almost_equal(res.detected[l], other.detected[l]);
        }
    };
    
    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare(res2);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare(res3);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare(res4);
}

INSTANTIATE_TEST_SUITE_P(
    AggregateAcrossCells,
    AggregateAcrossCellsTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(1, 3) // number of threads
    )
);

/*********************************************/

TEST(AggregateAcrossCells, Skipping) {
    auto input = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    auto grouping = create_groupings(input->ncol(), 2);

    scran::AggregateAcrossCells runner;
    auto ref = runner.run(input.get(), grouping.data());
    EXPECT_EQ(ref.sums.size(), 2);
    EXPECT_EQ(ref.detected.size(), 2);

    // Skipping works correctly when we don't want to compute things.
    runner.set_compute_sums(false);
    auto partial = runner.run(input.get(), grouping.data());
    EXPECT_EQ(partial.sums.size(), 0);
    EXPECT_EQ(partial.detected.size(), 2);
    
    runner.set_compute_detected(false);
    auto skipped = runner.run(input.get(), grouping.data());
    EXPECT_EQ(skipped.sums.size(), 0);
    EXPECT_EQ(skipped.detected.size(), 0);
}

TEST(CombineFactors, Simple) {
    // Simple sorted case.
    {
        std::vector<int> stuff{0, 0, 1, 1, 1, 2, 2, 2, 2 };
        auto combined = scran::AggregateAcrossCells::combine_factors(stuff.size(), std::vector<const int*>{stuff.data()});
        EXPECT_EQ(combined.second, stuff);

        EXPECT_EQ(combined.first.factors.size(), 1);
        std::vector<int> expected { 0, 1, 2 };
        EXPECT_EQ(combined.first.factors[0], expected);

        std::vector<size_t> counts { 2, 3, 4 };
        EXPECT_EQ(combined.first.counts, counts);
    }

    // Testing the unsorted case.
    {
        std::vector<int> stuff{ 1, 0, 1, 2, 1, 0, 2, 3, 2 };
        auto combined = scran::AggregateAcrossCells::combine_factors(stuff.size(), std::vector<const int*>{stuff.data()});
        EXPECT_EQ(combined.second, stuff);

        EXPECT_EQ(combined.first.factors.size(), 1);
        std::vector<int> expected { 0, 1, 2, 3 };
        EXPECT_EQ(combined.first.factors[0], expected);

        std::vector<size_t> counts { 2, 3, 3, 1 };
        EXPECT_EQ(combined.first.counts, counts);
    }

    // Non-consecutive still works.
    {
        std::vector<int> stuff{ 1, 3, 5, 7, 9 };
        auto combined = scran::AggregateAcrossCells::combine_factors(stuff.size(), std::vector<const int*>{stuff.data()});
        std::vector<int> expected { 0, 1, 2, 3, 4 };
        EXPECT_EQ(combined.second, expected);

        EXPECT_EQ(combined.first.factors.size(), 1);
        std::vector<int> levels { 1, 3, 5, 7, 9 };
        EXPECT_EQ(combined.first.factors[0], levels);

        std::vector<size_t> counts { 1, 1, 1, 1, 1 };
        EXPECT_EQ(combined.first.counts, counts);
    }
}

TEST(CombineFactors, Multiple) {
    {
        std::vector<int> stuff1{ 0, 0, 1, 1, 1, 2, 2, 2, 2 };
        std::vector<int> stuff2{ 0, 1, 2, 0, 1, 2, 0, 1, 2 };
        auto combined = scran::AggregateAcrossCells::combine_factors(stuff1.size(), std::vector<const int*>{stuff1.data(), stuff2.data()});

        std::vector<int> expected { 0, 1, 4, 2, 3, 7, 5, 6, 7 };
        EXPECT_EQ(combined.second, expected);

        EXPECT_EQ(combined.first.factors.size(), 2);
        std::vector<int> levels1 { 0, 0, 1, 1, 1, 2, 2, 2 };
        std::vector<int> levels2 { 0, 1, 0, 1, 2, 0, 1, 2 };
        EXPECT_EQ(combined.first.factors[0], levels1);
        EXPECT_EQ(combined.first.factors[1], levels2);

        std::vector<size_t> counts { 1, 1, 1, 1, 1, 1, 1, 2 };
        EXPECT_EQ(combined.first.counts, counts);
    }

    {
        std::map<std::pair<int, int>, std::vector<int> > collected;
        std::vector<int> stuff1;
        std::vector<int> stuff2;
        std::mt19937_64 rng(1000);

        int choice1 = 13, choice2 = 19;
        for (size_t i = 0; i < 100; ++i) {
            stuff1.push_back(rng() % choice1);
            stuff2.push_back(rng() % choice2);
            auto& current = collected[std::make_pair(stuff1.back(), stuff2.back())];
            current.push_back(i);
        }

        auto combined = scran::AggregateAcrossCells::combine_factors(stuff1.size(), std::vector<const int*>{stuff1.data(), stuff2.data()});

        // Reference calculation.
        std::vector<int> expected(stuff1.size());
        std::vector<int> factor1, factor2;
        std::vector<size_t> counts;

        size_t counter = 0;
        for (const auto& p : collected) {
            factor1.push_back(p.first.first);
            factor2.push_back(p.first.second);
            counts.push_back(p.second.size());

            for (auto i : p.second) {
                expected[i] = counter;
            }
            ++counter;
        }

        EXPECT_EQ(expected, combined.second);
        EXPECT_EQ(factor1, combined.first.factors[0]);
        EXPECT_EQ(factor2, combined.first.factors[1]);
        EXPECT_EQ(counts, combined.first.counts);
    }
}

