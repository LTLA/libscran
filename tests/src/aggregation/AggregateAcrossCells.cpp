#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/aggregation/AggregateAcrossCells.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"
#include <map>
#include <random>

class AggregateAcrossCellsTest : public ::testing::TestWithParam<int> {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }

    std::vector<int> create_groupings(size_t n, int ngroups) {
        std::vector<int> groupings(n);
        for (size_t g = 0; g < groupings.size(); ++g) {
            groupings[g] = g % ngroups;
        }
        return groupings;
    }
};

TEST_P(AggregateAcrossCellsTest, Basics) {
    auto ngroups = GetParam();
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::AggregateAcrossCells chd;
    auto res = chd.run(dense_row.get(), groupings.data());

    // Running cursory checks on the metrics.
    size_t ngenes = dense_row->nrow();
    for (size_t g = 0; g < ngenes; ++g) {
        std::vector<double> buf(dense_row->ncol());
        auto ptr = dense_row->row(g, buf.data());

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

INSTANTIATE_TEST_CASE_P(
    AggregateAcrossCells,
    AggregateAcrossCellsTest,
    ::testing::Values(2, 3, 4, 5) // number of clusters
);

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

