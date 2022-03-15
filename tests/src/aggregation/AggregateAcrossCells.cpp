#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/aggregation/AggregateAcrossCells.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

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
