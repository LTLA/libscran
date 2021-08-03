#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/ScoreMarkers.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

class ScoreMarkersTest : public ::testing::TestWithParam<std::tuple<int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        dense_row = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        dense_column = tatami::convert_to_dense(dense_row.get(), false);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), true);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), false);
    }

    std::vector<int> create_groupings(size_t n, int ngroups) {
        std::vector<int> groupings(n);
        for (size_t g = 0; g < groupings.size(); ++g) {
            groupings[g] = g % ngroups;
        }
        return groupings;
    }
};

TEST_P(ScoreMarkersTest, Basics) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    auto res = chd.run(dense_row.get(), groupings.data());

    // Running cursory checks on the metrics.
    size_t ngenes = dense_row->nrow();
    for (size_t g = 0; g < ngenes; ++g) {
        std::vector<double> buf(dense_row->ncol());
        auto ptr = dense_row->row(g, buf.data());

        for (int l = 0; l < ngroups; ++l) {
            double curmean = 0, curdetected = 0;
            int count = 0;
            for (int i = l; i < dense_row->ncol(); i += ngroups) {
                curmean += ptr[i];
                curdetected += (ptr[i] > 0);
                ++count;
            }

            curmean /= count;
            EXPECT_EQ(curmean, res.means[g + l * ngenes]);
            curdetected /= count;
            EXPECT_EQ(curdetected, res.detected[g + l * ngenes]);
        }
    }

    // Comparing to other implementations.
    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare_almost_equal(res.means, res2.means);
    compare_almost_equal(res.detected, res2.detected);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare_almost_equal(res.means, res3.means);
    compare_almost_equal(res.detected, res3.detected);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare_almost_equal(res.means, res4.means);
    compare_almost_equal(res.detected, res4.detected);
}

TEST_P(ScoreMarkersTest, CohensD) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    auto res = chd.run(dense_row.get(), groupings.data());

    size_t ngenes = dense_row->nrow();
    for (int l = 0; l < ngroups; ++l) {
        size_t ngenes = dense_row->nrow();

        for (size_t g = 0; g < ngenes; ++g) {
            double curmin = res.effects[0][0][g + l * ngenes];
            double curmean = res.effects[0][1][g + l * ngenes];
            double curmed = res.effects[0][2][g + l * ngenes];
            double currank = res.effects[0][3][g + l * ngenes];

            EXPECT_TRUE(curmin <= curmean);
            EXPECT_TRUE(curmin <= curmed);
            if (curmed > curmin) {
                EXPECT_TRUE(curmean > curmin);
            }
            EXPECT_TRUE(currank >= 1);
            EXPECT_TRUE(currank <= ngenes);
        }
    }

    // Don't compare min-rank here, as minor numerical differences
    // can change the ranks by a small amount when effects are tied.
    auto res2 = chd.run(sparse_row.get(), groupings.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(res.effects[0][l], res2.effects[0][l]);
    }

    auto res3 = chd.run(dense_column.get(), groupings.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(res.effects[0][l], res3.effects[0][l]);
    }

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(res.effects[0][l], res4.effects[0][l]);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreMarkers,
    ScoreMarkersTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4) // number of clusters
    )
);
