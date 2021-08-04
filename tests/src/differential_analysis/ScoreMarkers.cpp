#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DelayedBind.hpp"
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

TEST_P(ScoreMarkersTest, Blocked) {
    // Check that everything is more or less computed correctly,
    // by duplicating the matrices and blocking on them.
    auto combined = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_row, dense_row});

    auto NC = dense_row->ncol();
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(NC * 2, ngroups);
    std::vector<int> blocks(groupings.size());
    std::fill(blocks.begin() + NC, blocks.end(), 1);

    scran::ScoreMarkers chd;
    auto comres = chd.run_blocked(combined.get(), groupings.data(), blocks.data());
    std::vector<int> g1(groupings.begin(), groupings.begin() + NC);
    auto res1 = chd.run(dense_row.get(), g1.data());
    std::vector<int> g2(groupings.begin() + NC, groupings.end());
    auto res2 = chd.run(dense_row.get(), g2.data());

    if (NC % ngroups == 0) {
        // Everything should be equal to those in each batch, if the number of cells is a multiple
        // of the number of groups (and thus the `grouping` vector is perfectly recycled).
        for (int l = 0; l < 3; ++l) {
            compare_almost_equal(comres.effects[0][l], res1.effects[0][l]);
            compare_almost_equal(comres.effects[0][l], res2.effects[0][l]);
        }

        compare_almost_equal(comres.means, res1.means);
        compare_almost_equal(comres.means, res2.means);
        compare_almost_equal(comres.detected, res1.detected);
        compare_almost_equal(comres.detected, res2.detected);
    } else {
        // Otherwise, only the means and proportion detected are equal to an analysis without blocking.
        auto blindres = chd.run(combined.get(), groupings.data());
        compare_almost_equal(comres.means, blindres.means);
        compare_almost_equal(comres.detected, blindres.detected);
    }

    // Again, checking consistency across representations.
    auto combined2 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_row, sparse_row});
    auto comres2 = chd.run_blocked(combined2.get(), groupings.data(), blocks.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(comres.effects[0][l], comres2.effects[0][l]);
    }

    auto combined3 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_column, dense_column});
    auto comres3 = chd.run_blocked(combined3.get(), groupings.data(), blocks.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(comres.effects[0][l], comres3.effects[0][l]);
    }

    auto combined4 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_column, sparse_column});
    auto comres4 = chd.run_blocked(combined4.get(), groupings.data(), blocks.data());
    for (int l = 0; l < 3; ++l) {
        compare_almost_equal(comres.effects[0][l], comres4.effects[0][l]);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreMarkers,
    ScoreMarkersTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5) // number of clusters
    )
);

TEST(ScoreMarkers, MinRank) {
    // Checking that the minimum rank is somewhat sensible,
    // and we didn't feed in the wrong values somewhere.
    int ngenes = 10;
    int nsamples = 8;

    std::vector<double> buffer(ngenes * nsamples, 0);
    for (int i = 0; i < ngenes; ++i) {
        buffer[i * nsamples + 1] = i;
        buffer[i * nsamples + 3] = i+1;

        buffer[i * nsamples + 4] = -i;
        buffer[i * nsamples + 6] = -i + 1;
    }

    std::vector<int> grouping{0, 1, 0, 1, 2, 3, 2, 3};
    tatami::DenseRowMatrix<double, int> mat(ngenes, nsamples, std::move(buffer));

    scran::ScoreMarkers chd;
    auto res = chd.run(&mat, grouping.data());

    for (int i = 0; i < ngenes; ++i) {
        EXPECT_EQ(res.effects[0][3][ngenes + i], ngenes - i); // 0 for Cohen's d, 3 for minrank, and then +ngenes to get the first column.
        EXPECT_EQ(res.effects[0][3][ngenes* 2 + i], i + 1); // +ngenes*2 to get the second column.
    }
}
