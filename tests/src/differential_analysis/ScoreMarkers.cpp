#include <gtest/gtest.h>

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DelayedBind.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"
#include "scran/differential_analysis/ScoreMarkers.hpp"

#include "../data/data.h"
#include "../utils/compare_almost_equal.h"

class ScoreMarkersTest : public ::testing::TestWithParam<std::tuple<int, bool> > {
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
    chd.set_compute_auc(std::get<1>(GetParam())); // false, if we want to check the running implementations.
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
            EXPECT_EQ(curmean, res.means[l][0][g]);
            curdetected /= count;
            EXPECT_EQ(curdetected, res.detected[l][0][g]);
        }
    }

    // Comparing to other implementations. 
    auto compare = [&](const auto& other) -> void {
        for (int l = 0; l < ngroups; ++l) {
            EXPECT_EQ(other.means[l].size(), 1);
            EXPECT_EQ(other.detected[l].size(), 1);
            compare_almost_equal(res.means[l][0], other.means[l][0]);
            compare_almost_equal(res.detected[l][0], other.detected[l][0]);
        }
    };

    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare(res2);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare(res3);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare(res4);
}

TEST_P(ScoreMarkersTest, Effects) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(dense_row.get(), groupings.data());

    size_t ngenes = dense_row->nrow();
    for (int l = 0; l < ngroups; ++l) {
        size_t ngenes = dense_row->nrow();

        for (size_t g = 0; g < ngenes; ++g) {
            double curmin = res.cohen[0][l][g];
            double curmean = res.cohen[1][l][g];
            double curmed = res.cohen[2][l][g];
            double curmax = res.cohen[3][l][g];
            double currank = res.cohen[4][l][g];

            EXPECT_TRUE(curmin <= curmean);
            EXPECT_TRUE(curmin <= curmed);
            EXPECT_TRUE(curmax >= curmean);
            EXPECT_TRUE(curmax >= curmed);

            if (curmed > curmin) {
                EXPECT_TRUE(curmean > curmin);
            }
            if (curmed < curmax) {
                EXPECT_TRUE(curmean < curmax);
            }

            EXPECT_TRUE(currank >= 1);
            EXPECT_TRUE(currank <= ngenes);
        }

        if (do_auc) {
            for (size_t g = 0; g < ngenes; ++g) {
                double curmin = res.auc[0][l][g];
                double curmean = res.auc[1][l][g];
                double curmed = res.auc[2][l][g];
                double curmax = res.auc[3][l][g];
                double currank = res.auc[4][l][g];

                EXPECT_TRUE(curmin >= 0);
                EXPECT_TRUE(curmin <= curmean);
                EXPECT_TRUE(curmin <= curmed);
                EXPECT_TRUE(curmax >= curmean);
                EXPECT_TRUE(curmax >= curmed);
                EXPECT_TRUE(curmax <= 1);

                if (curmed > curmin) {
                    EXPECT_TRUE(curmean > curmin);
                }
                if (curmed < curmax) {
                    EXPECT_TRUE(curmean < curmax);
                }

                EXPECT_TRUE(currank >= 1);
                EXPECT_TRUE(currank <= ngenes);
            }
        }
    }

    auto compare = [&](const auto& other) -> void {
        // Don't compare min-rank here, as minor numerical differences
        // can change the ranks by a small amount when effects are tied.
        for (int s = 0; s < 4; ++s) {
            EXPECT_EQ(res.cohen[s].size(), other.cohen[s].size());
            for (int l = 0; l < ngroups; ++l) {
                compare_almost_equal(res.cohen[s][l], other.cohen[s][l]);
            }
        }
    };

    auto res2 = chd.run(sparse_row.get(), groupings.data());
    compare(res2);

    auto res3 = chd.run(dense_column.get(), groupings.data());
    compare(res3);

    auto res4 = chd.run(sparse_column.get(), groupings.data());
    compare(res4);

    if (do_auc) {
        auto compare_auc = [&](const auto& other) -> void {
            // Don't compare min-rank here, as minor numerical differences
            // can change the ranks by a small amount when effects are tied.
            for (int s = 0; s < 4; ++s) {
                EXPECT_EQ(res.auc[s].size(), other.auc[s].size());
                for (int l = 0; l < ngroups; ++l) {
                    compare_almost_equal(res.auc[s][l], other.auc[s][l]);
                }
            }
        };

        compare_auc(res2);
        compare_auc(res3);
        compare_auc(res4);
    } else {
        EXPECT_EQ(res.auc.size(), 0);
    }
}

TEST_P(ScoreMarkersTest, Self) {
    // Replicating the same matrix 'ngroups' times.
    int ngroups = std::get<0>(GetParam());
    std::vector<std::shared_ptr<tatami::NumericMatrix> > stuff;
    for (int i = 0; i < ngroups; ++i) {
        stuff.push_back(dense_row);
    }
    auto combined = tatami::make_DelayedBind<1>(std::move(stuff));

    // Creating two groups; second group can be larger than the first, to check
    // for correct behavior w.r.t. imbalanced groups.
    size_t NC = dense_row->ncol();
    std::vector<int> groupings(NC * ngroups);
    std::fill(groupings.begin(), groupings.begin() + NC, 0);
    std::fill(groupings.begin() + NC, groupings.end(), 1); 

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto res = chd.run(combined.get(), groupings.data());

    // All AUCs should be 0.5, all Cohen's should be 0.
    size_t ngenes = dense_row->nrow();
    for (int l = 0; l < 2; ++l) {
        size_t ngenes = dense_row->nrow();

        for (size_t g = 0; g < ngenes; ++g) {
            if (do_auc) {
                EXPECT_EQ(res.auc[0][l][g], 0.5);
                EXPECT_EQ(res.auc[3][l][g], 0.5);
            }

            // Handle some numerical imprecision...
            EXPECT_TRUE(std::abs(res.cohen[0][l][g]) < 1e-10);
            EXPECT_TRUE(std::abs(res.cohen[3][l][g]) < 1e-10);
        }
    }
}

TEST_P(ScoreMarkersTest, Thresholds) {
    auto ngroups = std::get<0>(GetParam());
    std::vector<int> groupings = create_groupings(dense_row->ncol(), ngroups);

    scran::ScoreMarkers chd;
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto ref = chd.run(dense_row.get(), groupings.data());
    auto out = chd.set_threshold(1).run(dense_row.get(), groupings.data());

    bool some_diff = false;
    for (int s = 0; s < 4; ++s) {
        for (int l = 0; l < ngroups; ++l) {
            for (size_t g = 0; g < dense_row->nrow(); ++g) {
                EXPECT_TRUE(ref.cohen[s][l][g] > out.cohen[s][l][g]);

                if (do_auc) {
                    // '>' is not guaranteed due to imprecision with ranks
                    some_diff |= (ref.auc[s][l][g] > out.auc[s][l][g]);
                    EXPECT_TRUE(ref.auc[s][l][g] >= out.auc[s][l][g]); 
                }
            }
        }
    }

    if (do_auc) {
        EXPECT_TRUE(some_diff); // but at least one is '>', hopefully.
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
    bool do_auc = std::get<1>(GetParam());
    chd.set_compute_auc(do_auc); // false, if we want to check the running implementations.
    auto comres = chd.run_blocked(combined.get(), groupings.data(), blocks.data());

    std::vector<int> g1(groupings.begin(), groupings.begin() + NC);
    auto res1 = chd.run(dense_row.get(), g1.data());
    std::vector<int> g2(groupings.begin() + NC, groupings.end());
    auto res2 = chd.run(dense_row.get(), g2.data());

    if (NC % ngroups == 0) {
        // Everything should be equal to those in each batch, if the number of cells is a multiple
        // of the number of groups (and thus the `grouping` vector is perfectly recycled).
        for (int s = 0; s < 4; ++s) {
            for (int l = 0; l < ngroups; ++l) {
                compare_almost_equal(comres.cohen[s][l], res1.cohen[s][l]);
                compare_almost_equal(comres.cohen[s][l], res2.cohen[s][l]);

                if (do_auc) {
                    compare_almost_equal(comres.auc[s][l], res1.auc[s][l]);
                    compare_almost_equal(comres.auc[s][l], res2.auc[s][l]);
                }
            }
        }
    }

    for (int l = 0; l < ngroups; ++l) {
        compare_almost_equal(comres.means[l][0], res1.means[l][0]);
        compare_almost_equal(comres.detected[l][0], res1.detected[l][0]);
        compare_almost_equal(comres.means[l][1], res2.means[l][0]);
        compare_almost_equal(comres.detected[l][1], res2.detected[l][0]);
    }

    // Again, checking consistency across representations.
    auto compare = [&](const auto& other) -> void {
        for (int s = 0; s < 4; ++s) {
            for (int l = 0; l < ngroups; ++l) {
                compare_almost_equal(comres.cohen[s][l], other.cohen[s][l]);
            }
        }
    };

    auto combined2 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_row, sparse_row});
    auto comres2 = chd.run_blocked(combined2.get(), groupings.data(), blocks.data());
    compare(comres2);

    auto combined3 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{dense_column, dense_column});
    auto comres3 = chd.run_blocked(combined3.get(), groupings.data(), blocks.data());
    compare(comres3);

    auto combined4 = tatami::make_DelayedBind<1>(std::vector<std::shared_ptr<tatami::NumericMatrix> >{sparse_column, sparse_column});
    auto comres4 = chd.run_blocked(combined4.get(), groupings.data(), blocks.data());
    compare(comres4);

    if (do_auc) {
        auto compare_auc = [&](const auto& other) -> void {
            for (int s = 0; s < 4; ++s) {
                for (int l = 0; l < ngroups; ++l) {
                    compare_almost_equal(comres.auc[s][l], other.auc[s][l]);
                }
            }
        };
        compare_auc(comres2);
        compare_auc(comres2);
        compare_auc(comres2);
    }
}

INSTANTIATE_TEST_CASE_P(
    ScoreMarkers,
    ScoreMarkersTest,
    ::testing::Combine(
        ::testing::Values(2, 3, 4, 5), // number of clusters
        ::testing::Values(false, true)  // with or without the AUC?
    )
);

TEST(ScoreMarkersTest, MinRank) {
    // Checking that the minimum rank is somewhat sensible,
    // and we didn't feed in the wrong values somewhere.
    int ngenes = 10;
    int nsamples = 8;

    std::vector<double> buffer(ngenes * nsamples, 0);
    for (int i = 0; i < ngenes; ++i) {
        buffer[i * nsamples + 1] = i; // second column gets increasing rank with later genes
        buffer[i * nsamples + 3] = i+1;

        buffer[i * nsamples + 4] = -i; // third column gets decreasing rank with later genes
        buffer[i * nsamples + 6] = -i + 1;
    }

    std::vector<int> grouping{0, 1, 0, 1, 2, 3, 2, 3};
    tatami::DenseRowMatrix<double, int> mat(ngenes, nsamples, std::move(buffer));

    scran::ScoreMarkers chd;
    auto res = chd.run(&mat, grouping.data());

    for (int i = 0; i < ngenes; ++i) {
        EXPECT_EQ(res.cohen[4][1][i], ngenes - i); // second column
        EXPECT_EQ(res.cohen[4][2][i], i + 1);  // third column
    }
}
