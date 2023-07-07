#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"
#include "../utils/compare_almost_equal.h"
#include "utils.h"

#include "tatami/tatami.hpp"

#include "scran/quality_control/PerCellQcMetrics.hpp"

#include <cmath>

class PerCellQcMetricsTestStandard : public ::testing::TestWithParam<int> {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void SetUp() {
        size_t nr = 100, nc = 50;
        auto mat = Simulator().matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }

public:
    template<bool exact = false, class Result>
    static void compare(const Result& res, const Result& other) {
        if constexpr(exact) {
            EXPECT_EQ(res.total, other.total);
        } else {
            compare_almost_equal(res.total, other.total);
        }
        EXPECT_EQ(res.detected, other.detected);
        EXPECT_EQ(res.max_count, other.max_count);
        EXPECT_EQ(res.max_index, other.max_index);

        ASSERT_EQ(res.subset_total.size(), other.subset_total.size());
        for (size_t i = 0; i < res.subset_total.size(); ++i) {
            if constexpr(exact) {
                EXPECT_EQ(res.subset_total[i], other.subset_total[i]);
            } else {
                compare_almost_equal(res.subset_total[i], other.subset_total[i]);
            }
        }

        ASSERT_EQ(res.subset_detected.size(), other.subset_detected.size());
        for (size_t i = 0; i < res.subset_detected.size(); ++i) {
            EXPECT_EQ(res.subset_detected[i], other.subset_detected[i]);
        }
    }
};

TEST_P(PerCellQcMetricsTestStandard, NoSubset) {
    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), {});

    {
        EXPECT_EQ(res.total, tatami::column_sums(dense_row.get()));
        EXPECT_EQ(res.detected, quality_control::compute_num_detected(dense_row.get()));
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), {});
    compare<true>(res, res1);

    auto res2 = qcfun.run(dense_column.get(), {});
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), {});
    compare(res, res3);

    auto res4 = qcfun.run(sparse_column.get(), {});
    compare(res, res3);
}

TEST_P(PerCellQcMetricsTestStandard, OneSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9, 10, 16, 17 };
    auto keep_s = quality_control::to_filter(dense_row->nrow(), keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), subs);

    {
        auto ref = tatami::make_DelayedSubset<0>(dense_row, keep_i);
        auto reftotal = tatami::column_sums(ref.get());
        EXPECT_EQ(reftotal, res.subset_total[0]);
        EXPECT_EQ(res.subset_detected[0], quality_control::compute_num_detected(ref.get()));
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);
    
    auto res1 = qcfun.run(dense_row.get(), subs);
    compare<true>(res, res1);

    auto res2 = qcfun.run(dense_column.get(), subs);
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), subs);
    compare(res, res3);
    
    auto res4 = qcfun.run(sparse_column.get(), subs);
    compare(res, res4);
}

TEST_P(PerCellQcMetricsTestStandard, TwoSubsets) {
    std::vector<size_t> keep_i1 = { 0, 5, 7, 8, 9, 10, 16, 17 };
    std::vector<size_t> keep_i2 = { 1, 8, 2, 6, 11, 5, 19, 17 };
    auto keep_s1 = quality_control::to_filter(dense_row->nrow(), keep_i1);
    auto keep_s2 = quality_control::to_filter(dense_row->nrow(), keep_i2);
    std::vector<const int*> subs = { keep_s1.data(), keep_s2.data() };

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), subs);

    {
        auto ref1 = tatami::make_DelayedSubset<0>(dense_row, keep_i1);
        auto refprop1 = tatami::column_sums(ref1.get());
        EXPECT_EQ(refprop1, res.subset_total[0]);
        EXPECT_EQ(res.subset_detected[0], quality_control::compute_num_detected(ref1.get()));

        auto ref2 = tatami::make_DelayedSubset<0>(dense_row, keep_i2);
        auto refprop2 = tatami::column_sums(ref2.get());
        compare_almost_equal(refprop2, res.subset_total[1]); // keep_i2 is scrambled, so exact equality can't be expected upon summation.
        EXPECT_EQ(res.subset_detected[1], quality_control::compute_num_detected(ref2.get()));
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), subs);
    compare<true>(res, res1);

    auto res2 = qcfun.run(dense_column.get(), subs);
    compare(res, res2);

    auto res3 = qcfun.run(sparse_row.get(), subs);
    compare(res, res3);
    
    auto res4 = qcfun.run(sparse_column.get(), subs);
    compare(res, res4);
}

INSTANTIATE_TEST_SUITE_P(
    PerCellQcMetrics,
    PerCellQcMetricsTestStandard,
    ::testing::Values(1, 3) // number of threads
);

class PerCellQcMetricsTestMaxed : public ::testing::TestWithParam<int> {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void propagate() {
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }
};

TEST_P(PerCellQcMetricsTestMaxed, SomeNegatives) {
    {
        Simulator sim;
        sim.lower = -2;
        sim.upper = -1;

        size_t nr = 99, nc =47;
        auto mat = sim.matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        propagate();
    }

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), {});

    { // max should be all-zeros.
        std::vector<double> expected(dense_row->ncol());
        EXPECT_EQ(res.max_count, expected);
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), {});
    EXPECT_EQ(res1.max_count, res.max_count);
    EXPECT_EQ(res1.max_index, res.max_index);

    auto res2 = qcfun.run(dense_column.get(), {});
    EXPECT_EQ(res2.max_count, res.max_count);
    EXPECT_EQ(res2.max_index, res.max_index);

    auto res3 = qcfun.run(sparse_row.get(), {});
    EXPECT_EQ(res3.max_count, res.max_count);
    EXPECT_EQ(res3.max_index, res.max_index);
    
    auto res4 = qcfun.run(sparse_column.get(), {});
    EXPECT_EQ(res4.max_count, res.max_count);
    EXPECT_EQ(res4.max_index, res.max_index);
}

TEST_P(PerCellQcMetricsTestMaxed, AllNegatives) {
    {
        Simulator sim;
        sim.lower = -5;
        sim.upper = -1;
        sim.density = 1;

        size_t nr = 20, nc = 100;
        auto mat = sim.matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        propagate();
    }

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), {});

    {
        bool max_neg = true;
        for (auto x : res.max_count) {
            if (x >= 0) {
                max_neg = false;
                break;
            }
        }
        EXPECT_TRUE(max_neg);
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), {});
    EXPECT_EQ(res1.max_count, res.max_count);
    EXPECT_EQ(res1.max_index, res.max_index);

    auto res2 = qcfun.run(dense_column.get(), {});
    EXPECT_EQ(res2.max_count, res.max_count);
    EXPECT_EQ(res2.max_index, res.max_index);

    auto res3 = qcfun.run(sparse_row.get(), {});
    EXPECT_EQ(res3.max_count, res.max_count);
    EXPECT_EQ(res3.max_index, res.max_index);
    
    auto res4 = qcfun.run(sparse_column.get(), {});
    EXPECT_EQ(res4.max_count, res.max_count);
    EXPECT_EQ(res4.max_index, res.max_index);
}

TEST_P(PerCellQcMetricsTestMaxed, AllZeros) {
    {
        size_t nr = 12, nc = 45;
        std::vector<double> empty(nr * nc);
        dense_row.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(empty)));
        propagate();
    }

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), {});

    { // max should be all-zeros.
        std::vector<double> expected(dense_row->ncol());
        EXPECT_EQ(res.max_count, expected);
        std::vector<int> indices(dense_row->ncol());
        EXPECT_EQ(res.max_index, indices);
    }

    int threads = GetParam();
    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), {});
    EXPECT_EQ(res1.max_count, res.max_count);
    EXPECT_EQ(res1.max_index, res.max_index);

    auto res2 = qcfun.run(dense_column.get(), {});
    EXPECT_EQ(res2.max_count, res.max_count);
    EXPECT_EQ(res2.max_index, res.max_index);

    auto res3 = qcfun.run(sparse_row.get(), {});
    EXPECT_EQ(res3.max_count, res.max_count);
    EXPECT_EQ(res3.max_index, res.max_index);
    
    auto res4 = qcfun.run(sparse_column.get(), {});
    EXPECT_EQ(res4.max_count, res.max_count);
    EXPECT_EQ(res4.max_index, res.max_index);
}

TEST_P(PerCellQcMetricsTestMaxed, StructuralZeros) {
    int threads = GetParam();

    {
        // Fewer rows, so we're more likely to get a row where the maximum
        // is determined by one of the structural zeros.
        size_t nr = 9, nc = 1001;

        std::vector<int> i;
        std::vector<double> x;
        std::vector<size_t> p(1);
        std::mt19937_64 rng(98712 * threads);
        std::uniform_real_distribution<> structural(0.0, 1.0);

        for (size_t c = 0; c < nc; ++c) {
            for (size_t r = 0; r < nr; ++r) {
                if (structural(rng) < 0.2) {
                    i.push_back(r);
                    auto choice = structural(rng);
                    if (choice < 0.3) {
                        x.push_back(-1);
                    } else if (choice < 0.7) {
                        x.push_back(0); // spiking in structural values that are actually non-zero.
                    } else {
                        x.push_back(1);
                    }
                }
            }
            p.push_back(i.size());
        }

        sparse_column.reset(new tatami::CompressedSparseColumnMatrix<double, int>(nr, nc, std::move(x), std::move(i), std::move(p)));
        sparse_row = tatami::convert_to_sparse(sparse_column.get(), 0);
        dense_column = tatami::convert_to_dense(sparse_column.get(), 1);
        dense_row = tatami::convert_to_dense(sparse_column.get(), 0);
    }

    scran::PerCellQcMetrics qcfun;
    auto res = qcfun.run(dense_row.get(), {});

    qcfun.set_num_threads(threads);

    auto res1 = qcfun.run(dense_row.get(), {});
    EXPECT_EQ(res1.max_count, res.max_count);
    EXPECT_EQ(res1.max_index, res.max_index);

    auto res2 = qcfun.run(dense_column.get(), {});
    EXPECT_EQ(res2.max_count, res.max_count);
    EXPECT_EQ(res2.max_index, res.max_index);

    auto res3 = qcfun.run(sparse_row.get(), {});
    EXPECT_EQ(res3.max_count, res.max_count);
    EXPECT_EQ(res3.max_index, res.max_index);
    
    auto res4 = qcfun.run(sparse_column.get(), {});
    EXPECT_EQ(res4.max_count, res.max_count);
    EXPECT_EQ(res4.max_index, res.max_index);
}

TEST_P(PerCellQcMetricsTestMaxed, OkayOnMissing) {
    {
        Simulator sim;
        size_t nr = 20, nc = 100;
        auto mat = sim.matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        propagate();
    }

    auto threads = GetParam();

    {
        scran::PerCellQcMetrics qcfun;
        qcfun.set_compute_max_count(false);
        qcfun.set_num_threads(threads);

        auto res1 = qcfun.run(dense_row.get(), {});
        EXPECT_TRUE(res1.max_count.empty());

        auto res2 = qcfun.run(dense_column.get(), {});
        EXPECT_TRUE(res2.max_count.empty());
        EXPECT_EQ(res2.max_index, res1.max_index);

        auto res3 = qcfun.run(sparse_row.get(), {});
        EXPECT_TRUE(res3.max_count.empty());
        EXPECT_EQ(res3.max_index, res1.max_index);
        
        auto res4 = qcfun.run(sparse_column.get(), {});
        EXPECT_TRUE(res4.max_count.empty());
        EXPECT_EQ(res4.max_index, res1.max_index);
    }

    {
        scran::PerCellQcMetrics qcfun;
        qcfun.set_compute_max_index(false);
        qcfun.set_num_threads(threads);

        auto res1 = qcfun.run(dense_row.get(), {});
        EXPECT_TRUE(res1.max_index.empty());

        auto res2 = qcfun.run(dense_column.get(), {});
        EXPECT_TRUE(res2.max_index.empty());
        EXPECT_EQ(res2.max_count, res1.max_count);

        auto res3 = qcfun.run(sparse_row.get(), {});
        EXPECT_TRUE(res3.max_index.empty());
        EXPECT_EQ(res3.max_count, res1.max_count);
        
        auto res4 = qcfun.run(sparse_column.get(), {});
        EXPECT_TRUE(res4.max_index.empty());
        EXPECT_EQ(res4.max_count, res1.max_count);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PerCellQcMetrics,
    PerCellQcMetricsTestMaxed,
    ::testing::Values(1, 3) // number of threads
);

TEST(PerCellQcMetrics, FullyDisabled) {
    size_t nr = 100, nc = 50;
    auto mat = Simulator().matrix(nr, nc);

    scran::PerCellQcMetrics qcfun;
    qcfun.set_compute_total(false);
    qcfun.set_compute_detected(false);
    qcfun.set_compute_max_count(false);
    qcfun.set_compute_max_index(false);
    qcfun.set_compute_subset_total(false);
    qcfun.set_compute_subset_detected(false);

    std::vector<size_t> keep_i = { 0, 1, 2, 3 };
    auto keep_s = quality_control::to_filter(nr, keep_i);
    std::vector<const int*> subs(1, keep_s.data());

    auto res = qcfun.run(&mat, subs);
    EXPECT_TRUE(res.total.empty());
    EXPECT_TRUE(res.detected.empty());
    EXPECT_TRUE(res.max_count.empty());
    EXPECT_TRUE(res.max_index.empty());

    EXPECT_TRUE(res.subset_total.empty());
    EXPECT_TRUE(res.subset_detected.empty());
}

TEST(PerCellQcMetrics, BufferFilling) {
    size_t nr = 100, nc = 50;
    auto mat = Simulator().matrix(nr, nc);

    scran::PerCellQcMetrics::Results output;
    scran::PerCellQcMetrics::Buffers<> buffers;

    // Prefilling each vector with a little bit of nonsense.
    {
        output.total.resize(nc, 99);
        buffers.total = output.total.data();

        output.detected.resize(nc, 111);
        buffers.detected = output.detected.data();

        output.max_index.resize(nc, 91);
        buffers.max_index = output.max_index.data();

        output.max_count.resize(nc, 23214);
        buffers.max_count = output.max_count.data();

        size_t nsubsets = 1;
        output.subset_total.resize(nsubsets);
        buffers.subset_total.resize(nsubsets);
        output.subset_detected.resize(nsubsets);
        buffers.subset_detected.resize(nsubsets);

        for (size_t s = 0; s < nsubsets; ++s) {
            output.subset_total[s].resize(nc, 99);
            buffers.subset_total[s] = output.subset_total[s].data();
            output.subset_detected[s].resize(nc, -100);
            buffers.subset_detected[s] = output.subset_detected[s].data();
        }
    }

    std::vector<size_t> keep_i = { 1, 5, 7, 9, 11 };
    auto keep_s = quality_control::to_filter(nr, keep_i);
    std::vector<const int*> subs(1, keep_s.data());
     
    scran::PerCellQcMetrics qcfun;
    qcfun.run(&mat, subs, buffers);
    auto ref = qcfun.run(&mat, subs);
    PerCellQcMetricsTestStandard::compare<true>(ref, output);
}
