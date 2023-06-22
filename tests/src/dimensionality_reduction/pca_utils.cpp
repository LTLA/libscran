#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/pca_utils.hpp"

#include <vector>
#include <random>

class ExtractForPcaTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> sparse_row, sparse_column;

    template<class Param>
    void assemble(const Param& param) {
        auto NR = std::get<0>(param);
        auto NC = std::get<1>(param);
        nthreads = std::get<2>(param);

        std::mt19937_64 rng(NR * NC / nthreads);
        std::normal_distribution<> ndist;
        std::uniform_real_distribution<> udist(0,1);

        ptrs.resize(NR + 1);
        for (size_t r = 0; r < NR; ++r) {
            for (size_t c = 0; c < NC; ++c) {
                if (udist(rng) < 0.2) {
                    auto val = ndist(rng);
                    values.push_back(val);
                    indices.push_back(c);
                    ++ptrs[r+1];
                }
            }
        }

        for (size_t i = 0; i < NR; ++i) {
            ptrs[i+1] += ptrs[i];
        }

        sparse_row.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, values, indices, ptrs));
        sparse_column = tatami::convert_to_sparse(sparse_row.get(), 1);
        return;
    }

    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> ptrs;
    int nthreads;
};

TEST_P(ExtractForPcaTest, Sparse) {
    auto param = GetParam();
    assemble(param);

    {
        auto extracted = scran::pca_utils::extract_sparse_for_pca(sparse_row.get(), nthreads);
        EXPECT_EQ(values, extracted.values);
        EXPECT_EQ(indices, extracted.indices);
        EXPECT_EQ(ptrs, extracted.ptrs);
    }

    {
        auto extracted = scran::pca_utils::extract_sparse_for_pca(sparse_column.get(), nthreads);
        EXPECT_EQ(values, extracted.values);
        EXPECT_EQ(indices, extracted.indices);
        EXPECT_EQ(ptrs, extracted.ptrs);
    }
}

TEST_P(ExtractForPcaTest, Dense) {
    auto param = GetParam();
    assemble(param);

    {
        auto extracted = scran::pca_utils::extract_dense_for_pca(sparse_row.get(), nthreads);

        size_t NR = extracted.rows(), NC = extracted.cols();
        EXPECT_EQ(NR, sparse_row->ncol());
        EXPECT_EQ(NC, sparse_row->nrow());

        auto ext = sparse_row->dense_column();
        for (size_t r = 0; r < NR; ++r) {
            auto row = extracted.row(r);
            EXPECT_EQ(std::vector<double>(row.begin(), row.end()), ext->fetch(r));
        }
    }

    {
        auto extracted = scran::pca_utils::extract_dense_for_pca(sparse_column.get(), nthreads);

        size_t NR = extracted.rows(), NC = extracted.cols();
        EXPECT_EQ(NR, sparse_row->ncol());
        EXPECT_EQ(NC, sparse_row->nrow());

        auto ext = sparse_row->dense_column();
        for (size_t r = 0; r < NR; ++r) {
            auto row = extracted.row(r);
            EXPECT_EQ(std::vector<double>(row.begin(), row.end()), ext->fetch(r));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ExtractForPca,
    ExtractForPcaTest,
    ::testing::Combine(
        ::testing::Values(12, 56, 123), // rows
        ::testing::Values(19, 47, 98), // columns
        ::testing::Values(1, 3) // number of threads
    )
);

/********************************************
 ********************************************/

class MeanVarPcaCalculationsTest : public ::testing::TestWithParam<int> {};

TEST_P(MeanVarPcaCalculationsTest, DenseCheck) {
    auto nthreads = GetParam();
    size_t NR = 99;
    size_t NC = 101;

    Simulator sim;
    auto mat = sim.matrix(NR, NC);
    auto extracted = scran::pca_utils::extract_dense_for_pca(&mat, nthreads);

    // Check that the mean and variance calculations are correct.
    Eigen::VectorXd center_v(NR);
    Eigen::VectorXd scale_v(NR);
    scran::pca_utils::compute_mean_and_variance_from_dense_columns(extracted, center_v, scale_v, nthreads);

    auto means = tatami::row_sums(&mat);
    for (auto& x : means) {
        x /= NC;
    }
    compare_almost_equal(means, std::vector<double>(center_v.begin(), center_v.end()));

    auto refvar = tatami::row_variances(&mat);
    compare_almost_equal(refvar, std::vector<double>(scale_v.begin(), scale_v.end()));

    // Check that processing works with centering only.
    {
        double total = scran::pca_utils::process_scale_vector(false, scale_v);
        EXPECT_EQ(total, std::accumulate(refvar.begin(), refvar.end(), 0.0));

        auto matcopy = extracted;
        scran::pca_utils::center_and_scale_dense_columns(matcopy, center_v, false, scale_v, nthreads);

        Eigen::VectorXd center_v2(NR);
        Eigen::VectorXd scale_v2(NR);
        scran::pca_utils::compute_mean_and_variance_from_dense_columns(matcopy, center_v2, scale_v2, nthreads);

        compare_almost_equal(std::vector<double>(NR), std::vector<double>(center_v2.begin(), center_v2.end()));
        compare_almost_equal(refvar, std::vector<double>(scale_v2.begin(), scale_v2.end()));
    }

    // Check that processing works with centering and scaling.
    {
        double total = scran::pca_utils::process_scale_vector(true, scale_v);
        EXPECT_EQ(total, NR);

        auto matcopy = extracted;
        scran::pca_utils::center_and_scale_dense_columns(matcopy, center_v, true, scale_v, nthreads);

        Eigen::VectorXd center_v2(NR);
        Eigen::VectorXd scale_v2(NR);
        scran::pca_utils::compute_mean_and_variance_from_dense_columns(matcopy, center_v2, scale_v2, nthreads);

        compare_almost_equal(std::vector<double>(NR), std::vector<double>(center_v2.begin(), center_v2.end()));
        compare_almost_equal(std::vector<double>(NR, 1), std::vector<double>(scale_v2.begin(), scale_v2.end()));
    }
}

TEST_P(MeanVarPcaCalculationsTest, SparseCheck) {
    auto nthreads = GetParam();
    size_t NR = 59;
    size_t NC = 251;

    Simulator sim;
    auto mat = sim.matrix(NR, NC);
    auto smat = tatami::convert_to_sparse(&mat, 1);
    auto extracted = scran::pca_utils::extract_sparse_for_pca(smat.get(), nthreads);

    // Check that the mean and variance calculations are correct.
    Eigen::VectorXd center_v(NR);
    Eigen::VectorXd scale_v(NR);
    scran::pca_utils::compute_mean_and_variance_from_sparse_components(NR, NC, extracted.values, extracted.indices, extracted.ptrs, center_v, scale_v, nthreads);

    auto means = tatami::row_sums(smat.get());
    for (auto& x : means) {
        x /= NC;
    }
    compare_almost_equal(means, std::vector<double>(center_v.begin(), center_v.end()));

    auto refvar = tatami::row_variances(smat.get());
    compare_almost_equal(refvar, std::vector<double>(scale_v.begin(), scale_v.end()));
}

INSTANTIATE_TEST_SUITE_P(
    MeanVarPcaCalculations,
    MeanVarPcaCalculationsTest,
    ::testing::Values(1, 3) // number of threads
);

/********************************************
 ********************************************/

TEST(ProcessScaleVectorTest, EdgeCases) {
    Eigen::VectorXd scale_v(5);
    scale_v.fill(0);
    scale_v[0] = 1;
    scale_v[1] = 2;
    scale_v[2] = 3;

    EXPECT_EQ(scran::pca_utils::process_scale_vector(false, scale_v), 6); // not mutating; just returns the total variance.

    EXPECT_EQ(scran::pca_utils::process_scale_vector(true, scale_v), 3); // mutating, and returns the total number of non-zero variances.

    std::vector<double> expected { 1, std::sqrt(2), std::sqrt(3), 1, 1 }; // takes the square root, and fills in zeros with 1's.
    EXPECT_EQ(std::vector<double>(scale_v.begin(), scale_v.end()), expected);
}
