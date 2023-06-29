#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/utils.hpp"
#include "scran/dimensionality_reduction/convert.hpp"

#include <vector>
#include <random>

/********************************************
 ********************************************/

class CenterScalePcaTest : public ::testing::TestWithParam<int> {};

TEST_P(CenterScalePcaTest, DenseCheck) {
    auto nthreads = GetParam();
    size_t NR = 99;
    size_t NC = 101;

    Simulator sim;
    auto mat = sim.matrix(NR, NC);
    auto extracted = scran::pca_utils::extract_dense_for_pca(&mat, nthreads);

    // Check that the mean and variance calculations are correct.
    Eigen::VectorXd center_v(NR);
    Eigen::VectorXd scale_v(NR);
    scran::pca_utils::compute_mean_and_variance_from_dense_matrix(extracted, center_v, scale_v, nthreads);

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
        scran::pca_utils::apply_center_and_scale_to_dense_matrix(matcopy, center_v, false, scale_v, nthreads);

        Eigen::VectorXd center_v2(NR);
        Eigen::VectorXd scale_v2(NR);
        scran::pca_utils::compute_mean_and_variance_from_dense_matrix(matcopy, center_v2, scale_v2, nthreads);

        compare_almost_equal(std::vector<double>(NR), std::vector<double>(center_v2.begin(), center_v2.end()));
        compare_almost_equal(refvar, std::vector<double>(scale_v2.begin(), scale_v2.end()));
    }

    // Check that processing works with centering and scaling.
    {
        double total = scran::pca_utils::process_scale_vector(true, scale_v);
        EXPECT_EQ(total, NR);

        auto matcopy = extracted;
        scran::pca_utils::apply_center_and_scale_to_dense_matrix(matcopy, center_v, true, scale_v, nthreads);

        Eigen::VectorXd center_v2(NR);
        Eigen::VectorXd scale_v2(NR);
        scran::pca_utils::compute_mean_and_variance_from_dense_matrix(matcopy, center_v2, scale_v2, nthreads);

        compare_almost_equal(std::vector<double>(NR), std::vector<double>(center_v2.begin(), center_v2.end()));
        compare_almost_equal(std::vector<double>(NR, 1), std::vector<double>(scale_v2.begin(), scale_v2.end()));
    }
}

TEST_P(CenterScalePcaTest, SparseCheck) {
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
    scran::pca_utils::SparseMatrix emat(NC, NR, std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads);
    scran::pca_utils::compute_mean_and_variance_from_sparse_matrix(emat, center_v, scale_v, nthreads);

    auto means = tatami::row_sums(smat.get());
    for (auto& x : means) {
        x /= NC;
    }
    compare_almost_equal(means, std::vector<double>(center_v.begin(), center_v.end()));

    auto refvar = tatami::row_variances(smat.get());
    compare_almost_equal(refvar, std::vector<double>(scale_v.begin(), scale_v.end()));
}

INSTANTIATE_TEST_SUITE_P(
    CenterScalePca,
    CenterScalePcaTest,
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
