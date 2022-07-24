#include <gtest/gtest.h>
#include "scran/dimensionality_reduction/CustomSparseMatrix.hpp"
#include "Eigen/Dense"
#include <random>
#include "compare_pcs.h"

class CustomSparseMatrixTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(CustomSparseMatrixTest, TestFillRows) {
    auto param = GetParam();
    size_t nr = std::get<0>(param);
    size_t nc = std::get<1>(param);
    int nt = std::get<2>(param);

    Eigen::MatrixXd control(nr, nc);
    control.setZero();

    std::mt19937_64 rng(nr * nc);
    std::uniform_real_distribution udist(0.0, 1.0);
    std::normal_distribution ndist;

    std::vector<std::vector<double> > vbyrow(nr), vbycol(nc);
    std::vector<std::vector<int> > ibyrow(nr), ibycol(nc);
    std::vector<int> col_nzeros(nc);

    for (size_t c = 0; c < nc; ++c) {
        for (size_t r = 0; r < nr; ++r) {
            if (udist(rng) < 0.2) {
                double val = ndist(rng);
                control(r, c) = val;
                vbyrow[r].push_back(val);
                ibyrow[r].push_back(c);
                vbycol[c].push_back(val);
                ibycol[c].push_back(r);
                ++(col_nzeros[c]);
            }
        }
    }

    for (int mode = 0; mode < 2; ++mode) {
        scran::pca_utils::CustomSparseMatrix A(nr, nc, nt);
        EXPECT_EQ(A.rows(), nr);
        EXPECT_EQ(A.cols(), nc);

        if (mode == 0) {
            A.fill_rows(std::move(vbyrow), std::move(ibyrow), std::move(col_nzeros));
        } else {
            A.fill_columns(std::move(vbycol), std::move(ibycol));
        }

        // Realizes correctly.
        auto realized = A.realize();
        bool okay = true;
        for (size_t c = 0; c < nc; ++c) {
            auto col = realized.col(c);
            auto ref = control.col(c);
            for (size_t r = 0; r < nr; ++r) {
                if (col[r] != ref[r]) {
                    okay = false;
                    break;
                }
            }
        }
        EXPECT_TRUE(okay);

        // Multiplies correctly.
        {
            Eigen::VectorXd vec(nc);
            for (auto& v : vec) {
                v = ndist(rng);
            }

            Eigen::VectorXd ref = control * vec;
            Eigen::VectorXd obs(nr);
            A.multiply(vec, obs);

            expect_equal_vectors(ref, obs, 0);
        }

        // Adjoint multiplies correctly.
        {
            Eigen::VectorXd vec(nr);
            for (auto& v : vec) {
                v = ndist(rng);
            }

            Eigen::VectorXd ref = control.adjoint() * vec;
            Eigen::VectorXd obs(nc);
            A.adjoint_multiply(vec, obs);

            expect_equal_vectors(ref, obs);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    CustomSparseMatrix,
    CustomSparseMatrixTest,
    ::testing::Combine(
        ::testing::Values(10, 49, 97), // number of rows
        ::testing::Values(10, 51, 88), // number of columns
        ::testing::Values(1, 3) // number of threads
    )
);
