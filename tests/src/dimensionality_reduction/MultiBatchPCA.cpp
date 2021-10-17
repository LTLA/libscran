#include <gtest/gtest.h>

#include "../data/data.h"
#include "compare_pcs.h"

#include "tatami/base/Matrix.hpp"
#include "tatami/base/DenseMatrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/MultiBatchPCA.hpp"
#include "scran/dimensionality_reduction/RunPCA.hpp"

TEST(MultiBatchMatrixTest, Test) {
    size_t NR = 30, NC = 10;

    Eigen::MatrixXd thing(NR, NC);  
    std::mt19937_64 rng;
    std::normal_distribution<> dist;
    for (size_t i = 0; i < NR; ++i) {
        for (size_t j = 0; j < NC; ++j) {
            thing(i, j) = dist(rng);
        }
    }

    Eigen::MatrixXd means(1, NC);
    for (size_t j = 0; j < NC; ++j) {
        means(0, j) = dist(rng);
    }

    std::vector<double> weights(NR);
    for (size_t j = 0; j < NR; ++j) {
        weights[j] = dist(rng);
        weights[j] *= weights[j];
    }

    scran::MultiBatchEigenMatrix<false, decltype(thing), double> batched(thing, weights.data(), means);
    auto realized = batched.realize();

    // Trying in the normal orientation.
    {
        size_t NRHS = 2;
        Eigen::MatrixXd rhs(NC, NRHS);
        for (size_t i = 0; i < NC; ++i) {
            for (size_t j = 0; j < NRHS; ++j) {
                rhs(i, j) = dist(rng);
            }
        }

        Eigen::MatrixXd prod1 = batched * rhs;
        Eigen::MatrixXd prod2 = realized * rhs;
        compare_almost_equal(prod1, prod2);
    }

    // Trying in the transposed orientation.
    {
        size_t NRHS = 2;
        Eigen::MatrixXd rhs(NR, NRHS);
        for (size_t i = 0; i < NR; ++i) {
            for (size_t j = 0; j < NRHS; ++j) {
                rhs(i, j) = dist(rng);
            }
        }

        Eigen::MatrixXd tprod1 = batched.adjoint() * rhs;
        Eigen::MatrixXd tprod2 = realized.adjoint() * rhs;
        compare_almost_equal(tprod1, tprod2);
    }
}
