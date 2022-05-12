#ifndef COMPARE_PCS_H
#define COMPARE_PCS_H

#include "Eigen/Dense"
#include <gtest/gtest.h>

inline bool same_same(double left, double right, double tol) {
    return std::abs(left - right) <= (std::abs(left) + std::abs(right)) * tol;
}

inline void expect_equal_pcs(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());

    for (size_t i = 0; i < left.rows(); ++i) {
        for (size_t j = 0; j < left.cols(); ++j) {
            EXPECT_TRUE(same_same(std::abs(left(i, j)), std::abs(right(i, j)), tol));
        }

        // PCs should average to zero.
        EXPECT_TRUE(std::abs(left.row(i).sum()) < tol);
        EXPECT_TRUE(std::abs(right.row(i).sum()) < tol);
    }
    return;
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tol=1e-8) {
    ASSERT_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        EXPECT_TRUE(same_same(left[i], right[i], tol));
    }
    return;
}

inline std::vector<int> generate_blocks(int nobs, int nblocks) {
    std::vector<int> blocks(nobs);
    for (int i = 0; i < nobs; ++i) {
        blocks[i] = i % nblocks;
    }
    return blocks;
}

inline void compare_almost_equal(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());
    
    for (size_t c = 0; c < left.cols(); ++c) {
        for (size_t r = 0; r < left.rows(); ++r) {
            EXPECT_FLOAT_EQ(left(r, c), right(r, c));
        }
    }
}

#endif
