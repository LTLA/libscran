#ifndef COMPARE_PCS_H
#define COMPARE_PCS_H

#include "Eigen/Dense"
#include <gtest/gtest.h>
#include "../utils/compare_almost_equal.h"
#include <vector>
#include "tatami/tatami.hpp"

inline void are_pcs_centered(const Eigen::MatrixXd& pcs, double tol = 1e-8) {
    int ndims = pcs.rows(), ncells = pcs.cols();
    for (int r = 0; r < ndims; ++r) {
        auto ptr = pcs.data() + r;

        double mean = 0;
        for (size_t c = 0; c < ncells; ++c, ptr += ndims) {
            mean += *ptr;
        }
        mean /= ncells;

        EXPECT_TRUE(std::abs(mean) < tol);
    }
}

inline std::vector<std::shared_ptr<tatami::NumericMatrix> > fragment_matrices_by_block(const std::shared_ptr<tatami::NumericMatrix>& x, const std::vector<int>& block, int nblocks) {
    std::vector<std::shared_ptr<tatami::NumericMatrix> > collected;

    for (int b = 0; b < nblocks; ++b) {
        std::vector<int> keep;

        for (size_t i = 0; i < block.size(); ++i) {
            if (block[i] == b) {
                keep.push_back(i);
            }
        }

        if (keep.size() > 1) {
            collected.push_back(tatami::make_DelayedSubset<1>(x, keep));
        }
    }

    return collected;
}

inline void expect_equal_pcs(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8, bool relative = true) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());

    for (size_t i = 0; i < left.rows(); ++i) {
        for (size_t j = 0; j < left.cols(); ++j) {
            auto aleft = std::abs(left(i, j));
            auto aright = std::abs(right(i, j));
            if (relative) {
                compare_almost_equal(aleft, aright, tol);
            } else if (std::abs(aleft - aright) > tol) {
                EXPECT_TRUE(false) << "mismatch in almost-equal floats (expected " << aleft << ", got " << aright << ")";
            }
        }

        // PCs should average to zero.
        EXPECT_TRUE(std::abs(left.row(i).sum()) < tol);
        EXPECT_TRUE(std::abs(right.row(i).sum()) < tol);
    }
    return;
}

inline void expect_equal_rotation(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());

    for (size_t i = 0; i < left.rows(); ++i) {
        for (size_t j = 0; j < left.cols(); ++j) {
            auto aleft = std::abs(left(i, j));
            auto aright = std::abs(right(i, j));
            compare_almost_equal(aleft, aright, tol);
        }
    }
    return;
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tol=1e-8) {
    ASSERT_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        compare_almost_equal(left[i], right[i], tol);
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
