#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/normalization/CenterSizeFactors.hpp"

#include <cmath>

class CenterSizeFactorsTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        sf = tatami::column_sums(mat.get());
    }

    std::vector<int> create_block(size_t n) {
        std::vector<int> output(n);
        for (size_t i = 0; i < n; ++i) { 
            output[i] = i % 3;
        }
        return output;
    }

    std::vector<double> blockwise_means(const std::vector<int>& block, const std::vector<double>& sf) {
        std::vector<double> blocked_sf(3), blocked_num(3);
        for (size_t i = 0; i < block.size(); ++i) { 
            auto index = block[i];
            blocked_sf[index] += sf[i];
            ++blocked_num[index];
        }
        for (size_t i = 0; i < 3; ++i) {
            blocked_sf[i] /= blocked_num[i];
        }
        return blocked_sf;
    }

protected:
    std::shared_ptr<tatami::NumericMatrix> mat;
    std::vector<double> sf;
};

TEST_F(CenterSizeFactorsTester, Simple) {
    scran::CenterSizeFactors cen;
    EXPECT_FALSE(cen.run(sf.size(), sf.data()));

    double middle = std::accumulate(sf.begin(), sf.end(), 0.0) / sf.size();
    EXPECT_FLOAT_EQ(middle, 1);
}

TEST_F(CenterSizeFactorsTester, BlockedLowest) {
    auto block = create_block(mat->ncol());
    scran::CenterSizeFactors cen;
    cen.run_blocked(sf.size(), sf.data(), block.data());

    auto blockmeans = blockwise_means(block, sf);
    int at1 = 0, above1 = 0;
    for (size_t i = 0; i < blockmeans.size(); ++i ){
        if (std::abs(blockmeans[i] - 1) < 1e-8) {
            ++at1;
        } else if (blockmeans[i] > 1) {
            ++above1;
        }
    }

    // One block should be at 1, the other blocks should be > 1.
    EXPECT_EQ(at1, 1);
    EXPECT_EQ(above1, 2);
}

TEST_F(CenterSizeFactorsTester, BlockCenter) {
    auto block = create_block(mat->ncol());

    scran::CenterSizeFactors cen;
    cen.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
    cen.run_blocked(sf.size(), sf.data(), block.data());

    // All blocks centered at unity.
    auto blockmeans = blockwise_means(block, sf);
    for (size_t i = 0; i < blockmeans.size(); ++i) {
        EXPECT_FLOAT_EQ(blockmeans[i], 1);
    }
}

TEST_F(CenterSizeFactorsTester, Error) {
    std::vector<double> empty(mat->ncol(), 1);
    EXPECT_FALSE(scran::CenterSizeFactors::validate(empty.size(), empty.data()));

    empty[0] = 0;
    EXPECT_TRUE(scran::CenterSizeFactors::validate(empty.size(), empty.data()));

    scran::CenterSizeFactors cen;
    std::fill(empty.begin(), empty.end(), 0);
    auto copy = empty;
    EXPECT_TRUE(cen.run(empty.size(), copy.data()));
    EXPECT_EQ(copy, empty); // avoids division by zero.

    empty[0] = -1;
    EXPECT_ANY_THROW({
        try {
            cen.run(empty.size(), empty.data());
        } catch (std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("negative") != std::string::npos);
            throw;
        }
    });
}
