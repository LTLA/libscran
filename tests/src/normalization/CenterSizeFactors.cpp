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

TEST_F(CenterSizeFactorsTester, IgnoreZeros) {
    sf[0] = 0;

    {
        auto sf1 = sf;
        auto sf2 = sf;

        scran::CenterSizeFactors cen;
        EXPECT_TRUE(cen.run(sf1.size(), sf1.data()));
        EXPECT_EQ(sf1[0], 0);

        double middle = std::accumulate(sf1.begin() + 1, sf1.end(), 0.0) / (sf1.size() - 1);
        EXPECT_FLOAT_EQ(middle, 1);

        // Zero is ignored as if it wasn't even there.
        EXPECT_FALSE(cen.run(sf2.size() - 1, sf2.data() + 1));
        EXPECT_EQ(sf1, sf2);
    }

    // Restoring our acceptance of zeros.
    {
        scran::CenterSizeFactors cen;
        cen.set_ignore_zeros(false);
        EXPECT_TRUE(cen.run(sf.size(), sf.data()));
        EXPECT_EQ(sf[0], 0);

        double middle = std::accumulate(sf.begin(), sf.end(), 0.0) / sf.size();
        EXPECT_FLOAT_EQ(middle, 1);
    }

    // All-zeros avoid division by zero.
    scran::CenterSizeFactors cen;
    std::vector<double> empty(sf.size());
    auto copy = empty;
    EXPECT_TRUE(cen.run(copy.size(), copy.data()));
    EXPECT_EQ(copy, empty); 
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

TEST_F(CenterSizeFactorsTester, BlockIgnoreZeros) {
    auto block = create_block(mat->ncol());
    sf[0] = 0;

    // Zeros are basically ignored.
    {
        auto sf1 = sf;
        auto sf2 = sf;

        scran::CenterSizeFactors cen;
        cen.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
        EXPECT_TRUE(cen.run_blocked(sf1.size(), sf1.data(), block.data()));
        EXPECT_FALSE(cen.run_blocked(sf2.size() - 1, sf2.data() + 1, block.data() + 1));
        EXPECT_EQ(sf1, sf2);
    }

    // Same for the LOWEST mode.
    {
        auto sf1 = sf;
        auto sf2 = sf;

        scran::CenterSizeFactors cen;
        cen.set_block_mode(scran::CenterSizeFactors::LOWEST);
        EXPECT_TRUE(cen.run_blocked(sf1.size(), sf1.data(), block.data()));
        EXPECT_FALSE(cen.run_blocked(sf2.size() - 1, sf2.data() + 1, block.data() + 1));

        EXPECT_EQ(sf1, sf2);
    }

    // Unless we force them to be acknowledged.
    {
        auto sf1 = sf;
        auto sf2 = sf;

        scran::CenterSizeFactors cen;
        cen.set_ignore_zeros(false);
        cen.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
        EXPECT_TRUE(cen.run_blocked(sf1.size(), sf1.data(), block.data()));

        auto blockmeans = blockwise_means(block, sf1);
        for (size_t i = 0; i < blockmeans.size(); ++i) {
            EXPECT_FLOAT_EQ(blockmeans[i], 1);
        }

        cen.set_ignore_zeros(true);
        EXPECT_TRUE(cen.run_blocked(sf2.size(), sf2.data(), block.data()));
        EXPECT_NE(sf1, sf2);
    }
}

TEST_F(CenterSizeFactorsTester, BlockAllZeros) {
    auto block = create_block(mat->ncol());
    for (size_t i = 0; i < sf.size(); ++i) {
        if (block[i] == 1) {
            sf[i] = 0;
        }
    }

    // Perblock can handle all-zero groups.
    {
        auto sf1 = sf;
        scran::CenterSizeFactors cen;
        cen.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
        EXPECT_TRUE(cen.run_blocked(sf1.size(), sf1.data(), block.data()));

        auto blockmeans = blockwise_means(block, sf1);
        for (size_t i = 0; i < blockmeans.size(); ++i) {
            if (i == 1) {
                EXPECT_FLOAT_EQ(blockmeans[i], 0);
            } else {
                EXPECT_FLOAT_EQ(blockmeans[i], 1);
            }
        }
    }

    // Lowest survives.
    {
        auto sf1 = sf;
        scran::CenterSizeFactors cen;
        cen.set_block_mode(scran::CenterSizeFactors::LOWEST);
        EXPECT_TRUE(cen.run_blocked(sf1.size(), sf1.data(), block.data()));

        auto blockmeans = blockwise_means(block, sf1);
        for (size_t i = 0; i < blockmeans.size(); ++i) {
            if (i == 1) {
                EXPECT_FLOAT_EQ(blockmeans[i], 0);
            } else {
                EXPECT_TRUE(blockmeans[i] >= 1);
            }
        }
    }

    // All-zeros avoid division by zero.
    {
        scran::CenterSizeFactors cen;
        std::vector<double> empty(sf.size());
        auto copy = empty;
        EXPECT_TRUE(cen.run_blocked(copy.size(), copy.data(), block.data()));
        EXPECT_EQ(copy, empty); 

        cen.set_block_mode(scran::CenterSizeFactors::LOWEST);
        EXPECT_TRUE(cen.run_blocked(copy.size(), copy.data(), block.data()));
        EXPECT_EQ(copy, empty); 
    }
}

TEST_F(CenterSizeFactorsTester, SimpleError) {
    scran::CenterSizeFactors cen;
    std::vector<double> empty(sf.size());
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

    auto block = create_block(sf.size());
    EXPECT_ANY_THROW({
        try {
            cen.run_blocked(empty.size(), empty.data(), block.data());

        } catch (std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("negative") != std::string::npos);
            throw;
        }
    });
}
