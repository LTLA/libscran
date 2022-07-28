#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/normalization/LogNormCounts.hpp"

#include <cmath>

class LogNormCountsTester : public ::testing::Test {
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
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;
    std::vector<double> sf;
};

TEST_F(LogNormCountsTester, Simple) {
    scran::LogNormCounts lnc;
    auto lognormed = lnc.run(mat, sf);

    // Reference calculation.
    scran::CenterSizeFactors cen;
    cen.run(sf.size(), sf.data());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lognormed->column(i);
        auto output2 = mat->column(i);
        for (auto& o : output2) {
            o = std::log1p(o/sf[i])/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, AnotherPseudo) {
    scran::LogNormCounts lnc;
    auto lognormed = lnc.set_pseudo_count(1.5).run(mat, sf);

    // Reference calculation.
    scran::CenterSizeFactors cen;
    cen.run(sf.size(), sf.data());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lognormed->column(i);
        auto output2 = mat->column(i);
        for (auto& o : output2) {
            o = std::log(o/sf[i] + 1.5)/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, Block) {
    // Assigning them to three blocks.
    std::vector<int> block = create_block(mat->ncol());

    scran::LogNormCounts lnc;
    lnc.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
    auto lognormed = lnc.run_blocked(mat, sf, block.data());

    // Comparing to a reference.
    scran::CenterSizeFactors cen;
    cen.set_block_mode(scran::CenterSizeFactors::PER_BLOCK);
    cen.run_blocked(sf.size(), sf.data(), block.data());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lognormed->column(i);
        auto output2 = mat->column(i);
        for (auto& o : output2) {
            o = std::log1p(o/sf[i])/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, NoCenter) {
    scran::LogNormCounts lnc;
    lnc.set_center(false);
    auto lognormed = lnc.run(mat, sf);

    // Reference calculation.
    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lognormed->column(i);
        auto output2 = mat->column(i);
        for (auto& o : output2) {
            o = std::log1p(o/sf[i])/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, SelfCompute) {
    scran::LogNormCounts lnc;
    auto lognormed = lnc.run(mat);
    auto ref = lnc.run(mat, sf);

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto out1 = lognormed->column(i);
        auto out2 = ref->column(i);
        EXPECT_EQ(out1, out2);
    }

    // Creating blocks.
    std::vector<int> block = create_block(mat->ncol());
    auto lognormed2 = lnc.run_blocked(mat, block.data());
    auto ref2 = lnc.run_blocked(mat, sf, block.data());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto out1 = lognormed2->column(i);
        auto out2 = ref2->column(i);
        EXPECT_EQ(out1, out2);
    }
}

TEST_F(LogNormCountsTester, Error) {
    scran::LogNormCounts lnc;
    auto sf2 = sf;
    sf2.resize(sf.size() - 1);
    EXPECT_ANY_THROW(lnc.run(mat, sf2));

    std::vector<double> empty(mat->ncol());
    EXPECT_ANY_THROW(lnc.run(mat, empty));
}

TEST_F(LogNormCountsTester, NonStrict) {
    scran::LogNormCounts lnc;
    lnc.set_handle_zeros(true);
    std::vector<double> empty(mat->ncol());

    // No division, effectively; all zeroes set to 1.
    {
        auto lognormed = lnc.run(mat, empty);
        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lognormed->column(i);
            auto output2 = mat->column(i);
            for (auto& o : output2) {
                o = std::log1p(o)/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }

    // Uses 0.5, which is the smallest non-zero.
    empty[0] = 0.5;
    empty[1] = 1;
    double center = 1.5 / empty.size();

    {
        auto lognormed = lnc.run(mat, empty);
        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lognormed->column(i);
            auto output2 = mat->column(i);

            auto sf = (i <= 1 ? empty[i] : 0.5) / center;
            for (auto& o : output2) {
                o = std::log1p(o/sf)/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }
}

