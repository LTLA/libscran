#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/tatami.hpp"

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

    auto lext = lognormed->dense_column();
    auto mext = mat->dense_column();

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lext->fetch(i);
        auto output2 = mext->fetch(i);
        for (auto& o : output2) {
            o = std::log1p(o/sf[i])/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, AnotherPseudo) {
    auto copy = sf;
    scran::CenterSizeFactors cen;
    cen.run(copy.size(), copy.data());

    scran::LogNormCounts lnc;
    lnc.set_pseudo_count(1.5);
    lnc.set_sparse_addition(false);
    {
        auto lognormed = lnc.run(mat, sf);
        auto lext = lognormed->dense_column();
        auto mext = mat->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lext->fetch(i);
            auto output2 = mext->fetch(i);
            for (auto& o : output2) {
                o = std::log(o/copy[i] + 1.5)/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }

    lnc.set_sparse_addition(true);
    {
        auto lognormed = lnc.run(mat, sf);
        auto lext = lognormed->dense_column();
        auto mext = mat->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lext->fetch(i);
            auto output2 = mext->fetch(i);
            for (auto& o : output2) {
                o = std::log1p(o/(copy[i]*1.5))/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }

    // Trying with a non-default pseudo-count.
    lnc.set_choose_pseudo_count(true);
    lnc.set_min_value(2);

    {
        double expected = scran::ChoosePseudoCount().set_min_value(2).run(copy.size(), copy.data());

        auto lognormed = lnc.run(mat, sf);
        auto lext = lognormed->dense_column();
        auto mext = mat->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lext->fetch(i);
            auto output2 = mext->fetch(i);
            for (auto& o : output2) {
                o = std::log1p(o/(copy[i]*expected))/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
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

    auto lext = lognormed->dense_column();
    auto mext = mat->dense_column();

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lext->fetch(i);
        auto output2 = mext->fetch(i);
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

    auto lext = lognormed->dense_column();
    auto mext = mat->dense_column();

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto output = lext->fetch(i);
        auto output2 = mext->fetch(i);
        for (auto& o : output2) {
            o = std::log1p(o/sf[i])/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, SelfCompute) {
    scran::LogNormCounts lnc;

    {
        auto lognormed = lnc.run(mat);
        auto ref = lnc.run(mat, sf);

        auto lext = lognormed->dense_column();
        auto rext = ref->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto out1 = lext->fetch(i);
            auto out2 = rext->fetch(i);
            EXPECT_EQ(out1, out2);
        }
    }

    // Creating blocks.
    {
        std::vector<int> block = create_block(mat->ncol());
        auto lognormed = lnc.run_blocked(mat, block.data());
        auto ref = lnc.run_blocked(mat, sf, block.data());

        auto lext = lognormed->dense_column();
        auto rext = ref->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto out1 = lext->fetch(i);
            auto out2 = rext->fetch(i);
            EXPECT_EQ(out1, out2);
        }
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
        auto lext = lognormed->dense_column();
        auto mext = mat->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lext->fetch(i);
            auto output2 = mext->fetch(i);
            for (auto& o : output2) {
                o = std::log1p(o)/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }

    empty[0] = 0.5;
    empty[1] = 1;
    double center = 1.5 / 2; // centering only considers non-zero elements by default.

    {
        auto lognormed = lnc.run(mat, empty);
        auto lext = lognormed->dense_column();
        auto mext = mat->dense_column();

        for (size_t i = 0; i < mat->ncol(); ++i) {
            auto output = lext->fetch(i);
            auto output2 = mext->fetch(i);

            // Uses 0.5 as the fill-in, which is the smallest non-zero.
            auto sf = (i <= 1 ? empty[i] : 0.5) / center;
            for (auto& o : output2) {
                o = std::log1p(o/sf)/std::log(2);
            }
            EXPECT_EQ(output, output2);
        }
    }
}

