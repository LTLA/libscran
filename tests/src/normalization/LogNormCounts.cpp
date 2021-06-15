#include <gtest/gtest.h>

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/stats/sums.hpp"

#include "scran/normalization/LogNormCounts.hpp"

#include <cmath>

class LogNormCountsTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::numeric_matrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
        sf = tatami::column_sums(mat.get());
    }
protected:
    std::shared_ptr<tatami::numeric_matrix> mat;
    std::vector<double> sf;
};

TEST_F(LogNormCountsTester, SimpleTest) {
    scran::LogNormCounts lnc;
    auto lognormed = lnc.run(mat, sf);

    // Reference calculation.
    double mean_sf = std::accumulate(sf.begin(), sf.end(), 0.0)/sf.size();
    std::vector<double> buffer(mat->nrow());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto ptr = lognormed->column(i, buffer.data());
        std::vector<double> output(ptr, ptr + mat->nrow());

        auto ptr2 = mat->column(i, buffer.data());
        std::vector<double> output2(ptr2, ptr2 + mat->nrow());

        for (auto& o : output2) {
            o = std::log1p(o/(sf[i]/mean_sf))/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, AnotherPseudoTest) {
    scran::LogNormCounts lnc;
    auto lognormed = lnc.set_pseudo_count(1.5).run(mat, sf);

    // Reference calculation.
    double mean_sf = std::accumulate(sf.begin(), sf.end(), 0.0)/sf.size();
    std::vector<double> buffer(mat->nrow());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto ptr = lognormed->column(i, buffer.data());
        std::vector<double> output(ptr, ptr + mat->nrow());

        auto ptr2 = mat->column(i, buffer.data());
        std::vector<double> output2(ptr2, ptr2 + mat->nrow());

        for (auto& o : output2) {
            o = std::log(o/(sf[i]/mean_sf) + 1.5)/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, BlockTest) {
    std::vector<int> block(mat->ncol());
    std::vector<double> blocked_sf(3), blocked_num(3);
    for (size_t i = 0; i < block.size(); ++i) { 
        auto index = i % 3;
        block[i] = index;
        blocked_sf[index] += sf[i];
        ++blocked_num[index];
    }
    for (size_t i = 0; i < 3; ++i) {
        blocked_sf[i] /= blocked_num[i];
    }

    scran::LogNormCounts lnc;
    auto lognormed = lnc.set_blocks(block.size(), block.data()).run(mat, sf);
    std::vector<double> buffer(mat->nrow());

    for (size_t i = 0; i < mat->ncol(); ++i) {
        auto ptr = lognormed->column(i, buffer.data());
        std::vector<double> output(ptr, ptr + mat->nrow());

        auto ptr2 = mat->column(i, buffer.data());
        std::vector<double> output2(ptr2, ptr2 + mat->nrow());

        for (auto& o : output2) {
            o = std::log1p(o/(sf[i]/blocked_sf[i%3]))/std::log(2);
        }
        EXPECT_EQ(output, output2);
    }
}

TEST_F(LogNormCountsTester, ErrorTest) {
    scran::LogNormCounts lnc;
    auto sf2 = sf;
    sf2.resize(sf.size() - 1);
    EXPECT_ANY_THROW(lnc.run(mat, sf2));

    std::vector<double> empty(mat->ncol());
    EXPECT_ANY_THROW(lnc.run(mat, empty));
}