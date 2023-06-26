#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/tatami.hpp"

#include "scran/quality_control/FilterCells.hpp"

#include <cmath>

class FilterCellsTester : public ::testing::Test {
protected:
    void SetUp() {
        mat = std::unique_ptr<tatami::NumericMatrix>(new tatami::DenseRowMatrix<double>(sparse_nrow, sparse_ncol, sparse_matrix));
    }
protected:
    std::shared_ptr<tatami::NumericMatrix> mat;

    std::vector<int> to_filter (const std::vector<size_t>& indices) {
        std::vector<int> keep_s(mat->ncol());
        for (auto i : indices) { keep_s[i] = 1; }
        return keep_s;        
    }
};

TEST_F(FilterCellsTester, RetainSubset) {
    std::vector<size_t> keep_i = { 0, 5, 7, 8, 9 };
    auto keep_s = to_filter(keep_i);

    scran::FilterCells filter;
    auto filtered = filter.set_retain().run(mat, keep_s.data());
    EXPECT_EQ(filtered->nrow(), mat->nrow());
    EXPECT_EQ(filtered->ncol(), keep_i.size());
    
    std::vector<double> buffer(mat->nrow());
    auto mext = mat->dense_column();
    auto fext = filtered->dense_column();

    for (size_t c = 0; c < keep_i.size(); ++c) {
        auto filt = fext->fetch(c);
        auto ref = mext->fetch(keep_i[c]);
        EXPECT_EQ(filt, ref);
    }
}

TEST_F(FilterCellsTester, DiscardSubset) {
    std::vector<size_t> discard_i = { 1, 5, 7, 8 };
    auto discard_s = to_filter(discard_i);

    scran::FilterCells filter;
    auto filtered = filter.run(mat, discard_s.data());
    EXPECT_EQ(filtered->nrow(), mat->nrow());
    EXPECT_EQ(filtered->ncol(), mat->ncol() - discard_i.size());

    std::vector<double> buffer(mat->nrow());
    auto mext = mat->dense_column();
    auto fext = filtered->dense_column();

    size_t counter = 0;
    for (size_t c = 0; c < mat->ncol(); ++c) {
        if (!discard_s[c]) {
            auto filt = fext->fetch(counter);
            auto ref = mext->fetch(c);
            EXPECT_EQ(filt, ref);
            ++counter;
        }
    }
}
