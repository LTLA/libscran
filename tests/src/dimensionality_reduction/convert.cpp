#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"

#include "scran/dimensionality_reduction/convert.hpp"

#include <vector>
#include <random>

class ExtractForPcaTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    std::shared_ptr<tatami::NumericMatrix> sparse_row, sparse_column;

    template<class Param>
    void assemble(const Param& param) {
        auto NR = std::get<0>(param);
        auto NC = std::get<1>(param);
        nthreads = std::get<2>(param);

        std::mt19937_64 rng(NR * NC / nthreads);
        std::normal_distribution<> ndist;
        std::uniform_real_distribution<> udist(0,1);

        ptrs.resize(NR + 1);
        for (size_t r = 0; r < NR; ++r) {
            for (size_t c = 0; c < NC; ++c) {
                if (udist(rng) < 0.2) {
                    auto val = ndist(rng);
                    values.push_back(val);
                    indices.push_back(c);
                    ++ptrs[r+1];
                }
            }
        }

        for (size_t i = 0; i < NR; ++i) {
            ptrs[i+1] += ptrs[i];
        }

        sparse_row.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, values, indices, ptrs));
        sparse_column = tatami::convert_to_sparse(sparse_row.get(), 1);
        return;
    }

    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> ptrs;
    int nthreads;
};

TEST_P(ExtractForPcaTest, Sparse) {
    auto param = GetParam();
    assemble(param);

    {
        EXPECT_TRUE(sparse_row->prefer_rows());

        auto extracted = scran::pca_utils::extract_sparse_for_pca(sparse_row.get(), nthreads);
        EXPECT_EQ(values, extracted.values);
        EXPECT_EQ(indices, extracted.indices);
        EXPECT_EQ(ptrs, extracted.ptrs);
    }

    {
        EXPECT_FALSE(sparse_column->prefer_rows());

        auto extracted = scran::pca_utils::extract_sparse_for_pca(sparse_column.get(), nthreads);
        EXPECT_EQ(values, extracted.values);
        EXPECT_EQ(indices, extracted.indices);
        EXPECT_EQ(ptrs, extracted.ptrs);
    }
}

TEST_P(ExtractForPcaTest, Dense) {
    auto param = GetParam();
    assemble(param);

    {
        EXPECT_TRUE(sparse_row->prefer_rows());

        auto extracted = scran::pca_utils::extract_dense_for_pca(sparse_row.get(), nthreads);

        size_t NR = extracted.rows(), NC = extracted.cols();
        EXPECT_EQ(NR, sparse_row->ncol());
        EXPECT_EQ(NC, sparse_row->nrow());

        auto ext = sparse_row->dense_column();
        for (size_t r = 0; r < NR; ++r) {
            auto row = extracted.row(r);
            EXPECT_EQ(std::vector<double>(row.begin(), row.end()), ext->fetch(r));
        }
    }

    {
        EXPECT_FALSE(sparse_column->prefer_rows());

        auto extracted = scran::pca_utils::extract_dense_for_pca(sparse_column.get(), nthreads);

        size_t NR = extracted.rows(), NC = extracted.cols();
        EXPECT_EQ(NR, sparse_row->ncol());
        EXPECT_EQ(NC, sparse_row->nrow());

        auto ext = sparse_row->dense_column();
        for (size_t r = 0; r < NR; ++r) {
            auto row = extracted.row(r);
            EXPECT_EQ(std::vector<double>(row.begin(), row.end()), ext->fetch(r));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ExtractForPca,
    ExtractForPcaTest,
    ::testing::Combine(
        ::testing::Values(12, 56, 123), // rows
        ::testing::Values(19, 47, 98), // columns
        ::testing::Values(1, 3) // number of threads
    )
);
