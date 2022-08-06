#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "tatami/base/CompressedSparseMatrix.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "scran/dimensionality_reduction/pca_utils.hpp"

#include <vector>
#include <random>

class FillCompressedSparseTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
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

TEST_P(FillCompressedSparseTest, Basic) {
    auto param = GetParam();
    assemble(param);

    {
        std::vector<double> values2;
        std::vector<int> indices2;
        std::vector<size_t> ptrs2 = scran::pca_utils::fill_transposed_compressed_sparse_vectors(sparse_row.get(), values2, indices2, nthreads);

        EXPECT_EQ(values, values2);
        EXPECT_EQ(indices, indices2);
        EXPECT_EQ(ptrs, ptrs2);
    }

    {
        std::vector<double> values2;
        std::vector<int> indices2;
        std::vector<size_t> ptrs2 = scran::pca_utils::fill_transposed_compressed_sparse_vectors(sparse_column.get(), values2, indices2, nthreads);

        EXPECT_EQ(values, values2);
        EXPECT_EQ(indices, indices2);
        EXPECT_EQ(ptrs, ptrs2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FillCompressedSparse,
    FillCompressedSparseTest,
    ::testing::Combine(
        ::testing::Values(12, 56, 123), // rows
        ::testing::Values(19, 47, 98), // columns
        ::testing::Values(1, 3) // number of threads
    )
);
