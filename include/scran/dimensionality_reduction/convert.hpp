#ifndef SCRAN_PCA_CONVERT_HPP
#define SCRAN_PCA_CONVERT_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tatami/tatami.hpp"
#include "Eigen/Dense"
#include "irlba/parallel.hpp"

namespace scran {

namespace pca_utils {

struct SparseComponents {
    std::vector<size_t> ptrs;
    std::vector<double> values;
    std::vector<int> indices;
};

/*
 * We use a two-pass philosophy to create a CSR matrix, which is a little
 * slower but reduces memory usage. Otherwise, we'd either have to have to
 * store two realized copies of a double-precision sparse matrix, if we
 * extracted each row/column into its own vector, before consolidating into a
 * single CSR matrix; or we'd have to suffer repeated reallocations or 
 * overallocations to accommodate an unknown total number of non-zeros.
*/

namespace extract_for_pca_internal {

template<typename T, typename IDX>
SparseComponents sparse_by_row(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    SparseComponents output;
    auto NR = mat->nrow(), NC = mat->ncol();
    auto& ptrs = output.ptrs;
    ptrs.resize(NR + 1);

    /*** First round, to fetch the number of zeros in each row. ***/
    tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
        tatami::Options opt;
        opt.sparse_extract_value = false;
        opt.sparse_extract_index = false;
        auto ext = tatami::consecutive_extractor<true, true>(mat, start, length, opt);

        for (IDX r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(r, NULL, NULL);
            ptrs[r + 1] = range.number;
        }
    }, NR, nthreads);

    /*** Second round, to populate the vectors. ***/
    for (size_t r = 0; r < NR; ++r) {
        ptrs[r + 1] += ptrs[r];
    }
    output.values.resize(ptrs.back());
    output.indices.resize(ptrs.back());

    tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
        auto ext = tatami::consecutive_extractor<true, true>(mat, start, length);
        for (IDX r = start, end = start + length; r < end; ++r) {
            auto offset = ptrs[r];
            ext->fetch_copy(r, output.values.data() + offset, output.indices.data() + offset);
        }
    }, NR, nthreads);

    return output;
}

template<typename T, typename IDX>
SparseComponents sparse_by_column(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    auto NR = mat->nrow(), NC = mat->ncol();

    /*** First round, to fetch the number of zeros in each row. ***/
    std::vector<std::vector<size_t> > threaded_nonzeros_per_row(nthreads);
    for (auto& nonzeros : threaded_nonzeros_per_row) {
        nonzeros.resize(NR);
    }

    tatami::parallelize([&](size_t t, IDX start, IDX length) -> void {
        tatami::Options opt;
        opt.sparse_extract_value = false;
        opt.sparse_ordered_index = false;
        auto ext = tatami::consecutive_extractor<false, true>(mat, start, length, opt);

        std::vector<IDX> ibuffer(NR);
        auto& nonzeros = threaded_nonzeros_per_row[t];
        for (IDX c = start, end = start + length; c < end; ++c) {
            auto range = ext->fetch(c, NULL, ibuffer.data());
            for (IDX j = 0; j < range.number; ++j) {
                ++nonzeros[range.index[j]];
            }
        }
    }, NC, nthreads);

    // There had better be at least one thread!
    std::vector<size_t> nonzeros_per_row = std::move(threaded_nonzeros_per_row[0]);
    for (int t = 1; t < nthreads; ++t) {
        auto it = nonzeros_per_row.begin();
        for (auto x : threaded_nonzeros_per_row[t]) {
            *it += x;
            ++it;
        }
    }

    /*** Second round, to populate the vectors. ***/
    SparseComponents output;
    output.ptrs.resize(NR + 1);
    size_t total_nzeros = 0;
    for (size_t r = 0; r < NR; ++r) {
        total_nzeros += nonzeros_per_row[r];
        output.ptrs[r + 1] = total_nzeros;
    }
    output.values.resize(total_nzeros);
    output.indices.resize(total_nzeros);

    // Splitting by row this time, as it can't otherwise be safely parallelized.
    auto ptr_copy = output.ptrs;
    tatami::parallelize([&](size_t t, IDX start, IDX length) -> void {
        tatami::Options opt;
        opt.sparse_ordered_index = false;
        auto ext = tatami::consecutive_extractor<false, true>(mat, 0, NC, start, length, opt);

        std::vector<T> vbuffer(length);
        std::vector<IDX> ibuffer(length);

        for (size_t c = 0; c < NC; ++c) {
            auto range = ext->fetch(c, vbuffer.data(), ibuffer.data());
            for (size_t i = 0; i < range.number; ++i) {
                auto r = range.index[i];
                auto& offset = ptr_copy[r];
                output.values[offset] = range.value[i];
                output.indices[offset] = c;
                ++offset;
            }
        }
    }, NR, nthreads);

    return output;
}

}

template<typename T, typename IDX>
SparseComponents extract_sparse_for_pca(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    if (mat->prefer_rows()) {
        return extract_for_pca_internal::sparse_by_row(mat, nthreads);
    } else {
        return extract_for_pca_internal::sparse_by_column(mat, nthreads);
    }
}

/************************************************
 ************************************************/

namespace extract_for_pca_internal {

template<typename T, typename IDX>
Eigen::MatrixXd dense_by_row(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    auto NR = mat->nrow(), NC = mat->ncol();
    Eigen::MatrixXd output(NC, NR); // transposed, we want our genes in the columns.
    auto ptr = output.data();

    tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
        auto ext = tatami::consecutive_extractor<true, false>(mat, start, length);
        for (IDX r = start, end = start + length; r < end; ++r) {
            ext->fetch_copy(r, ptr + r * NC);
        }
    }, NR, nthreads);

    return output;
}

template<typename T, typename IDX>
Eigen::MatrixXd dense_by_column(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    auto NR = mat->nrow(), NC = mat->ncol();
    Eigen::MatrixXd output(NC, NR); // transposed, we want our genes in the columns.

    tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
        auto ext = tatami::consecutive_extractor<false, false>(mat, 0, NC, start, length);
        std::vector<T> buffer(length);

        for (size_t c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(c, buffer.data());
            for (size_t r = 0; r < length; ++r) {
                output(c, r + start) = ptr[r];
            }
        }
    }, NR, nthreads);

    return output;
}

}

template<typename T, typename IDX>
Eigen::MatrixXd extract_dense_for_pca(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    if (mat->prefer_rows()) {
        return extract_for_pca_internal::dense_by_row(mat, nthreads);
    } else {
        return extract_for_pca_internal::dense_by_column(mat, nthreads);
    }
}

}

}

#endif
