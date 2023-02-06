#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tatami/tatami.hpp"
#include "Eigen/Dense"

namespace scran {

namespace pca_utils {

inline void clean_up(size_t NC, Eigen::MatrixXd& U, Eigen::VectorXd& D) {
    auto uIt = U.data();
    auto dIt = D.data();
    for (int i = 0, iend = U.cols(); i < iend; ++i, ++dIt) {
        for (int j = 0, jend = U.rows(); j < jend; ++j, ++uIt) {
            (*uIt) *= (*dIt);
        }
    }

    for (auto& d : D) {
        d = d * d / static_cast<double>(NC - 1);
    }
    return;
}

inline double variance_to_scale(bool scale, Eigen::VectorXd& scale_v) {
    if (scale) {
        double total_var = 0;
        for (auto& s : scale_v) {
            if (s) {
                s = std::sqrt(s);
                ++total_var;
            } else {
                s = 1; // avoid division by zero.
            }
        }
        return total_var;
    } else {
        return std::accumulate(scale_v.begin(), scale_v.end(), 0.0);
    }
}

template<typename T, typename IDX, typename X>
std::shared_ptr<const tatami::Matrix<T, IDX> > subset_matrix_by_features(const tatami::Matrix<T, IDX>* mat, const X * features) {
    std::vector<int> subset;
    subset.reserve(mat->nrow());
    for (size_t r = 0; r < mat->nrow(); ++r) {
        if (features[r]) {
            subset.push_back(r);
        }
    }
    return tatami::make_DelayedSubset<0>(tatami::wrap_shared_ptr(mat), std::move(subset));
}

// Compute mean and variance from CSR components.
inline void compute_mean_and_variance_from_sparse_components( 
    size_t NR,
    size_t NC,
    const std::vector<double>& values,
    const std::vector<int>& indices,
    const std::vector<size_t>& ptrs,
    Eigen::VectorXd& centers,
    Eigen::VectorXd& variances,
    int nthreads
) {

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t r = 0; r < NR; ++r) {
#else
    SCRAN_CUSTOM_PARALLEL(NR, [&](size_t first, size_t last) -> void {
    for (size_t r = first; r < last; ++r) {
#endif

        auto offset = ptrs[r];
        tatami::SparseRange<double, int> range;
        range.number = ptrs[r+1] - offset;
        range.value = values.data() + offset;
        range.index = indices.data() + offset;

        auto results = tatami::stats::variances::compute_direct(range, NC);
        centers.coeffRef(r) = results.first;
        variances.coeffRef(r) = results.second;

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

    return;
}

// Compute mean and variance from column-major dense matrix.
inline void compute_mean_and_variance_from_dense_columns(const Eigen::MatrixXd& mat, Eigen::VectorXd& centers, Eigen::VectorXd& variances, int nthreads) {
    size_t NC = mat.cols();
    size_t NR = mat.rows();
    const double* ptr = mat.data();

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t c = 0; c < NC; ++c) {
#else
    SCRAN_CUSTOM_PARALLEL(NC, [&](size_t first, size_t last) -> void {
    for (size_t c = first; c < last; ++c) {
#endif

        auto curptr = ptr + NR * c;
        auto results = tatami::stats::variances::compute_direct(curptr, NR);
        centers[c] = results.first;
        variances[c] = results.second;

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

    return;
}

inline void center_and_scale_dense_columns(Eigen::MatrixXd& mat, const Eigen::VectorXd& centers, bool use_scale, const Eigen::VectorXd& scale, int nthreads) {
    size_t NC = mat.cols();
    size_t NR = mat.rows();
    double* ptr = mat.data();

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t c = 0; c < NC; ++c) {
#else
    SCRAN_CUSTOM_PARALLEL(NC, [&](size_t first, size_t last) -> void {
    for (size_t c = first; c < last; ++c) {
#endif

        auto curptr = ptr + NR * c;
        auto mean = centers[c];
        for (size_t r = 0; r < NR; ++r) {
            curptr[r] -= mean;
        }

        if (use_scale) {
            auto sd = scale[c];
            for (size_t r = 0; r < NR; ++r) {
                // variance_to_scale should already protect against division by zero.
                curptr[r] /= sd;
            }
        }

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif
}

/************************************************
 ************************************************/

struct SparseComponents {
    std::vector<size_t> ptrs;
    std::vector<double> values;
    std::vector<int> indices;
};

namespace extract_for_pca_internal {

/*
 * We use a two-pass philosophy to create a CSR matrix, which is a little
 * slower but reduces memory usage. Otherwise, we'd either have to have to
 * store two realized copies of a double-precision sparse matrix, if we
 * extracted each row/column into its own vector, before consolidating into a
 * single CSR matrix; or we'd have to suffer repeated reallocations or 
 * overallocations to accommodate an unknown total number of non-zeros.
*/

template<typename T, typename IDX>
SparseComponents sparse_by_row(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    SparseComponents output;
    size_t NR = mat->nrow(), NC = mat->ncol();
    auto& ptrs = output.ptrs;
    ptrs.resize(NR + 1);

    /*** First round, to fetch the number of zeros in each row. ***/
#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(nthreads)
    {
#else
    SCRAN_CUSTOM_PARALLEL(NR, [&](size_t start, size_t end) -> void {
#endif

        std::vector<double> xbuffer(NC);
        std::vector<int> ibuffer(NC);
        auto wrk = mat->new_workspace(true);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp for
        for (size_t r = 0; r < NR; ++r) {
#else
        for (size_t r = start; r < end; ++r) {
#endif

            auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data(), wrk.get());
            ptrs[r + 1] = range.number;

#ifndef SCRAN_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    /*** Second round, to populate the vectors. ***/
    for (size_t r = 0; r < NR; ++r) {
        ptrs[r + 1] += ptrs[r];
    }
    output.values.resize(ptrs.back());
    output.indices.resize(ptrs.back());

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(nthreads)
    {
#else
    SCRAN_CUSTOM_PARALLEL(NR, [&](size_t start, size_t end) -> void {
#endif

        auto wrk = mat->new_workspace(true);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp for
        for (size_t r = 0; r < NR; ++r) {
#else
        for (size_t r = start; r < end; ++r) {
#endif

            auto offset = ptrs[r];
            mat->sparse_row_copy(r, output.values.data() + offset, output.indices.data() + offset, tatami::SPARSE_COPY_BOTH, wrk.get());

#ifndef SCRAN_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    return output;
}

template<typename T, typename IDX>
SparseComponents sparse_by_column(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    size_t NR = mat->nrow(), NC = mat->ncol();

    /*** First round, to fetch the number of zeros in each row. ***/
    size_t cols_per_thread = std::ceil(static_cast<double>(NC) / nthreads);
    std::vector<std::vector<size_t> > threaded_nonzeros_per_row(nthreads);

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
#else
    SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
    for (int t = start; t < end; ++t) {
#endif

        size_t startcol = cols_per_thread * t, endcol = std::min(startcol + cols_per_thread, NC);
        if (startcol < endcol) {
            std::vector<size_t>& nonzeros_per_row = threaded_nonzeros_per_row[t];
            nonzeros_per_row.resize(NR);
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);
            auto wrk = mat->new_workspace(false);

            for (size_t c = startcol; c < endcol; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), wrk.get());
                for (size_t i = 0; i < range.number; ++i) {
                    ++(nonzeros_per_row[range.index[i]]);
                }
            }
        }

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

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

    // Splitting by row this time, because columnar extraction can't be done safely.
    size_t rows_per_thread = std::ceil(static_cast<double>(NR) / nthreads);
    auto ptr_copy = output.ptrs;

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
#else
    SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
    for (int t = start; t < end; ++t) {
#endif

        size_t startrow = rows_per_thread * t, endrow = std::min(startrow + rows_per_thread, NR);
        if (startrow < endrow) {
            auto wrk = mat->new_workspace(false);
            std::vector<double> xbuffer(endrow - startrow);
            std::vector<int> ibuffer(endrow - startrow);

            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), startrow, endrow, wrk.get());
                for (size_t i = 0; i < range.number; ++i) {
                    auto r = range.index[i];
                    auto& offset = ptr_copy[r];
                    output.values[offset] = range.value[i];
                    output.indices[offset] = c;
                    ++offset;
                }
            }
        }

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

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
    size_t NC = mat->ncol();
    size_t NR = mat->nrow();
    Eigen::MatrixXd output(NC, NR); // transposed, we want our genes in the columns.
    auto ptr = output.data();

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(nthreads)
    {
        auto wrk = mat->new_workspace(true);

        #pragma omp for
        for (int r = 0; r < NR; ++r) {
#else
    SCRAN_CUSTOM_PARALLEL(NR, [&](int start, int end) -> void {
        auto wrk = mat->new_workspace(true);
        for (int r = start; r < end; ++r) {
#endif

            mat->row_copy(r, ptr + r * NC, wrk.get());

#ifndef SCRAN_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    return output;
}

template<typename T, typename IDX>
Eigen::MatrixXd dense_by_column(const tatami::Matrix<T, IDX>* mat, int nthreads) {
    size_t NC = mat->ncol();
    size_t NR = mat->nrow();
    Eigen::MatrixXd output(NC, NR); // transposed, we want our genes in the columns.

    // Splitting by row this time, to avoid false sharing across threads.
    size_t rows_per_thread = std::ceil(static_cast<double>(NR) / nthreads);

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
#else
    SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
    for (int t = start; t < end; ++t) {
#endif

        size_t startrow = rows_per_thread * t, endrow = std::min(startrow + rows_per_thread, NR);
        if (startrow < endrow) {
            auto wrk = mat->new_workspace(false);
            std::vector<double> buffer(endrow - startrow);

            for (size_t c = 0; c < NC; ++c) {
                auto ptr = mat->column(c, buffer.data(), startrow, endrow, wrk.get());
                for (size_t r = startrow; r < endrow; ++r) {
                    output(c, r) = ptr[r - startrow];
                }
            }
        }

#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

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
