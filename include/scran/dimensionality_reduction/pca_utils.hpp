#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include "../utils/macros.hpp"

#include "Eigen/Dense"
#include <algorithm>
#include <cmath>
#include <vector>
#include "tatami/tatami.hpp"

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


template<bool byrow, class SparseMat>
void fill_sparse_matrix(SparseMat& A, 
    const std::vector<std::vector<int> >& indices, 
    const std::vector<std::vector<double> >& values, 
    const std::vector<int>& nonzeros)
{
    // Actually filling the matrix. Note the implicit transposition.
    A.reserve(nonzeros);
    for (size_t z = 0; z < values.size(); ++z) {
        const auto& curi = indices[z];
        const auto& curv = values[z];
        for (size_t i = 0; i < curi.size(); ++i) {
            if constexpr(byrow) {
                A.insert(curi[i], z) = curv[i];
            } else {
                A.insert(z, curi[i]) = curv[i];
            }
        }
    }
}

inline void set_scale(bool scale, Eigen::VectorXd& scale_v, double& total_var) {
    if (scale) {
        total_var = 0;
        for (auto& s : scale_v) {
            if (s) {
                s = std::sqrt(s);
                ++total_var;
            } else {
                s = 1;
            }
        }
    } else {
        total_var = std::accumulate(scale_v.begin(), scale_v.end(), 0.0);
    }
}

inline void apply_scale(bool scale, double val, size_t n, double* ptr, double& total_var) {
    if (scale) {
        if (val) {
            double sd = std::sqrt(val);
            for (size_t c = 0; c < n; ++c, ++ptr) {
                *ptr /= sd;
            }
            ++total_var;
        }
    } else {
        total_var += val;
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

    // Using a no-op deleter in a shared pointer to get it to work with the
    // DelayedSubset without actually taking ownership of 'mat'. This hacky
    // shared pointer should only be used as long as 'mat' is alive.
    std::shared_ptr<const tatami::Matrix<T, IDX> > ptr(mat, [](const tatami::Matrix<T, IDX>*){});

    return tatami::make_DelayedSubset<0>(std::move(ptr), std::move(subset));
}

template<class Matrix>
std::vector<size_t> fill_transposed_compressed_sparse_vectors(const Matrix* mat, std::vector<double>& values, std::vector<int>& indices, int nthreads) {
    size_t NR = mat->nrow(), NC = mat->ncol();
    std::vector<size_t> ptrs(NR + 1);

    if (mat->prefer_rows()) {
        /*** First round, to fetch the number of zeros in each row. ***/
        {
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
        }

        /*** Second round, to populate the vectors. ***/
        {
            for (size_t r = 0; r < NR; ++r) {
                ptrs[r + 1] += ptrs[r];
            }
            values.resize(ptrs.back());
            indices.resize(ptrs.back());

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
                    mat->sparse_row_copy(r, values.data() + offset, indices.data() + offset, tatami::SPARSE_COPY_BOTH, wrk.get());

#ifndef SCRAN_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, nthreads);
#endif
        }

    } else {
        /*** First round, to fetch the number of zeros in each row. ***/
        std::vector<size_t> nonzeros_per_row;
        {
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
                    std::vector<size_t> nonzeros_per_row(NR);
                    std::vector<double> xbuffer(NC);
                    std::vector<int> ibuffer(NC);
                    auto wrk = mat->new_workspace(false);

                    for (size_t c = startcol; c < endcol; ++c) {
                        auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), wrk.get());
                        for (size_t i = 0; i < range.number; ++i) {
                            ++(nonzeros_per_row[range.index[i]]);
                        }
                    }

                    threaded_nonzeros_per_row[t] = std::move(nonzeros_per_row);
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif

            // There had better be at least one thread!
            nonzeros_per_row = std::move(threaded_nonzeros_per_row[0]);
            for (int t = 1; t < nthreads; ++t) {
                auto it = nonzeros_per_row.begin();
                for (auto x : threaded_nonzeros_per_row[t]) {
                    *it += x;
                    ++it;
                }
            }
        }

        /*** Second round, to populate the vectors. ***/
        {
            size_t total_nzeros = 0;
            for (size_t r = 0; r < NR; ++r) {
                total_nzeros += nonzeros_per_row[r];
                ptrs[r + 1] = total_nzeros;
            }
            values.resize(total_nzeros);
            indices.resize(total_nzeros);

            // Splitting by row this time, because columnar extraction can't be done safely.
            size_t rows_per_thread = std::ceil(static_cast<double>(NR) / nthreads);
            auto ptr_copy = ptrs;

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
                            values[offset] = range.value[i];
                            indices[offset] = c;
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
        }
    }

    return ptrs;
}

}

}

#endif
