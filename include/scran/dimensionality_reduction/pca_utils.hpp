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

inline void set_scale(bool scale, Eigen::VectorXd& scale_v, double& total_var) {
    if (scale) {
        total_var = 0;
        for (auto& s : scale_v) {
            if (s) {
                s = std::sqrt(s);
                ++total_var;
            } else {
                s = 1; // avoid division by zero.
            }
        }
    } else {
        total_var = std::accumulate(scale_v.begin(), scale_v.end(), 0.0);
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
    return tatami::make_DelayedSubset<0>(tatami::wrap_shared_ptr(ptr), std::move(subset));
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

// Compute mean and variance from column-major matrix.
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
                // set_scale should avoid division by zero.
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

}

}

#endif
