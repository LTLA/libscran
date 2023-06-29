#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tatami/tatami.hpp"
#include "Eigen/Dense"
#include "irlba/parallel.hpp"

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

inline double process_scale_vector(bool scale, Eigen::VectorXd& scale_v) {
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

typedef irlba::ParallelSparseMatrix<> SparseMatrix;

inline void compute_mean_and_variance_from_sparse_matrix(const SparseMatrix& emat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, int nthreads) {
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t ncells = emat.rows();
        auto& ptrs = emat.get_pointers();
        auto& values = emat.get_values();
        auto& indices = emat.get_indices();

        for (int r = start, end = start + length; r < end; ++r) {
            auto offset = ptrs[r];

            tatami::SparseRange<double, int> range;
            range.number = ptrs[r + 1] - offset;
            range.value = values.data() + offset;
            range.index = indices.data() + offset;

            auto results = tatami::stats::variances::compute_direct(range, ncells);
            center_v.coeffRef(r) = results.first;
            scale_v.coeffRef(r) = results.second;
        }
    }, emat.cols(), nthreads);
}

inline void compute_mean_and_variance_from_dense_matrix(const Eigen::MatrixXd& emat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, int nthreads) {
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t ncells = emat.rows();
        const double* ptr = emat.data() + static_cast<size_t>(start) * ncells; // enforce size_t to avoid overflow issues.
        for (size_t c = start, end = start + length; c < end; ++c, ptr += ncells) {
            auto results = tatami::stats::variances::compute_direct(ptr, ncells);
            center_v.coeffRef(c) = results.first;
            scale_v.coeffRef(c) = results.second;
        }
    }, emat.cols(), nthreads);
}

inline void apply_center_and_scale_to_dense_matrix(Eigen::MatrixXd& emat, const Eigen::VectorXd& center_v, bool scale, const Eigen::VectorXd& scale_v, int nthreads) {
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NR = emat.rows();
        double* ptr = emat.data() + static_cast<size_t>(start) * NR;
        for (size_t c = start, end = start + length; c < end; ++c, ptr += NR) {
            auto mean = center_v[c];
            for (size_t r = 0; r < NR; ++r) {
                ptr[r] -= mean;
            }

            if (scale) {
                auto sd = scale_v[c];
                for (size_t r = 0; r < NR; ++r) {
                    ptr[r] /= sd; // process_scale_vector should already protect against division by zero.
                }
            }
        }
    }, emat.cols(), nthreads);
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

}

}

#endif
