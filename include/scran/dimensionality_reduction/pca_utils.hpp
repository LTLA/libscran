#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include "Eigen/Dense"
#include <algorithm>
#include <cmath>
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

class EigenThreadScope {
public:
#ifdef _OPENMP
    EigenThreadScope(int n) : previous(Eigen::nbThreads()) {
        Eigen::setNbThreads(n);        
    }

    EigenThreadScope(const EigenThreadScope&) = delete;
    EigenThreadScope(EigenThreadScope&&) = delete;
    EigenThreadScope& operator=(const EigenThreadScope&) = delete;
    EigenThreadScope& operator=(EigenThreadScope&&) = delete;
    
    ~EigenThreadScope() { 
        Eigen::setNbThreads(previous);
    }
#else
    EigenThreadScope(int dummy) {}
#endif
private:
    int previous;
};

}

}

#endif
