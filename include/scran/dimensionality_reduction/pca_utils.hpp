#ifndef SCRAN_PCA_UTILS_HPP
#define SCRAN_PCA_UTILS_HPP

#include "Eigen/Dense"

namespace scran {

namespace pca_utils {

inline void clean_up(size_t NC, const Eigen::MatrixXd& U, const Eigen::VectorXd& D, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained) {
    pcs = U;
    for (int i = 0; i < U.cols(); ++i) {
        for (size_t j = 0; j < NC; ++j) {
            pcs(j, i) *= D[i];
        }
    }

    variance_explained.resize(D.size());
    for (int i = 0; i < D.size(); ++i) {
        variance_explained[i] = D[i] * D[i] / static_cast<double>(NC - 1);
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

}

}

#endif
