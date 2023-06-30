#ifndef SCRAN_PCA_WRAPPERS_HPP
#define SCRAN_PCA_WRAPPERS_HPP

#include "../utils/macros.hpp"

#include "Eigen/Dense"
#include <type_traits>
#include "irlba/irlba.hpp"

namespace scran {

namespace pca_utils {

/*
 * Scales each sample (i.e., row of the matrix) before multiplication.
 * This should be applied after all centering is performed.
 */
template<class Matrix_>
using SampleScaledWrapper = irlba::Scaled<Matrix_, false, false>;

/* 
 * Multiplication of the residuals after centering on block-specific means.
 */
template<class Matrix_, typename Block_>
struct RegressWrapper {
    RegressWrapper(const Matrix_* m, const Block_* b, const Eigen::MatrixXd* mx) : mat(m), block(b), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

public:
    struct Workspace {
        Workspace(size_t nblocks, irlba::WrappedWorkspace<Matrix_> c) : sub(nblocks), child(std::move(c)) {}
        Eigen::VectorXd sub;
        irlba::WrappedWorkspace<Matrix_> child;
    };

    Workspace workspace() const {
        return Workspace(means->rows(), irlba::wrapped_workspace(mat));
    }

    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, Eigen::VectorXd& output) const {
        irlba::wrapped_multiply(mat, rhs, work.child, output);

        work.sub.noalias() = (*means) * rhs;
        for (Eigen::Index i = 0, end = output.size(); i < end; ++i) {
            auto& val = output.coeffRef(i);
            val -= work.sub.coeff(block[i]);
        }
    }

public:
    struct AdjointWorkspace {
        AdjointWorkspace(size_t nblocks, irlba::WrappedAdjointWorkspace<Matrix_> c) : aggr(nblocks), child(std::move(c)) {}
        Eigen::VectorXd aggr;
        irlba::WrappedWorkspace<Matrix_> child;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace(means->rows(), irlba::wrapped_adjoint_workspace(mat));
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, Eigen::VectorXd& output) const {
        irlba::wrapped_adjoint_multiply(mat, rhs, work.child, output);

        work.aggr.setZero();
        for (Eigen::Index i = 0, end = rhs.size(); i < end; ++i) {
            work.aggr.coeffRef(block[i]) += rhs.coeff(i); 
        }

        output.noalias() -= means->adjoint() * work.aggr;
    }

public:
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = irlba::wrapped_realize(mat);

        for (Eigen::Index c = 0, cend = output.cols(); c < cend; ++c) {
            for (Eigen::Index r = 0, rend = output.rows(); r < rend; ++r) {
                auto& val = output.coeffRef(r, c);
                val -= means->coeff(block[r], c);
            }
        }

        return output;
    }

private:
    const Matrix_* mat;
    const Block_* block;
    const Eigen::MatrixXd* means;
};

}

}

#endif
