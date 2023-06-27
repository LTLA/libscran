#ifndef SCRAN_PCA_WRAPPERS_HPP
#define SCRAN_PCA_WRAPPERS_HPP

#include "Eigen/Dense"
#include <type_traits>

namespace scran {

namespace pca_utils {

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
        Workspace(size_t nblocks, irlba::WrappedWorkspace<Matrix> c) : sub(nblocks), child(std::move(c)) {}
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
            output.coeffRef(i) -= work.sub.coeff(block[i]);
        }
        return;
    }

public:
    struct AdjointWorkspace {
        AdjointWorkspace(size_t nblocks, irlba::WrappedAdjointWorkspace<Matrix> c) : aggr(nblocks), child(std::move(c)) {}
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
        return;
    }

public:
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = irlba::wrapped_realize(mat);

        for (Eigen::Index c = 0, cend = output.cols(); c < cend; ++c) {
            for (Eigen::Index r = 0, rend = output.rows(); r < rend; ++r) {
                output.coeffRef(r, c) -= means->coeff(block[r], c);
            }
        }

        return output;
    }

private:
    const Matrix_* mat;
    const Block_* block;
    const Eigen::MatrixXd* means;
};

/*
 * Multiplication of the weighted matrix where each block is downscaled according to its size.
 */
template<class Matrix_>
struct EquiweightWrapper {
    EquiweightWrapper(const Matrix_* m, const Eigen::VectorXd* w, const Eigen::VectorXd* mx) : mat(m), weights(w), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

public:
    typedef irlba::WrappedWorkspace<Matrix_> Workspace;

    Workspace workspace() const {
        return irlba::wrapped_workspace(mat);
    }

    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, Eigen::VectorXd& output) const {
        irlba::wrapped_multiply(mat, rhs, work, output);
        double sub = means->dot(rhs);
        output.noalias() -= sub;
        output.noalias() *= (*weights);
    }

public:
    struct AdjointWorkspace {
        AdjointWorkspace(size_t n, irlba::WrappedAdjointWorkspace<Matrix_> c) : combined(n), child(std::move(c)) {}
        Eigen::VectorXd combined;
        irlba::WrappedAdjointWorkspace<Matrix_> child;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace(weights->size(), irlba::wrapped_adjoint_workspace(mat));
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& output) const {
        work.combined.noalias() = weights->cwiseProduct(rhs);
        double sum = work.combined.sum();
        irlba::wrapped_adjoint_multiply(mat, work.combined, work.child, output);
        output.noalias() -= (*means) * sum;
        return;
    }

public:
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = irlba::wrapped_realize(mat);

        for (Eigen::Index c = 0, cend = output.cols(); c < cend; ++c) {
            for (Eigen::Index r = 0, rend = output.rows(); r < rend; ++r) {
                auto& val = output.coeffRef(r, c);
                val -= means->coeff(c);
                val *= weights->coeff(r);
            }
        }
        return output;
    }

private:
    const Matrix_* mat;
    const Eigen::VectorXd* weights;
    const Eigen::VectorXd* means;
};

/*
 * Multiplication of residual matrix where each block is downscaled according to its size.
 */
template<class Matrix_, typename Block_>
struct EquiweightRegressWrapper {
    EquiweightRegressWrapper(const Matrix_* m, const Block_* b, const Eigen::VectorXd* w, const Eigen::MatrixXd* mx) : mat(m), block(b), weights(w), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

public:
    struct Workspace {
        Workspace(size_t nblocks, irlba::WrappedWorkspace<Matrix> c) : sub(nblocks), child(std::move(c)) {}
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
            val *= weights->coeff(i);
        }
    }

public:
    struct AdjointWorkspace {
        AdjointWorkspace(size_t n, irlba::WrappedAdjointWorkspace<Matrix> c) : combined(n), child(std::move(c)) {}
        Eigen::VectorXd combined;
        irlba::WrappedAdjointWorkspace<Matrix> child;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace(weights->size(), irlba::wrapped_adjoint_workspace(mat));
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& output) const {
        irlba::wrapped_adjoint_multiply(mat, work.combined, work.child, output);

        work.aggr.setZero();
        for (Eigen::Index i = 0, end = rhs.size(); i < end; ++i) {
            work.aggr.coeffRef(block[i]) += rhs.coeff(i); 
        }

        work.combined.noalias() = weights->cwiseProduct(rhs);
        double sum = work.combined.sum();

        output.noalias() -= means->adjoint() * work.aggr * sum;
    }

public:
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = irlba::wrapped_realize(mat);

        for (Eigen::Index c = 0, cend = output.cols(); c < cend; ++c) {
            for (Eigen::Index r = 0, rend = output.rows(); r < rend; ++r) {
                auto& val = output.coeffRef(r, c);
                val -= means->coeff(block[r], c);
                val *= weights->coeff(r);
            }
        }

        return output;
    }

private:
    const Matrix_* mat;
    const Block_* block;
    const Eigen::VectorXd* weights;
    const Eigen::MatrixXd* means;
};

}

}

#endif
