#ifndef SCRAN_MULTI_BATCH_PCA
#define SCRAN_MULTI_BATCH_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"

namespace scran {

template<bool transposed, class Matrix, typename Weight>
struct MultiBatchEigenMatrix {
    MultiBatchEigenMatrix(const Matrix& m, const Weight* w, const Eigen::MatrixXd& mx) : mat(m), weights(w), means(mx) {}

    auto rows() const { return mat.rows(); }
    auto cols() const { return mat.cols(); }

    template<class Right>
    auto operator*(const Right& rhs) const {
        if constexpr(!transposed) {
            Eigen::MatrixXd raw = mat * rhs;
            Eigen::MatrixXd sub = means * rhs;
            for (Eigen::Index c = 0; c < raw.cols(); ++c) {
                auto meanval = sub(0, c);
                for (Eigen::Index r = 0; r < raw.rows(); ++r) {
                    auto& curval = raw(r, c); 
                    curval -= meanval;
                    curval *= weights[r];
                }
            }
            return raw;
        } else {
            Eigen::MatrixXd rhs_copy(rhs);
            std::vector<double> sums(rhs_copy.cols());
            for (Eigen::Index c = 0; c < rhs_copy.cols(); ++c) {
                for (Eigen::Index r = 0; r < rhs_copy.rows(); ++r) {
                    auto& val = rhs_copy(r, c);
                    val *= weights[r];
                    sums[c] += val;
                }
            }

            Eigen::MatrixXd output = mat.adjoint() * rhs_copy;
            for (Eigen::Index c = 0; c < output.cols(); ++c) {
                for (Eigen::Index r = 0; r < output.rows(); ++r) {
                    output(r, c) -= means(0, r) * sums[c];
                }
            }

            return output;
        }
    }

    MultiBatchEigenMatrix<!transposed, Matrix, Weight> adjoint() const {
        return MultiBatchEigenMatrix<!transposed, Matrix, Weight>(mat, weights, means);
    }

    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output(mat);
        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                auto& val = output(r, c);
                val -= means(0, c);
                val *= weights[r];
            }
        }
        if constexpr(transposed) {
            output.adjointInPlace();
        }
        return output;
    }
private:
    const Matrix& mat;
    const Weight* weights;
    const Eigen::MatrixXd& means;
};

}

#endif
