#ifndef SCRAN_RUN_PCA
#define SCRAN_RUN_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>

namespace scran {

class RunPCA {
    int rank = 10;
    bool scale = false;
    double sparsity = 0.1;
public:
    irlba::Irlba irb;
public:
    RunPCA& set_rank(int r = 10) {
        rank = r;
        return *this;
    }

    RunPCA& set_scale(bool s = false) {
        scale = s;
        return *this;
    }

    RunPCA& set_sparsity(double s = 0.1) {
        sparsity = s;
        return *this;
    }

private:
    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* mat, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained, double& total_var) {
        irlba::NormalSampler norm(42);
        irb.set_number(rank);

        if (mat->sparse()) {
            Eigen::VectorXd center_v(mat->nrow()), scale_v(mat->nrow());
            auto emat = create_eigen_matrix_sparse(mat, center_v, scale_v, total_var);
            auto result = irb.run(emat, center_v, scale_v, norm);
            clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
        } else {
            auto emat = create_eigen_matrix_dense(mat, total_var);
            auto result = irb.run(emat, norm); // already centered and scaled, if relevant.
            clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
        }

        return;
    }

public:
    struct Results {
        Eigen::MatrixXd pcs;
        Eigen::VectorXd variance_explained;
        double total_variance = 0;
    };

    template<typename T, typename IDX>
    Results run(const tatami::Matrix<T, IDX>* mat) {
        Results output;
        run(mat, output.pcs, output.variance_explained, output.total_variance);
        return output;
    }

    template<class MAT, typename X>
    Results run(std::shared_ptr<MAT> mat, const X* features) {
        Results output;

        if (!features) {
            run(mat.get(), output.pcs, output.variance_explained, output.total_variance);
        } else {
            std::vector<int> subset;
            subset.reserve(mat->nrow());
            for (size_t r = 0; r < mat->nrow(); ++r) {
                if (features[r]) {
                    subset.push_back(r);
                }
            }

            auto subsetted = tatami::make_DelayedSubset<0>(std::move(mat), std::move(subset));
            run(subsetted.get(), output.pcs, output.variance_explained, output.total_variance);
        }

        return output;
    }

private:
    void clean_up(size_t NC, const Eigen::MatrixXd& U, const Eigen::VectorXd& D, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained) {
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

private:
    template<typename T, typename IDX> 
    Eigen::SparseMatrix<double> create_eigen_matrix_sparse(const tatami::Matrix<T, IDX>* mat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, double& total_var) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        // Filling up the vector of triplets. We pre-allocate at an assumed 10%
        // density, so as to avoid unnecessary movements.
        typedef Eigen::Triplet<double> TRIPLET;
        std::vector<TRIPLET> triplets;
        triplets.reserve(static_cast<double>(NR * NC) * sparsity);

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                auto stats = tatami::stats::variances::compute_direct(range, NC);
                center_v[r] = stats.first;
                if (scale) {
                    ++total_var;
                    scale_v[r] = std::sqrt(stats.second);
                } else {
                    total_var += stats.second;
                    scale_v[r] = 1;
                }

                for (size_t i = 0; i < range.number; ++i) {
                    triplets.push_back(TRIPLET(range.index[i], r, range.value[i])); // transposing.
                }
            }

        } else {
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);

            center_v.setZero();
            scale_v.setZero();
            std::vector<int> nonzeros(NR);
            int count = 0;
            
            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                tatami::stats::variances::compute_running(range, center_v.data(), scale_v.data(), nonzeros.data(), count);
                for (size_t i = 0; i < range.number; ++i) {
                    triplets.push_back(TRIPLET(c, range.index[i], range.value[i])); // transposing.
                }
            }

            tatami::stats::variances::finish_running(NR, center_v.data(), scale_v.data(), nonzeros.data(), count);

            if (scale) {
                for (auto& s : scale_v) { s = std::sqrt(s); }
                total_var = NR;
            } else {
                total_var = std::accumulate(scale_v.begin(), scale_v.end(), 0.0);
                std::fill(scale_v.begin(), scale_v.end(), 1);
            }
        }

        Eigen::SparseMatrix<double> A(NC, NR); // transposed
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        return A;
    }

private:
    template<typename T, typename IDX> 
    Eigen::MatrixXd create_eigen_matrix_dense(const tatami::Matrix<T, IDX>* mat, double& total_var) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::MatrixXd output(NC, NR); // transposed.
        std::vector<double> xbuffer(NC);
        double* outIt = output.data();

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);
            auto stats = tatami::stats::variances::compute_direct(outIt, NC);

            auto copy = outIt;
            for (size_t c = 0; c < NC; ++c, ++copy) {
                *copy -= stats.first;
            }

            if (scale) {
                double sd = std::sqrt(stats.second);
                auto copy = outIt;
                for (size_t c = 0; c < NC; ++c, ++copy) {
                    *copy /= sd;
                }
                ++total_var;
            } else {
                total_var += stats.second;
            }
        }

        return output;
    }

};

}

#endif
