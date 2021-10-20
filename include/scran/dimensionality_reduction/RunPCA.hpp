#ifndef SCRAN_RUN_PCA
#define SCRAN_RUN_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"

namespace scran {

class RunPCA {
public:
    struct Defaults {
        static constexpr int rank = 10;

        static constexpr bool scale = false;
    };
private:
    int rank = Defaults::rank;
    bool scale = Defaults::scale;
    irlba::Irlba irb;

public:
    RunPCA& set_rank(int r = Defaults::rank) {
        irb.set_number(r);
        return *this;
    }

    RunPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

private:
    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* mat, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained, double& total_var) {

        if (mat->sparse()) {
            Eigen::VectorXd center_v(mat->nrow()), scale_v(mat->nrow());
            auto emat = create_eigen_matrix_sparse(mat, center_v, scale_v, total_var);

#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::RunPCA", "Running the IRLBA algorithm");
#endif

            irlba::Centered<decltype(emat)> centered(&emat, &center_v);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                auto result = irb.run(scaled);
                pca_utils::clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
            } else {
                auto result = irb.run(centered);
                pca_utils::clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
            }
        } else {
            auto emat = create_eigen_matrix_dense(mat, total_var);

#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::RunPCA", "Running the IRLBA algorithm");
#endif

            auto result = irb.run(emat); // already centered and scaled, if relevant.
            pca_utils::clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
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

    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) {
        Results output;

        if (!features) {
            run(mat, output.pcs, output.variance_explained, output.total_variance);
        } else {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("RunPCA", "Subsetting to features of interest");
#endif
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), output.pcs, output.variance_explained, output.total_variance);
        }

        return output;
    }

private:
    template<typename T, typename IDX> 
    Eigen::SparseMatrix<double> create_eigen_matrix_sparse(const tatami::Matrix<T, IDX>* mat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, double& total_var) {
        size_t NR = mat->nrow(), NC = mat->ncol();

        Eigen::SparseMatrix<double> A(NC, NR); // transposed; we want genes in the columns.
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);
            std::vector<int> nonzeros(NR);
            values.reserve(NR);
            indices.reserve(NR);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                auto stats = tatami::stats::variances::compute_direct(range, NC);
                center_v[r] = stats.first;
                scale_v[r] = stats.second;

                nonzeros[r] = range.number;
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            pca_utils::set_scale(scale, scale_v, total_var);
            pca_utils::fill_sparse_matrix<true>(A, indices, values, nonzeros);
        } else {
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);
            values.reserve(NC);
            indices.reserve(NC);

            center_v.setZero();
            scale_v.setZero();
            std::vector<int> nonzeros(NR);
            int count = 0;

            // First pass to compute variances and extract non-zero values.
            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                tatami::stats::variances::compute_running(range, center_v.data(), scale_v.data(), nonzeros.data(), count);
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            tatami::stats::variances::finish_running(NR, center_v.data(), scale_v.data(), nonzeros.data(), count);

            pca_utils::set_scale(scale, scale_v, total_var);
            pca_utils::fill_sparse_matrix<false>(A, indices, values, nonzeros);
        }

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

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);
            auto stats = tatami::stats::variances::compute_direct(outIt, NC);

            auto copy = outIt;
            for (size_t c = 0; c < NC; ++c, ++copy) {
                *copy -= stats.first;
            }

            pca_utils::apply_scale(scale, stats.second, NC, outIt, total_var);
        }

        return output;
    }

};

}

#endif
