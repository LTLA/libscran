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

/**
 * @file RunPCA.hpp
 *
 * @brief Perform PCA on a gene-by-cell matrix.
 */

namespace scran {

/**
 * @brief Perform PCA on a gene-cell matrix.
 *
 * Principal components analysis (PCA) is a helpful technique for data compression and denoising.
 * The idea is that the earlier PCs capture most of the systematic biological variation while the later PCs capture random technical noise.
 * Thus, we can reduce the size of the data and eliminate noise by only using the earlier PCs for further analyses.
 * Most practitioners will keep the first 10-50 PCs, though the exact choice is fairly arbitrary.
 * For speed, we use the [**CppIrlba**](https://github.com/LTLA/CppIrlba) package to perform an approximate PCA.
 */
class RunPCA {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_rank()` for more details.
         */
        static constexpr int rank = 10;

        /**
         * See `set_scale()` for more details.
         */
        static constexpr bool scale = false;
    };
private:
    bool scale = Defaults::scale;
    irlba::Irlba irb;

public:
    /**
     * Constructor. 
     */
    RunPCA() {
        irb.set_number(Defaults::rank);
        return;
    }

    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `RunPCA` instance.
     */
    RunPCA& set_rank(int r = Defaults::rank) {
        irb.set_number(r);
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `RunPCA` instance.
     */
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
    /**
     * @brief Container for the PCA results.
     */
    struct Results {
        /**
         * Matrix of principal components.
         * Each row corresponds to a cell while each column corresponds to a PC,
         * with number of columns determined by `set_rank()`.
         */
        Eigen::MatrixXd pcs;

        /**
         * Variance explained by each PC.
         * Each entry corresponds to a column in `pcs` and is in decreasing order.
         */
        Eigen::VectorXd variance_explained;

        /**
         * Total variance of the dataset (possibly after scaling, if `set_scale()` is set to `true`).
         * This can be used to divide `variance_explained` to obtain the percentage of variance explained.
         */
        double total_variance = 0;
    };

    /**
     * Run PCA on an input gene-by-cell matrix.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX>
    Results run(const tatami::Matrix<T, IDX>* mat) {
        Results output;
        run(mat, output.pcs, output.variance_explained, output.total_variance);
        return output;
    }

    /**
     * Run PCA on an input gene-by-cell matrix after filtering for genes of interest.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of genes.
     * Each entry treated as a boolean specifying whether the corresponding genes should be used in the PCA.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
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
