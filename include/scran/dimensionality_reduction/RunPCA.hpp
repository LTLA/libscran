#ifndef SCRAN_RUN_PCA
#define SCRAN_RUN_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"
#include "CustomSparseMatrix.hpp"

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

        /**
         * See `set_transpose()` for more details.
         */
        static constexpr bool transpose = true;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;
    };
private:
    bool scale = Defaults::scale;
    bool transpose = Defaults::transpose;
    int rank = Defaults::rank;
    int nthreads = Defaults::num_threads;

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    bool use_eigen = false;
#endif

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `RunPCA` instance.
     */
    RunPCA& set_rank(int r = Defaults::rank) {
        rank = r;
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

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `RunPCA` instance.
     */
    RunPCA& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `RunPCA` instance.
     */
    RunPCA& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    RunPCA& set_use_eigen(bool e = false) {
        use_eigen = true;
        return *this;
    }
#endif

private:
    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* mat, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        pca_utils::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        if (mat->sparse()) {
            Eigen::VectorXd center_v(mat->nrow()), scale_v(mat->nrow());
            auto emat = create_custom_sparse_matrix(mat, center_v, scale_v, total_var);

            irlba::Centered<decltype(emat)> centered(&emat, &center_v);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, variance_explained);
            } else {
                irb.run(centered, pcs, rotation, variance_explained);
            }
        } else {
            auto emat = create_eigen_matrix_dense(mat, total_var);
            irb.run(emat, pcs, rotation, variance_explained); // already centered and scaled, if relevant.
        }

        pca_utils::clean_up(mat->ncol(), pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }

        return;
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `RunPCA::run()` methods.
     */
    struct Results {
        /**
         * Matrix of principal components.
         * By default, each row corresponds to a PC while each column corresponds to a cell in the input matrix.
         * If `set_transpose()` is set to `false`, rows are cells instead.
         * The number of PCs is determined by `set_rank()`.
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

        /**
         * Rotation matrix.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;
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
    Results run(const tatami::Matrix<T, IDX>* mat) const {
        Results output;
        run(mat, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        return output;
    }

    /**
     * Run PCA on an input gene-by-cell matrix after filtering for genes of interest.
     * We typically use the set of highly variable genes from `ChooseHVGs`, 
     * with the aim being to improve computational efficiency and avoid random noise by removing lowly variable genes.
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
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) const {
        Results output;

        if (!features) {
            run(mat, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        } else {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("RunPCA", "Subsetting to features of interest");
#endif
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), output.pcs, output.rotation, output.variance_explained, output.total_variance);
        }

        return output;
    }

private:
    template<typename T, typename IDX> 
    pca_utils::CustomSparseMatrix create_custom_sparse_matrix(const tatami::Matrix<T, IDX>* mat, Eigen::VectorXd& center_v, Eigen::VectorXd& scale_v, double& total_var) const {
        size_t NR = mat->nrow(), NC = mat->ncol();

        pca_utils::CustomSparseMatrix A(NC, NR, nthreads); // transposed; we want genes in the columns.
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (use_eigen) {
            A.use_eigen();
        }
#endif

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);
            values.reserve(NR);
            indices.reserve(NR);
            auto wrk = mat->new_workspace(true);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data(), wrk.get());

                auto stats = tatami::stats::variances::compute_direct(range, NC);
                center_v[r] = stats.first;
                scale_v[r] = stats.second;

                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            pca_utils::set_scale(scale, scale_v, total_var);
            A.fill_columns(values, indices);
        } else {
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);
            values.reserve(NC);
            indices.reserve(NC);
            auto wrk = mat->new_workspace(false);

            center_v.setZero();
            scale_v.setZero();
            std::vector<int> nonzeros(NR);
            int count = 0;

            // First pass to compute variances and extract non-zero values.
            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), wrk.get());
                tatami::stats::variances::compute_running(range, center_v.data(), scale_v.data(), nonzeros.data(), count);
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            tatami::stats::variances::finish_running(NR, center_v.data(), scale_v.data(), nonzeros.data(), count);
            pca_utils::set_scale(scale, scale_v, total_var);
            A.fill_rows(values, indices, nonzeros);
        }

        return A;
    }

private:
    template<typename T, typename IDX> 
    Eigen::MatrixXd create_eigen_matrix_dense(const tatami::Matrix<T, IDX>* mat, double& total_var) const {
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
