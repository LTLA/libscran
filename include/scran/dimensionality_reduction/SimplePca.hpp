#ifndef SCRAN_SIMPLE_PCA_HPP
#define SCRAN_SIMPLE_PCA_HPP

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "convert.hpp"
#include "utils.hpp"

/**
 * @file SimplePca.hpp
 *
 * @brief Perform a simple PCA on a gene-by-cell matrix.
 */

namespace scran {

/**
 * @brief Perform a simple PCA on a gene-cell matrix.
 *
 * Principal components analysis (PCA) is a helpful technique for data compression and denoising.
 * The idea is that the earlier PCs capture most of the systematic biological variation while the later PCs capture random technical noise.
 * Thus, we can reduce the size of the data and eliminate noise by only using the earlier PCs for further analyses.
 * Most practitioners will keep the first 10-50 PCs, though the exact choice is fairly arbitrary.
 * For speed, we use the [**CppIrlba**](https://github.com/LTLA/CppIrlba) package to perform an approximate PCA.
 */
class SimplePca {
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

        /**
         * See `set_return_rotation()` for more details.
         */
        static constexpr bool return_rotation = false;

        /**
         * See `set_return_center()` for more details.
         */
        static constexpr bool return_center = false;

        /**
         * See `set_return_scale()` for more details.
         */
        static constexpr bool return_scale = false;
    };
private:
    bool scale = Defaults::scale;
    bool transpose = Defaults::transpose;
    int rank = Defaults::rank;

    bool return_rotation = Defaults::return_rotation;
    bool return_center = Defaults::return_center;
    bool return_scale = Defaults::return_scale;

    int nthreads = Defaults::num_threads;

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param r Should the rotation matrix be returned in the output?
     * 
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_return_rotation(bool r = Defaults::return_rotation) {
        return_rotation = r;
        return *this;
    }

    /**
     * @param r Should the center vector be returned in the output?
     * 
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_return_center(bool r = Defaults::return_center) {
        return_center = r;
        return *this;
    }

    /**
     * @param r Should the scale vector be returned in the output?
     * 
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_return_scale(bool r = Defaults::return_scale) {
        return_scale = r;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `SimplePca` instance.
     */
    SimplePca& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    template<typename Data_, typename Index_>
    void run_sparse(
        const tatami::Matrix<Data_, Index_>* mat, 
        const irlba::Irlba& irb, 
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained,
        Eigen::VectorXd& center_v,
        Eigen::VectorXd& scale_v,
        double& total_var)
    const {
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction to create a CSR matrix.
        pca_utils::SparseMatrix emat(mat->ncol(), mat->nrow(), std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        size_t ngenes = emat.cols();
        center_v.resize(ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_from_sparse_matrix(emat, center_v, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        irlba::Centered<decltype(emat)> centered(&emat, &center_v);
        if (scale) {
            irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
            irb.run(scaled, pcs, rotation, variance_explained);
        } else {
            irb.run(centered, pcs, rotation, variance_explained);
        }
    }

    template<typename Data_, typename Index_>
    void run_dense(
        const tatami::Matrix<Data_, Index_>* mat, 
        const irlba::Irlba& irb, 
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::VectorXd& center_v,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // get a column-major matrix with genes in columns.

        size_t ngenes = emat.cols();
        center_v.resize(ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_from_dense_matrix(emat, center_v, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Applying the centering and scaling now so we can do the PCA without any wrappers.
        pca_utils::apply_center_and_scale_to_dense_matrix(emat, center_v, scale, scale_v, nthreads);
        irb.run(emat, pcs, rotation, variance_explained); 
    }

    template<typename Data_, typename Index_>
    void run_internal(
        const tatami::Matrix<Data_, Index_>* mat, 
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::VectorXd& center_v,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        if (mat->sparse()) {
            run_sparse(mat, irb, pcs, rotation, variance_explained, center_v, scale_v, total_var);
        } else {
            run_dense(mat, irb, pcs, rotation, variance_explained, center_v, scale_v, total_var);
        }

        pca_utils::clean_up(mat->ncol(), pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `SimplePca::run()` methods.
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
         * Rotation matrix, only returned if `SimplePca::set_return_rotation()` is `true`.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;

        /**
         * Centering vector, only returned if `SimplePca::set_return_center()` is `true`.
         * Each entry corresponds to a row in the matrix and contains the mean value for that feature.
         * If feature filtering was performed, the length is equal to the number of features remaining after filtering.
         */
        Eigen::VectorXd center;

        /**
         * Scaling vector, only returned if `SimplePca::set_return_center()` is `true`.
         * Each entry corresponds to a row in the matrix and contains the scaling factor used to divide the feature values if `SimplePca::set_scale()` is `true`.
         * If feature filtering was performed, the length is equal to the number of features remaining after filtering.
         */
        Eigen::VectorXd scale;
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

        Eigen::MatrixXd rotation;
        Eigen::VectorXd center, scale;
        run_internal(mat, output.pcs, rotation, output.variance_explained, center, scale, output.total_variance);

        // Shifting them if we want to keep them.
        if (return_rotation) {
            output.rotation = std::move(rotation);
        }
        if (return_center) {
            output.center = std::move(center);
        }
        if (return_scale) {
            output.scale = std::move(scale);
        }

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
        if (!features) {
            return run(mat);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            return run(subsetted.get());
        }
    }
};

}

#endif
