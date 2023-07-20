#ifndef SCRAN_RESIDUAL_PCA_HPP
#define SCRAN_RESIDUAL_PCA_HPP

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "utils.hpp"
#include "convert.hpp"
#include "wrappers.hpp"
#include "blocking.hpp"

/**
 * @file ResidualPca.hpp
 *
 * @brief Compute PCA after regressing out an uninteresting factor.
 */

namespace scran {

/**
 * @brief Compute PCA after regressing out an uninteresting factor.
 *
 * A simple batch correction method involves centering the expression of each gene in each batch to remove systematic differences between batches.
 * The residuals are then used in PCA to obtain a batch-corrected low-dimensional representation of the dataset.
 * Unfortunately, naively centering the expression values will discard sparsity and reduce the computational efficiency of the PCA.
 * To avoid these drawbacks, `ResidualPca` defers the residual calculation until the matrix multiplication of the IRLBA step.
 * This yields the same results as the naive approach but is much faster as it can take advantage of efficient sparse operations.
 *
 * We can optionally scale each batch so that they contribute equally to the rotation vectors, regardless of their size.
 * This is achieved using the same approach described in `MultiBatchPca`,
 * whereby batches with more cells are downscaled during calculation of the rotation vectors.
 * The final PCs are then obtained by projecting the residuals onto the space defined by the rotation vectors.
 * This ensures that larger batches do not mask interesting variation in other batches.
 *
 * Note that the use of residuals for batch correction makes some strong assumptions about the batches,
 * e.g., they have the same composition and the batch effect is a consistent shift for all populations.
 * Non-linear correction algorithms are usually more effective, e.g., [MNN correction](https://github.com/LTLA/CppMnnCorrect).
 */
class ResidualPca {
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
         * See `set_block_weight_policy()` for more details.
         */
        static constexpr WeightPolicy block_weight_policy = WeightPolicy::VARIABLE;

        /**
         * See `set_variable_block_weight_parameters()` for more details.
         */
        static constexpr VariableBlockWeightParameters variable_block_weight_parameters = VariableBlockWeightParameters();

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

    WeightPolicy block_weight_policy = Defaults::block_weight_policy;
    VariableBlockWeightParameters variable_block_weight_parameters = Defaults::variable_block_weight_parameters;

    int nthreads = Defaults::num_threads;

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output PC matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param r Should the rotation matrix be returned in the output?
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_return_rotation(bool r = Defaults::return_rotation) {
        return_rotation = r;
        return *this;
    }

    /**
     * @param r Should the center vector be returned in the output?
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_return_center(bool r = Defaults::return_center) {
        return_center = r;
        return *this;
    }

    /**
     * @param r Should the scale vector be returned in the output?
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_return_scale(bool r = Defaults::return_scale) {
        return_scale = r;
        return *this;
    }

    /**
     * @param w Policy to use for weighting batches of different size.
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_block_weight_policy(WeightPolicy w = Defaults::block_weight_policy) {
        block_weight_policy = w;
        return *this;
    }

    /**
     * @param v Parameters for the variable block weights, see `variable_block_weight()` for more details.
     * Only used when the block weight policy is set to `WeightPolicy::VARIABLE`.
     * 
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_variable_block_weight_parameters(VariableBlockWeightParameters v = Defaults::variable_block_weight_parameters) {
        variable_block_weight_parameters = v;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `ResidualPca` instance.
     */
    ResidualPca& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    template<bool weight_, typename Data_, typename Index_, typename Block_>
    void run_sparse(
        const tatami::Matrix<Data_, Index_>* mat, 
        const Block_* block, 
        const pca_utils::BlockingDetails<weight_>& block_details, 
        const irlba::Irlba& irb,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::MatrixXd& center_m,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        auto ngenes = mat->nrow(), ncells = mat->ncol(); 
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major filling.
        pca_utils::SparseMatrix emat(ncells, ngenes, std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        auto nblocks = block_details.num_blocks();
        center_m.resize(nblocks, ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_regress<weight_>(emat, block, block_details, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        pca_utils::RegressWrapper<decltype(emat), Block_> centered(&emat, block, &center_m);
        if constexpr(weight_) {
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&scaled, &(block_details.expanded_weights));
                irb.run(weighted, pcs, rotation, variance_explained);
            } else {
                pca_utils::SampleScaledWrapper<decltype(centered)> weighted(&centered, &(block_details.expanded_weights));
                irb.run(weighted, pcs, rotation, variance_explained);
            }

            // This transposes 'pcs' to be a NDIM * NCELLS matrix.
            pca_utils::project_sparse_matrix(emat, pcs, rotation, scale, scale_v, nthreads);

            // Subtracting each block's mean from the PCs.
            Eigen::MatrixXd centering;
            if (scale) {
                centering = (center_m * (rotation.array().colwise() / scale_v.array()).matrix()).adjoint();
            } else {
                centering = (center_m * rotation).adjoint();
            }
            for (size_t i = 0, iend = pcs.cols(); i < iend; ++i) {
                pcs.col(i) -= centering.col(block[i]);
            }

            pca_utils::clean_up_projected<true>(pcs, variance_explained);
            if (!transpose) {
                pcs.adjointInPlace();
            }

        } else {
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, variance_explained);
            } else {
                irb.run(centered, pcs, rotation, variance_explained);
            }

            pca_utils::clean_up(mat->ncol(), pcs, variance_explained);
            if (transpose) {
                pcs.adjointInPlace();
            }
        }
    }

    template<bool weight_, typename Data_, typename Index_, typename Block_>
    void run_dense(
        const tatami::Matrix<Data_, Index_>* mat, 
        const Block_* block,
        const pca_utils::BlockingDetails<weight_>& block_details, 
        const irlba::Irlba& irb,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::MatrixXd& center_m,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // get a column-major matrix with genes in columns.

        auto ngenes = emat.cols();
        auto nblocks = block_details.num_blocks();
        center_m.resize(nblocks, ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_regress<weight_>(emat, block, block_details, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Applying the centering and scaling directly so that we can run the PCA with no or fewer layers.
        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            size_t ncells = emat.rows();
            double* ptr = emat.data() + static_cast<size_t>(start) * ncells;
            for (size_t g = start, end = start + length; g < end; ++g, ptr += ncells) {
                for (size_t c = 0; c < ncells; ++c) {
                    ptr[c] -= center_m.coeff(block[c], g);
                }

                if (scale) {
                    auto sd = scale_v[g];
                    for (size_t c = 0; c < ncells; ++c) {
                        ptr[c] /= sd; // process_scale_vector should already protect against division by zero.
                    }
                }
            }
        }, ngenes, nthreads);

        if constexpr(weight_) {
            pca_utils::SampleScaledWrapper<decltype(emat)> weighted(&emat, &(block_details.expanded_weights));
            irb.run(weighted, pcs, rotation, variance_explained);
            pcs.noalias() = emat * rotation;
            pca_utils::clean_up_projected<false>(pcs, variance_explained);
        } else {
            irb.run(emat, pcs, rotation, variance_explained);
            pca_utils::clean_up(pcs.rows(), pcs, variance_explained);
        }

        if (transpose) {
            pcs.adjointInPlace();
        }
    }

    template<typename Data_, typename Index_, typename Block_>
    void run_internal(
        const tatami::Matrix<Data_, Index_>* mat, 
        const Block_* block,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::MatrixXd& center_m,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        if (block_weight_policy == WeightPolicy::NONE) {
            auto bdetails = pca_utils::compute_blocking_details(mat->ncol(), block);
            if (mat->sparse()) {
                run_sparse<false>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
            } else {
                run_dense<false>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
            }

        } else {
            auto bdetails = pca_utils::compute_blocking_details(mat->ncol(), block, block_weight_policy, variable_block_weight_parameters);
            if (mat->sparse()) {
                run_sparse<true>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
            } else {
                run_dense<true>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
            }
        }
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `ResidualPca::run()` methods.
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
         * Rotation matrix, only returned if `ResidualPca::set_return_rotation()` is `true`.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;

        /**
         * Centering matrix, only returned if `ResidualPca::set_return_center()` is `true`.
         * Each row corresponds to a row in the input matrix and each column corresponds to a block, 
         * such that each entry contains the mean for a particular feature in the corresponding block.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd center;

        /**
         * Scaling vector, only returned if `ResidualPca::set_return_center()` is `true`.
         * Each entry corresponds to a row in the input matrix and contains the scaling factor used to divide the feature values if `ResidualPca::set_scale()` is `true`.
         * If feature filtering was performed, the length is equal to the number of features remaining after filtering.
         */
        Eigen::VectorXd scale;
    };

    /**
     * Run the blocked PCA on an input gene-by-cell matrix.
     *
     * @tparam Data_ Floating point type for the data.
     * @tparam Index_ Integer type for the indices.
     * @tparam Block_ Integer type for the blocking factor.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] block Pointer to an array of length equal to the number of cells, 
     * containing the block assignment for each cell - see `count_blocks()` for details.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename Data_, typename Index_, typename Block_>
    Results run(const tatami::Matrix<Data_, Index_>* mat, const Block_* block) const {
        Results output;
        Eigen::MatrixXd rotation, center;
        Eigen::VectorXd scale;

        run_internal(mat, block, output.pcs, rotation, output.variance_explained, center, scale, output.total_variance);

        // Shifting them if we want to keep them.
        if (return_rotation) {
            output.rotation = std::move(rotation);
        }
        if (return_center) {
            output.center = center.adjoint();
        }
        if (return_scale) {
            output.scale = std::move(scale);
        }

        return output;
    }

    /**
     * Run the blocked PCA on an input gene-by-cell matrix after filtering for genes of interest.
     * We typically use the set of highly variable genes from `ChooseHVGs`, 
     * with the aim being to improve computational efficiency and avoid random noise by removing lowly variable genes.
     *
     * @tparam Data_ Floating point type for the data.
     * @tparam Index_ Integer type for the indices.
     * @tparam Block_ Integer type for the blocking factor.
     * @tparam Subset_ Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] block Pointer to an array of length equal to the number of cells, 
     * containing the block assignment for each cell - see `count_blocks()` for details.
     * @param[in] features Pointer to an array of length equal to the number of genes.
     * Each entry is a boolean specifying whether the corresponding gene should be used in the PCA.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename Data_, typename Index_, typename Block_, typename Subset_>
    Results run(const tatami::Matrix<Data_, Index_>* mat, const Block_* block, const Subset_* features) const {
        if (!features) {
            return run(mat, block);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            return run(subsetted.get(), block);
        }
    }
};

}

#endif
