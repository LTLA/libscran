#ifndef SCRAN_MULTI_BATCH_PCA
#define SCRAN_MULTI_BATCH_PCA

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
 * @file MultiBatchPca.hpp
 *
 * @brief Compute PCA after adjusting for differences between batch sizes.
 */

namespace scran {

/**
 * @brief Compute PCA after adjusting for differences between batch sizes.
 *
 * In multi-batch scenarios, we may wish to compute a PCA involving data from multiple batches.
 * However, if one batch has many more cells, it will dominate the PCA by driving the definition of the rotation vectors.
 * This may mask interesting aspects of variation in the smaller batches.
 * To overcome this problem, we scale each batch in inverse proportion to its size.
 * This ensures that each batch contributes equally to the (conceptual) gene-gene covariance matrix, the eigenvectors of which are used as the rotation vectors.
 * Cells are then projected to the subspace defined by these rotation vectors to obtain PC coordinates.
 *
 * Alternatively, we can compute rotation vectors from the residuals, i.e., after centering each batch.
 * The gene-gene covariance matrix will thus focus on variation within each batch, ensuring that the top PCs capture biological heterogeneity instead of batch effects.
 * (This is particularly important in applications with many batches, where batch effects might otherwise displace biology from the top PCs.)
 * However, unlike `ResidualPca`, it is important to note that the residuals are only used here for calculating the rotation vectors.
 * We still project the input matrix to obtain the PCs, so batch effects will likely still be present (though hopefully less pronounced) and must be removed with methods like [MNN correction](https://github.com/LTLA/CppMnnCorrect).
 *
 * Finally, we can combine these mechanisms to compute rotation vectors from residuals with equal weighting.
 * This gives us the benefits of both approaches as described above.
 */
class MultiBatchPca {
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
         * See `set_use_residuals()` for more details.
         */
        static constexpr bool use_residuals = false;
        
        /**
         * See `set_block_weight_policy()` for more details.
         */
        static constexpr WeightPolicy block_weight_policy = WeightPolicy::VARIABLE;

        /**
         * See `set_variable_block_weight_parameters()` for more details.
         */
        static constexpr VariableBlockWeightParameters variable_block_weight_parameters = VariableBlockWeightParameters();

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

    bool use_residuals = Defaults::use_residuals;
    WeightPolicy block_weight_policy = Defaults::block_weight_policy;
    VariableBlockWeightParameters variable_block_weight_parameters = Defaults::variable_block_weight_parameters;

    bool return_rotation = Defaults::return_rotation;
    bool return_center = Defaults::return_center;
    bool return_scale = Defaults::return_scale;

    int nthreads = Defaults::num_threads;

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param u Whether to compute the rotation vectors from the residuals after centering each batch.
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_use_residuals(bool u = Defaults::use_residuals) {
        use_residuals = u;
        return *this;
    }

    /**
     * @param w Policy to use for weighting batches of different size.
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_block_weight_policy(WeightPolicy w = Defaults::block_weight_policy) {
        block_weight_policy = w;
        return *this;
    }

    /**
     * @param v Parameters for the variable block weights, see `variable_block_weight()` for more details.
     * Only used when the block weight policy is set to `WeightPolicy::VARIABLE`.
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_variable_block_weight_parameters(VariableBlockWeightParameters v = Defaults::variable_block_weight_parameters) {
        variable_block_weight_parameters = v;
        return *this;
    }

    /**
     * @param r Should the rotation matrix be returned in the output?
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_return_rotation(bool r = Defaults::return_rotation) {
        return_rotation = r;
        return *this;
    }

    /**
     * @param r Should the center vector be returned in the output?
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_return_center(bool r = Defaults::return_center) {
        return_center = r;
        return *this;
    }

    /**
     * @param r Should the scale vector be returned in the output?
     * 
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_return_scale(bool r = Defaults::return_scale) {
        return_scale = r;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `MultiBatchPca` instance.
     */
    MultiBatchPca& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    template<typename Data_, typename Index_, typename Block_>
    void run_sparse_simple(
        const tatami::Matrix<Data_, Index_>* mat, 
        const Block_* block, 
        const pca_utils::BlockingDetails<true> block_details,
        const irlba::Irlba& irb,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::VectorXd& center_v,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        pca_utils::SparseMatrix emat(mat->ncol(), mat->nrow(), std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        size_t ngenes = emat.cols();
        center_v.resize(ngenes);
        scale_v.resize(ngenes);

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            const auto& values = emat.get_values();
            const auto& indices = emat.get_indices();
            const auto& ptrs = emat.get_pointers();

            const auto& block_size = block_details.block_size;
            size_t nblocks = block_size.size();
            std::vector<int> block_count(nblocks);
            const auto& block_weight = block_details.per_element_weight;

            for (size_t r = start, end = start + length; r < end; ++r) {
                auto offset = ptrs[r];
                size_t num_entries = ptrs[r+1] - offset;
                auto value_ptr = values.data() + offset;
                auto index_ptr = indices.data() + offset;

                std::fill(block_count.begin(), block_count.end(), 0);

                // Computing the grand mean across all blocks.
                double& grand_mean = center_v[r];
                grand_mean = 0;
                for (size_t i = 0; i < num_entries; ++i) {
                    auto b = block[index_ptr[i]];
                    grand_mean += value_ptr[i] * block_weight[b];
                    ++(block_count[b]);
                }
                grand_mean /= block_details.total_block_weight;

                // Computing pseudo-variances where each block's contribution
                // is weighted inversely proportional to its size. This aims to
                // match up with the variances used in the PCA but not the
                // variances of the output components (where weightings are not used).
                double& proxyvar = scale_v[r];
                proxyvar = 0;
                for (size_t b = 0; b < nblocks; ++b) {
                    double zero_sum = (block_size[b] - block_count[b]) * grand_mean * grand_mean;
                    proxyvar += zero_sum * block_weight[b];
                }

                for (size_t i = 0; i < num_entries; ++i) {
                    double diff = value_ptr[i] - grand_mean;
                    proxyvar += diff * diff * block_weight[block[index_ptr[i]]];
                }

                proxyvar /= emat.rows() - 1;
            }
        }, ngenes, nthreads);

        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Now actually performing the PCA.
        irlba::Centered<decltype(emat)> centered(&emat, &center_v);
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

        pca_utils::clean_up_projected<true>(pcs, variance_explained);
        if (!transpose) {
            pcs.adjointInPlace();
        }
    }

    template<typename Data_, typename Index_, typename Block_>
    void run_dense_simple(
        const tatami::Matrix<Data_, Index_>* mat, 
        const Block_* block, 
        const pca_utils::BlockingDetails<true>& block_details,
        const irlba::Irlba& irb,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained, 
        Eigen::VectorXd& center_v,
        Eigen::VectorXd& scale_v,
        double& total_var) 
    const {
        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // row-major extraction.

        size_t ngenes = emat.cols();
        center_v.resize(ngenes);
        scale_v.resize(ngenes);

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            size_t nblocks = block_details.num_blocks();
            std::vector<double> mean_buffer(nblocks);
            const auto& block_weight = block_details.per_element_weight;
            size_t ncells = emat.rows();

            for (size_t c = start, end = start + length; c < end; ++c) {
                auto ptr = emat.data() + c * ncells;

                double& grand_mean = center_v[c];
                grand_mean = 0;
                for (size_t r = 0; r < ncells; ++r) {
                    grand_mean += ptr[r] * block_weight[block[r]];
                }
                grand_mean /= block_details.total_block_weight;

                // We don't actually compute the batchwise variance, but instead
                // the weighted sum of squared deltas, which is what PCA actually sees.
                double& proxyvar = scale_v[c];
                proxyvar = 0;
                for (size_t r = 0; r < ncells; ++r) {
                    double diff = ptr[r] - grand_mean;
                    proxyvar += diff * diff * block_weight[block[r]];
                }

                proxyvar /= emat.rows() - 1;
            }
        }, ngenes, nthreads);

        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Applying the centering and scaling now so we can do the PCA with fewer wrappers.
        pca_utils::apply_center_and_scale_to_dense_matrix(emat, center_v, scale, scale_v, nthreads);

        pca_utils::SampleScaledWrapper<decltype(emat)> weighted(&emat, &(block_details.expanded_weights));
        irb.run(weighted, pcs, rotation, variance_explained);

        pcs.noalias() = emat * rotation;
        pca_utils::clean_up_projected<false>(pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }
    }

private:
    template<bool weight_, typename Matrix_, typename Block_>
    void run_residuals_internal(
        const Matrix_& emat, 
        const Block_* block, 
        const pca_utils::BlockingDetails<weight_>& block_details,
        const Eigen::MatrixXd& center_m,
        const Eigen::VectorXd& scale_v,
        const irlba::Irlba& irb,
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation, 
        Eigen::VectorXd& variance_explained)
    const {
        pca_utils::RegressWrapper<Matrix_, Block_> centered(&emat, block, &center_m);

        if constexpr(weight_) {
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&scaled, &(block_details.expanded_weights));
                irb.run(weighted, pcs, rotation, variance_explained);
            } else {
                pca_utils::SampleScaledWrapper<decltype(centered)> weighted(&centered, &(block_details.expanded_weights));
                irb.run(weighted, pcs, rotation, variance_explained);
            }

        } else {
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, variance_explained);
            } else {
                irb.run(centered, pcs, rotation, variance_explained);
            }
        }
    }

    template<bool weight_, typename Data_, typename Index_, typename Block_>
    void run_sparse_residuals(
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
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        pca_utils::SparseMatrix emat(mat->ncol(), mat->nrow(), std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        size_t ngenes = emat.cols();
        center_m.resize(block_details.num_blocks(), ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_regress<weight_>(emat, block, block_details, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        run_residuals_internal<weight_>(
            emat, 
            block, 
            block_details,
            center_m,
            scale_v,
            irb,
            pcs, 
            rotation, 
            variance_explained
        );

        // This transposes 'pcs' to be a NDIM * NCELLS matrix.
        pca_utils::project_sparse_matrix(emat, pcs, rotation, scale, scale_v, nthreads);

        pca_utils::clean_up_projected<true>(pcs, variance_explained);
        if (!transpose) {
            pcs.adjointInPlace();
        }
    }

    template<bool weight_, typename Data_, typename Index_, typename Block_>
    void run_dense_residuals(
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

        size_t ngenes = emat.cols();
        center_m.resize(block_details.num_blocks(), ngenes);
        scale_v.resize(ngenes);
        pca_utils::compute_mean_and_variance_regress<weight_>(emat, block, block_details, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // No choice but to use wrappers here, as we still need the original matrix for projection.
        run_residuals_internal<weight_>(
            emat, 
            block, 
            block_details,
            center_m,
            scale_v,
            irb,
            pcs, 
            rotation, 
            variance_explained
        );

        if (scale) {
            pcs.noalias() = emat * (rotation.array().colwise() / scale_v.array()).matrix();
        } else {
            pcs.noalias() = emat * rotation;
        }

        pca_utils::clean_up_projected<false>(pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }
    }

private:
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

        if (use_residuals) {
            if (block_weight_policy == WeightPolicy::NONE) {
                auto bdetails = pca_utils::compute_blocking_details(mat->ncol(), block);
                if (mat->sparse()) {
                    run_sparse_residuals<false>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
                } else {
                    run_dense_residuals<false>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
                }

            } else {
                auto bdetails = pca_utils::compute_blocking_details(mat->ncol(), block, block_weight_policy, variable_block_weight_parameters);
                if (mat->sparse()) {
                    run_sparse_residuals<true>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
                } else {
                    run_dense_residuals<true>(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_m, scale_v, total_var);
                }
            }

        } else {
            if (block_weight_policy == WeightPolicy::NONE) {
                throw std::runtime_error("block weight policy cannot be NONE when 'use_residuals = true', use SimplePca instead"); 
            }

            auto bdetails = pca_utils::compute_blocking_details(mat->ncol(), block, block_weight_policy, variable_block_weight_parameters);

            Eigen::VectorXd center_v;
            if (mat->sparse()) {
                run_sparse_simple(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_v, scale_v, total_var);
            } else {
                run_dense_simple(mat, block, bdetails, irb, pcs, rotation, variance_explained, center_v, scale_v, total_var);
            }

            if (return_center) {
                center_m.resize(1, center_v.size());
                center_m.row(0) = center_v;
            }
        }
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `MultiBatchPca::run()` methods.
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
         *
         * Note that the absolute magnitude of the variance is quite difficult to interpret due to the weighting.
         * We suggest dividing by `total_variance` and working with the proportion of variance explained instead.
         */
        Eigen::VectorXd variance_explained;

        /**
         * Total variance of the dataset (possibly after scaling, if `set_scale()` is set to `true`).
         * This can be used to divide `variance_explained` to obtain the percentage of variance explained.
         */
        double total_variance = 0;

        /**
         * Rotation matrix, only returned if `MultiBatchPca::set_return_rotation()` is `true`.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;

        /**
         * Centering matrix, only returned if `MultiBatchPca::set_return_center()` is `true`.
         *
         * If `MultiBatchPca::set_block_policy()` is `MultiBatchPca::BlockPolicy::RESIDUAL_ONLY` or `MultiBatchPca::BlockPolicy::WEIGHTED_RESIDUAL`,
         * the number of columns is equal to the number of unique blocking levels.
         * Each row corresponds to a row in the matrix and each column corresponds to a block, 
         * such that each entry contains the mean for a particular feature in the corresponding block.
         *
         * Otherwise, the number of columns is equal to 1.
         * Each row corresponds to a row in the matrix and contains the (weighted) grand mean for that feature across all blocks.
         *
         * If feature filtering was performed, the length is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd center;

        /**
         * Scaling vector, only returned if `MultiBatchPca::set_return_center()` is `true`.
         * Each entry corresponds to a row in the matrix and contains the scaling factor used to divide the feature values if `MultiBatchPca::set_scale()` is `true`.
         * If feature filtering was performed, the length is equal to the number of features remaining after filtering.
         */
        Eigen::VectorXd scale;
    };

    /**
     * Run the multi-batch PCA on an input gene-by-cell matrix.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam Batch Integer type for the batch assignments.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] batch Pointer to an array of length equal to the number of cells.
     * This should contain a 0-based batch assignment for each cell
     * (i.e., for `n` batches, batch identities should run from 0 to `n-1` with at least one entry for each batch.)
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX, typename Batch>
    Results run(const tatami::Matrix<T, IDX>* mat, const Batch* batch) const {
        Results output;
        Eigen::MatrixXd rotation, center_m;
        Eigen::VectorXd scale_v;

        run_internal(mat, batch, output.pcs, rotation, output.variance_explained, center_m, scale_v, output.total_variance);

        // Shifting them if we want to keep them.
        if (return_rotation) {
            output.rotation = std::move(rotation);
        }
        if (return_center) {
            output.center = center_m.adjoint();
        }
        if (return_scale) {
            output.scale = std::move(scale_v);
        }

        return output;
    }

    /**
     * Run the multi-batch PCA on an input gene-by-cell matrix after filtering for genes of interest.
     * We typically use the set of highly variable genes from `ChooseHVGs`, 
     * with the aim being to improve computational efficiency and avoid random noise by removing lowly variable genes.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam Batch Integer type for the batch assignments
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] batch Pointer to an array of length equal to the number of cells.
     * This should contain a 0-based batch assignment for each cell
     * (i.e., for `n` batches, batch identities should run from 0 to `n-1` with at least one entry for each batch.)
     * @param[in] features Pointer to an array of length equal to the number of genes.
     * Each entry treated as a boolean specifying whether the corresponding genes should be used in the PCA.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX, typename Batch, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const Batch* batch, const X* features) const {
        Results output;
        if (!features) {
            return run(mat, batch);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            return run(subsetted.get(), batch);
        }
    }
};

}

#endif
