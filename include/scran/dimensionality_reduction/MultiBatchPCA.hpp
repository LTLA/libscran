#ifndef SCRAN_MULTI_BATCH_PCA
#define SCRAN_MULTI_BATCH_PCA

#include "tatami/tatami.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"
#include "../utils/block_indices.hpp"

/**
 * @file MultiBatchPCA.hpp
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
 * 
 * To overcome this problem, we weight each batch in inverse proportion to its size.
 * This ensures that each batch contributes equally to the (conceptual) gene-gene covariance matrix, the eigenvectors of which are used as the rotation vectors.
 * Cells are then projected to the subspace defined by these rotation vectors to obtain PC coordinates.
 *
 * Unlike `BlockedPCA`, this class will not actually perform any batch correction.
 * Any batch effects will be preserved in the low-dimensional space and require further processing to remove.
 */
class MultiBatchPCA {
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

public:
    /**
     * @param r Number of PCs to compute.
     * This should be smaller than the smaller dimension of the input matrix.
     *
     * @return A reference to this `MultiBatchPCA` instance.
     */
    MultiBatchPCA& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `MultiBatchPCA` instance.
     */
    MultiBatchPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `MultiBatchPCA` instance.
     */
    MultiBatchPCA& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `MultiBatchPCA` instance.
     */
    MultiBatchPCA& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    template<typename Batch>
    static Eigen::VectorXd compute_weights(size_t NC, const Batch* batch, const std::vector<int>& batch_size) {
        Eigen::VectorXd weights(NC);
        for (size_t i = 0; i < NC; ++i) {
            weights[i] = 1/std::sqrt(static_cast<double>(batch_size[batch[i]]));
        }
        return weights;
    }

private:
    template<typename Data_, typename Index_, typename Block_>
    void run_sparse_simple(const tatami::Matrix<Data_, Index_>* mat, const Block_* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        size_t NR = mat->nrow(), NC = mat->ncol();
        auto block_size = pca_utils::compute_block_size(NC, block);
        auto block_weight = pca_utils::compute_block_weight<weight_>(block_size);
        auto total_weight = total_block_weight(block_weight);

        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        pca_utils::SparseMatrix emat(NC, NR, std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        Eigen::VectorXd center_v(NR);
        Eigen::VectorXd scale_v(NR);

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            const auto& ptrs = extracted.ptrs;
            const auto& values = extracted.values;
            const auto& indices = extracted.indices;

            size_t nblocks = block_size.size();
            auto block_count = block_Size;

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
                grand_mean /= total_weight;

                // Computing pseudo-variances where each block's contribution
                // is weighted inversely proportional to its size. This aims to
                // match up with the variances used in the PCA but not the
                // variances of the output components (where weightings are not used).
                double& proxyvar = scale_v[r];
                proxyvar = 0;
                for (size_t b = 0; b < nblocks; ++b) {
                    double zero_sum = (block_size[b] - batch_count[b]) * grand_mean * grand_mean;
                    proxyvar += zero_sum * block_weight[b];
                }

                for (size_t i = 0; i < num_entries; ++i) {
                    double diff = value_ptr[i] - grand_mean;
                    proxyvar += diff * diff * block_weight[block[index_ptr[i]]];
                }
            }
        }, NR, nthreads);

        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Now actually performing the PCA.
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        irlba::Centered<decltype(emat), Block_> centered(&emat, &center_v);
        auto expanded = expand_block_weight(NC, block, block_weight);

        if (scale) {
            irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v)
            pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&scaled, &expanded);
            irb.run(weighted, pcs, rotation, variance_explained);
        } else {
            pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&centered, &expanded);
            irb.run(weighted, pcs, rotation, variance_explained);
        }

        // This transposes 'pcs' to be a NDIM * NCELLS matrix.
        pca_utils::project_sparse_matrix(emat, pcs, rotation, scale, scale_v, nthreads);

        pca_utils::clean_up_projected<true>(pcs, variance_explained);
        if (!transpose) {
            pcs.adjointInPlace();
        }
    }

    template<bool weight_, typename Data_, typename Index_, typename Block_>
    void run_sparse_residuals(const tatami::Matrix<Data_, Index_>* mat, const Block_* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        const size_t NR = mat->nrow(), NC = mat->ncol();
        auto block_size = pca_utils::compute_block_size(NC, block);
        auto nblocks = block_size.size();
        auto block_weight = pca_utils::compute_block_weight<weight_>(block_size);

        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        pca_utils::SparseMatrix emat(NC, NR, std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        Eigen::MatrixXd center_m(nblocks, NR);
        Eigen::VectorXd scale_v(NR);
        pca_utils::compute_mean_and_variance_regress(emat, block, block_size, block_weight, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Now actually performing the PCA.
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        pca_utils::RegressWrapper<decltype(emat), Block_> centered(&emat, block, &center_m);
        if constexpr(weight_) {
            auto expanded = expand_block_weight(NC, block, block_weight);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&scaled, &expanded);
                irb.run(weighted, pcs, rotation, variance_explained);
            } else {
                pca_utils::SampleScaledWrapper<decltype(centered)> weighted(&centered, &expanded);
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

        // This transposes 'pcs' to be a NDIM * NCELLS matrix.
        pca_utils::project_sparse_matrix(emat, pcs, rotation, scale, scale_v, nthreads);

        pca_utils::clean_up_projected<true>(pcs, variance_explained);
        if (!transpose) {
            pcs.adjointInPlace();
        }
    }

    template<typename Data_, typename Index_, typename Block_>
    void run_dense_simple(const tatami::Matrix<Data_, Index_>* mat, const Block_* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        size_t NR = mat->nrow(), NC = mat->ncol();
        auto block_size = pca_utils::compute_block_size(NC, block);
        auto block_weight = pca_utils::compute_block_weight<weight_>(block_size);
        auto total_weight = total_block_weight(block_weight);

        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        pca_utils::SparseMatrix emat(NC, NR, std::move(extracted.values), std::move(extracted.indices), std::move(extracted.ptrs), nthreads); // CSC with genes in columns.

        Eigen::VectorXd center_v(NR);
        Eigen::VectorXd scale_v(NR);

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            std::vector<double> mean_buffer(nbatchs);
            for (size_t c = start, end = start + length; c < end; ++c) {
                auto ptr = emat.data() + c * NR;

                double& grand_mean = center_v[c];
                grand_mean = 0;
                for (size_t r = 0; r < NR; ++r) {
                    grand_mean += ptr[r] * block_weight[block[r]];
                }

                // We don't actually compute the batchwise variance, but instead
                // the weighted sum of squared deltas, which is what PCA actually sees.
                double& proxyvar = scale_v[c];
                proxyvar = 0;
                for (size_t r = 0; r < NR; ++r) {
                    double diff = ptr[r] - grand_mean;
                    proxyvar += diff * diff * block_weight[block[r]];
                }
            }
        }, NC, nthreads);

        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // Applying the centering and scaling now so we can do the PCA with fewer wrappers.
        pca_utils::apply_dense_center_and_scale(emat, center_v, scale, scale_v, nthreads);

        // Now actually performing the PCA.
        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        auto expanded = expand_block_weight(NC, block, block_weight);
        pca_utils::SampleScaledWrapper<decltype(scaled)> weighted(&centered, &expanded);
        irb.run(weighted, pcs, rotation, variance_explained);

        pcs.noalias() = emat * rotation;
        pca_utils::clean_up_projected<false>(pcs, variance_explained);
        if (transpose) {
            pcs.adjointInPlace();
        }
    }

    template<typename Data_, typename Index_, typename Block_>
    void run_dense_residuals(const tatami::Matrix<Data_, Index_>* mat, const Block_* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        const size_t NR = mat->nrow(), NC = mat->ncol();
        auto block_size = pca_utils::compute_block_size(NC, block);
        auto nblocks = block_size.size();
        auto block_weight = pca_utils::compute_block_weight<weight_>(block_size);

        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // get a column-major matrix with genes in columns.

        Eigen::MatrixXd center_m(nblocks, NR);
        Eigen::VectorXd scale_v(NR);
        pca_utils::compute_mean_and_variance_regress(emat, block, block_size, block_weight, center_m, scale_v, nthreads);
        total_var = pca_utils::process_scale_vector(scale, scale_v);

        // No choice but to use wrappers here, as we still need the original matrix for projection.
        pca_utils::RegressWrapper<decltype(emat), Block_> centered(&emat, block, &center_m);
        if constexpr(weight_) {
            auto expanded = expand_block_weight(NC, block, block_weight);
            pca_utils::SampleScaledWrapper<decltype(centered)> weighted(&centered, &expanded);
            irb.run(weighted, scale_v, pcs, rotation, variance_explained);
        } else {
            irb.run(centered, pcs, rotation, variance_explained);
        }

        pcs.noalias() = emat * rotation;
        pca_utils::clean_up_projected<false>(pcs, variance_explained);
        if (!transpose) {
            pcs.adjointInPlace();
        }
    }

    template<typename Data_, typename Index_, typename Block_>
    void run(const tatami::Matrix<Data_, Index_>* mat, const Block_* block, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        if (block_policy == BlockPolicy::WEIGHT_ONLY) {
            if (mat->sparse()) {
                run_sparse_simple(mat, block, pcs, rotation, variance_explained, total);
            } else {
                run_dense_simple(mat, block, pcs, rotation, variance_explained, total);
            }

        } else if (block_policy == BlockPolicy::RESIDUAL_ONLY) {
            if (mat->sparse()) {
                run_sparse_residuals<false>(mat, block, pcs, rotation, variance_explained, total);
            } else {
                run_dense_residuals<false>(mat, block, pcs, rotation, variance_explained, total);
            }

        } else {
            if (mat->sparse()) {
                run_sparse_residuals<true>(mat, block, pcs, rotation, variance_explained, total);
            } else {
                run_dense_residuals<true>(mat, block, pcs, rotation, variance_explained, total);
            }
        }
    }

public:
    /**
     * @brief Container for the PCA results.
     *
     * Instances should be constructed by the `MultiBatchPCA::run()` methods.
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
         * Rotation matrix.
         * Each row corresponds to a feature while each column corresponds to a PC.
         * The number of PCs is determined by `set_rank()`.
         * If feature filtering was performed, the number of rows is equal to the number of features remaining after filtering.
         */
        Eigen::MatrixXd rotation;
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
        run(mat, batch, output.pcs, output.rotation, output.variance_explained, output.total_variance);
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
            run(mat, batch, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), batch, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        }
        return output;
    }

private:
    template<typename T, typename IDX, typename Batch> 
    Eigen::MatrixXd create_eigen_matrix_dense(
        const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& center_v, 
        Eigen::VectorXd& scale_v, 
        const Batch* batch,
        const std::vector<int>& batch_size,
        double& total_var) 
    const {

        // Extract a column-major matrix with genes in columns.
        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); 

        // Looping across genes (i.e., columns) to fill center_v and scale_v.
        size_t NC = emat.cols(), NR = emat.rows();
        size_t nbatchs = batch_size.size();

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            std::vector<double> mean_buffer(nbatchs);
            for (size_t c = start, end = start + length; c < end; ++c) {
                auto ptr = emat.data() + c * NR;
                std::fill(mean_buffer.begin(), mean_buffer.end(), 0.0);
                for (size_t r = 0; r < NR; ++r) {
                    mean_buffer[batch[r]] += ptr[r];
                }

                double& grand_mean = center_v[c];
                grand_mean = 0;
                int non_empty_batches = 0;
                for (size_t b = 0; b < nbatchs; ++b) {
                    const auto& bsize = batch_size[b];
                    if (bsize) {
                        grand_mean += mean_buffer[b] / bsize;
                        ++non_empty_batches;
                    }
                }
                grand_mean /= non_empty_batches;

                // We don't actually compute the batchwise variance, but instead
                // the weighted sum of squared deltas, which is what PCA actually sees.
                double& proxyvar = scale_v[c];
                proxyvar = 0;
                for (size_t r = 0; r < NR; ++r) {
                    const auto& bsize = batch_size[batch[r]];
                    if (bsize) {
                        double diff = ptr[r] - grand_mean;
                        proxyvar += diff * diff / bsize;
                    }
                }
            }
        }, NC, nthreads);

        total_var = pca_utils::process_scale_vector(scale, scale_v);
        return emat;
    }
};

}

#endif
