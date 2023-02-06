#ifndef SCRAN_BLOCKED_PCA_HPP
#define SCRAN_BLOCKED_PCA_HPP

#include "../utils/macros.hpp"

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"
#include "CustomSparseMatrix.hpp"

/**
 * @file BlockedPCA.hpp
 *
 * @brief Compute PCA after blocking on an uninteresting factor.
 */

namespace scran {

/**
 * @cond
 */
template<class Matrix, typename Block>
struct BlockedEigenMatrix {
    BlockedEigenMatrix(const Matrix* m, const Block* b, const Eigen::MatrixXd* mx) : mat(m), block(b), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& output) const {
        if constexpr(irlba::has_multiply_method<Matrix>::value) {
            output.noalias() = (*mat) * rhs;
        } else {
            mat->multiply(rhs, output);
        }

        Eigen::VectorXd sub = (*means) * rhs;
        for (Eigen::Index i = 0; i < output.size(); ++i) {
            output.coeffRef(i) -= sub.coeff(block[i]);
        }
        return;
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& output) const {
        if constexpr(irlba::has_adjoint_multiply_method<Matrix>::value) {
            output.noalias() = mat->adjoint() * rhs;
        } else {
            mat->adjoint_multiply(rhs, output);
        }

        Eigen::VectorXd aggr(means->rows());
        aggr.setZero();
        for (Eigen::Index i = 0; i < rhs.size(); ++i) {
            aggr.coeffRef(block[i]) += rhs.coeff(i); 
        }

        output.noalias() -= means->adjoint() * aggr;
        return;
    }

    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output;
        if constexpr(irlba::has_realize_method<Matrix>::value) {
            output = mat->realize();
        } else {
            output = Eigen::MatrixXd(*mat);
        }

        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                output.coeffRef(r, c) -= means->coeff(block[r], c);
            }
        }
        return output;
    }
private:
    const Matrix* mat;
    const Block* block;
    const Eigen::MatrixXd* means;
};
/**
 * @endcond
 */

/**
 * @brief Compute PCA after blocking on an uninteresting factor
 *
 * A simple batch correction method involves centering the expression of each gene in each batch to remove systematic differences between batches.
 * The corrected values are then used in PCA to obtain a batch-corrected low-dimensional representation of the dataset.
 * Unfortunately, naively centering the expression values will discard sparsity and reduce the computational efficiency of the PCA.
 * To avoid these drawbacks, `BlockedPCA` defers the centering until the matrix multiplication of the IRLBA step.
 * This yields the same results as the naive approach but is much faster as it can take advantage of efficient sparse operations.
 */
class BlockedPCA {
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
         * See `set_num_threads` for more details.
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
     * @return A reference to this `BlockedPCA` instance.
     */
    BlockedPCA& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    /**
     * @param s Should genes be scaled to unit variance?
     *
     * @return A reference to this `BlockedPCA` instance.
     */
    BlockedPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

    /**
     * @param t Should the PC matrix be transposed on output?
     * If `true`, the output PC matrix is column-major with cells in the columns, which is compatible with downstream **libscran** steps.
     * 
     * @return A reference to this `BlockedPCA` instance.
     */
    BlockedPCA& set_transpose(bool t = Defaults::transpose) {
        transpose = t;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `BlockedPCA` instance.
     */
    BlockedPCA& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    BlockedPCA& set_use_eigen(bool e = false) {
        use_eigen = true;
        return *this;
    }
#endif

private:
    template<typename T, typename IDX, typename Block>
    void run(const tatami::Matrix<T, IDX>* mat, const Block* block, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        const size_t NC = mat->ncol();
        const int nblocks = (NC ? *std::max_element(block, block + NC) + 1 : 1);
        std::vector<int> block_size(nblocks);
        for (size_t j = 0; j < NC; ++j) {
            ++block_size[block[j]];
        }

        irlba::EigenThreadScope t(nthreads);
        irlba::Irlba irb;
        irb.set_number(rank);

        if (mat->sparse()) {
            Eigen::MatrixXd center_m(nblocks, mat->nrow());
            Eigen::VectorXd scale_v(mat->nrow());
            auto emat = create_custom_sparse_matrix(mat, center_m, scale_v, block, block_size, total_var);

            BlockedEigenMatrix<decltype(emat), Block> centered(&emat, block, &center_m);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, variance_explained);
            } else {
                irb.run(centered, pcs, rotation, variance_explained);
            }
        } else {
            auto emat = create_eigen_matrix_dense(mat, block, block_size, total_var);
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
     * Instances should be constructed by the `BlockedPCA::run()` methods.
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
     * Run the blocked PCA on an input gene-by-cell matrix.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam Block Integer type for the blocking factor.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] block Pointer to an array of length equal to the number of cells.
     * This should contain the blocking factor as 0-based block assignments 
     * (i.e., for `n` blocks, block identities should run from 0 to `n-1` with at least one entry for each block.)
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX, typename Block>
    Results run(const tatami::Matrix<T, IDX>* mat, const Block* block) const {
        Results output;
        run(mat, block, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        return output;
    }

    /**
     * Run the blocked PCA on an input gene-by-cell matrix after filtering for genes of interest.
     * We typically use the set of highly variable genes from `ChooseHVGs`, 
     * with the aim being to improve computational efficiency and avoid random noise by removing lowly variable genes.
     *
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam Block Integer type for the blocking factor.
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] block Pointer to an array of length equal to the number of cells.
     * This should contain the blocking factor as 0-based block assignments 
     * (i.e., for `n` blocks, block identities should run from 0 to `n-1` with at least one entry for each block.)
     * @param[in] features Pointer to an array of length equal to the number of genes.
     * Each entry treated as a boolean specifying whether the corresponding genes should be used in the PCA.
     *
     * @return A `Results` object containing the PCs and the variance explained.
     */
    template<typename T, typename IDX, typename Block, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const Block* block, const X* features) const {
        Results output;
        if (!features) {
            run(mat, block, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), block, output.pcs, output.rotation, output.variance_explained, output.total_variance);
        }
        return output;
    }

private:
    template<typename T, typename IDX, typename Block> 
    pca_utils::CustomSparseMatrix create_custom_sparse_matrix(const tatami::Matrix<T, IDX>* mat, 
        Eigen::MatrixXd& center_m, 
        Eigen::VectorXd& scale_v, 
        const Block* block, 
        const std::vector<int>& block_size, 
        double& total_var) 
    const {

        size_t NR = mat->nrow(), NC = mat->ncol();
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major filling.
        auto& ptrs = extracted.ptrs;
        auto& values = extracted.values;
        auto& indices = extracted.indices;

        // Computing block-specific means and variances.
        {
            const size_t nblocks = block_size.size();

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (size_t r = 0; r < NR; ++r) {
#else
            SCRAN_CUSTOM_PARALLEL(NR, [&](size_t first, size_t last) -> void {
            for (size_t r = first; r < last; ++r) {
#endif

                auto offset = ptrs[r];
                size_t num_entries = ptrs[r+1] - offset;
                auto value_ptr = values.data() + offset;
                auto index_ptr = indices.data() + offset;

                // Computing the block-wise means.
                auto mbuffer = center_m.data() + nblocks * r;
                std::fill(mbuffer, mbuffer + nblocks, 0);
                for (size_t i = 0; i < num_entries; ++i) {
                    mbuffer[block[index_ptr[i]]] += value_ptr[i];
                }
                for (size_t b = 0; b < nblocks; ++b) {
                    mbuffer[b] /= block_size[b];
                }

                // Computing the variance from the sum of squared differences.
                // This is technically not the correct variance estimate if we
                // were to consider the loss of residual d.f. from estimating
                // the block means, but it's what the PCA sees, so whatever.
                double& proxyvar = scale_v[r];
                proxyvar = 0;
                {
                    auto block_copy = block_size;
                    for (size_t i = 0; i < num_entries; ++i) {
                        Block curb = block[index_ptr[i]];
                        double diff = value_ptr[i] - mbuffer[curb];
                        proxyvar += diff * diff;
                        --block_copy[curb];
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        proxyvar += mbuffer[b] * mbuffer[b] * block_copy[b];
                    }

                    proxyvar /= NC - 1;
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif

            total_var = pca_utils::process_scale_vector(scale, scale_v);
        }

        pca_utils::CustomSparseMatrix A(NC, NR, nthreads); // transposed; we want genes in the columns.
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (use_eigen) {
            A.use_eigen();
        }
#endif
        A.fill_direct(std::move(values), std::move(indices), std::move(ptrs));

        return A;
    }

private:
    template<typename T, typename IDX, typename Block> 
    Eigen::MatrixXd create_eigen_matrix_dense(
        const tatami::Matrix<T, IDX>* mat, 
        const Block* block, 
        const std::vector<int>& block_size, 
        double& total_var) 
    const {

        auto emat = pca_utils::extract_dense_for_pca(mat, nthreads); // get a column-major matrix with genes in columns.

        // Compute mean and variance of each block.
        size_t NR = emat.rows(), NC = emat.cols();
        int nblocks = block_size.size();
        Eigen::VectorXd scale_v(NC);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<double> mean_buffer(nblocks);
            #pragma omp for
            for (size_t c = 0; c < NC; ++c) {
#else
        SCRAN_CUSTOM_PARALLEL(NC, [&](size_t first, size_t last) -> void {
            std::vector<double> mean_buffer(nblocks);
            for (size_t c = first; c < last; ++c) {
#endif

                auto ptr = emat.data() + c * NR;
                std::fill(mean_buffer.begin(), mean_buffer.end(), 0);
                for (size_t r = 0; r < NR; ++r) {
                    mean_buffer[block[r]] += ptr[r];
                }

                for (int b = 0; b < nblocks; ++b) {
                    const auto& bsize = block_size[b];
                    if (bsize) {
                        mean_buffer[b] /= bsize;
                    }
                }

                // We don't actually compute the blockwise variance, but
                // instead the squared sum of deltas from each block's means,
                // divided by the degrees of freedom as if there weren't any
                // blocks... as this is what PCA actually sees.
                double& proxyvar = scale_v[c];
                proxyvar = 0;
                for (size_t r = 0; r < NR; ++r) {
                    auto& current = ptr[r];
                    current -= mean_buffer[block[r]]; // centering happens here!
                    proxyvar += current * current;
                }
                proxyvar /= NR - 1;

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif

        total_var = pca_utils::process_scale_vector(scale, scale_v);

        if (scale) {
#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (size_t c = 0; c < NC; ++c) {
#else
            SCRAN_CUSTOM_PARALLEL(NC, [&](size_t first, size_t last) -> void {
            for (size_t c = first; c < last; ++c) {
#endif

                auto ptr = emat.data() + c * NR;
                auto sd = scale_v[c];
                for (size_t r = 0; r < NR; ++r) {
                    ptr[r] /= sd; // process_scale_vector() should already protect against div-by-zero.
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif
        }

        return emat;
    }
};

}

#endif
