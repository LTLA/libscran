#ifndef SCRAN_MULTI_BATCH_PCA
#define SCRAN_MULTI_BATCH_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"
#include "CustomSparseMatrix.hpp"
#include "../utils/block_indices.hpp"

/**
 * @file MultiBatchPCA.hpp
 *
 * @brief Compute PCA after adjusting for differences between batch sizes.
 */

namespace scran {

/**
 * @cond
 */
template<class Matrix>
struct MultiBatchEigenMatrix {
    MultiBatchEigenMatrix(const Matrix* m, const Eigen::VectorXd* w, const Eigen::VectorXd* mx) : mat(m), weights(w), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

public:
    typedef irlba::WrappedWorkspace<Matrix> Workspace;

    Workspace workspace() const {
        return mat->workspace();
    }

    template<class Right>
    void multiply(const Right& rhs, Workspace& work, Eigen::VectorXd& output) const {
        irlba::wrapped_multiply(mat, rhs, work, output);

        double sub = means->dot(rhs);
        for (Eigen::Index i = 0; i < output.size(); ++i) {
            auto& val = output.coeffRef(i);
            val -= sub;
            val *= weights->coeff(i);
        }
    }

public:
    struct AdjointWorkspace {
        AdjointWorkspace(size_t n, irlba::WrappedWorkspace<Matrix> c) : combined(nblocks), child(std::move(c)) {}
        Eigen::VectorXd combined;
        irlba::WrappedWorkspace<Matrix> child;
    };

    AdjointWorkspace adjoint_workspace() const {
        return Workspace(weights.size(), mat->adjoint_workspace());
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& output) const {
        work.combined.noalias() = weights->cwiseProduct(rhs);

        irlba::wrapped_adjoint_multiply(mat, work.combined, work.child, output);

        double sum = combined.sum();
        for (Eigen::Index i = 0; i < output.size(); ++i) {
            output.coeffRef(i) -= means->coeff(i) * sum;
        }
        return;
    }

public:
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output;
        if constexpr(irlba::has_realize_method<Matrix>::value) {
            output = mat->realize();
        } else {
            output = Eigen::MatrixXd(*mat);
        }

        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                auto& val = output.coeffRef(r, c);
                val -= means->coeff(c);
                val *= weights->coeff(r);
            }
        }
        return output;
    }
private:
    const Matrix* mat;
    const Eigen::VectorXd* weights;
    const Eigen::VectorXd* means;
};
/**
 * @endcond
 */

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

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    bool use_eigen = false;
#endif

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

#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    MultiBatchPCA& set_use_eigen(bool e = false) {
        use_eigen = true;
        return *this;
    }
#endif

private:
    template<typename Batch>
    static Eigen::VectorXd compute_weights(size_t NC, const Batch* batch, const std::vector<int>& batch_size) {
        Eigen::VectorXd weights(NC);
        for (size_t i = 0; i < NC; ++i) {
            weights[i] = 1/std::sqrt(static_cast<double>(batch_size[batch[i]]));
        }
        return weights;
    }

public:
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
    template<typename T, typename IDX, typename Batch>
    Eigen::MatrixXd test_realize(const tatami::Matrix<T, IDX>* mat, const Batch* batch) const {
        const size_t NC = mat->ncol();
        auto batch_size = block_sizes(NC, batch); 
        auto weights = compute_weights(NC, batch, batch_size);

        Eigen::VectorXd center_v(mat->nrow());
        Eigen::VectorXd scale_v(mat->nrow());

        auto executor = [&](const auto& emat) -> Eigen::MatrixXd {
            MultiBatchEigenMatrix<typename std::remove_reference<decltype(emat)>::type> thing(&emat, &weights, &center_v);
            if (scale) {
                irlba::Scaled<decltype(thing)> scaled(&thing, &scale_v); 
                return scaled.realize();
            } else {
                return thing.realize();
            }
        };

        double total_var = 0; // dummy value.
        if (mat->sparse()) {
            auto emat = create_custom_sparse_matrix(mat, center_v, scale_v, batch, batch_size, total_var);
            return executor(emat);
        } else {
            auto emat = create_eigen_matrix_dense(mat, center_v, scale_v, batch, batch_size, total_var);
            return executor(emat);
        }
    }
#endif

private:
    template<class Matrix, class Rotation>
    void reapply(const Matrix& emat,
        const Eigen::VectorXd& center_v, 
        Eigen::MatrixXd& pcs, 
        const Rotation& rotation,
        Eigen::VectorXd& variance_explained)
    const {
        if constexpr(irlba::has_multiply_method<Matrix>::value) {
            pcs.noalias() = emat * rotation;
        } else {
            pcs.resize(emat.rows(), rotation.cols());
            emat.multiply(rotation, pcs);
        }

        // Effective centering because I don't want to modify 'emat'.
        auto pIt = pcs.data();
        for (size_t i = 0, iend = pcs.cols(); i < iend; ++i) {
            double meanval = center_v.dot(rotation.col(i));
            for (size_t j = 0, jend = pcs.rows(); j < jend; ++j, ++pIt) {
                *pIt -= meanval;
            }
        }

        // Variance is a somewhat murky concept with weights, so we just square
        // it and assume that only the relative value matters.
        for (auto& d : variance_explained) {
            d = d * d;
        }
        return;
    }

private:
    template<typename T, typename IDX, typename Batch>
    void run(const tatami::Matrix<T, IDX>* mat, const Batch* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) const {
        const size_t NC = mat->ncol();
        auto batch_size = block_sizes(NC, batch); 
        auto weights = compute_weights(NC, batch, batch_size);

        Eigen::VectorXd center_v(mat->nrow());
        Eigen::VectorXd scale_v(mat->nrow());

        // Remember, we want to run the PCA on the modified matrix,
        // but we want to apply the rotation vectors to the original matrix
        // (after any centering/scaling but without the batch weights).
        auto executor = [&](const auto& emat) -> void {
            irlba::EigenThreadScope t(nthreads);
            irlba::Irlba irb;
            irb.set_number(rank);

            MultiBatchEigenMatrix<typename std::remove_reference<decltype(emat)>::type> thing(&emat, &weights, &center_v);
            if (scale) {
                irb.run(irlba::Scaled<decltype(thing)>(&thing, &scale_v), pcs, rotation, variance_explained);

                // Dividing the rotation vectors by the scaling factor to mimic
                // the division of 'emat' by the scaling factor (after centering). 
                Eigen::MatrixXd temp = rotation.array().colwise() / scale_v.array();
                reapply(emat, center_v, pcs, temp, variance_explained);
            } else {
                irb.run(thing, pcs, rotation, variance_explained);
                reapply(emat, center_v, pcs, rotation, variance_explained);
            }
        };

        if (mat->sparse()) {
            auto emat = create_custom_sparse_matrix(mat, center_v, scale_v, batch, batch_size, total_var);
            executor(emat);
        } else {
            auto emat = create_eigen_matrix_dense(mat, center_v, scale_v, batch, batch_size, total_var);
            executor(emat);
        }

        if (transpose) {
            pcs.adjointInPlace();
        }

        return;
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
    pca_utils::CustomSparseMatrix create_custom_sparse_matrix(const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& center_v, 
        Eigen::VectorXd& scale_v, 
        const Batch* batch,
        const std::vector<int>& batch_size,
        double& total_var) 
    const {

        size_t NR = mat->nrow(), NC = mat->ncol();
        auto extracted = pca_utils::extract_sparse_for_pca(mat, nthreads); // row-major extraction.
        auto& ptrs = extracted.ptrs;
        auto& values = extracted.values;
        auto& indices = extracted.indices;

        // Computing grand means and variances with weights.
        {
            size_t nbatchs = batch_size.size();

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel num_threads(nthreads)
            {
#else
            SCRAN_CUSTOM_PARALLEL(NR, [&](size_t first, size_t last) -> void {
#endif

                std::vector<double> batch_means(nbatchs);
                std::vector<int> batch_count(nbatchs);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp for
                for (size_t r = 0; r < NR; ++r) {
#else
                for (size_t r = first; r < last; ++r) {
#endif

                    auto offset = ptrs[r];
                    size_t num_entries = ptrs[r+1] - offset;
                    auto value_ptr = values.data() + offset;
                    auto index_ptr = indices.data() + offset;

                    // Computing the grand mean across all batchs.
                    std::fill(batch_means.begin(), batch_means.end(), 0);
                    std::fill(batch_count.begin(), batch_count.end(), 0);
                    for (size_t i = 0; i < num_entries; ++i) {
                        auto b = batch[index_ptr[i]];
                        batch_means[b] += value_ptr[i];
                        ++batch_count[b];
                    }

                    double& grand_mean = center_v[r];
                    grand_mean = 0;
                    for (size_t b = 0; b < nbatchs; ++b) {
                        auto bsize = batch_size[b];
                        if (bsize) {
                            grand_mean += batch_means[b] / bsize;
                        }
                    }
                    grand_mean /= nbatchs;

                    // Computing pseudo-variances where each batch's contribution
                    // is weighted inversely proportional to its size. This aims to
                    // match up with the variances used in the PCA but not the
                    // variances of the output components (where weightings are not used).
                    double& proxyvar = scale_v[r];
                    proxyvar = 0;
                    for (size_t b = 0; b < nbatchs; ++b) {
                        double zero_sum = (batch_size[b] - batch_count[b]) * grand_mean * grand_mean;
                        proxyvar += zero_sum / batch_size[b];
                    }

                    for (size_t i = 0; i < num_entries; ++i) {
                        double diff = value_ptr[i] - grand_mean;
                        auto bsize = batch_size[batch[index_ptr[i]]];
                        if (bsize) {
                            proxyvar += diff * diff / bsize;
                        }
                    }

#ifndef SCRAN_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, nthreads);
#endif

            total_var = pca_utils::process_scale_vector(scale, scale_v);
        }

        // Actually filling the sparse matrix. Note that this is transposed
        // because we want genes in the columns.
        pca_utils::CustomSparseMatrix A(NC, NR, nthreads); 
#ifdef TEST_SCRAN_CUSTOM_SPARSE_MATRIX
        if (use_eigen) {
            A.use_eigen();
        }
#endif
        A.fill_direct(std::move(values), std::move(indices), std::move(ptrs));

        return A;
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

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<double> mean_buffer(nbatchs);
            #pragma omp for
            for (size_t c = 0; c < NC; ++c) {
#else
        SCRAN_CUSTOM_PARALLEL(NC, [&](size_t first, size_t last) -> void {
            std::vector<double> mean_buffer(nbatchs);
            for (size_t c = first; c < last; ++c) {
#endif

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

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif

        total_var = pca_utils::process_scale_vector(scale, scale_v);
        return emat;
    }
};

}

#endif
