#ifndef SCRAN_MULTI_BATCH_PCA
#define SCRAN_MULTI_BATCH_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

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
 * @cond
 */
template<class Matrix>
struct MultiBatchEigenMatrix {
    MultiBatchEigenMatrix(const Matrix* m, const Eigen::VectorXd* w, const Eigen::VectorXd* mx) : mat(m), weights(w), means(mx) {}

    auto rows() const { return mat->rows(); }
    auto cols() const { return mat->cols(); }

    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& output) const {
        output.noalias() = *mat * rhs;
        double sub = means->dot(rhs);
        for (Eigen::Index i = 0; i < output.size(); ++i) {
            output[i] -= sub;
            output[i] *= (*weights)[i];
        }
    }

    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& output) const {
        auto combined = weights->cwiseProduct(rhs);
        output.noalias() = mat->adjoint() * combined;
        double sum = combined.sum();
        for (Eigen::Index i = 0; i < output.size(); ++i) {
            output[i] -= (*means)[i] * sum;
        }
        return;
    }

    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output(*mat);
        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                auto& val = output(r, c);
                val -= (*means)[c];
                val *= (*weights)[r];
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
    template<class Matrix>
    void reapply(const Matrix& emat, 
        const Eigen::VectorXd& center_v, 
        const Eigen::VectorXd& scale_v, 
        Eigen::MatrixXd& pcs, 
        Eigen::MatrixXd& rotation,
        Eigen::VectorXd& variance_explained)
    {
        // Dividing by the scaling factor to mimic the division of 'emat'
        // by the scaling factor (after centering).
        auto rIt = rotation.data();
        for (size_t i = 0, iend = rotation.cols(); i < iend; ++i) {
            auto sIt = scale_v.data();
            for (size_t j = 0, jend = rotation.rows(); j < jend; ++j, ++rIt, ++sIt) {
                (*rIt) /= *sIt;
            }
        }

        reapply(emat, center_v, pcs, rotation, variance_explained);
    }

    template<class Matrix>
    void reapply(const Matrix& emat,
        const Eigen::VectorXd& center_v, 
        Eigen::MatrixXd& pcs, 
        const Eigen::MatrixXd& rotation,
        Eigen::VectorXd& variance_explained)
    {
        pcs = emat * rotation;

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
    void run(const tatami::Matrix<T, IDX>* mat, const Batch* batch, Eigen::MatrixXd& pcs, Eigen::MatrixXd& rotation, Eigen::VectorXd& variance_explained, double& total_var) {
        const size_t NC = mat->ncol();
        const auto& batch_size = block_sizes(NC, batch); 
        const size_t nbatchs = batch_size.size();

        // Computing weights.
        Eigen::VectorXd weights(NC);
        for (size_t i = 0; i < NC; ++i) {
            weights[i] = 1/std::sqrt(static_cast<double>(batch_size[batch[i]]));
        }

        Eigen::VectorXd center_v(mat->nrow());
        Eigen::VectorXd scale_v(mat->nrow());

        // Remember, we want to run the PCA on the modified matrix,
        // but we want to apply the rotation vectors to the original matrix
        // (after any centering/scaling but without the batch weights).
        auto executor = [&](const auto& emat) -> void {
            pca_utils::EigenThreadScope t(nthreads);
            irlba::Irlba irb;
            irb.set_number(rank);

            MultiBatchEigenMatrix<typename std::remove_reference<decltype(emat)>::type> thing(&emat, &weights, &center_v);
            if (scale) {
                irb.run(irlba::Scaled<decltype(thing)>(&thing, &scale_v), pcs, rotation, variance_explained);
                reapply(emat, center_v, scale_v, pcs, rotation, variance_explained);
            } else {
                irb.run(thing, pcs, rotation, variance_explained);
                reapply(emat, center_v, pcs, rotation, variance_explained);
            }
        };

        if (mat->sparse()) {
            auto emat = create_eigen_matrix_sparse(mat, center_v, scale_v, batch, batch_size, total_var);
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
    Results run(const tatami::Matrix<T, IDX>* mat, const Batch* batch) {
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
    Results run(const tatami::Matrix<T, IDX>* mat, const Batch* batch, const X* features) {
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
    Eigen::SparseMatrix<double> create_eigen_matrix_sparse(const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& mean_v, 
        Eigen::VectorXd& scale_v, 
        const Batch* batch,
        const std::vector<int>& batch_size,
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        size_t nbatchs = batch_size.size();
        total_var = 0;

        Eigen::SparseMatrix<double> A(NC, NR); // transposed; we want genes in the columns.
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);
            std::vector<int> nnzeros(NR);
            values.reserve(NR);
            indices.reserve(NR);
            std::vector<double> batch_means(nbatchs);
            std::vector<int> batch_count(nbatchs);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                // Computing the grand mean across all batchs.
                std::fill(batch_means.begin(), batch_means.end(), 0);
                std::fill(batch_count.begin(), batch_count.end(), 0);
                for (size_t i = 0; i < range.number; ++i) {
                    auto b = batch[range.index[i]];
                    batch_means[b] += range.value[i];
                    ++batch_count[b];
                }

                double& grand_mean = mean_v[r];
                grand_mean = 0;
                for (size_t b = 0; b < nbatchs; ++b) {
                    grand_mean += batch_means[b] / batch_size[b];
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

                for (size_t i = 0; i < range.number; ++i) {
                    double diff = range.value[i] - grand_mean;
                    auto b = batch[range.index[i]];
                    proxyvar += diff * diff / batch_size[b];
                }

                nnzeros[r] = range.number;
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);
            }

            pca_utils::set_scale(scale, scale_v, total_var);
            pca_utils::fill_sparse_matrix<true>(A, indices, values, nnzeros);
        } else {
            std::vector<double> xbuffer(NR);
            std::vector<int> ibuffer(NR);
            values.reserve(NC);
            indices.reserve(NC);
            std::vector<int> nnzeros(NR);

            std::vector<std::vector<double> > tmp_means(nbatchs, std::vector<double>(NR));
            std::vector<std::vector<int> > tmp_nonzero(nbatchs, std::vector<int>(NR));

            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);

                // Collecting values for the means.
                Batch curb = batch[c];
                auto& cur_means = tmp_means[curb];
                auto& cur_nonzero= tmp_nonzero[curb];
                for (size_t i = 0; i < range.number; ++i) {
                    auto r = range.index[i];
                    cur_means[r] += range.value[i];
                    ++cur_nonzero[r];
                    ++nnzeros[r];
                }
            }

            // Finalizing the means.
            mean_v.setZero();
            for (size_t b = 0; b < nbatchs; ++b) {
                const auto& cur_means = tmp_means[b];
                for (size_t r = 0; r < NR; ++r) {
                    mean_v[r] += cur_means[r] / batch_size[b];
                }
            }
            for (size_t r = 0; r < NR; ++r) {
                mean_v[r] /= nbatchs;
            }

            // Computing the pseudo-variances for each gene.
            scale_v.setZero();
            for (size_t c = 0; c < NC; ++c) {
                const auto& cur_vals = values[c];
                const auto& cur_idx = indices[c];
                auto bs = batch_size[batch[c]];

                for (size_t i = 0; i < cur_idx.size(); ++i) {
                    auto r = cur_idx[i];
                    double diff = cur_vals[i] - mean_v[r];
                    scale_v[r] += diff * diff / bs;
                }
            }

            for (size_t b = 0; b < nbatchs; ++b) {
                const auto& used = tmp_nonzero[b];
                for (size_t r = 0; r < NR; ++r) {
                    double zero_sum = mean_v[r] * mean_v[r] * (batch_size[b] - used[r]);
                    scale_v[r] += zero_sum / batch_size[b];
                }
            }

            pca_utils::set_scale(scale, scale_v, total_var);
            pca_utils::fill_sparse_matrix<false>(A, indices, values, nnzeros);
        }

        A.makeCompressed();
        return A;
    }

private:
    template<typename T, typename IDX, typename Batch> 
    Eigen::MatrixXd create_eigen_matrix_dense(
        const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& mean_v, 
        Eigen::VectorXd& scale_v, 
        const Batch* batch,
        const std::vector<int>& batch_size,
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::MatrixXd output(NC, NR); // transposed.
        std::vector<double> xbuffer(NC);
        double* outIt = output.data();

        size_t nbatchs = batch_size.size();
        std::vector<double> mean_buffer(nbatchs);

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);

            std::fill(mean_buffer.begin(), mean_buffer.end(), 0);
            for (size_t c = 0; c < NC; ++c) {
                mean_buffer[batch[c]] += ptr[c];
            }
            double& grand_mean = mean_v[r];
            grand_mean = 0;
            for (size_t b = 0; b < nbatchs; ++b) {
                grand_mean += mean_buffer[b] / batch_size[b];
            }
            grand_mean /= nbatchs; 

            // We don't actually compute the batchwise variance, but instead
            // the weighted sum of squared deltas, which is what PCA actually sees.
            double& proxyvar = scale_v[r];
            proxyvar = 0;
            for (size_t c = 0; c < NC; ++c) {
                double diff = outIt[c] - grand_mean;
                proxyvar += diff * diff / batch_size[batch[c]];
            }
        }

        pca_utils::set_scale(scale, scale_v, total_var);
        return output;
    }
};

}

#endif
