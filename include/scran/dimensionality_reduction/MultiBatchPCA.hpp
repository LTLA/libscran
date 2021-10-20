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

namespace scran {

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

class MultiBatchPCA {
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
    MultiBatchPCA& set_rank(int r = Defaults::rank) {
        irb.set_number(r);
        return *this;
    }

    MultiBatchPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

private:
    template<class Matrix>
    void reapply(const Matrix& emat, 
        const Eigen::VectorXd& center_v, 
        const Eigen::VectorXd& scale_v, 
        Eigen::MatrixXd V, 
        const Eigen::VectorXd& D, 
        Eigen::MatrixXd& pcs, 
        Eigen::VectorXd& variance_explained)
    {
        for (Eigen::Index i = 0; i < V.cols(); ++i) {
            for (Eigen::Index j = 0; j < V.rows(); ++j) {
                V(j, i) /= scale_v[j];
            }
        }
        reapply(emat, center_v, V, D, pcs, variance_explained);
    }

    template<class Matrix>
    void reapply(const Matrix& emat,
        const Eigen::VectorXd& center_v, 
        const Eigen::MatrixXd& V, 
        const Eigen::VectorXd& D, 
        Eigen::MatrixXd& pcs, 
        Eigen::VectorXd& variance_explained)
    {
        pcs = emat * V;
        for (Eigen::Index i = 0; i < pcs.cols(); ++i) {
            double meanval = center_v.dot(V.col(i));
            for (Eigen::Index j = 0; j < pcs.rows(); ++j) {
                pcs(j, i) -= meanval;
            }
        }

        variance_explained.resize(D.size());
        for (int i = 0; i < D.size(); ++i) {
            variance_explained[i] = D[i] * D[i];
        }
        return;
    }

private:
    template<typename T, typename IDX, typename Block>
    void run(const tatami::Matrix<T, IDX>* mat, const Block* block, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained, double& total_var) {
        const size_t NC = mat->ncol();
        const auto& block_size = block_sizes(NC, block); 
        const size_t nblocks = block_size.size();

        // Computing weights.
        Eigen::VectorXd weights(NC);
        for (size_t i = 0; i < NC; ++i) {
            weights[i] = 1/std::sqrt(static_cast<double>(block_size[block[i]]));
        }

        Eigen::VectorXd center_v(mat->nrow());
        Eigen::VectorXd scale_v(mat->nrow());

        // Remember, we want to run the PCA on the modified matrix,
        // but we want to apply the rotation vectors to the original matrix
        // (after any centering/scaling but without the batch weights).
        auto executor = [&](const auto& emat) -> void {
            MultiBatchEigenMatrix<typename std::remove_reference<decltype(emat)>::type> thing(&emat, &weights, &center_v);
            if (scale) {
                auto result = irb.run(irlba::Scaled<decltype(thing)>(&thing, &scale_v));
                reapply(emat, center_v, scale_v, result.V, result.D, pcs, variance_explained);
            } else {
                auto result = irb.run(thing);
                reapply(emat, center_v, result.V, result.D, pcs, variance_explained);
            }
        };

        if (mat->sparse()) {
            auto emat = create_eigen_matrix_sparse(mat, center_v, scale_v, block, block_size, total_var);
            executor(emat);
        } else {
            auto emat = create_eigen_matrix_dense(mat, center_v, scale_v, block, block_size, total_var);
            executor(emat);
        }

        return;
    }

public:
    struct Results {
        Eigen::MatrixXd pcs;
        Eigen::VectorXd variance_explained;
        double total_variance = 0;
    };

    template<typename T, typename IDX, typename Block>
    Results run(const tatami::Matrix<T, IDX>* mat, const Block* block) {
        Results output;
        run(mat, block, output.pcs, output.variance_explained, output.total_variance);
        return output;
    }

    template<typename T, typename IDX, typename Block, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const Block* block, const X* features) {
        Results output;
        if (!features) {
            run(mat, block, output.pcs, output.variance_explained, output.total_variance);
        } else {
            auto subsetted = pca_utils::subset_matrix_by_features(mat, features);
            run(subsetted.get(), block, output.pcs, output.variance_explained, output.total_variance);
        }
        return output;
    }

private:
    template<typename T, typename IDX, typename Block> 
    Eigen::SparseMatrix<double> create_eigen_matrix_sparse(const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& mean_v, 
        Eigen::VectorXd& scale_v, 
        const Block* block,
        const std::vector<int>& block_size,
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        size_t nblocks = block_size.size();
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
            std::vector<double> block_means(nblocks);
            std::vector<int> block_count(nblocks);

            for (size_t r = 0; r < NR; ++r) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                // Computing the grand mean across all blocks.
                std::fill(block_means.begin(), block_means.end(), 0);
                std::fill(block_count.begin(), block_count.end(), 0);
                for (size_t i = 0; i < range.number; ++i) {
                    auto b = block[range.index[i]];
                    block_means[b] += range.value[i];
                    ++block_count[b];
                }

                double& grand_mean = mean_v[r];
                grand_mean = 0;
                for (size_t b = 0; b < nblocks; ++b) {
                    grand_mean += block_means[b] / block_size[b];
                }
                grand_mean /= nblocks;

                // Computing pseudo-variances where each block's contribution
                // is weighted inversely proportional to its size. This aims to
                // match up with the variances used in the PCA but not the
                // variances of the output components (where weightings are not used).
                double& proxyvar = scale_v[r];
                proxyvar = 0;
                for (size_t b = 0; b < nblocks; ++b) {
                    double zero_sum = (block_size[b] - block_count[b]) * grand_mean * grand_mean;
                    proxyvar += zero_sum / block_size[b];
                }

                for (size_t i = 0; i < range.number; ++i) {
                    double diff = range.value[i] - grand_mean;
                    auto b = block[range.index[i]];
                    proxyvar += diff * diff / block_size[b];
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

            std::vector<std::vector<double> > tmp_means(nblocks, std::vector<double>(NR));
            std::vector<std::vector<int> > tmp_nonzero(nblocks, std::vector<int>(NR));

            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);

                // Collecting values for the means.
                Block curb = block[c];
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
            for (size_t b = 0; b < nblocks; ++b) {
                const auto& cur_means = tmp_means[b];
                for (size_t r = 0; r < NR; ++r) {
                    mean_v[r] += cur_means[r] / block_size[b];
                }
            }
            for (size_t r = 0; r < NR; ++r) {
                mean_v[r] /= nblocks;
            }

            // Computing the pseudo-variances for each gene.
            scale_v.setZero();
            for (size_t c = 0; c < NC; ++c) {
                const auto& cur_vals = values[c];
                const auto& cur_idx = indices[c];
                auto bs = block_size[block[c]];

                for (size_t i = 0; i < cur_idx.size(); ++i) {
                    auto r = cur_idx[i];
                    double diff = cur_vals[i] - mean_v[r];
                    scale_v[r] += diff * diff / bs;
                }
            }

            for (size_t b = 0; b < nblocks; ++b) {
                const auto& used = tmp_nonzero[b];
                for (size_t r = 0; r < NR; ++r) {
                    double zero_sum = mean_v[r] * mean_v[r] * (block_size[b] - used[r]);
                    scale_v[r] += zero_sum / block_size[b];
                }
            }

            pca_utils::set_scale(scale, scale_v, total_var);
            pca_utils::fill_sparse_matrix<false>(A, indices, values, nnzeros);
        }

        A.makeCompressed();
        return A;
    }

private:
    template<typename T, typename IDX, typename Block> 
    Eigen::MatrixXd create_eigen_matrix_dense(
        const tatami::Matrix<T, IDX>* mat, 
        Eigen::VectorXd& mean_v, 
        Eigen::VectorXd& scale_v, 
        const Block* block,
        const std::vector<int>& block_size,
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::MatrixXd output(NC, NR); // transposed.
        std::vector<double> xbuffer(NC);
        double* outIt = output.data();

        size_t nblocks = block_size.size();
        std::vector<double> mean_buffer(nblocks);

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);

            std::fill(mean_buffer.begin(), mean_buffer.end(), 0);
            for (size_t c = 0; c < NC; ++c) {
                mean_buffer[block[c]] += ptr[c];
            }
            double& grand_mean = mean_v[r];
            grand_mean = 0;
            for (size_t b = 0; b < nblocks; ++b) {
                grand_mean += mean_buffer[b] / block_size[b];
            }
            grand_mean /= nblocks; 

            // We don't actually compute the blockwise variance, but instead
            // the weighted sum of squared deltas, which is what PCA actually sees.
            double& proxyvar = scale_v[r];
            proxyvar = 0;
            for (size_t c = 0; c < NC; ++c) {
                double diff = outIt[c] - grand_mean;
                proxyvar += diff * diff / block_size[block[c]];
            }
        }

        pca_utils::set_scale(scale, scale_v, total_var);
        return output;
    }
};

}

#endif
