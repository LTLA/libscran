#ifndef SCRAN_BLOCKED_PCA
#define SCRAN_BLOCKED_PCA

#include "tatami/stats/variances.hpp"
#include "tatami/base/DelayedSubset.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>

#include "pca_utils.hpp"

namespace scran {

template<bool transposed, class Matrix, typename Block>
struct BlockedEigenMatrix {
    BlockedEigenMatrix(const Matrix& m, const Block* b, const Eigen::MatrixXd& mx) : mat(m), block(b), means(mx) {}

    auto rows() const { return mat.rows(); }
    auto cols() const { return mat.cols(); }

    template<class Right>
    auto operator*(const Right& rhs) const {
        if constexpr(!transposed) {
            Eigen::MatrixXd raw = mat * rhs;
            Eigen::MatrixXd sub = means * rhs;
            for (Eigen::Index c = 0; c < raw.cols(); ++c) {
                for (Eigen::Index r = 0; r < raw.rows(); ++r) {
                    raw(r, c) -= sub(block[r], c);
                }
            }
            return raw;
        } else {
            Eigen::MatrixXd aggr(means.rows(), rhs.cols());
            aggr.setZero(means.rows(), rhs.cols());

            for (Eigen::Index c = 0; c < rhs.cols(); ++c) {
                for (Eigen::Index i = 0; i < rhs.rows(); ++i) {
                    aggr(block[i], c) += rhs(i, c); 
                }
            }

            Eigen::MatrixXd output = mat.adjoint() * rhs;
            output.noalias() -= means.adjoint() * aggr;
            return output;
        }
    }

    BlockedEigenMatrix<!transposed, Matrix, Block> adjoint() const {
        return BlockedEigenMatrix<!transposed, Matrix, Block>(mat, block, means);
    }

    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output(mat);
        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                output(r, c) -= means(block[r], c);
            }
        }
        if constexpr(transposed) {
            output.adjointInPlace();
        }
        return output;
    }
private:
    const Matrix& mat;
    const Block* block;
    const Eigen::MatrixXd& means;
};

class BlockedPCA {
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
    BlockedPCA& set_rank(int r = Defaults::rank) {
        rank = r;
        return *this;
    }

    BlockedPCA& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

private:
    template<typename T, typename IDX, typename Block>
    void run(const tatami::Matrix<T, IDX>* mat, const Block* block, Eigen::MatrixXd& pcs, Eigen::VectorXd& variance_explained, double& total_var) {
        irb.set_number(rank);

        const size_t NC = mat->ncol();
        const int nblocks = (NC ? *std::max_element(block, block + NC) + 1 : 1);
        std::vector<int> block_size(nblocks);
        for (size_t j = 0; j < NC; ++j) {
            ++block_size[block[j]];
        }

        if (mat->sparse()) {
            Eigen::MatrixXd center_m(nblocks, mat->nrow());
            Eigen::VectorXd scale_v(mat->nrow());
            auto emat = create_eigen_matrix_sparse(mat, center_m, scale_v, block, block_size, total_var);

            // Dummy vector, the real centering is done in center_m.
            Eigen::VectorXd center_v(mat->nrow()); 
            center_v.setZero(); 

            auto result = irb.run(BlockedEigenMatrix<false, decltype(emat), Block>(emat, block, center_m), center_v, scale_v);
            pca_utils::clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
        } else {
            auto emat = create_eigen_matrix_dense(mat, block, block_size, total_var);
            auto result = irb.run(emat); // already centered and scaled, if relevant.
            pca_utils::clean_up(mat->ncol(), result.U, result.D, pcs, variance_explained);
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
        Eigen::MatrixXd& center_m, 
        Eigen::VectorXd& scale_v, 
        const Block* block, 
        const std::vector<int>& block_size, 
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::SparseMatrix<double> A(NC, NR); // transposed; we want genes in the columns.
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;

        const size_t nblocks = block_size.size();
        double * mbuffer = center_m.data();

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        if (mat->prefer_rows()) {
            std::vector<double> xbuffer(NC);
            std::vector<int> ibuffer(NC);
            std::vector<int> nnzeros(NR);
            values.reserve(NR);
            indices.reserve(NR);

            for (size_t r = 0; r < NR; ++r, mbuffer += nblocks) {
                auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data());

                // Computing the block-wise means.
                std::fill(mbuffer, mbuffer + nblocks, 0);
                for (size_t i = 0; i < range.number; ++i) {
                    mbuffer[block[range.index[i]]] += range.value[i];
                }
                for (size_t b = 0; b < nblocks; ++b) {
                    mbuffer[b] /= block_size[b];
                }

                // Computing the variance from the sum of squared differences.
                // This is technically not the correct variance estimate if 
                // we were to consider the blocks, but it's what the PCA sees.
                double& proxyvar = scale_v[r];
                proxyvar = 0;
                {
                    auto block_copy = block_size;
                    for (size_t i = 0; i < range.number; ++i) {
                        Block curb = block[range.index[i]];
                        double diff = range.value[i] - mbuffer[curb];
                        proxyvar += diff * diff;
                        --block_copy[curb];
                    }
                    
                    for (size_t b = 0; b < nblocks; ++b) {
                        proxyvar += mbuffer[b] * mbuffer[b] * block_copy[b];
                    }

                    proxyvar /= NC - 1;
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
            std::vector<std::vector<int> > tmp_nonzeros(nblocks, std::vector<int>(NR));

            for (size_t c = 0; c < NC; ++c) {
                auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data());
                values.emplace_back(range.value, range.value + range.number);
                indices.emplace_back(range.index, range.index + range.number);

                // Collecting values for the means.
                Block curb = block[c];
                auto& cur_means = tmp_means[curb];
                auto& cur_nonzeros = tmp_nonzeros[curb];
                for (size_t i = 0; i < range.number; ++i) {
                    auto r = range.index[i];
                    cur_means[r] += range.value[i];
                    ++cur_nonzeros[r];
                    ++nnzeros[r];
                }
            }

            // Finalizing the means.
            for (size_t b = 0; b < nblocks; ++b) {
                auto& cur_means = tmp_means[b];
                for (size_t r = 0; r < cur_means.size(); ++r) {
                    cur_means[r] /= block_size[b];
                    center_m(b, r) = cur_means[r];
                }
            }

            // Computing the pseudo-variances for each gene.
            {
                scale_v.setZero();

                for (size_t c = 0; c < NC; ++c) {
                    const auto& cur_means = tmp_means[block[c]];
                    const auto& cur_vals = values[c];
                    const auto& cur_idx = indices[c];

                    for (size_t i = 0; i < cur_idx.size(); ++i) {
                        auto r = cur_idx[i];
                        double diff = cur_vals[i] - cur_means[r];
                        scale_v[r] += diff * diff;
                    }
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    const auto& cur_means = tmp_means[b];
                    const auto& cur_nonzeros = tmp_nonzeros[b];
                    for (size_t r = 0; r < NR; ++r) {
                        scale_v[r] += cur_means[r] * cur_means[r] * (block_size[b] - cur_nonzeros[r]);
                    }
                }

                for (size_t r = 0; r < NR; ++r) {
                    scale_v[r] /= NC - 1;
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
        const Block* block, 
        const std::vector<int>& block_size, 
        double& total_var) 
    {
        size_t NR = mat->nrow(), NC = mat->ncol();
        total_var = 0;

        Eigen::MatrixXd output(NC, NR); // transposed.
        std::vector<double> xbuffer(NC);
        double* outIt = output.data();

        int nblocks = block_size.size();
        std::vector<double> mean_buffer(nblocks);

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::RunPCA", "Preparing the input matrix");
#endif

        for (size_t r = 0; r < NR; ++r, outIt += NC) {
            auto ptr = mat->row_copy(r, outIt);
        
            // Compute the means of each block.
            std::fill(mean_buffer.begin(), mean_buffer.end(), 0);
            {
                auto copy = outIt;
                auto bcopy = block;
                for (size_t c = 0; c < NC; ++c, ++copy, ++bcopy) {
                    mean_buffer[*bcopy] += *copy;
                }
            }
            for (int b = 0; b < nblocks; ++b) {
                if (block_size[b]) {
                    mean_buffer[b] /= block_size[b];
                }
            }

            // We don't actually compute the blockwise variance, but instead
            // the squared sum of deltas, which is what PCA actually sees.
            double proxyvar = 0;
            {
                auto copy = outIt;
                auto bcopy = block;
                for (size_t c = 0; c < NC; ++c, ++copy, ++bcopy) {
                    *copy -= mean_buffer[*bcopy];
                    proxyvar += *copy * *copy;
                }
                proxyvar /= NC - 1;
            }

            pca_utils::apply_scale(scale, proxyvar, NC, outIt, total_var);
        }

        return output;
    }

};

}

#endif
