#ifndef SCRAN_PCA_UTILS_MOMENTS_HPP
#define SCRAN_PCA_UTILS_MOMENTS_HPP

#include "tatami/tatami.hpp"
#include "Eigen/Dense"

#include <vector>

namespace scran {

namespace pca_utils {

std::vector<int> compute_block_size(size_t n, const Block_* block) {
    auto nblocks = static_cast<size_t>(n ? *std::max_element(block, block + n) : 0) + 1;
    std::vector<int> block_size(nblocks);
    for (size_t j = 0; j < NC; ++j) {
        ++block_size[block[j]];
    }
    return block_size;
}

inline void rescale_block_weights(const std::vector<int>& block_size, std::vector<double>& block_weight) {
    // Computing effective block weights that also incorporate division by the
    // block size. This avoids having to do the division by block size in the
    // 'compute_mean_and_variance_regress()' function.
    for (size_t i = 0, end = block_size.size(); i < end; ++i) {
        if (block_size[i]) {
            block_weight[i] /= block_size[i];
        } else {
            block_weight[i] = 0;
        }
    }
}

template<bool weight_> 
auto compute_block_weight(const std::vector<int>& block_size) {
    if constexpr(weight_) {
        std::vector<double> block_weight(block.size(), 1);
        std::fill(block_weight.begin(), block_weight.end(), 1.0); // TODO: allow variable weights here based on some convergence function.
        pca_utils::rescale_block_weights(block_size, block_weight);
        return block_weight;
    } else {
        return false;
    }
}

template<typename Block_>
Eigen::VectorXd expand_block_weight(size_t n, const Block_* block, const std::vector<double>& block_weight) {
    Eigen::VectorXd output(n);
    for (size_t i = 0; i < n; ++i) {
        output.coeffRef(i) = block_weight[block[i]];
    }
    return output;
}

inline double total_block_weight(const std::vector<double>& block_weight) {
    double total_weight = std::accumulate(block_weight.begin(), block_weight.end(), 0.0);
    if (total_weight == 0) {
        return 1; // placeholder value.
    } else {
        return total_weight;
    }
}

template<typename Block_, typename Weights_>
void compute_mean_and_variance_regress(
    const SparseMatrix& emat,
    const Block* block, 
    const std::vector<int>& block_size, 
    const Weights_& block_weight,
    Eigen::MatrixXd& centers,
    Eigen::VectorXd& variances,
    int nthreads) 
{
    // We assume that block_weight has already been passed through rescale_block_weight,
    // such that they are already downscaled according to the size of the blocks.
    constexpr bool use_weights = std::is_same<Weights_, std::vector<double> >::value;
    double total_weight = 0;
    if constexpr(use_weights) {
        total_weight = total_block_weight(block_weight);
    }

    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NR = emat.rows(), NC = emat.cols();
        const auto& x = emat.get_values();
        const auto& i = emat.get_indices();
        const auto& p = emat.get_pointers();

        size_t nblocks = block_size.size();
        auto mptr = centers.data() + static_cast<size_t>(start) * nblocks; // coerce to size_t to avoid overflow.
        auto block_copy = block_size;

        for (size_t c = start, end = start + length; c < end; ++c, mptr += nblocks) {
            auto offset = ptrs[c];
            size_t num_entries = ptrs[c + 1] - offset;
            auto value_ptr = values.data() + offset;
            auto index_ptr = indices.data() + offset;

            // Computing the block-wise means.
            std::fill(mptr, mptr + nblocks, 0);
            for (size_t i = 0; i < num_entries; ++i) {
                mptr[block[index_ptr[i]]] += value_ptr[i];
            }
            for (size_t b = 0; b < nblocks; ++b) {
                auto bsize = block_size[b];
                if (bsize) {
                    mptr[b] /= bsize;
                }
            }

            // Computing the variance from the sum of squared differences.
            // This is technically not the correct variance estimate if we
            // were to consider the loss of residual d.f. from estimating
            // the block means, but it's what the PCA sees, so whatever.
            double& proxyvar = variances[r];
            proxyvar = 0;
            std::copy(block_size.begin(), block_size.end(), block_copy.begin());

            for (size_t i = 0; i < num_entries; ++i) {
                Block curb = block[index_ptr[i]];
                double diff = value_ptr[i] - mptr[curb];

                double squared = diff * diff;
                if constexpr(use_weights) {
                    mptr[b] *= block_weight[curb];
                }

                proxyvar += squared;
                --block_copy[curb];
            }

            for (size_t b = 0; b < nblocks; ++b) {
                auto extra = mptr[b] * mptr[b] * block_copy[b];
                if constexpr(use_weights) {
                    extra *= block_weight[b];
                }
                proxyvar += extra;
            }

            // If weighting is involved, we've already scaled it by the number of cells,
            // so there's no need to do this extra step. As before, the concept of the
            // variance is kinda murky when we do all this weighting, but just remember
            // that PCA only looks at the sum of squares from the mean.
            if constexpr(!use_weights) {
                proxyvar /= NC - 1;
            }
        }
    }, emat.cols(), nthreads);
}

template<typename Block_, typename Weights_>
void compute_mean_and_variance_regress(
    const Eigen::MatrixXd& mat,
    const Block_* block,
    const std::vector<int>& block_size,
    const Weights_& block_weight,
    Eigen::MatrixXd& centers, 
    Eigen::VectorXd& variances, 
    int nthreads)
{
    constexpr bool use_weights = std::is_same<Weights_, std::vector<double> >::value;
    double total_weight = 0;
    if constexpr(use_weights) {
        total_weight = total_block_weight(block_weight);
    }

    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NC = emat.cols(), NR = emat.rows();
        auto ptr = emat.data() + static_cast<size_t>(start) * NR; 
        size_t nblocks = block_size.size();
        auto mptr = centers.data() + static_cast<size_t>(start) * nblocks;

        for (size_t c = start, end = start + length; c < end; ++c, ptr += NR, mptr += nblocks) {
            std::fill(mptr, mptr + nblocks, 0);
            for (size_t r = 0; r < NR; ++r) {
                mptr[block[r]] += ptr[r];
            }

            for (int b = 0; b < nblocks; ++b) {
                const auto& bsize = block_size[b];
                if (bsize) {
                    mptr[b] /= bsize;
                }
            }

            // We don't actually compute the blockwise variance, but
            // instead the squared sum of deltas from each block's means,
            // divided by the degrees of freedom as if there weren't any
            // blocks... as this is what PCA actually sees.
            double& proxyvar = variances[c];
            proxyvar = 0;

            for (size_t r = 0; r < NR; ++r) {
                auto curb = block[r];
                double delta = ptr[r] - mptr[curb];
                auto squared = current * current;
                if constexpr(use_weights) {
                    squared *= block_weight[curb];
                }
                proxyvar += squared;
            }

            if constexpr(!use_weights) {
                proxyvar /= NR - 1;
            }
        }
    }, emat.cols(), nthreads);
}

inline void project_sparse_matrix(const SparseMatrix& emat, Eigen::MatrixXd& pcs, const Eigen::MatrixXd& rotation, bool scale, const Eigen::VectorXd& scale_v, int nthreads) {
    size_t nvec = rotation.cols();
    size_t nrow = emat.rows();
    size_t ncol = emat.cols();

    pcs.resize(nvec, nrow); // used a transposed version for more cache efficiency.
    pcs.setZero();

    const auto& x = emat.get_values();
    const auto& i = emat.get_indices();
    const auto& p = emat.get_pointers();

    if (nthreads == 1) {
        Eigen::VectorXd multipliers(nvec);
        for (size_t c = 0; c < ncol; ++c) {
            multipliers.noalias() = rotation.row(c);
            if (scale) {
                multipliers.noalias() *= scale_v[c];
            }

            auto start = p[c], end = p[c + 1];
            for (size_t s = start; s < end; ++s) {
                pcs.col(i[s]).noalias() += x[s] * multipliers;
            }
        }
    } else {
        const auto& row_nonzero_starts = emat.get_secondary_nonzero_starts();

        IRLBA_CUSTOM_PARALLEL(nthreads, [&](size_t t) -> void { 
            const auto& starts = row_nonzero_starts[t];
            const auto& ends = row_nonzero_starts[t + 1];
            Eigen::VectorXd multipliers(nvec);

            for (size_t c = 0; c < ncol; ++c) {
                multipliers.noalias() = rotation.row(c);
                if (scale) {
                    multipliers.noalias() *= scale_v[c];
                }

                auto start = starts[c], end = ends[c];
                for (size_t s = start; s < end; ++s) {
                    pcs.col(i[s]).noalias() += x[s] * multipliers;
                }
            }
        });
    }
}

}

}

#endif
