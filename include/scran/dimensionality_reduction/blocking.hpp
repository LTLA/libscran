#ifndef SCRAN_PCA_UTILS_MOMENTS_HPP
#define SCRAN_PCA_UTILS_MOMENTS_HPP

#include "tatami/tatami.hpp"
#include "Eigen/Dense"

#include <vector>

namespace scran {

namespace pca_utils {

struct UnweightedBlockingDetails {
    UnweightedBlockingDetails(size_t nblocks, size_t) : block_size(nblocks) {}
    std::vector<int> block_size;
    size_t num_blocks() const { return block_size.size(); }
};

struct WeightedBlockingDetails {
    WeightedBlockingDetails(size_t nblocks, size_t ncells) : block_size(nblocks), per_element_weight(nblocks), expanded_weights(ncells) {}
    std::vector<int> block_size;
    std::vector<double> per_element_weight;
    double total_block_weight = 0;
    Eigen::VectorXd expanded_weights;
    size_t num_blocks() const { return block_size.size(); }
};

template<bool weight_>
using BlockingDetails = typename std::conditional<weight_, WeightedBlockingDetails, UnweightedBlockingDetails>::type;

template<bool weight_, typename Block_>
BlockingDetails<weight_> compute_blocking_details(size_t ncells, const Block_* block) {
    auto nblocks = static_cast<size_t>(ncells ? *std::max_element(block, block + ncells) : 0) + 1;
    typename std::conditional<weight_, WeightedBlockingDetails, UnweightedBlockingDetails>::type output(nblocks, ncells);

    auto& block_size = output.block_size;
    for (size_t j = 0; j < ncells; ++j) {
        ++block_size[block[j]];
    }

    if constexpr(weight_) {
        auto& total_weight = output.total_block_weight;
        auto& element_weight = output.per_element_weight;

        for (size_t i = 0; i < nblocks; ++i) {
            // This per-block total weight can be adjusted for more weighting schemes.
            // By default, we assume that each block should be weighted equally, but
            // that need not be the case if we want to penalize very small blocks.
            double block_weight = 1;

            // Computing effective block weights that also incorporate division by the
            // block size. This avoids having to do the division by block size in the
            // 'compute_mean_and_variance_regress()' function.
            const auto& bs = block_size[i];
            if (bs) {
                total_weight += block_weight;
                element_weight[i] = block_weight / bs;
            } else {
                element_weight[i] = 0;
            }
        }

        auto& expanded = output.expanded_weights;
        for (size_t i = 0; i < ncells; ++i) {
            expanded.coeffRef(i) = element_weight[block[i]];
        }

        // Setting a placeholder value to avoid problems with division by zero.
        if (total_weight == 0) {
            total_weight = 1; 
        }
    }

    return output;
}

template<bool weight_, typename Block_>
void compute_mean_and_variance_regress(
    const SparseMatrix& emat,
    const Block_* block, 
    const BlockingDetails<weight_>& block_details,
    Eigen::MatrixXd& centers,
    Eigen::VectorXd& variances,
    int nthreads) 
{
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NR = emat.rows();
        const auto& values = emat.get_values();
        const auto& indices = emat.get_indices();
        const auto& ptrs = emat.get_pointers();

        size_t nblocks = block_details.num_blocks();
        const auto& block_size = block_details.block_size;

        auto mptr = centers.data() + static_cast<size_t>(start) * nblocks; // coerce to size_t to avoid overflow.
        auto block_copy = block_details.block_size;

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
            double& proxyvar = variances[c];
            proxyvar = 0;
            std::copy(block_size.begin(), block_size.end(), block_copy.begin());

            for (size_t i = 0; i < num_entries; ++i) {
                Block_ curb = block[index_ptr[i]];
                double diff = value_ptr[i] - mptr[curb];

                double squared = diff * diff;
                if constexpr(weight_) {
                    squared *= block_details.per_element_weight[curb];
                }

                proxyvar += squared;
                --block_copy[curb];
            }

            for (size_t b = 0; b < nblocks; ++b) {
                auto extra = mptr[b] * mptr[b] * block_copy[b];
                if constexpr(weight_) {
                    extra *= block_details.per_element_weight[b];
                }
                proxyvar += extra;
            }

            // If we're not dealing with weights, we compute the actual sample
            // variance for easy interpretation (and to match up with the
            // per-PC calculations in pca_utils::clean_up).
            //
            // If we're dealing with weights, the concept of the sample
            // variance becomes somewhat weird. So, we just keep proxyvar as
            // the sum of squares, which is equivalent to the D^2 after SVD
            // (computed by pca_utils::clean_up_projected). Magnitude doesn't
            // matter when scaling for pca_utils::process_scale_vector anyway.
            if constexpr(!weight_) {
                proxyvar /= NR - 1;
            }
        }
    }, emat.cols(), nthreads);
}

template<bool weight_, typename Block_>
void compute_mean_and_variance_regress(
    const Eigen::MatrixXd& emat,
    const Block_* block,
    const BlockingDetails<weight_>& block_details,
    Eigen::MatrixXd& centers, 
    Eigen::VectorXd& variances, 
    int nthreads)
{
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NR = emat.rows();
        auto ptr = emat.data() + static_cast<size_t>(start) * NR; 

        size_t nblocks = block_details.num_blocks();
        const auto& block_size = block_details.block_size;

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

            double& proxyvar = variances[c];
            proxyvar = 0;

            for (size_t r = 0; r < NR; ++r) {
                auto curb = block[r];
                double delta = ptr[r] - mptr[curb];
                auto squared = delta * delta;
                if constexpr(weight_) {
                    squared *= block_details.per_element_weight[curb];
                }
                proxyvar += squared;
            }

            if constexpr(!weight_) {
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
                multipliers.array() *= scale_v[c];
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
                    multipliers.array() *= scale_v[c];
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
