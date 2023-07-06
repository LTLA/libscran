#ifndef SCRAN_PCA_UTILS_MOMENTS_HPP
#define SCRAN_PCA_UTILS_MOMENTS_HPP

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"
#include "Eigen/Dense"

#include <vector>

#include "utils.hpp"
#include "../utils/blocking.hpp"

namespace scran {

namespace pca_utils {

struct UnweightedBlockingDetails {
    UnweightedBlockingDetails(size_t nblocks) : block_size(nblocks) {}
    UnweightedBlockingDetails(size_t nblocks, size_t) : UnweightedBlockingDetails(nblocks) {}
    std::vector<int> block_size;
    size_t num_blocks() const { return block_size.size(); }
};

template<typename Block_>
UnweightedBlockingDetails compute_blocking_details(size_t ncells, const Block_* block) {
    auto bsizes = tabulate_ids(ncells, block);
    auto nblocks = bsizes.size();
    UnweightedBlockingDetails output(nblocks);
    output.block_size = std::move(bsizes);
    return output;
}

struct WeightedBlockingDetails {
    WeightedBlockingDetails(size_t nblocks, size_t ncells) : block_size(nblocks), per_element_weight(nblocks), expanded_weights(ncells) {}
    std::vector<int> block_size;
    std::vector<double> per_element_weight;
    double total_block_weight = 0;
    Eigen::VectorXd expanded_weights;
    size_t num_blocks() const { return block_size.size(); }
};

template<typename Block_>
WeightedBlockingDetails compute_blocking_details(size_t ncells, const Block_* block, WeightPolicy block_weight_policy, const VariableBlockWeightParameters& variable_block_weight_parameters) {
    auto bsizes = tabulate_ids(ncells, block);
    auto nblocks = bsizes.size();
    WeightedBlockingDetails output(nblocks, ncells);
    output.block_size = std::move(bsizes);

    auto& total_weight = output.total_block_weight;
    auto& element_weight = output.per_element_weight;

    for (size_t i = 0; i < nblocks; ++i) {
        auto block_size = output.block_size[i];

        // Computing effective block weights that also incorporate division by the
        // block size. This avoids having to do the division by block size in the
        // 'compute_mean_and_variance_regress()' function.
        if (block_size) {
            double block_weight = 1;
            if (block_weight_policy == WeightPolicy::VARIABLE) {
                block_weight = variable_block_weight(block_size, variable_block_weight_parameters);
            }

            element_weight[i] = block_weight / block_size;
            total_weight += block_weight;
        } else {
            element_weight[i] = 0;
        }
    }

    // Setting a placeholder value to avoid problems with division by zero.
    if (total_weight == 0) {
        total_weight = 1; 
    }

    // Expanding them for multiplication in pca_utils::SampleScaledWrapper.
    auto sqrt_weights = element_weight;
    for (auto& s : sqrt_weights) {
        s = std::sqrt(s);
    }

    auto& expanded = output.expanded_weights;
    for (size_t i = 0; i < ncells; ++i) {
        expanded.coeffRef(i) = sqrt_weights[block[i]];
    }

    return output;
}

template<bool weight_>
using BlockingDetails = typename std::conditional<weight_, WeightedBlockingDetails, UnweightedBlockingDetails>::type;

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
            // variance becomes somewhat weird, but we just use the same
            // denominator for consistency in pca_utils::clean_up_projected.
            // Magnitude doesn't matter when scaling for
            // pca_utils::process_scale_vector anyway.
            proxyvar /= NR - 1;
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

            proxyvar /= NR - 1;
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
                multipliers.array() /= scale_v[c];
            }

            auto start = p[c], end = p[c + 1];
            for (size_t s = start; s < end; ++s) {
                pcs.col(i[s]).noalias() += x[s] * multipliers;
            }
        }
    } else {
        const auto& row_nonzero_starts = emat.get_secondary_nonzero_starts();

#ifndef IRLBA_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (size_t t = 0; t < nthreads; ++t) {
#else
        IRLBA_CUSTOM_PARALLEL(nthreads, [&](size_t t) -> void { 
#endif

            const auto& starts = row_nonzero_starts[t];
            const auto& ends = row_nonzero_starts[t + 1];
            Eigen::VectorXd multipliers(nvec);

            for (size_t c = 0; c < ncol; ++c) {
                multipliers.noalias() = rotation.row(c);
                if (scale) {
                    multipliers.array() /= scale_v[c];
                }

                auto start = starts[c], end = ends[c];
                for (size_t s = start; s < end; ++s) {
                    pcs.col(i[s]).noalias() += x[s] * multipliers;
                }
            }

#ifndef IRLBA_CUSTOM_PARALLEL
        }
#else
        });
#endif
    }
}

template<bool rows_are_dims_>
void clean_up_projected(Eigen::MatrixXd& proj, Eigen::VectorXd& D) {
    // Empirically centering to give nice centered PCs, because we can't
    // guarantee that the projection is centered in this manner.
    if constexpr(rows_are_dims_) {
        for (size_t i = 0, iend = proj.rows(); i < iend; ++i) {
            proj.row(i).array() -= proj.row(i).sum() / proj.cols();
        }
    } else {
        for (size_t i = 0, iend = proj.cols(); i < iend; ++i) {
            proj.col(i).array() -= proj.col(i).sum() / proj.rows();
        }
    }

    // Just dividing by the number of observations - 1 regardless of weighting.
    double denom = (rows_are_dims_ ? proj.cols() : proj.rows()) - 1;
    for (auto& d : D) {
        d = d * d / denom;
    }
}

}

}

#endif
