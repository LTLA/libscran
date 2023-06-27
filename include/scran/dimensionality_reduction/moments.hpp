#ifndef SCRAN_PCA_MOMENTS_HPP
#define SCRAN_PCA_MOMENTS_HPP

#include "tatami/tatami.hpp"
#include "Eigen/Dense"

namespace scran {

namespace pca_utils {

/******************************************
 *** Genewise moments from a CSR matrix ***
 ******************************************/

inline void compute_mean_and_variance_simple( 
    int NR,
    int NC,
    const std::vector<double>& values,
    const std::vector<int>& indices,
    const std::vector<size_t>& ptrs,
    Eigen::VectorXd& centers,
    Eigen::VectorXd& variances,
    int nthreads) 
{
    tatami::parallelize([&](size_t, int start, int length) -> void {
        for (int r = start, end = start + length; r < end; ++r) {
            auto offset = ptrs[r];

            tatami::SparseRange<double, int> range;
            range.number = ptrs[r+1] - offset;
            range.value = values.data() + offset;
            range.index = indices.data() + offset;

            auto results = tatami::stats::variances::compute_direct(range, NC);
            centers.coeffRef(r) = results.first;
            variances.coeffRef(r) = results.second;
        }
    }, NR, nthreads);
}

template<typename Block_>
void compute_mean_and_variance_equiweight( 
    int NR,
    const std::vector<double>& values,
    const std::vector<int>& indices,
    const std::vector<size_t>& ptrs,
    const Block_* block, 
    const std::vector<int>& block_size, 
    Eigen::VectorXd& centers,
    Eigen::VectorXd& variances,
    int nthreads) 
{
    tatami::parallelize([&](size_t, int start, int length) -> void {
        size_t nblocks = block_size.size();
        std::vector<double> block_means(nblocks);
        std::vector<int> block_count(nblocks);

        for (int r = start, end = start + length; r < end; ++r) {
            auto offset = ptrs[r];
            size_t num_entries = ptrs[r+1] - offset;
            auto value_ptr = values.data() + offset;
            auto index_ptr = indices.data() + offset;

            // Computing the grand mean across all blocks.
            std::fill(block_means.begin(), block_means.end(), 0);
            std::fill(block_count.begin(), block_count.end(), 0);
            for (size_t i = 0; i < num_entries; ++i) {
                auto b = block[index_ptr[i]];
                block_means[b] += value_ptr[i];
                ++block_count[b];
            }

            double& grand_mean = centers[r];
            grand_mean = 0;
            for (size_t b = 0; b < nblocks; ++b) {
                auto bsize = block_size[b];
                if (bsize) {
                    grand_mean += block_means[b] / bsize;
                }
            }
            grand_mean /= nblocks;

            // Computing pseudo-variances where each block's contribution
            // is weighted inversely proportional to its size. This aims to
            // match up with the PCA's perception of 'variance', but not the
            // variances of the output components (where weightings are not used).
            double& proxyvar = variances[r];
            proxyvar = 0;
            for (size_t b = 0; b < nblocks; ++b) {
                double zero_sum = (block_size[b] - block_count[b]) * grand_mean * grand_mean;
                proxyvar += zero_sum / block_size[b];
            }

            for (size_t i = 0; i < num_entries; ++i) {
                double diff = value_ptr[i] - grand_mean;
                auto bsize = block_size[block[index_ptr[i]]];
                if (bsize) {
                    proxyvar += diff * diff / bsize;
                }
            }
        }
    }, NR, nthreads);
}

template<bool equiweight_, typename Block_>
void compute_mean_and_variance_regress( 
    int NR,
    const std::vector<double>& values,
    const std::vector<int>& indices,
    const std::vector<size_t>& ptrs,
    const Block* block, 
    const std::vector<int>& block_size, 
    Eigen::MatrixXd& centers,
    Eigen::VectorXd& variances,
    int nthreads) 
{
    tatami::parallelize([&](size_t, int start, int length) -> void {
        const size_t nblocks = block_size.size();
        for (int r = start, end = start + length; r < end; ++r) {
            auto offset = ptrs[r];
            size_t num_entries = ptrs[r+1] - offset;
            auto value_ptr = values.data() + offset;
            auto index_ptr = indices.data() + offset;

            // Computing the block-wise means.
            auto mbuffer = centers.data() + nblocks * static_cast<size_t>(r);
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
            double& proxyvar = variances[r];
            proxyvar = 0;
            {
                auto block_copy = block_size;
                for (size_t i = 0; i < num_entries; ++i) {
                    Block curb = block[index_ptr[i]];
                    double diff = value_ptr[i] - mbuffer[curb];

                    double squared = diff * diff;
                    if constexpr(equiweight_) {
                        squared /= block_size[curb];
                    }

                    proxyvar += squared;
                    --block_copy[curb];
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    auto extra = mbuffer[b] * mbuffer[b] * block_copy[b];
                    if constexpr(equiweight_) {
                        extra /= block_size[b];
                    }
                    proxyvar += extra;
                }

                // If weighting is involved, we've already scaled it by the number of cells,
                // so there's no need to do this extra step. As before, the concept of the
                // variance is kinda murky when we do all this weighting, but just remember
                // that PCA only looks at the sum of squares from the mean.
                if constexpr(!equiweight_) {
                    proxyvar /= NC - 1;
                }
            }
        }
    }, NR, nthreads);
}

/********************************************
 *** Genewise moments from a dense matrix ***
 ********************************************/

inline void compute_mean_and_variance_simple(
    const Eigen::MatrixXd& emat, 
    Eigen::VectorXd& centers, 
    Eigen::VectorXd& variances, 
    int nthreads)
{
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NR = emat.rows();
        const double* ptr = emat.data() + static_cast<size_t>(start) * NR; // enforce size_t to avoid overflow issues.
        for (size_t c = start, end = start + length; c < end; ++c, ptr += NR) {
            auto results = tatami::stats::variances::compute_direct(ptr, NR);
            centers.coeffRef(c) = results.first;
            variances.coeffRef(c) = results.second;
        }
    }, emat.cols(), nthreads);
}

template<typename Block_>
void compute_mean_and_variance_equiweight(
    const Eigen::MatrixXd& emat,
    const Block_* block,
    const std::vector<int>& block_size,
    Eigen::VectorXd& centers, 
    Eigen::VectorXd& variances, 
    int nthreads)
{
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NC = emat.cols(), NR = emat.rows();
        size_t nblocks = block_size.size();
        std::vector<double> mean_buffer(nblocks);
        auto ptr = emat.data() + static_cast<size_t>(start) * NR;

        for (size_t c = start, end = start + length; c < end; ++c, ptr += NR) {
            std::fill(mean_buffer.begin(), mean_buffer.end(), 0.0);
            for (size_t r = 0; r < NR; ++r) {
                mean_buffer[block[r]] += ptr[r];
            }

            double& grand_mean = center_v[c];
            grand_mean = 0;
            int non_empty_blockes = 0;
            for (size_t b = 0; b < nblocks; ++b) {
                const auto& bsize = block_size[b];
                if (bsize) {
                    grand_mean += mean_buffer[b] / bsize;
                    ++non_empty_blockes;
                }
            }
            grand_mean /= non_empty_blockes;

            // We don't actually compute the blockwise variance, but instead
            // the weighted sum of squared deltas, which is what PCA actually sees.
            double& proxyvar = scale_v[c];
            proxyvar = 0;
            for (size_t r = 0; r < NR; ++r) {
                const auto& bsize = block_size[block[r]];
                if (bsize) {
                    double diff = ptr[r] - grand_mean;
                    proxyvar += diff * diff / bsize;
                }
            }
        }
    }, emat.cols(), nthreads);
}

template<bool equiweight_, typename Block_>
void compute_mean_and_variance_regress(
    const Eigen::MatrixXd& mat,
    const Block_* block,
    const std::vector<int>& block_size,
    Eigen::MatrixXd& centers, 
    Eigen::VectorXd& variances, 
    int nthreads)
{
    tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
        size_t NC = emat.cols(), NR = emat.rows();
        size_t nblocks = block_size.size();
        std::vector<double> mean_buffer(nblocks);
        auto ptr = emat.data() + static_cast<size_t>(start) * NR; 

        for (size_t c = start, end = start + length; c < end; ++c, ptr += NR) {
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
                double delta = ptr[r] - mean_buffer[block[r]]; 
                auto squared = current * current;
                if constexpr(equiweight_) {
                    squared /= block_size[curb];
                }
                proxyvar += squared;
            }

            if constexpr(!equiweight_) {
                proxyvar /= NR - 1;
            }
        }
    }, emat.cols(), nthreads);
}
