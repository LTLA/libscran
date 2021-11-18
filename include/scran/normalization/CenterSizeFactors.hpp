#ifndef SCRAN_CENTER_SIZE_FACTORS_HPP
#define SCRAN_CENTER_SIZE_FACTORS_HPP

#include <stdexcept>
#include "../utils/block_indices.hpp"

/**
 * @file CenterSizeFactors.hpp
 * @brief Center size factors prior to normalization.
 */

namespace scran {

/**
 * @brief Center size factors prior to scaling normalization.
 *
 * The aim of centering is to ensure that the normalized expression values are on roughly the same scale as the original counts.
 * This simplifies interpretation and ensures that any added pseudo-count prior to log-transformation has a predictable shrinkage effect.
 * The functionality in this class is used automatically by `LogNormCounts` but can be called separately if desired.
 */
class CenterSizeFactors {
public:
    /**
     * Strategy for handling blocks, see `set_block_mode()` for details.
     */
    enum BlockMode { PER_BLOCK, LOWEST };

    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_block_mode()` for details.
         */
        static constexpr BlockMode block_mode = LOWEST;
    };

private:
    BlockMode block_mode = Defaults::block_mode;

public:
    /**
     * Validate size factors by checking that they are all positive.
     *
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     *
     * @param[in] size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     *
     * @return An error is raised if a negative or zero size factor is detected.
     */
    template<class V>
    static void validate(const V& size_factors) {
        for (auto x : size_factors) {
            if (x <= 0) {
                throw std::runtime_error("non-positive size factors detected");
            }
        }
    }

public:
    /**
     * @param b Strategy for handling blocks in `run_blocked()`.
     *
     * @return A reference to this `CenterSizeFactors` object.
     *
     * With the `PER_BLOCK` strategy, size factors are scaled separately for each block so that they have a mean of 1 within each block.
     * The scaled size factors are identical to those obtained by separate invocations of `CenterSizeFactors::run()` on the size factors for each block.
     * This can be desirable to ensure consistency with independent analyses of each block - otherwise, the centering would depend on the size factors across all blocks.
     * 
     * With the `LOWEST` strategy, we compute the mean size factor for each block and we divide all size factors by the minimum mean.
     * In effect, we downscale all blocks to match the coverage of the lowest-coverage block.
     * This is useful for datasets with highly heterogeneous coverage of different blocks as it avoids egregious upscaling of low-coverage blocks.
     * (By contrast, downscaling is always safe as it simply discards information across all blocks by shrinking log-fold changes towards zero at low expression.) 
     */
    CenterSizeFactors& set_block_mode(BlockMode b = Defaults::block_mode) {
        block_mode = b;
        return *this;
    }

public:
    /**
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     *
     * @param[in,out] size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     *
     * @return Entries in `size_factors` are scaled so that their mean is equal to 1.
     */
    template<class V>
    void run(V& size_factors) {
        if (size_factors.size()) { // avoid division by zero
            validate(size_factors);
            double mean = std::accumulate(size_factors.begin(), size_factors.end(), static_cast<double>(0)) / size_factors.size();

            if (mean) { // avoid division by zero
                for (auto& x : size_factors) {
                    x /= mean;
                }
            }
        }
    }

    /**
     * @tparam V A vector class supporting `size()`, random access via `[`, `begin()`, `end()` and `data()`.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param size_factors A vector of positive size factors, of length equal to the number of columns in `mat`.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to the length of `size_factors`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Entries in `size_factors` are scaled according to the mode defined by `set_block_mode()`.
     */
    template<class V, typename B>
    void run_blocked(V& size_factors, const B* block) {
        if (block == NULL) {
            run(size_factors);
            return;
        }

        validate(size_factors);
        auto by_group = block_indices(size_factors.size(), block);
        std::vector<double> group_mean(by_group.size());

        for (size_t g = 0; g < by_group.size(); ++g) {
            const auto& current = by_group[g];
            if (current.size()) {
                double& mean = group_mean[g];
                for (auto i : current) {
                    mean += size_factors[i];
                }
                mean /= current.size();
            }
        }

        if (block_mode == PER_BLOCK) {
            for (size_t g = 0; g < by_group.size(); ++g) {
                const auto& mean = group_mean[g];
                if (mean > 0) {
                    const auto& current = by_group[g];
                    for (auto i : current) {
                        size_factors[i] /= mean;
                    }
                }
            }
        } else if (block_mode == LOWEST) {
            if (group_mean.size()) {
                double min = *std::min_element(group_mean.begin(), group_mean.end());
                if (min) {
                    for (auto& x : size_factors) {
                        x /= min;
                    }
                }
            }
        }
    }
};

}

#endif
