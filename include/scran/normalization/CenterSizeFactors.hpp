#ifndef SCRAN_CENTER_SIZE_FACTORS_HPP
#define SCRAN_CENTER_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include <stdexcept>
#include <vector>
#include <algorithm>

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
     * Validate size factors by checking that they are all non-zero.
     * We also check whether any size factors are zero, which may be handled separately by callers. 
     *
     * @tparam T Floating-point type for the size factors.
     *
     * @param n Number of size factors.
     * @param[in] size_factors Pointer to an array of non-negative size factors of length `n`.
     *
     * @return Whether a size factor of zero was detected.
     * An error is raised if a negative size factor is detected.
     */
    template<typename T>
    static bool validate(size_t n, const T* size_factors) {
        bool is_zero = false;
        for (size_t i = 0; i < n; ++i) {
             if (size_factors[i] < 0) {
                throw std::runtime_error("negative size factors detected");
            } else if (!is_zero && size_factors[i] == 0) {
                is_zero = true;
            }
        }
        return is_zero;
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
     * @tparam T Floating-point type for the size factors.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of positive size factors of length `n`.
     *
     * @return Entries in `size_factors` are scaled so that their mean is equal to 1.
     * A boolean is returned indicating whether size factors of zero were detected.
     */
    template<typename T>
    bool run(size_t n, T* size_factors) {
        bool has_zero = false;

        if (n) { // avoid division by zero
            has_zero = validate(n, size_factors);
            double mean = std::accumulate(size_factors, size_factors + n, static_cast<double>(0)) / n;

            if (mean) { // avoid division by zero
                for (size_t i = 0; i < n; ++i){
                    size_factors[i] /= mean;
                }
            }
        }

        return has_zero;
    }

    /**
     * @tparam T Floating-point type for the size factors.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of positive size factors of length 1n1.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `n`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Entries in `size_factors` are scaled according to the mode defined by `set_block_mode()`.
     * A boolean is returned indicating whether size factors of zero were detected.
     */
    template<typename T, typename B>
    bool run_blocked(size_t n, T* size_factors, const B* block) {
        if (block == NULL) {
            return run(n, size_factors);
        } 
        
        bool has_zero = false;
        if (n) {
            has_zero = validate(n, size_factors);

            size_t ngroups = *std::max_element(block, block + n) + 1;
            std::vector<double> group_mean(ngroups);
            std::vector<double> group_num(ngroups);

            for (size_t i = 0; i < n; ++i) {
                group_mean[block[i]] += size_factors[i];
                ++group_num[block[i]];
            }
            for (size_t g = 0; g < ngroups; ++g) {
                group_mean[g] /= group_num[g];
            }

            if (block_mode == PER_BLOCK) {
                for (size_t i = 0; i < n; ++i) {
                    size_factors[i] /= group_mean[block[i]];
                }
            } else if (block_mode == LOWEST) {
                double min = *std::min_element(group_mean.begin(), group_mean.end());
                for (size_t i = 0; i < n; ++i) {
                    size_factors[i] /= min;
                }
            }
        }

        return has_zero;
    }
};

}

#endif
