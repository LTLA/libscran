#ifndef SCRAN_CENTER_SIZE_FACTORS_HPP
#define SCRAN_CENTER_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include "../utils/blocking.hpp"

#include "SanitizeSizeFactors.hpp"
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

        /**
         * Set `set_ignore_zeros()` for more details.
         */
        static constexpr bool ignore_zeros = true;
    };

private:
    BlockMode block_mode = Defaults::block_mode;

    bool ignore_zeros = Defaults::ignore_zeros;

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

    /**
     * @param i Whether to ignore zeros when computing the mean size factor.
     *
     * @return A reference to this `CenterSizeFactors` object.
     *
     * While size factors of zero are generally invalid, they may occur in datasets that have not been properly filtered to remove low-quality cells.
     * In such cases, we may wish to ignore size factors of zero so as to avoid a spurious deflation of the mean during centering.
     * This is useful if some filtering is to be applied after normalization - by ignoring zeros now, we ensure that we get the same result as if we had removed those zeros prior to centering. 
     *
     * Note that non-finite size factors (e.g., Inf, NaN) are always ignored when computing the mean.
     */
    CenterSizeFactors& set_ignore_zeros(bool i = Defaults::ignore_zeros) {
        ignore_zeros = i;
        return *this;
    }

public:
    /**
     * @tparam T Floating-point type for the size factors.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of size factors of length `n`.
     * On output, entries are scaled so that their mean is equal to 1.
     * Note that this only considers the mean across finite (and, if `set_ignore_zeros()` is `true`, positive) entries.
     * If there are no non-zero finite size factors, no centering is performed.
     *
     * @return Object indicating whether invalid size factors (zero or non-finite) were detected.
     */
    template<typename T>
    SizeFactorValidity run(size_t n, T* size_factors) const {
        size_t num_used = 0;
        SizeFactorValidity output;

        double mean = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& current = size_factors[i];
            if (current < 0) {
                output.has_negative = true;
                continue;
            } else if (current == 0) {
                output.has_zero = true;
                if (ignore_zeros) {
                    continue;
                }
            } else if (std::isnan(current)) {
                output.has_nan = true;
                continue;
            } else if (std::isinf(current)) {
                output.has_infinite = true;
                continue;
            }

            ++num_used;
            mean += current;
        }

        if (mean) {
            mean /= num_used;
            for (size_t i = 0; i < n; ++i){
                size_factors[i] /= mean;
            }
        }

        return output;
    }

    /**
     * @tparam T Floating-point type for the size factors.
     * @tparam B An integer type, to hold the block IDs.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of positive size factors of length 1n1.
     * On output, entries are scaled so that their mean is equal to 1 according to the strategy defined in `set_block_mode()`.
     * This only considers the mean across finite (and, if `set_ignore_zeros()` is `true`, positive) entries within each block.
     * If there are no non-zero finite size factors in a block, no centering is performed for that block.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `n`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * This can also be a `NULL`, in which case all cells are assumed to belong to the same block.
     *
     * @return Object indicating whether invalid size factors (zero or non-finite) were detected.
     */
    template<typename T, typename B>
    SizeFactorValidity run_blocked(size_t n, T* size_factors, const B* block) const {
        if (block == NULL) {
            return run(n, size_factors);
        } 

        size_t ngroups = count_ids(n, block);
        std::vector<double> group_mean(ngroups);
        std::vector<double> group_num(ngroups);

        SizeFactorValidity output;
        for (size_t i = 0; i < n; ++i) {
            const auto& current = size_factors[i];
            if (current < 0) {
                output.has_negative = true;
                continue;
            } else if (current == 0) {
                output.has_zero = true;
                if (ignore_zeros) {
                    continue;
                }
            } else if (std::isinf(current)) {
                output.has_infinite = true;
                continue;
            } else if (std::isnan(current)) {
                output.has_nan = true;
                continue;
            }

            const auto& b = block[i];
            group_mean[b] += size_factors[i];
            ++group_num[b];
        }

        for (size_t g = 0; g < ngroups; ++g) {
            if (group_num[g]) {
                group_mean[g] /= group_num[g];
            }
        }

        if (block_mode == PER_BLOCK) {
            for (size_t i = 0; i < n; ++i) {
                const auto& div = group_mean[block[i]];
                if (div) {
                    size_factors[i] /= div;
                }
            }

        } else if (block_mode == LOWEST) {
            // Ignore groups with means of zeros, either because they're full of zeros themselves
            // or they have no cells associated with them.
            double min = std::numeric_limits<double>::infinity();
            for (size_t b = 0; b < ngroups; ++b) {
                const auto& div = group_mean[b];
                if (div && div < min) {
                    min = div;
                }
            }

            if (std::isfinite(min)) {
                for (size_t i = 0; i < n; ++i) {
                    size_factors[i] /= min;
                }
            }
        }

        return output;
    }
};

}

#endif
