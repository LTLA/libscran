#ifndef SCRAN_UTILS_BLOCKING_HPP
#define SCRAN_UTILS_BLOCKING_HPP

#include "macros.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>

/**
 * @file blocking.hpp
 * @brief Utilities for handling blocks of cells.
 */

namespace scran {

/**
 * Count the number of unique 0-based IDs, e.g., for block or group assignments.
 * All IDs are assumed to be integers in `[0, x)` where `x` is the return value of this function.
 *
 * @tparam Id_ Integer type for the IDs.
 *
 * @param length Length of the array in `ids`.
 * @param[in] ids Pointer to an array containing 0-based IDs of some kind.
 *
 * @return Number of IDs, or 0 if `length = 0`.
 */
template<typename Id_>
size_t count_ids(size_t length, const Id_* value) {
    if (!length) {
        return 0;
    } else {
        return static_cast<size_t>(*std::max_element(value, value + length)) + 1;
    }
}

/**
 * Count the frequency of 0-based IDs, e.g., for block or group assignments.
 * All IDs are assumed to be integers in `[0, x)` where `x` is the number of unique IDs.
 *
 * @tparam Output_ Numeric type for the output frequencies.
 * @tparam Id_ Integer type for the IDs.
 *
 * @param length Length of the array in `ids`.
 * @param[in] ids Pointer to an array containing 0-based IDs of some kind.
 * @param allow_zeros Whether to throw an error if frequencies of zero are detected.
 *
 * @return Vector of length equal to the number of IDs, containing the frequency of each ID.
 * An error is raised if an ID has zero frequency and `allow_zeros = false`.
 */
template<typename Output_ = int, bool allow_zeros_ = false, typename Id_> 
std::vector<Output_> tabulate_ids(size_t num, const Id_* ids, bool allow_zeros = false) {
    size_t nidss = count_ids(num, ids);

    std::vector<Output_> ids_size(nidss);
    for (size_t j = 0; j < num; ++j) {
        ++ids_size[ids[j]];
    }

    if (!allow_zeros) {
        for (auto b : ids_size) {
            if (b == 0) {
                throw std::runtime_error("IDs must be 0-based and consecutive with no empty blocks");
            }
        }
    }

    return ids_size;
}

/**
 * Policy to use for weighting blocks based on their size, i.e., the number of cells in each block.
 * This controls the calculation of weighted averages across blocks.
 *
 * - `NONE`: no weighting is performed.
 *   Larger blocks will contribute more to the weighted average. 
 * - `EQUAL`: each block receives equal weight, regardless of its size.
 *   Equivalent to averaging across blocks without weights.
 * - `VARIABLE`: each batch is weighted using the logic in `variable_block_weight()`.
 *   This penalizes small blocks with unreliable statistics while equally weighting all large blocks.
 */
enum class WeightPolicy : char { NONE, VARIABLE, EQUAL };

/**
 * @brief Parameters for `variable_block_weight()`.
 */
struct VariableBlockWeightParameters {
    /**
     * @param l Lower bound for the block weight calculation.
     * @param u Upper bound for the block weight calculation.
     * This should be greater than `l`.
     */
    constexpr VariableBlockWeightParameters(double l = 0, double u = 1000) : upper_bound(u), lower_bound(l) {}

    /**
     * Lower bound for the block weight calculation.
     */
    double lower_bound;

    /**
     * Upper bound for the block weight calculation.
     */
    double upper_bound;
};

/**
 * Weight each block of cells for use in computing a weighted average across blocks.
 * The weight for each block is calcualted from the size of that block.
 *
 * - If the block is smaller than some lower bound, it has zero weight.
 * - If the block is greater than some upper bound, it has weight of 1.
 * - Otherwise, the block has weight proportional to its size, increasing linearly from 0 to 1 between the two bounds.
 *
 * Blocks that are "large enough" are considered to be equally trustworthy and receive the same weight, ensuring that each block contributes equally to the weighted average.
 * By comparison, very small blocks receive lower weight as their statistics are generally less stable.
 *
 * @param s Size of the block, in terms of the number of cells in that block.
 * @param params Parameters for the weight calculation, consisting of the lower and upper bounds.
 *
 * @return Weight of the block, to use for computing a weighted average across blocks. 
 */
inline double variable_block_weight(double s, const VariableBlockWeightParameters& params) {
    if (s < params.lower_bound) {
        return 0;
    }

    if (s > params.upper_bound) {
        return 1;
    }

    return (s - params.lower_bound) / (params.upper_bound - params.lower_bound);
}

}

#endif
