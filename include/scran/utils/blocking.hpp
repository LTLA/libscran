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
 * Compute the number of cells in each block.
 *
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Block_ Integer type for the block IDs.
 *
 * @param num Number of cells.
 * @param[in] block Pointer to an array of length equal to the number of cells.
 * This should contain a 0-based block assignment for each cell
 * (i.e., for `B` block, batch identities should run from 0 to `B-1` with at least one entry for each block.)
 *
 * @return Number of unique blocks in `block`.
 */
template<typename Index_, typename Block_>
Index_ count_block_levels(Index_ num, const Block_* block) {
    if (!num) {
        return 0;
    } else {
        return static_cast<Index_>(*std::max_element(block, block + num)) + 1;
    }
}

/**
 * Compute the number of cells in each block.
 *
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Block_ Integer type for the block IDs.
 *
 * @param num Number of cells.
 * @param[in] block Pointer to an array of length equal to the number of cells.
 * This should contain a 0-based block assignment for each cell
 * (i.e., for `B` block, batch identities should run from 0 to `B-1` with at least one entry for each block.)
 *
 * @return Vector of length equal to the number of blocks, containing the number of cells in each block.
 * An error is raised if an empty block is detected.
 */
template<typename Index_, typename Block_>
std::vector<Index_> count_blocks(Index_ num, const Block_* block) {
    Index_ nblocks = count_block_levels(num, block);

    std::vector<Index_> block_size(nblocks);
    for (Index_ j = 0; j < num; ++j) {
        ++block_size[block[j]];
    }

    for (auto b : block_size) {
        if (b == 0) {
            throw std::runtime_error("block IDs must be 0-based and consecutive with no empty blocks");
        }
    }

    return block_size;
}

/**
 * Weight each block of cells when averaging statistics across blocks.
 * Each weight is defined as `min(1, block_size / cap)`.
 * Blocks that are "large enough" (i.e., `block_size >= cap`) are considered to be equally trustworthy and receive the same weight,
 * ensuring that each block contributes equally to the weighted average.
 * By comparison, very small blocks receive lower weight as their statistics are generally less stable.
 * If both `cap` and `block_size` are zero, the weight is also set to zero.
 *
 * @tparam Size_ Numeric type for the sizes.
 *
 * @param block_size Size of the block, in terms of the number of cells in that block.
 * @param cap Minimum number of cells for a block to be considered "large enough".
 *
 * @return Weight of the block, to use for computing a weighted average across blocks. 
 */
template<class Size_>
double weight_block(Size_ block_size, Size_ cap) {
    if (block_size >= cap) {
        return (block_size > 0);
    } else if (cap) {
        return static_cast<double>(block_size)/static_cast<double>(cap);
    } else {
        return 0;
    }
};

}

#endif
