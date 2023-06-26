#ifndef SCRAN_BLOCKED_VARIANCES_HPP
#define SCRAN_BLOCKED_VARIANCES_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <limits>

#include "tatami/tatami.hpp"

namespace scran {

namespace feature_selection {

template<bool blocked_, class Block_>
Block_ get_block(size_t j, const Block_* block) {
    if constexpr(blocked_) {
        return block[j];
    } else {
        return 0;
    }
}

template<typename BlockSize_, typename Stat_>
void finish_means(size_t nblocks, const BlockSize_* block_size, Stat_* tmp_means) {
    for (size_t b = 0; b < nblocks; ++b) {
        if (block_size[b]) {
            tmp_means[b] /= block_size[b];
        } else {
            tmp_means[b] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

template<typename BlockSize_, typename Stat_>
void finish_variances(size_t nblocks, const BlockSize_* block_size, Stat_* tmp_vars) {
    for (size_t b = 0; b < nblocks; ++b) {
        if (block_size[b] > 1) {
            tmp_vars[b] /= block_size[b] - 1;
        } else {
            tmp_vars[b] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

template<bool blocked_, typename Data_, typename Index_, typename Block_, typename BlockSize_, typename Stat_>
void blocked_variance_with_mean(const Data_* ptr, Index_ len, const Block_* block, size_t nblocks, const BlockSize_* block_size, Stat_* tmp_means, Stat_* tmp_vars) {
    std::fill(tmp_means, tmp_means + nblocks, 0);
    for (size_t j = 0; j < len; ++j) {
        auto b = get_block<blocked_>(j, block);
        tmp_means[b] += ptr[j];
    }
    finish_means(nblocks, block_size, tmp_means);

    std::fill(tmp_vars, tmp_vars + nblocks, 0);
    for (size_t j = 0; j < len; ++j) {
        auto b = get_block<blocked_>(j, block);
        tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
    }
    finish_variances(nblocks, block_size, tmp_vars);
}

template<bool blocked_, typename Data_, typename Index_, typename Block_, typename BlockSize_, typename Stat_, typename Count_>
void blocked_variance_with_mean(const tatami::SparseRange<Data_, Index_>& range, const Block_* block, size_t nblocks, const BlockSize_* block_size, Stat_* tmp_means, Stat_* tmp_vars, Count_* tmp_nzero) {
    std::fill(tmp_means, tmp_means + nblocks, 0);
    std::fill(tmp_nzero, tmp_nzero + nblocks, 0);
    for (size_t j = 0; j < range.number; ++j) {
        if (range.value[j]) { // ensure correct calculation of tmp_nzero if there are zeros in the values.
            auto b = get_block<blocked_>(range.index[j], block);
            tmp_means[b] += range.value[j];
            ++tmp_nzero[b];
        }
    }
    finish_means(nblocks, block_size, tmp_means);

    std::fill(tmp_vars, tmp_vars + nblocks, 0);
    for (size_t j = 0; j < range.number; ++j) {
        auto b = get_block<blocked_>(range.index[j], block);
        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
    }
    for (size_t b = 0; b < nblocks; ++b) {
        tmp_vars[b] += tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
    }
    finish_variances(nblocks, block_size, tmp_vars);
}

}

}

#endif
