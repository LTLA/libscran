#ifndef SCRAN_BLOCK_INDICES_H
#define SCRAN_BLOCK_INDICES_H

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace scran {

typedef std::vector<std::vector<std::size_t> > BlockIndices;

template<class SIT>
BlockIndices block_indices(size_t n, SIT p) {
    if (p) {
        int ngroups = (n ? *std::max_element(p, p + n) + 1 : 0);

        BlockIndices by_group(ngroups);
        for (auto& g : by_group) { 
            g.clear();
        }

        for (size_t i = 0; i < n; ++i, ++p) {
            by_group[*p].push_back(i);
        }

        return by_group;
    } else {
        return BlockIndices();
    }
}

template<typename Block>
std::vector<int> block_sizes(size_t n, const Block* block) {
    const Block nblocks = (n ? *std::max_element(block, block + n) + 1 : 1);

    std::vector<int> block_size(nblocks);
    for (size_t j = 0; j < n; ++j) {
        ++block_size[block[j]];
    }

    for (auto b : block_size) {
        if (b == 0) {
            throw std::runtime_error("block IDs must be 0-based and consecutive with no empty blocks");
        }
    }

    return block_size;
}

}

#endif
