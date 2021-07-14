#ifndef SCRAN_BLOCK_INDICES_H
#define SCRAN_BLOCK_INDICES_H

#include <vector>
#include <algorithm>

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

}

#endif
