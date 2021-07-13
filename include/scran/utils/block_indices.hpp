#ifndef SCRAN_BLOCK_INDICES_H
#define SCRAN_BLOCK_INDICES_H

#include <vector>
#include <algorithm>

namespace scran {

typedef std::vector<std::vector<std::size_t> > BlockIndices;

template<class SIT>
inline void block_indices(size_t n, SIT p, BlockIndices& by_group) {
    int ngroups = (n ? *std::max_element(p, p + n) + 1 : 0);

    by_group.resize(ngroups);
    for (auto& g : by_group) { 
        g.clear();
    }

    for (size_t i = 0; i < n; ++i, ++p) {
        by_group[*p].push_back(i);
    }
    return;
}

template<class SIT>
BlockIndices block_indices(size_t n, SIT p) {
    BlockIndices output;
    if constexpr(!std::is_same<SIT, std::nullptr_t>::value) {
        block_indices(n, p, output);
    }
    return output;
}

}

#endif
