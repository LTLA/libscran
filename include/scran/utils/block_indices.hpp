#ifndef SCRAN_BLOCK_INDICES_H
#define SCRAN_BLOCK_INDICES_H

#include <vector>
#include <algorithm>

namespace scran {

template<class SIT>
inline void block_indices(size_t n, SIT p, std::vector<std::vector<size_t> >& by_group) {
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

}

#endif
