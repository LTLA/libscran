#ifndef SCRAN_BLOCK_INDICES_H
#define SCRAN_BLOCK_INDICES_H

#include <vector>
#include <algorithm>

namespace scran {

template<class SIT>
inline std::vector<std::vector<size_t> > block_indices(size_t n, SIT p) {
    int ngroups = (n ? *std::max_element(p, p + n) + 1 : 1);
    std::vector<std::vector<size_t> > by_group(ngroups);
    for (size_t i = 0; i < n; ++i, ++p) {
        by_group[*p].push_back(i);
    }
    return by_group;
}

}

#endif
