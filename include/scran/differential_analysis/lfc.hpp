#ifndef SCRAN_LFC_HPP
#define SCRAN_LFC_HPP

#include "../utils/macros.hpp"
#include "simple_diff.hpp"

#include <vector>
#include <limits>

namespace scran {

namespace differential_analysis {

template<typename Stat, typename Ls>
void compute_pairwise_lfc (const Stat* means, const Ls& level_size, int ngroups, int nblocks, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            auto d = compute_pairwise_simple_diff(g1, g2, means, level_size, ngroups, nblocks);
            output[g1 * ngroups + g2] = d;
            output[g2 * ngroups + g1] = -d;
        }
    }
}

}

}

#endif
