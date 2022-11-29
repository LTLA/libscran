#ifndef SCRAN_SIMPLE_DIFF_HPP
#define SCRAN_SIMPLE_DIFF_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

namespace scran {

namespace differential_analysis {

template<typename Stat, typename Ls>
double compute_pairwise_simple_diff(int g1, int g2, const Stat* values, const Ls& level_size, int ngroups, int nblocks) {
    double total_weight = 0;
    double output = 0;

    for (int b = 0; b < nblocks; ++b) {
        int offset1 = g1 * nblocks + b;
        const auto& left = values[offset1];
        const auto& left_size = level_size[offset1];
        if (!left_size) {
            continue;
        }

        int offset2 = g2 * nblocks + b;
        const auto& right = values[offset2]; 
        const auto& right_size = level_size[offset2];
        if (!right_size) {
            continue;
        }

        double weight = left_size * right_size;
        total_weight += weight;
        output += (left - right) * weight;
    }

    if (total_weight) {
        output /= total_weight;
    } else {
        output = std::numeric_limits<double>::quiet_NaN();
    }

    return output;
}

template<typename Stat, typename Ls>
void compute_pairwise_simple_diff(const Stat* values, const Ls& level_size, int ngroups, int nblocks, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            auto d = compute_pairwise_simple_diff(g1, g2, values, level_size, ngroups, nblocks);
            output[g1 * ngroups + g2] = d;
            output[g2 * ngroups + g1] = -d;
        }
    }
}

}

}

#endif
