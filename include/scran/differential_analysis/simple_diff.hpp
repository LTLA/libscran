#ifndef SCRAN_SIMPLE_DIFF_HPP
#define SCRAN_SIMPLE_DIFF_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

namespace scran {

/**
 * @cond
 */
namespace differential_analysis {

template<typename Stat_, typename Weights_>
double compute_pairwise_simple_diff(int g1, int g2, const Stat_* values, const Weights_& weights, int ngroups, int nblocks) {
    double total_weight = 0;
    double output = 0;

    for (int b = 0; b < nblocks; ++b) {
        int offset1 = g1 * nblocks + b;
        auto left = values[offset1];
        auto lweight = weights[offset1];
        if (!lweight) {
            continue;
        }

        int offset2 = g2 * nblocks + b;
        auto right = values[offset2]; 
        auto rweight = weights[offset2];
        if (!rweight) {
            continue;
        }

        double weight = lweight * rweight;
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
/**
 * @endcond
 */

}

#endif
