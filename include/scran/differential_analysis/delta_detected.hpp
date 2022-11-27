#ifndef SCRAN_DELTA_DETECTED_HPP
#define SCRAN_DELTA_DETECTED_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>

namespace scran {

namespace differential_analysis {

template<typename Count, typename Ls>
double compute_pairwise_delta_detected (const Count* detected, const Ls& level_size, int ngroups, int nblocks) {
    double total_weight = 0;
    double output = 0;

    for (int b = 0; b < nblocks; ++b) {
        int offset1 = g1 * nblocks + b;
        const auto& left_detected = detected[offset1];
        const auto& left_size = level_size[offset1];
        if (!left_size) {
            continue;
        }

        int offset2 = g2 * nblocks + b;
        const auto& right_detected = detected[offset2]; 
        const auto& right_size = level_size[offset2];
        if (!right_size) {
            continue;
        }

        double weight = left_size * right_size;
        total_weight += weight;
        output += (static_cast<double>(left_detected)/left_size - static_cast<double>(right_detected)/right_size) * weight;
    }

    if (total_weight) {
        output /= total_weight;
    } else {
        output = std::numeric_limits<double>::quiet_NaN();
    }

    return output;
}

template<typename Count, typename Stat, typename Ls>
void compute_pairwise_delta_detected (const Count* detected, const Ls& level_size, int ngroups, int nblocks, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            if (g1 == g2) {
                continue;
            }
            if (g2 < g1) {
                output[g1 * ngroups + g2] = -output[g2 * ngroups + g1];
            } else {
                output[g1 * ngroups + g2] = compute_pairwise_delta_detected(g1, g2, detected, level_size, ngroups, nblocks);
            }
        }
    }
}

}

}

#endif
