#ifndef SCRAN_LFC_HPP
#define SCRAN_LFC_HPP

#include <vector>
#include <limits>

namespace scran {

namespace differential_analysis {

template<typename Stat, typename Ls>
void compute_pairwise_lfc (const Stat* means, const Ls& level_size, int ngroups, int nblocks, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            double total_weight = 0;
            double& total_d1 = output[g1 * ngroups + g2];
            total_d1 = 0;

            for (int b = 0; b < nblocks; ++b) {
                int offset1 = g1 * nblocks + b;
                const auto& left_mean = means[offset1];
                const auto& left_size = level_size[offset1];
                if (!left_size) {
                    continue;
                }

                int offset2 = g2 * nblocks + b;
                const auto& right_mean = means[offset2]; 
                const auto& right_size = level_size[offset2];
                if (!right_size) {
                    continue;
                }

                double weight = left_size * right_size;
                total_weight += weight;
                total_d1 += (left_mean - right_mean) * weight;
            }

            double& total_d2 = output[g2 * ngroups + g1];
            if (total_weight) {
                total_d1 /= total_weight;
                total_d2 = -total_d1;
            } else {
                total_d1 = std::numeric_limits<double>::quiet_NaN();
                total_d2 = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

}

}

#endif
