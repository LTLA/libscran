#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../feature_selection/block_summaries.hpp"
#include "summarize_comparisons.hpp"

#include <vector>
#include <limits>
#include <algorithm>

namespace scran {

namespace differential_analysis {

inline double compute_cohens_d(double m1, double m2, double sd, double threshold) {
    if (std::isnan(sd)) {
        return std::numeric_limits<double>::quiet_NaN();
    } 
    
    double delta = m1 - m2 - threshold;
    if (sd == 0 && delta == 0) {
        return 0;
    } else if (sd == 0) {
        if (delta > 0) {
            return std::numeric_limits<double>::infinity();
        } else {
            return -std::numeric_limits<double>::infinity();
    } else {
        return delta / sd;
    }
}

inline double cohen_denominator(double left_var, double right_var) {
    if (std::isnan(left_var) && std::isnan(right_var)) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (std::isnan(left_var)) {
        return std::sqrt(right_var);
    } else if (std::isnan(right_var)) {
        return std::sqrt(left_var);
    } else {
        return std::sqrt((left_var + right_var)/2);
    }
}

template<typename Stat, typename Level, typename Ls>
void compute_pairwise_cohens_d (const Stat& means, const Stat& vars, const Level& levels, const Ls& level_size, int ngroups, int nblocks, Stat& output, Stat& weightsum, double threshold = 0) {
    std::fill(output, output + ngroups * ngroups, 0);
    std::fill(weightsum, weightsum + ngroups * ngroups, 0);

    for (int b = 0; b < nblocks; ++b) {
        int offset = b * ngroups;

        for (int g1 = 0; g1 < ngroups; ++g1) {
            const auto& left_mean = means[offset + g1]; 
            const auto& left_var = vars[offset + g1];
            const auto& left_size = level_size[offset + g1];
            if (!left_size) {
                continue;
            }

            for (int g2 = 0; g2 < g1; ++g2) {
                const auto& right_mean = means[offset + g2]; 
                const auto& right_var = vars[offset + g2];
                const auto& right_size = level_size[offset + g2];
                if (!right_size) {
                    continue;
                }

                double denom = cohen_denominator(vars[g1], vars[g2]);
                if (std::isnan(denom)) {
                    continue;
                }

                double weight = left_size * right_size;
                weightsum[g1 * ngroups + g2] += weight;

                output[g1 * ngroups + g2] += compute_cohens_d(means[g1], means[g2], denom, threshold) * weight;
                if (threshold != 0) { 
                    output[g2 * ngroups + g1] = compute_cohens_d(means[g2], means[g1], denom, threshold) * weight;
                }
            }
        }
    }

    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            double total_weight = weightsum[g1 * ngroups + g2];
            if (total_weight) {
                output[g1 * ngroups + g2] /= total_weight;
                if (threshold == 0) {
                    output[g2 * ngroups + g1] = -output[g1 * ngroups + g2];
                } else {
                    output[g2 * ngroups + g1] /= total_weight;
                }
            } else {
                output[g1 * ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
                output[g2 * ngroups + g1] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

}

}

#endif
