#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

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
        }
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

template<typename Stat, typename Ls>
void compute_pairwise_cohens_d (const Stat* means, const Stat* vars, const Ls& level_size, int ngroups, int nblocks, Stat* output, double threshold = 0) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            double total_weight = 0;
            double& total_d1 = output[g1 * ngroups + g2];
            total_d1 = 0;
            double& total_d2 = output[g2 * ngroups + g1];
            total_d2 = 0;

            for (int b = 0; b < nblocks; ++b) {
                int offset1 = g1 * nblocks + b;
                const auto& left_mean = means[offset1];
                const auto& left_var = vars[offset1];
                const auto& left_size = level_size[offset1];
                if (!left_size) {
                    continue;
                }

                int offset2 = g2 * nblocks + b;
                const auto& right_mean = means[offset2]; 
                const auto& right_var = vars[offset2];
                const auto& right_size = level_size[offset2];
                if (!right_size) {
                    continue;
                }

                double denom = cohen_denominator(left_var, right_var);
                if (std::isnan(denom)) {
                    continue;
                }

                double weight = left_size * right_size;
                total_weight += weight;

                total_d1 += compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;
                if (threshold != 0) { 
                    total_d2 += compute_cohens_d(right_mean, left_mean, denom, threshold) * weight;
                }
            }

            if (total_weight) {
                total_d1 /= total_weight;
                if (threshold == 0) {
                    total_d2 = -total_d1;
                } else {
                    total_d2 /= total_weight;
                }
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
