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
double compute_pairwise_cohens_d(int g1, int g2, const Stat* means, const Stat* vars, const Ls& level_size, int ngroups, int nblocks, double threshold) {
    double total_weight = 0;
    double output = 0;

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
        output += compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;
    }

    if (total_weight) {
        output /= total_weight;
    } else {
        output = std::numeric_limits<double>::quiet_NaN();
    }

    return output;
}

template<typename Stat, typename Ls>
void compute_pairwise_cohens_d(int g1, const Stat* means, const Stat* vars, const Ls& level_size, int ngroups, int nblocks, double threshold, Stat* output) {
    for (int g2 = 0; g2 < ngroups; ++g2) {
        if (g1 == g2) {
            continue;
        }
        output[g2] = compute_pairwise_cohens_d(g1, g2, means, vars, level_size, ngroups, nblocks, threshold);
    }
}

template<typename Stat, typename Ls>
void compute_pairwise_cohens_d (const Stat* means, const Stat* vars, const Ls& level_size, int ngroups, int nblocks, double threshold, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < ngroups; ++g2) {
            if (g1 == g2) {
                continue;
            }
            if (threshold == 0 && g2 < g1) {
                output[g1 * ngroups + g2] = -output[g2 * ngroups + g1];
            } else {
                output[g1 * ngroups + g2] = compute_pairwise_cohens_d(g1, g2, means, vars, level_size, ngroups, nblocks, threshold);
            }
        }
    }
}

}

}

#endif
