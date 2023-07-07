#ifndef SCRAN_COHEN_D_HPP
#define SCRAN_COHEN_D_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <type_traits>

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

template<typename Stat, typename Weights_, class Output>
void compute_pairwise_cohens_d_internal(int g1, int g2, const Stat* means, const Stat* vars, const Weights_& weights, int ngroups, int nblocks, double threshold, Output& output) {
    double total_weight = 0;
    constexpr bool do_both_sides = !std::is_same<Stat, Output>::value;

    for (int b = 0; b < nblocks; ++b) {
        int offset1 = g1 * nblocks + b;
        const auto& left_mean = means[offset1];
        const auto& left_var = vars[offset1];
        const auto& left_weight = weights[offset1];
        if (!left_weight) {
            continue;
        }

        int offset2 = g2 * nblocks + b;
        const auto& right_mean = means[offset2]; 
        const auto& right_var = vars[offset2];
        const auto& right_weight = weights[offset2];
        if (!right_weight) {
            continue;
        }

        double denom = cohen_denominator(left_var, right_var);
        if (std::isnan(denom)) {
            continue;
        }

        double weight = left_weight * right_weight;
        total_weight += weight;

        double extra = compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;
        if constexpr(do_both_sides) {
            output.first += extra;
            if (threshold) {
                output.second += compute_cohens_d(right_mean, left_mean, denom, threshold) * weight;
            }
        } else {
            output += extra;
        }
    }

    if constexpr(do_both_sides) {
        if (total_weight) {
            output.first /= total_weight;
            if (threshold) {
                output.second /= total_weight;
            } else {
                output.second = -output.first;
            }
        } else {
            output.first = std::numeric_limits<double>::quiet_NaN();
            output.second = std::numeric_limits<double>::quiet_NaN();
        }
    } else {
        if (total_weight) {
            output /= total_weight;
        } else {
            output = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

template<bool both, typename Stat, typename Weights_>
auto compute_pairwise_cohens_d(int g1, int g2, const Stat* means, const Stat* vars, const Weights_& weights, int ngroups, int nblocks, double threshold) {
    if constexpr(!both) {
        Stat output = 0;
        compute_pairwise_cohens_d_internal(g1, g2, means, vars, weights, ngroups, nblocks, threshold, output);
        return output;
    } else {
        std::pair<Stat, Stat> output(0, 0);
        compute_pairwise_cohens_d_internal(g1, g2, means, vars, weights, ngroups, nblocks, threshold, output);
        return output;
    }
}

template<typename Stat, typename Weights_>
void compute_pairwise_cohens_d (const Stat* means, const Stat* vars, const Weights_& weights, int ngroups, int nblocks, double threshold, Stat* output) {
    for (int g1 = 0; g1 < ngroups; ++g1) {
        for (int g2 = 0; g2 < g1; ++g2) {
            auto tmp = compute_pairwise_cohens_d<true>(g1, g2, means, vars, weights, ngroups, nblocks, threshold);
            output[g1 * ngroups + g2] = tmp.first;
            output[g2 * ngroups + g1] = tmp.second;
        }
    }
}

}

}

#endif
