#ifndef AVERAGE_VECTORS_HPP
#define AVERAGE_VECTORS_HPP

#include "macros.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <numeric>

/**
 * @file average_vectors.hpp
 *
 * @brief Average parallel elements across vectors.
 */

namespace scran {

/**
 * @cond
 */
template<bool check_nan_ = true, bool weighted_, typename Stat_, typename Weight_, typename Output_>
void average_vectors_internal(size_t n, std::vector<Stat_*> in, const Weight_* w, Output_* out) {
    if (in.empty()) {
        std::fill(out, out + n, std::numeric_limits<Output_>::quiet_NaN());
        return;
    } else if (in.size() == 1) {
        if constexpr(weighted_) {
            if (w[0] == 0) {
                std::fill(out, out + n, std::numeric_limits<Output_>::quiet_NaN());
                return;
            }
        } 
        std::copy(in[0], in[0] + n, out);
        return;
    }

    std::fill(out, out + n, 0);
    typename std::conditional<check_nan_, std::vector<Weight_>, size_t>::type accumulated(n);

    auto wcopy = w;
    for (auto current : in) {
        auto copy = out;

        Weight_ weight = 0;
        if constexpr(weighted_) {
            weight = *(wcopy++);
            if (weight == 0) {
                continue;
            }
        }

        if constexpr(weighted_) {
            if (weight != 1) {
                for (size_t i = 0; i < n; ++i, ++current, ++copy) {
                    auto x = *current * weight;
                    if constexpr(!check_nan_) {
                        copy += x;
                    } else if (!std::isnan(x)) {
                        *copy += x; 
                        accumulated[i] += weight;
                    }
                }
                continue;
            }
        }

        // Avoid an extra multiplication if unweighted OR the weight = 1.
        for (size_t i = 0; i < n; ++i, ++current, ++copy) {
            auto x = *current;
            if constexpr(!check_nan_) {
                copy += x;
            } else if (!std::isnan(x)) {
                *copy += x; 
                ++accumulated[i];
            }
        }
    }

    if constexpr(check_nan_) {
        for (size_t i = 0; i < n; ++i, ++out) {
            *out /= accumulated[i];
        }
    } else {
        double denom = 1;
        if constexpr(weighted_) {
            denom /= std::accumulate(w, w + in.size(), 0.0);
        } else {
            denom /= in.size();
        }
        for (size_t i = 0; i < n; ++i, ++out) {
            *out *= denom;
        }
    }
}
/**
 * @endcond
 */

/**
 * Average parallel elements across multiple arrays.
 *
 * @tparam check_nan_ Whether to check for NaNs.
 * If `true`, NaNs are ignored in the average calculations for each element, at the cost of some efficiency.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Output_ Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[out] out Pointer to an output array of length `n`.
 * On completion, `out` is filled with the average of all arrays in `in`.
 * Specifically, each element of `out` is set to the average of the corresponding elements across all `in` arrays.
 */
template<bool check_nan_ = true, typename Stat_, typename Output_>
void average_vectors(size_t n, std::vector<Stat_*> in, Output_* out) {
    average_vectors_internal<check_nan_, false>(n, std::move(in), static_cast<int*>(NULL), out);
    return;
}

/**
 * Average parallel elements across multiple arrays.
 *
 * @tparam check_nan_ Whether to check for NaNs, see `average_vectors()`.
 * @tparam Output Floating-point output type.
 * @tparam Stat Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 *
 * @return A vector of length `n` is returned, containing the average of all arrays in `in`.
 */
template<bool check_nan_ = true, typename Output_ = double, typename Stat_>
std::vector<Output_> average_vectors(size_t n, std::vector<Stat_*> in) {
    std::vector<Output_> out(n);
    average_vectors<check_nan_>(n, std::move(in), out.data());
    return out;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
 *
 * @tparam check_nan_ Whether to check for NaNs.
 * If `true`, NaNs are ignored in the average calculations for each element, at the cost of some efficiency.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Weight_ Type of the weight, typically floating point.
 * @tparam Output_ Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * Weights should be non-negative and finite.
 * @param[out] out Pointer to an output array of length `n`.
 * On output, `out` is filled with the weighted average of all arrays in `in`.
 * Specifically, each element of `out` is set to the weighted average of the corresponding elements across all `in` arrays.
 */
template<bool check_nan_ = true, typename Stat_, typename Weight_, typename Output_>
void average_vectors_weighted(size_t n, std::vector<Stat_*> in, const Weight_* w, Output_* out) {
    if (!in.empty()) {
        bool same = true;
        for (size_t i = 1, end = in.size(); i < end; ++i) {
            if (w[i] != w[0]) {
                same = false;
                break;
            }
        }

        if (same) {
            if (w[0] == 0) {
                std::fill(out, out + n, std::numeric_limits<Output_>::quiet_NaN());
            } else {
                average_vectors<check_nan_>(n, std::move(in), out);
            }
            return;
        }
    }

    average_vectors_internal<check_nan_, true>(n, std::move(in), w, out);
    return;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
 *
 * @tparam check_nan_ Whether to check for NaNs, see `average_vectors_weighted()` for details.
 * @tparam Output_ Floating-point output type.
 * @tparam Weight_ Type of the weight, typically floating point.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of the same length.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * Weights should be non-negative and finite.
 *
 * @return A vector is returned containing with the average of all arrays in `in`.
 */
template<bool check_nan_ = true, typename Output_ = double, typename Stat_, typename Weight_>
std::vector<Output_> average_vectors_weighted(size_t n, std::vector<Stat_*> in, const Weight_* w) {
    std::vector<Output_> out(n);
    average_vectors_weighted<check_nan_>(n, std::move(in), w, out.data());
    return out;
}

}

#endif
