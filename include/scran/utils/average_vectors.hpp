#ifndef AVERAGE_VECTORS_HPP
#define AVERAGE_VECTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

/**
 * @file average_vectors.hpp
 *
 * @brief Average parallel elements across vectors.
 */

namespace scran {

/**
 * @cond
 */
template<bool weighted = true, typename Stat, typename Weight, typename Output>
void average_vectors_internal(size_t n, std::vector<Stat*> in, const Weight* w, Output* out) {
    std::fill(out, out + n, 0);
    std::vector<Weight> accumulated(n);

    for (auto current : in) {
        auto copy = out;
        for (size_t i = 0; i < n; ++i, ++current, ++copy) {
            auto x = *current;
            if (!std::isnan(x)) {
                auto& a = accumulated[i];
                if constexpr(weighted) {
                    *copy += x * (*w); 
                    a += (*w);
                } else {
                    *copy += x; 
                    ++a;
                }
            }
        }

        if constexpr(weighted) {
            ++w;
        }
    }

    for (size_t i = 0; i < n; ++i, ++out) {
        if (accumulated[i]) {
            *out /= accumulated[i];
        } else {
            *out = std::numeric_limits<Output>::quiet_NaN();
        }
    }
}
/**
 * @endcond
 */

/**
 * Average parallel elements across multiple arrays.
 *
 * @tparam Stat Type of the input statistic, typically floating point.
 * @tparam Output Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[out] out Pointer to an output array of length `n`.
 *
 * @return `out` is filled with the average of all arrays in `in`.
 * Specifically, each element of `out` is set to the average of the corresponding elements across all `in` arrays.
 */
template<typename Stat, typename Output>
void average_vectors(size_t n, std::vector<Stat*> in, Output* out) {
    average_vectors_internal<false>(n, std::move(in), (double*)NULL, out);
    return;
}

/**
 * Average parallel elements across multiple arrays.
 *
 * @tparam Stat Type of the input statistic, typically floating point.
 * @tparam Output Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 *
 * @return A vector of length `n` is returned, containing the average of all arrays in `in`.
 */
template<typename Stat, typename Output = Stat>
std::vector<Output> average_vectors(size_t n, std::vector<Stat*> in) {
    std::vector<Output> out(n);
    average_vectors(n, std::move(in), out.data());
    return out;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
 *
 * @tparam Stat Type of the input statistic, typically floating point.
 * @tparam Weight Type of the weight, typically floating point.
 * @tparam Output Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * @param[out] out Pointer to an output array of length `n`.
 *
 * @return `out` is filled with the weighted average of all arrays in `in`.
 * Specifically, each element of `out` is set to the weighted average of the corresponding elements across all `in` arrays.
 */
template<typename Stat, typename Weight, typename Output>
void average_vectors_weighted(size_t n, std::vector<Stat*> in, const Weight* w, Output* out) {
    average_vectors_internal<true>(n, std::move(in), w, out);
    return;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
 *
 * @tparam Stat Type of the input statistic, typically floating point.
 * @tparam Weight Type of the weight, typically floating point.
 * @tparam Output Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of the same length.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 *
 * @return A vector is returned containing with the average of all arrays in `in`.
 */
template<typename Stat, typename Weight, typename Output = Stat>
std::vector<Output> average_vectors_weighted(size_t n, std::vector<Stat*> in, const Weight* w) {
    std::vector<Output> out(n);
    average_vectors_weighted(n, std::move(in), w, out.data());
    return out;
}

}

#endif
