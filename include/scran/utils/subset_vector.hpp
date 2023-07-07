#ifndef SUBSET_VECTOR_HPP
#define SUBSET_VECTOR_HPP

#include "macros.hpp"

/**
 * @file subset_vector.hpp
 * @brief Subset a vector easily.
 */

namespace scran {

/**
 * Subset a vector to retain/discard elements.
 *
 * @tparam retain Should the non-zero elements in `sub` be retained (`true`) or discarded (`false`)?
 * @tparam Vector A vector class that has a `size()` method, a `[]` operator, and a constructor that initializes the vector to a specified length.
 * @tparam Subset Integer/boolean type for the subsetting vector.
 *
 * @param vec A vector of arbitrary (copy-able) elements.
 * @param sub Pointer to an array of integer/boolean elements of length equal to `vec`.
 * Non-zero values indicate that an element should be retained (if `retain=true`) or discarded (otherwise).
 *
 * @return A vector containing the desired subset of elements in `vec`.
 */
template<bool retain, class Vector, typename Subset>
Vector subset_vector(const Vector& vec, const Subset* sub) {
    int n = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (sub[i]) {
            ++n;
        }
    }

    if constexpr(!retain) {
        n = vec.size() - n;        
    }

    Vector output(n);
    size_t counter = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
        if constexpr(retain) {
            if (sub[i]) {
                output[counter] = vec[i];
                ++counter;
            }
        } else {
            if (!sub[i]) {
                output[counter] = vec[i];
                ++counter;
            }
        }
    }

    return output;
}

}

#endif
