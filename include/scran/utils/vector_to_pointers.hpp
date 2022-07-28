#ifndef VECTOR_TO_POINTERS
#define VECTOR_TO_POINTERS

#include "macros.hpp"

#include <vector>

/**
 * @file vector_to_pointers.hpp
 *
 * @brief Create a vector of pointers from a vector of vectors.
 */

namespace scran {

/**
 * Extract a vector of pointers from a vector of vectors.
 * This is a convenient utility as many **scran** functions accept the former but can return the latter.
 *
 * @tparam T Type of data.
 * @param input Vector of vector of values.
 *
 * @return A vector of pointers to each inner vector in `input`.
 */
template<typename T>
inline std::vector<T*> vector_to_pointers(std::vector<std::vector<T> >& input) {
    std::vector<T*> output(input.size());
    auto oIt = output.begin();
    for (auto& i : input) {
        *oIt = i.data();
        ++oIt;
    }
    return output;
}

/**
 * Extract a vector of `const` pointers from a vector of vectors.
 *
 * @tparam T Type of data.
 * @param input Vector of vector of values.
 *
 * @return A vector of pointers to each inner vector in `input`.
 */
template<typename T>
inline std::vector<const T*> vector_to_pointers(const std::vector<std::vector<T> >& input) {
    std::vector<const T*> output(input.size());
    auto oIt = output.begin();
    for (auto& i : input) {
        *oIt = i.data();
        ++oIt;
    }
    return output;
}

}

#endif
