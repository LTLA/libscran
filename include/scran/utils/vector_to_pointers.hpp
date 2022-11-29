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
std::vector<T*> vector_to_pointers(std::vector<std::vector<T> >& input) {
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
std::vector<const T*> vector_to_pointers(const std::vector<std::vector<T> >& input) {
    std::vector<const T*> output(input.size());
    auto oIt = output.begin();
    for (auto& i : input) {
        *oIt = i.data();
        ++oIt;
    }
    return output;
}

/**
 * Extract a vector of vector of pointers from a vector of vectors of vectors..
 *
 * @tparam T Type of data.
 * @param input Vector of vector of vector of values.
 *
 * @return A vector of vector of pointers to each inner vector in `input`.
 */
template<typename T>
std::vector<std::vector<T*> > vector_to_pointers(std::vector<std::vector<std::vector<T> > >& input) {
    std::vector<std::vector<T*> > output;
    output.reserve(input.size());
    for (auto& current : input) {
        output.emplace_back(vector_to_pointers(current));
    }
    return output;
}

/**
 * @cond
 */

// Convenience method to get the pointers if each middle vector contains exactly one inner vector.
// This allows us to create pointer vectors in the same format as the first vector_to_pointers overload.
template<typename T>
std::vector<T*> vector_to_front_pointers(std::vector<std::vector<std::vector<T> > >& input) {
    std::vector<T*> ptrs;
    ptrs.reserve(input.size());
    for (auto& current : input) {
        ptrs.push_back(current.front().data()); // first vector from each element.
    }
    return ptrs;
}

/**
 * @endcond
 */

}

#endif
