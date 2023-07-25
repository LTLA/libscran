#ifndef SCRAN_NORMALIZATION_UTILS_HPP
#define SCRAN_NORMALIZATION_UTILS_HPP

#include <limits>
#include <cmath>

namespace scran {

template<typename V>
void sanitize_zeros(V& size_factors) {
    auto smallest = std::numeric_limits<double>::infinity();
    for (auto s : size_factors) {
        if (s && smallest > s) { // NaN returns false here, and Inf won't be less than the starting 'smallest', so no need to handle these separately.
            smallest = s;
        }
    }

    if (std::isinf(smallest)) {
        smallest = 1;
    }

    for (auto& s : size_factors) {
        if (s == 0) {
            s = smallest;
        }
    }
}

template<typename V>
void sanitize_non_finite(V& size_factors) {
    // Replacing them with the smallest non-zero size factor, or 1.
    double largest = 0;
    for (auto s : size_factors) {
        if (std::isfinite(s) && largest < s) {
            largest = s;
        }
    }

    if (largest == 0) {
        largest = 1;
    }

    for (auto& s : size_factors) {
        if (std::isinf(s)) {
            s = largest;
        } else if (std::isnan(s)) { // no-op scaling if size factor is missing.
            s = 1;
        }
    }
}

}

#endif
