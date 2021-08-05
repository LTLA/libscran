#ifndef AVERAGE_VECTORS_HPP
#define AVERAGE_VECTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace scran {

/**
 * @cond
 */
template<bool weighted = true, typename Stat, typename Weights, typename Output>
void average_vectors_internal(size_t n, std::vector<Stat*> in, Weights* w, Output* out) {
    std::fill(out, out + n, 0);
    std::vector<Weights> accumulated(n);

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

template<typename Stat, typename Output>
void average_vectors(size_t n, std::vector<Stat*> in, Output* out) {
    average_vectors_internal<false>(n, std::move(in), (double*)NULL, out);
    return;
}

template<typename Stat, typename Output = Stat>
std::vector<Output> average_vectors(size_t n, std::vector<Stat*> in) {
    std::vector<Output> out(n);
    average_vectors(n, std::move(in), out.data());
    return out;
}

template<typename Stat, typename Weights, typename Output>
void average_vectors_weighted(size_t n, std::vector<Stat*> in, Weights* w, Output* out) {
    average_vectors_internal<true>(n, std::move(in), w, out);
    return;
}

template<typename Stat, typename Weights, typename Output = Stat>
std::vector<Output> average_vectors_weighted(size_t n, std::vector<Stat*> in, Weights* w) {
    std::vector<Output> out(n);
    average_vectors_weighted(n, std::move(in), w, out.data());
    return out;
}

}

#endif
