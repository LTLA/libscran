#ifndef SCRAN_QUALITY_CONTROL_UTILS_HPP
#define SCRAN_QUALITY_CONTROL_UTILS_HPP

#include <limits>
#include <vector>

namespace scran {

/**
 * @cond
 */
namespace quality_control {

template<typename Float>
void safe_divide(size_t n, Float* top, const Float* bottom) {
    for (size_t c = 0; c < n; ++c) {
        if (bottom[c]) {
            top[c] /= bottom[c];
        } else {
            if constexpr(std::numeric_limits<Float>::has_quiet_NaN) {
                top[c] = std::numeric_limits<Float>::quiet_NaN();
            } else {
                top[c] = 0;
            }
        }
    }
}

template<typename Float>
bool is_less_than(Float value, Float threshold) {
    return !std::isnan(value) && !std::isnan(threshold) && value < threshold;
}

template<typename Float>
bool is_greater_than(Float value, Float threshold) {
    return !std::isnan(value) && !std::isnan(threshold) && value > threshold;
}

}
/**
 * @endcond
 */

}

#endif
