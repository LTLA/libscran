#ifndef SCRAN_QUALITY_CONTROL_UTILS_HPP
#define SCRAN_QUALITY_CONTROL_UTILS_HPP

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

}
/**
 * @endcond
 */

}

#endif
