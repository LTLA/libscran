#ifndef SCRAN_QUALITY_CONTROL_UTILS_HPP
#define SCRAN_QUALITY_CONTROL_UTILS_HPP

#include <limits>
#include <vector>

namespace scran {

/**
 * @brief Thresholds for QC filtering.
 */
struct FilterThresholds {
    /**
     * @cond
     */
    FilterThresholds(size_t nblocks = 0) :
        lower(nblocks, -std::numeric_limits<double>::infinity()),
        upper(nblocks, std::numeric_limits<double>::infinity()) {}
    /**
     * @endcond
     */

    /**
     * Vector of lower thresholds, one per batch.
     * Cells where the relevant QC metric is below this threshold are considered to be low quality.
     */
    std::vector<double> lower;

    /**
     * Vector of upper thresholds, one per batch.
     * Cells where the relevant QC metric is above this threshold are considered to be low quality.
     */
    std::vector<double> upper;

};

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
