#ifndef SCRAN_FILTER_OUTLIERS_HPP
#define SCRAN_FILTER_OUTLIERS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <cmath>

#include "utils.hpp"
#include "ComputeMedianMad.hpp"

/**
 * @file FilterOutliers.hpp
 *
 * @brief Define outlier filters using a MAD-based approach.
 */

namespace scran {

/**
 * @brief Define outlier filters from the median and MAD.
 *
 * Given an array of values, outliers are defined as those that are more than some number of median absolute deviations (MADs) from the median value.
 * This class uses the output from `ComputeMedianMad` to define filter thresholds for outliers on the QC metrics.
 * By default, we require 3 MADs, which is motivated by the low probability (less than 1%) of obtaining such a value under the normal distribution.
 * Outliers can be defined in both or either direction.
 */
struct FilterOutliers {
public:
    /**
     * Should low values be considered as potential outliers?
     * If `false`, no lower threshold is applied when defining outliers.
     */
    bool lower = true;

    /**
     * Should high values be considered as potential outliers?
     * If `false`, no upper threshold is applied when defining outliers.
     */
    bool upper = true;

    /**
     * Number of MADs to use to define outliers.
     */
    int num_mads = 3;

    /**
     * Minimum difference from the median to define outliers.
     * If the median and MADs are log-transformed, this difference is interpreted as a unit on the log-scale.
     */
    double min_diff = 0;

private:
    double sanitize(double proposed, bool log) const {
        if (log) {
            if (std::isinf(proposed)) {
                if (proposed < 0) {
                    proposed = 0;
                }
            } else {
                proposed = std::exp(proposed);
            }
        } 
        return proposed;
    }

public:
    /**
     * @param x Median and MAD computed from `ComputeMedianMad::run()` or `ComputeMedianMad::run_blocked()`.
     * @return Outlier filter thresholds defined in terms of MADs from the median.
     *
     * If `x.log` is true, the returned thresholds are defined in the original space, i.e., not log-transformed.
     * They can be directly used for comparison to the original metrics without further exponentiation.
     *
     * If multiple blocks are present in `x`, one upper/lower threshold is computed for each block.
     */
    FilterThresholds run(ComputeMedianMad::Results x) const {
        size_t nblocks = x.medians.size();
        for (size_t b = 0; b < nblocks; ++b) {
            auto med = x.medians[b];
            auto mad = x.mads[b];

            double& lthresh = x.medians[b];
            double& uthresh = x.mads[b];

            if (std::isnan(med)) {
                lthresh = std::numeric_limits<double>::quiet_NaN();
                uthresh = std::numeric_limits<double>::quiet_NaN();
            } else {
                auto delta = std::max(min_diff, num_mads * mad);
                lthresh = (lower ? sanitize(med - delta, x.log) : -std::numeric_limits<double>::infinity());
                uthresh = (upper ? sanitize(med + delta, x.log) : std::numeric_limits<double>::infinity());
            }
        }

        FilterThresholds output;
        output.lower = std::move(x.medians);
        output.upper = std::move(x.mads);
        return output;
    }
};

}

#endif
