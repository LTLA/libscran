#ifndef SCRAN_CHOOSE_OUTLIER_FILTERS_HPP
#define SCRAN_CHOOSE_OUTLIER_FILTERS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <cmath>

#include "utils.hpp"
#include "ComputeMedianMad.hpp"

/**
 * @file ChooseOutlierFilters.hpp
 *
 * @brief Define outlier filters using a MAD-based approach.
 */

namespace scran {

/**
 * @brief Define outlier filters from the median and MAD.
 *
 * This class uses the output from `ComputeMedianMad` to define filter thresholds for outliers on the QC metrics.
 * Given an array of values, outliers are defined as those that are more than some number of median absolute deviations (MADs) from the median value.
 * By default, we require 3 MADs, which is motivated by the low probability (less than 1%) of obtaining such a value under the normal distribution.
 * Outliers can be defined in both directions, or just a single direction, depending on the interpretation of the QC metric.
 * Any log-transformation used to compute the MAD (see `ComputeMedianMad::set_log()`) is automatically reversed in the reported metrics.
 */
struct ChooseOutlierFilters {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_lower()` for details.
         */
        static constexpr bool lower = true;

        /**
         * See `set_upper()` for details.
         */
        static constexpr bool upper = true;

        /**
         * See `set_num_mads()` for details.
         */
        static constexpr double num_mads = 3;

        /**
         * See `set_min_diff()` for details.
         */
        static constexpr double min_diff = 0.1;
    };

private:
    bool lower = Defaults::lower;
    bool upper = Defaults::upper;
    double num_mads = Defaults::num_mads;
    double min_diff = Defaults::min_diff;

public:
    /**
     * @param l Should low values be considered as potential outliers?
     * If `false`, no lower threshold is applied when defining outliers.
     *
     * @return A reference to this `ChooseOutlierFilters` object.
     */
    ChooseOutlierFilters& set_lower(bool l = Defaults::lower) {
        lower = l;
        return *this;
    }

    /**
     * @param u Should high values be considered as potential outliers?
     * If `false`, no upper threshold is applied when defining outliers.
     *
     * @return A reference to this `ChooseOutlierFilters` object.
     */
    ChooseOutlierFilters& set_upper(bool u = Defaults::upper) {
        upper = u;
        return *this;
    }

    /**
     * @param n Number of MADs to use to define outliers.
     * Larger values result in more relaxed thresholds.
     * 
     * @return A reference to this `ChooseOutlierFilters` object.
     */
    ChooseOutlierFilters& set_num_mads(double n = Defaults::num_mads) {
        num_mads = n;
        return *this;
    }

    /**
     * @param m Minimum difference from the median to define outliers.
     * This enforces a more relaxed threshold in cases where the MAD may be too small.
     * If the median and MADs are log-transformed, this difference is interpreted as a unit on the log-scale.
     *
     * @return A reference to this `ChooseOutlierFilters` object.
     */
    ChooseOutlierFilters& set_min_diff(double m = Defaults::min_diff) {
        min_diff = m;
        return *this;
    }

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
