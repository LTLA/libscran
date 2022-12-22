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
 * Any log-transformation used to compute the MAD (see `ComputeMedianMad::set_log()`) is automatically reversed in the reported thresholds.
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
        static constexpr double min_diff = 0;
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
     * @brief Outlier thresholds for QC filtering.
     */
    struct Thresholds {
        /**
         * @cond
         */
        Thresholds(size_t nblocks = 0) : lower(nblocks), upper(nblocks) {}
        /**
         * @endcond
         */

        /**
         * Vector of lower thresholds, one per batch.
         * Cells where the relevant QC metric is below this threshold are considered to be low quality.
         * If empty, no lower threshold is to be used.
         */
        std::vector<double> lower;

        /**
         * Vector of upper thresholds, one per batch.
         * Cells where the relevant QC metric is above this threshold are considered to be low quality.
         * If empty, no upper threshold is to be used.
         */
        std::vector<double> upper;

    public:
        /**
         * @tparam overwrite Whether to overwrite existing truthy entries in `output`.
         * @tparam Input Numeric type for the values.
         * @tparam Output Boolean type for the outlier calls.
         *
         * @param n Number of observations.
         * @param[in] input Pointer to an array of length `n`, containing the values to be filtered.
         * @param[out] output Pointer to an array of length `n`, to store the outlier calls.
         * Values are set to `true` for outliers.
         * If `overwrite = true`, values are set to `false` for non-outliers, otherwise the existing entry is preserved.
         *
         * Use `filter_blocked()` instead for multi-block datasets. 
         */
        template<bool overwrite = true, typename Input, typename Output>
        void filter(size_t n, Input* input, Output* output) const {
            if (upper.size() > 1 || lower.size() > 1) {
                throw std::runtime_error("should use filter_blocked() for multiple batches");
            }
            filter_<overwrite>(n, input, output, [](size_t i) -> size_t { return 0; });
        }

        /**
         * @overload
         * @tparam Output Boolean type for the outlier calls.
         * @tparam Input Numeric type for the values.
         *
         * @param n Number of observations.
         * @param[in] input Pointer to an array of length `n`, containing the values to be filtered.
         *
         * @return Vector of outlier calls, of length equal to `n`.
         */
        template<typename Output = uint8_t, typename Input>
        std::vector<Output> filter(size_t n, const Input* input) const {
            std::vector<Output> output(n);
            filter(n, input, output.data());
            return output;
        }

    public:
        /**
         * @tparam overwrite Whether to overwrite existing truthy entries in `output`.
         * @tparam Block Integer type for the block assignments.
         * @tparam Input Numeric type for the values.
         * @tparam Output Boolean type for the outlier calls.
         *
         * @param n Number of cells.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         * @param[in] input Pointer to an array of length `n`, containing the values to be filtered.
         * @param[out] output Pointer to an array of length `n`, to store the outlier calls.
         * Values are set to `true` for outliers.
         * If `overwrite = true`, values are set to `false` for non-outliers, otherwise the existing entry is preserved.
         */
        template<bool overwrite = true, typename Block, typename Input, typename Output>
        void filter_blocked(size_t n, const Block* block, const Input* input, Output* output) const {
            if (block) {
                filter_<overwrite>(n, input, output, [&](size_t i) -> Block { return block[i]; });
            } else {
                filter<overwrite>(n, input, output);
            }
        }

        /**
         * @overload
         *
         * @tparam Output Boolean type for the outlier calls.
         * @tparam Block Integer type for the block assignments.
         * @tparam Input Numeric type for the values.
         *
         * @param n Number of cells.
         * @param[in] block Pointer to an array of length `n`, containing the block assignment for each cell.
         * This may be `NULL`, in which case all cells are assumed to belong to the same block.
         * @param[in] input Pointer to an array of length `n`, containing the values to be filtered.
         *
         * @return Vector of low-quality calls, of length equal to the number of cells in `metrics`.
         */
        template<typename Output = uint8_t, typename Block, typename Input>
        std::vector<Output> filter_blocked(size_t n, const Block* block, const Input* input) const {
            std::vector<Output> output(n);
            filter_blocked(n, block, input, output.data());
            return output;
        }

    private:
        template<bool overwrite, typename Input, typename Output, typename Function>
        void filter_(size_t n, const Input* input, Output* output, Function indexer) const {
            for (size_t i = 0; i < n; ++i) {
                auto b = indexer(i);

                if (!lower.empty()) {
                    if (quality_control::is_less_than(input[i], lower[b])) {
                        output[i] = true;
                        continue;
                    }
                }

                if (!upper.empty()) {
                    if (quality_control::is_greater_than(input[i], upper[b])) {
                        output[i] = true;
                        continue;
                    }
                }

                if constexpr(overwrite) {
                    output[i] = false;
                }
            }

            return;
        }
    };

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
    Thresholds run(ComputeMedianMad::Results x) const {
        size_t nblocks = x.medians.size();

        for (size_t b = 0; b < nblocks; ++b) {
            auto med = x.medians[b];
            auto mad = x.mads[b];

            double& lthresh = x.medians[b];
            double& uthresh = x.mads[b];

            if (std::isnan(med)) {
                if (lower) {
                    lthresh = std::numeric_limits<double>::quiet_NaN();
                }
                if (upper) {
                    uthresh = std::numeric_limits<double>::quiet_NaN();
                }
            } else {
                auto delta = std::max(min_diff, num_mads * mad);
                if (lower) {
                    lthresh = sanitize(med - delta, x.log);
                }
                if (upper) {
                    uthresh = sanitize(med + delta, x.log);
                }
            }
        }

        Thresholds output;
        if (lower) {
            output.lower = std::move(x.medians);
        }
        if (upper) {
            output.upper = std::move(x.mads);
        }
        return output;
    }
};

}

#endif
