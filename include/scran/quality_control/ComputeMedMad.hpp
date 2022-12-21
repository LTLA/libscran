#ifndef SCRAN_COMPUTE_MED_MAD_H
#define SCRAN_COMPUTE_MED_MAD_H

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "utils.hpp"
#include "../utils/block_indices.hpp"

/**
 * @file ComputeMedMad.hpp
 *
 * @brief Compute the median and MAD from an array of values.
 */

namespace scran {

/**
 * @brief Compute the median and MAD from an array of values.
 *
 * Pretty much as it says on the can; computes the median of an array of values first,
 * and uses the median to then compute the median absolute deviation (MAD) from that array.
 * We support calculation of medians and MADs for separate "blocks" of observations in the same array.
 */
class ComputeMedMad {
public:
    /**
     * Whether to compute the median and MAD after log-transformation of the values.
     * If `true`, all values are assumed to be non-negative.
     */
    bool log = false;

public:
    /**
     * @brief Medians and MADs, possibly for multiple blocks.
     *
     * Meaningful instances of this object should generally be constructed by calling the `IsOutlier::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results(int nblocks=0) : medians(nblocks), mads(nblocks) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the median for that block.
         *
         * Note that this may be NaN if there are no non-NaN observations,
         * or infinite if observations contain non-finite values.
         */
        std::vector<double> medians;

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the MAD for that block.
         *
         * Note that this may be NaN if there are no non-NaN observations.
         */
        std::vector<double> mads;

        /**
         * Whether the medians and MADs are computed on a (natural) log scale.
         */
        bool log = false;
    };

public:
    /**
     * @tparam Input Numeric type for the input.
     * @tparam Buffer Floating-point type for the buffer.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * NaNs are ignored.
     * @param buffer Pointer to an array of length `n`.
     *
     * @return Median and MAD for the `metrics`.
     */
    template<typename Input, typename Buffer> 
    Results run(size_t n, const Input* metrics, Buffer* buffer) const {
        Results output(1);
        output.log = log;

        auto copy = buffer;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isnan(metrics[i])) {
                *copy = metrics[i];
                ++copy;
            }
        }

        compute(copy - buffer, buffer, output.medians[0], output.mads[0]);
        return output;
    }

    /**
     * @tparam Input Numeric type for the input.
     * @tparam Buffer Floating-point type for the buffer.
     *
     * @param n Number of observations.
     * @param by_block Vector of length equal to the number of blocks.
     * Each entry is a vector of integers, containing the indices of the elements in `metrics` in the corresponding block.
     * All indices should lie in `[0, n)`.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * NaNs are ignored.
     * @param buffer Pointer to an array of length `n`.
     *
     * @return Medians and MADs for each block in `metrics`.
     */
    template<typename Input, typename Buffer>
    Results run_blocked(size_t n, const std::vector<std::vector<size_t> >& by_block, const Input* metrics, Buffer* buffer) const {
        Results output(by_block.size());
        output.log = log;

        for (size_t g = 0; g < by_block.size(); ++g) {
            auto copy = buffer;
            const auto& curblock = by_block[g];
            for (auto s : curblock) {
                if (!std::isnan(metrics[s])) {
                    *copy = metrics[s];
                    ++copy;
                }
            }
            compute(copy - buffer, buffer, output.medians[g], output.mads[g]);
        }

        return output;
    }

public:
    /**
     * @tparam Input Numeric type for the input.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * NaNs are ignored.
     *
     * @return Median and the MAD for `metrics`.
     */
    template<typename Input>
    Results run(size_t n, const Input* metrics) const {
        std::vector<double> buffer(n);
        return run(n, metrics, buffer.data());
    } 

    /**
     * @tparam Block Integer type, containing the block IDs.
     * @tparam Input Type of observation, numeric.
     *
     * @param n Number of observations.
     * @param[in] block Optional pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * If a null pointer is supplied, all observations are assumed to belong to the same block.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     *
     * @return Median and the MAD for each block in `metrics`.
     */
    template<typename Block, typename Input>
    Results run_blocked(size_t n, const Block * block, const Input* metrics) const {
        if (block) {
            BlockIndices by_block = block_indices(n, block);
            std::vector<double> buffer(n);
            return run_blocked(n, by_block, metrics, buffer.data());
        } else {
            return run(n, metrics);
        }
    }

private:
    template<typename Buffer>
    void compute(size_t n, Buffer* ptr, double& median, double& mad) const {
        if (!n) {
            median = std::numeric_limits<double>::quiet_NaN();
            mad = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        if (log) {
            auto copy = ptr;
            for (size_t i = 0; i < n; ++i, ++copy) {
                if (*copy > 0) {
                    *copy = std::log(*copy);
                } else if (*copy == 0) {
                    *copy = -std::numeric_limits<double>::infinity();
                } else {
                    throw std::runtime_error("cannot log-transform negative values");
                }
            }
        }

        // First, getting the median.
        size_t halfway = n / 2;
        median = compute_median(ptr, halfway, n);

        if (std::isnan(median)) {
            // Giving up.
            mad = std::numeric_limits<double>::quiet_NaN();
            return;
        } else if (std::isinf(median)) {
            // MADs should be no-ops when added/subtracted from infinity. Any
            // finite value will do here, so might as well keep it simple.
            mad = 0;
            return;
        }

        // Now getting the MAD. 
        auto copy = ptr;
        for (size_t i = 0; i < n; ++i, ++copy) {
            *copy = std::abs(*copy - median);
        }
        mad = compute_median(ptr, halfway, n);
        mad *= 1.4826; // for equivalence with the standard deviation under normality.
        return;
    }

    static double compute_median (double* ptr, size_t halfway, size_t n) {
        std::nth_element(ptr, ptr + halfway, ptr + n);
        double medtmp = *(ptr + halfway);

        if (n % 2 == 0) {
            std::nth_element(ptr, ptr + halfway - 1, ptr + n);
            double left = *(ptr + halfway - 1);

            bool inf_left = std::isinf(left);
            bool inf_right = std::isinf(medtmp);
            if (inf_left && inf_right) {
                if ((inf_left < 0) != (inf_right > 0)) {
                    return std::numeric_limits<double>::quiet_NaN();
                } else {
                    return inf_left;
                }
            } 

            medtmp = (medtmp + left)/2;
        }

        return medtmp;
    }

public:
    /**
     * @brief Define outlier filters from the median and MAD.
     *
     * Given an array of values, outliers are defined as those that are more than some number of median absolute deviations (MADs) from the median value.
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
         * @param x Median and MAD computed from `ComputeMedMad::run()` or `ComputeMedMad::run_blocked()`.
         * @return Outlier filter thresholds defined in terms of MADs from the median.
         */
        FilterThresholds run(ComputeMedMad::Results x) const {
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
};

}

#endif
