#ifndef SCRAN_IS_OUTLIER_H
#define SCRAN_IS_OUTLIER_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "../utils/block_indices.hpp"

/**
 * @file IsOutlier.hpp
 *
 * @brief Define outliers using a simple MAD-based approach.
 */

namespace scran {

/**
 * @brief Define outlier values based on MADs.
 *
 * Given an array of values, outliers are defined as those that are more than some number of median absolute deviations (MADs) from the median value.
 * By default, we require 3 MADs, which is motivated by the low probability (less than 1%) of obtaining such a value under the normal distribution.
 * Outliers can be defined in both or either direction, and also after log-transformation.
 */
class IsOutlier {
public:
    /**
     * Compute outliers after log-transformation of the input values.
     * This is useful for improving resolution close to zero as well as mimicking normality in right-skewed distributions.
     *
     * @param l Whether the values should be log-transformed.
     *
     * @return A reference to this `IsOutlier` object.
     */
    IsOutlier& set_log(bool l = false) {
        log = l;
        return *this;
    }

    /**
     * Should low values be considered as potential outliers?
     * If `false`, no lower threshold is applied when defining outliers.
     *
     * @param l Whether low outlier values should be identified.
     *
     * @return A reference to this `IsOutlier` object.
     */
    IsOutlier& set_lower(bool l = true) {
        lower = l;
        return *this;
    }

    /**
     * Should high values be considered as potential outliers?
     * If `false`, no upper threshold is applied when defining outliers.
     *
     * @param u Whether high outlier values should be identified.
     *
     * @return A reference to this `IsOutlier` object.
     */
    IsOutlier& set_upper(bool u = true) {
        upper = u;
        return *this;
    }

    /**
     * @param n Number of MADs to use to define outliers.
     *
     * @return A reference to this `IsOutlier` object.
     */
    IsOutlier& set_nmads(double n = 3) {
        nmads = n;
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     *
     * Meaningful instances of this object should generally be constructed by calling the `IsOutlier::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Thresholds {
        /**
         * @cond
         */
        Thresholds() {}

        Thresholds(int nblocks=1) : lower(nblocks), upper(nblocks) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the lower threshold used to define outliers (i.e., all values below this threshold are considered outliers).
         *
         * If `set_lower()` was set to `false`, all values are set to negative infinity.
         *
         * Note that these thresholds are always reported on the original scale, regardless of `set_log()`.
         * Any log-transformation used during outlier identification is reversed when reporting the thresholds.
         */
        std::vector<double> lower;

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the upper threshold used to define outliers (i.e., all values above this threshold are considered outliers).
         *
         * If `set_upper()` was set to `false`, all values are set to positive infinity.
         *
         * Note that these thresholds are always reported on the original scale, regardless of `set_log()`.
         * Any log-transformation used during outlier identification is reversed when reporting the thresholds.
         */
        std::vector<double> upper;
    };

public:
    /**
     * Identify outliers in an array of observations. 
     * This is done according to the parameters specified in the various setters, i.e., `set_nmads()`, `set_lower()`, etc.
     *
     * @tparam T Type of observation, numeric.
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * @param[out] outliers Pointer to an output array of length `n`,
     * where the outlier identification results are to be stored.
     *
     * @return A `Thresholds` object containing the thresholds used to identify outliers.
     * `outliers` is filled with the outlier calls for each observation in `metrics`.
     */
    template<typename T, typename X>
    Thresholds run(size_t n, const T* metrics, X* outliers) {
        Thresholds output(1);
        buffer.resize(n);
        std::copy(metrics, metrics + n, buffer.data());

        double& lthresh = output.lower[0];
        double& uthresh = output.upper[0];
        compute_thresholds(n, buffer.data(), lthresh, uthresh);

        auto copy = metrics;
        for (size_t i = 0; i < n; ++i, ++copy, ++outliers) {
            const double val = *copy;
            *outliers = (val < lthresh || val > uthresh);
        }

        return output;
    } 

    /**
     * Identify outliers in an array of observations with blocking.
     * Outlier detection is performed separately within each block of observations, as if `run()` was called on each subset of observations with the same ID in `block`.
     * This is occasionally useful when dealing with multi-batch datasets, see `PerCellQCFilters::run()` for an example.
     *
     * @tparam B Integer type, containing the block IDs.
     * @tparam T Type of observation, numeric.
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     *
     * @param n Number of observations.
     * @param[in] block Optional pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * If a null pointer is supplied, all observations are assumed to belong to the same block.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * @param[out] outliers Pointer to an output array of length `n`,
     * where the outlier identification results are to be stored.
     *
     * @return A `Thresholds` object containing the thresholds used to identify outliers in each block.
     * `outliers` is filled with the outlier calls for each observation in `metrics`.
     */
    template<typename B, typename T, typename X>
    Thresholds run_blocked(size_t n, const B * block, const T* metrics, X* outliers) {
        if (block) {
            BlockIndices by_block = block_indices(n, block);
            return run_blocked(n, by_block, metrics, outliers);
        } else {
            return run(n, metrics, outliers);
        }
    }

    /**
     * @cond
     */
    template<typename T, typename X>
    Thresholds run_blocked(size_t n, const BlockIndices& by_group, const T* metrics, X* outliers) {
        Thresholds output(by_group.size());
        buffer.resize(n);

        for (size_t g = 0; g < by_group.size(); ++g) {
            auto copy = buffer.data();
            const auto& curgroup = by_group[g];
            for (auto s : curgroup) {
                *copy = metrics[s];
                ++copy;
            }

            double& lthresh = output.lower[g];
            double& uthresh = output.upper[g];
            compute_thresholds(curgroup.size(), buffer.data(), lthresh, uthresh);

            for (auto s : curgroup) {
                const double val = metrics[s];
                outliers[s] = (val < lthresh || val > uthresh);
            }
        }

        return output;
    }
    /**
     * @endcond
     */

public:
    /**
     * @brief Results of the outlier identification.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     */
    template<typename X = uint8_t>
    struct Results {
        /**
         * @param n Number of observations.
         */
        Results(size_t n) : outliers(n), thresholds(0) {}

        /**
         * A vector of boolean types indicating whether an observation should be considered an outlier.
         */
        std::vector<X> outliers;

        /**
         * The thresholds used to define outliers.
         */
        Thresholds thresholds;
    };

public:
    /**
     * Identify outliers in an array of observations, see `run()` for details.
     * 
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam T Type of observation, numeric.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     *
     * @return A `Results` object containing the outlier identification results.
     */
    template<typename X = uint8_t, typename T>
    Results<X> run(size_t n, const T* metrics) {
        Results<X> output(n); 
        output.thresholds = run(n, metrics, output.outliers.data());
        return output;
    }

    /**
     * Identify outliers in an array of observations with blocking, see `run_blocked()` for details.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam T Type of observation, numeric.
     * @tparam B Integer type, containing the block IDs.
     *
     * @param n Number of observations.
     * @param[in] block Optional pointer to an array of block identifiers, see `run_blocked()` for details.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     *
     * @return A `Results` object containing the outlier identification results.
     */
    template<typename X = uint8_t, typename B, typename T>
    Results<X> run_blocked(size_t n, const B* block, const T* metrics) {
        Results<X> output(n); 
        if (block) {
            output.thresholds = run_blocked(n, block, metrics, output.outliers.data());
        } else {
            output.thresholds = run(n, metrics, output.outliers.data());
        }
        return output;
    }

public:
    /**
     * Identify outliers in an vector of observations, see `run()` for details.
     * 
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam V Vector class - typically a `std::vector`, but anything that supports the `size()` and `data()` methods can be used here.
     *
     * @param metrics Vector of observations.
     *
     * @return A `Results` object containing the outlier identification results.
     */
    template<typename X = uint8_t, class V>
    Results<X> run(const V& metrics) {
        return run(metrics.size(), metrics.data());
    }

    /**
     * Identify outliers in an vector of observations with blocking, see `run_blocked()` for details.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam V Vector class - typically a `std::vector`, but anything that supports the `size()` and `data()` methods can be used here.
     * @tparam B Integer type, containing the block IDs.
     *
     * @param metrics Vector of observations.
     * @param[in] block Optional pointer to an array of block identifiers of length equal to `metrics`, see `run_blocked()` for details.
     *
     * @return A `Results` object containing the outlier identification results.
     */
    template<typename X = uint8_t, class V, typename B>
    Results<X> run_blocked(const V& metrics, const B* block) {
        return run_blocked(metrics.size(), block, metrics.data());
    }

private:
    void compute_thresholds(size_t n, double* ptr, double& lower_threshold, double& upper_threshold) {
        if (!n) {
            lower_threshold = std::numeric_limits<double>::quiet_NaN();
            upper_threshold = std::numeric_limits<double>::quiet_NaN();
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
        double medtmp = compute_median(ptr, halfway, n);

        if (log && std::isinf(medtmp)) {
            if (lower) {
                lower_threshold = 0;
            } else {
                lower_threshold = -std::numeric_limits<double>::infinity();
            }

            if (upper) {
                upper_threshold = 0;
            } else {
                upper_threshold = std::numeric_limits<double>::infinity();
            }

            return;
        }

        // Now getting the MAD. No need to protect against -inf here;
        // if the median is finite, the MAD must also be finite.
        auto copy = ptr;
        for (size_t i = 0; i < n; ++i, ++copy) {
            *copy = std::abs(*copy - medtmp);
        }
        double madtmp = compute_median(ptr, halfway, n);

        madtmp *= 1.4826; // for equivalence with the standard deviation under normality.

        // Computing the outputs.
        if (lower) {
            lower_threshold = medtmp - nmads * madtmp;
            if (log) {
                lower_threshold = std::exp(lower_threshold);
            }
        } else {
            lower_threshold = -std::numeric_limits<double>::infinity();
        }

        if (upper) {
            upper_threshold = medtmp + nmads * madtmp;
            if (log) {
                upper_threshold = std::exp(upper_threshold);
            }
        } else {
            upper_threshold = std::numeric_limits<double>::infinity();
        }

        return;
    }

    static double compute_median (double* ptr, size_t halfway, size_t n) {
        std::nth_element(ptr, ptr + halfway, ptr + n);
        double medtmp = *(ptr + halfway);

        if (n % 2 == 0) {
            std::nth_element(ptr, ptr + halfway - 1, ptr + n);
            medtmp = (medtmp + *(ptr + halfway - 1))/2;
        }

        return medtmp;
    }

private:
    double nmads = 3;
    bool lower = true, upper = true, log = false;
    std::vector<double> buffer;
};

}

#endif
