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
 * Define outliers using a simple MAD-based approach.
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
     * Define blocks of observations where outliers are computed within each block.
     * In other words, only the observations in each block are used to define the MAD and median (and subsequent thresholds) for identifying outliers in that block.
     *
     * @tparam V Class of the blocking vector.
     * @param p Blocking vector of length equal to the number of input values.
     * Values should be integer block IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     *
     * @return A reference to this `IsOutlier` object. 
     */
    template<class V>
    IsOutlier& set_blocks(const V& p) {
        return set_blocks(p.size(), p.begin());
    }

    /**
     * Define blocks of observations, see `set_blocks(const V& p)`.
     *
     * @tparam SIT Iterator class for the blocking vector.
     *
     * @param n Length of the blocking vector, should be equal to the number of observations.
     * @param p Pointer or iterator to the start of the blocking vector.
     *
     * @return A reference to this `IsOutlier` object. 
     */
    template<typename SIT>
    IsOutlier& set_blocks(size_t n, SIT p) {
        group_ncells = n;
        block_indices(n, p, by_group);
        return *this;
    }

    /**
     * Unset any previous blocking structure.
     *
     * @return A reference to this `IsOutlier` object. 
     */
    IsOutlier& set_blocks() {
        group_ncells = 0;
        by_group.clear();
        return *this;
    }

public:
    /**
     * @brief Thresholds to define outliers on each metric.
     */
    struct Thresholds {
        /**
         * @param nblocks Number of blocks, see `set_blocks(const V& p)`.
         */
        Thresholds(int nblocks=1) : lower(nblocks), upper(nblocks) {}

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
     * Identify outliers in an array of observations.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam V Vector class - typically a `std::vector`, but anything that supports the `size()` and `data()` methods can be used here.
     *
     * @param metrics A vector of observations.
     *
     * @return A `Results` object containing the outlier identification results.
     */
    template<typename X = uint8_t, class V>
    Results<X> run(const V& metrics) {
        Results<X> output(metrics.size()); 
        output.thresholds = run(metrics.size(), metrics.data(), output.outliers.data());
        return output;
    }

    /**
     * Identify outliers in an array of observations.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam T Type of observation.
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
     * Identify outliers in an array of observations.
     *
     * @tparam X Boolean type to indicate whether an observation is an outlier.
     * @tparam T Type of observation.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * @param[out] outliers Pointer to an output array of length `n`,
     * where the outlier identification results are to be stored.
     *
     * @return A `Thresholds` object containing the thresholds used to identify outliers.
     * `outliers` is filled with the outlier calls for each observation in `metrics`.
     */
    template<typename X = uint8_t, typename T>
    Thresholds run(size_t n, const T* metrics, X* outliers) {
        Thresholds output(std::max(static_cast<size_t>(1), by_group.size()));
        buffer.resize(n);

        if (by_group.size() == 0) {
            std::copy(metrics, metrics + n, buffer.data());

            double& lthresh = output.lower[0];
            double& uthresh = output.upper[0];
            compute_thresholds(n, buffer.data(), lthresh, uthresh);

            auto copy = metrics;
            for (size_t i = 0; i < n; ++i, ++copy, ++outliers) {
                const double val = *copy;
                *outliers = (val < lthresh || val > uthresh);
            }
        } else {
            if (group_ncells != n) {
                throw std::runtime_error("length of grouping vector and number of cells are not equal");
            }

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
        }

        return output;
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

    std::vector<std::vector<size_t> > by_group;
    size_t group_ncells = 0;
};

}

#endif
