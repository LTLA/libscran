#ifndef SCRAN_CHOOSE_PSEUDO_COUNT_HPP
#define SCRAN_CHOOSE_PSEUDO_COUNT_HPP

#include <algorithm>
#include <vector>
#include <cmath>

/**
 * @file ChoosePseudoCount.hpp
 * @brief Choose a suitable pseudo-count.
 */

namespace scran {

/**
 * @brief Choose a pseudo-count for log-transformation.
 *
 * This class chooses a pseudo-count for log-transformation (see `LogNormCounts::set_pseudo_count()`) that aims to control the log-transform bias.
 * Specifically, the log-transform can introduce spurious differences in the expected log-normalized expression between cells with very different size factors.
 * This bias can be mitigated by increasing the pseudo-count, which effectively shrinks all log-expression values towards the zero-expression baseline.
 * The increased shrinkage is strongest at low counts where the log-transform bias is most pronounced, while the transformation of large counts is mostly unaffected.
 *
 * In practice, the log-transformation bias is modest in datasets where there are stronger sources of variation.
 * When observed, it manifests as a library size-dependent trend in the log-normalized expression values.
 * This is difficult to regress out without also removing biology that is associated with, e.g., total RNA content;
 * rather, a simpler solution is to increase the pseudo-count to suppress the bias.
 *
 * No centering is performed by this function, so if centering is required, the size factors should be used in `CenterSizeFactors` first.
 *
 * @see
 * Lun ATL (2018).
 * Overcoming systematic errors caused by log-transformation of normalized single-cell RNA sequencing data
 * _biorXiv_ doi:10.1101/404962
 */
class ChoosePseudoCount {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_quantile()` for more details.
         */
        static constexpr double quantile = 0.05;

        /**
         * See `set_max_bias()` for more details.
         */
        static constexpr double max_bias = 0.1;

        /**
         * See `set_min_value()` for more details.
         */
        static constexpr double min_value = 1;
    };

private:
    double quantile = Defaults::quantile;
    double max_bias = Defaults::max_bias;
    double min_value = Defaults::min_value;

public:
    /**
     * @param q Quantile to use for finding the smallest/largest size factors.
     * Setting this to zero will use the observed minimum and maximum, though this is usually too extreme in practice.
     * The default is to take the 5th and 95th percentile, yielding a range that is still representative of most cells.
     *
     * @return Reference to this `ChoosePseudoCount` class.
     */
    ChoosePseudoCount& set_quantile(double q = Defaults::quantile) {
        quantile = q;
        return *this;
    }

    /**
     * @param b Acceptable upper bound on the log-transformation bias.
     *
     * @return Reference to this `ChoosePseudoCount` class.
     */
    ChoosePseudoCount& set_max_bias(double b = Defaults::max_bias) {
        max_bias = b;
        return *this;
    }

    /**
     * @param v Minimum value for the pseudo-count returned by `run()`.
     * Defaults to 1 to stabilize near-zero normalized expression values, otherwise these manifest as avoid large negative values.
     *
     * @return Reference to this `ChoosePseudoCount` class.
     */
    ChoosePseudoCount& set_min_value(double v = Defaults::min_value) {
        min_value = v;
        return *this;
    }

public:
    /**
     * @cond
     */
    static double find_quantile(double quantile, size_t n, double* ptr) {
        double raw = static_cast<double>(n - 1) * quantile;
        size_t index = std::ceil(raw);
        std::nth_element(ptr, ptr + index, ptr + n);
        double upper = *(ptr + index);
        std::nth_element(ptr, ptr + index - 1, ptr + index);
        double lower = *(ptr + index - 1);
        return lower * (index - raw) + upper * (raw - (index - 1));
    }
    /**
     * @endcond
     */

public:
    /**
     * @param n Number of size factors.
     * @param[in] size_factors Pointer to an array of size factors of length `n`.
     * Values should be positive, and all non-positive values are ignored.
     * @param buffer Pointer to an array of length `n`, to be used as a workspace.
     *
     * @return The suggested pseudo-count to control the log-transformation-induced bias below the specified threshold.
     */
    double run(size_t n, const double* size_factors, double* buffer) const {
        if (n <= 1) {
            return min_value;
        }

        // Avoid problems with zeros.
        size_t counter = 0;
        for (size_t i = 0; i < n; ++i) {
            if (size_factors[i] > 0) {
                buffer[counter] = size_factors[i];
                ++counter;
            }
        }
        n = counter;

        if (n <= 1) {
            return min_value;
        }

        double lower_sf, upper_sf;
        if (quantile == 0) {
            lower_sf = *std::min_element(buffer, buffer + n);
            upper_sf = *std::max_element(buffer, buffer + n);
        } else {
            lower_sf = find_quantile(quantile, n, buffer);
            upper_sf = find_quantile(1 - quantile, n, buffer);
        }

        // Very confusing formulation in Equation 3, but whatever.
        double pseudo_count = (1 / lower_sf - 1 / upper_sf) / (8 * max_bias);

        return std::max(min_value, pseudo_count);
    }

    /**
     * @param n Number of size factors.
     * @param[in] size_factors Pointer to an array of size factors of length `n`.
     * Values should be positive, and all non-positive values are ignored.
     *
     * @return The suggested pseudo-count to control the log-transformation-induced bias below the specified threshold.
     */
    double run(size_t n, const double* size_factors) const {
        std::vector<double> buffer(n);
        return run(n, size_factors, buffer.data());
    }
};

}

#endif
