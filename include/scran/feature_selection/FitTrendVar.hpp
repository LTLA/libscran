#ifndef SCRAN_FIT_TREND_VAR_H
#define SCRAN_FIT_TREND_VAR_H

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include "WeightedLowess/WeightedLowess.hpp"

/**
 * @file FitTrendVar.hpp
 *
 * @brief Fit a mean-variance trend to log-count data.
 */

namespace scran {

/**
 * @brief Fit a mean-variance trend to log-count data.
 *
 * Fit a trend to the per-feature variances against the means, both of which are computed from log-normalized expression data.
 * We use a LOWESS smoother in several steps:
 *
 * 1. Filter out low-abundance genes, to ensure the span of the smoother is not skewed by many low-abundance genes.
 * 2. Take the quarter-root of the variances, to squeeze the trend towards 1.
 * This makes the trend more "linear" to improve the performance of the LOWESS smoother;
 * it also reduces the chance of obtaining negative fitted values.
 * 3. Apply the LOWESS smoother to the quarter-root variances.
 * This is done using the implementation in the **WeightedLowess** library.
 * 4. Reverse the quarter-root transformation to obtain the fitted values for all non-low-abundance genes.
 * 5. Extrapolate linearly from the left-most fitted value to the origin to obtain fitted values for the previously filtered genes.
 * This is empirically justified by the observation that mean-variance trends of log-expression data are linear at very low abundances.
 */
class FitTrendVar {
public:
    /**
     * @brief Parameter defaults.
     */
    struct Defaults {
        /**
         * See `set_minimum_mean()` for details.
         */
        static constexpr double minimum_mean = 0.1;

        /**
         * See `set_filter()` for details.
         */
        static constexpr bool filter = true;

        /**
         * See `set_transform()` for details.
         */
        static constexpr bool transform = true;

        /**
         * See `set_span()` for details.
         */
        static constexpr double span = 0.3;
    };

public:
    /** 
     * Set the span for the LOWESS smoother.
     *
     * @param s Span for the smoother, as a proportion of the total number of points.
     *
     * @return A reference to this `FitTrendVar` object.
     */
    FitTrendVar& set_span(double s = Defaults::span) {
        span = s;
        return *this;
    }

    /**
     * Set the minimum mean log-expression, to use for filtering out low-abundance genes.
     *
     * @param m Minimum value for the mean log-expression.
     *
     * @return A reference to this `FitTrendVar` object.
     */
    FitTrendVar& set_minimum_mean(double m = Defaults::minimum_mean) {
        min_mean = m;
        return *this;
    }

    /**
     * Should any filtering be performed by log-expression?
     * This may need to be disabled if `FitTrendVar` is not being used on statistics computed from log-expression values.
     *
     * @param f Whether filtering should be performed.
     *
     * @return A reference to this `FitTrendVar` object.
     */
    FitTrendVar& set_filter(bool f = Defaults::filter) {
        filter = f;
        return *this;
    }

    /**
     * Should any transformation of the variances be performed prior to LOWESS smoothing.
     * This may need to be disabled if `FitTrendVar` is not being used on statistics computed from log-expression values.
     *
     * @param t Whether transformation should be performed.
     *
     * @return A reference to this `FitTrendVar` object.
     */
    FitTrendVar& set_transform(bool t = Defaults::transform) {
        transform = t;
        return *this;
    }

private:
    double span = Defaults::span;
    double min_mean = Defaults::minimum_mean;
    bool filter = Defaults::filter;
    bool transform = Defaults::transform;

    static double quad(double x) {
        return x*x*x*x;
    }

public:
    /**
     * Run the trend fitting on the means and variances across all features.
     *
     * @param n Number of features.
     * @param[in] mean Pointer to an array of length `n`, containing the means for all features.
     * @param[in] variance Pointer to an array of length `n`, containing the variances for all features.
     * @param[out] fitted Pointer to an array of length `n`, to store the fitted values.
     * @param[out] residuals Pointer to an array of length `n`, to store the residuals.
     * 
     * @return `fitted` and `residuals` are filled with the fitted values and residuals of the trend.
     */
    void run(size_t n, const double* mean, const double* variance, double* fitted, double* residuals) const {
        std::vector<double> xbuffer(n), ybuffer(n);

        size_t counter = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!filter || mean[i] >= min_mean) {
                xbuffer[counter] = mean[i];
                if (transform) {
                    ybuffer[counter] = std::pow(variance[i], 0.25); // Using the same quarter-root transform that limma::voom uses.
                } else {
                    ybuffer[counter] = variance[i];
                }
                ++counter;
            }
        }

        if (counter < 2) {
            throw std::runtime_error("not enough observations above the minimum mean");
        }

        // Determining the left edge. This needs to be done before
        // run_in_place, which mutates the input xbuffer.
        size_t left_index = std::min_element(xbuffer.begin(), xbuffer.begin() + counter) - xbuffer.begin();
        double left_x = xbuffer[left_index];

        WeightedLowess::WeightedLowess<> smoother;
        smoother.set_span(span);

        std::vector<double> fbuffer(counter), rbuffer(counter);
        smoother.run(counter, xbuffer.data(), ybuffer.data(), NULL, fbuffer.data(), rbuffer.data());

        // Identifying the left-most fitted value.
        double left_fitted = (transform ? quad(fbuffer[left_index]) : fbuffer[left_index]);

        counter = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!filter || mean[i] >= min_mean) {
                fitted[i] = (transform ? quad(fbuffer[counter]) : fbuffer[counter]);
                ++counter;
            } else {
                fitted[i] = mean[i] / left_x * left_fitted; // draw a y = x line to the origin from the left of the fitted trend.
            }
            residuals[i] = variance[i] - fitted[i];
        }
        return;
    }

public:
    /**
     * @brief Results of the trend fit.
     *
     * Meaningful instances of this object should generally be constructed by calling the `FitTrendVar::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t n) : fitted(n), residuals(n) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of features, containing fitted values from the trend.
         */
        std::vector<double> fitted;

        /**
         * Vector of length equal to the number of features, containing residuals from the trend.
         */
        std::vector<double> residuals;
    };

    /**
     * Run the trend fitting on the means and variances across all features.
     *
     * @param n Number of features.
     * @param[in] mean Pointer to an array of length `n`, containing the means for all features.
     * @param[in] variance Pointer to an array of length `n`, containing the variances for all features.
     * 
     * @return A `Results` object containing the fitted values and residuals of the trend.
     */
    Results run(size_t n, const double* mean, const double* variance) const {
        Results output(n);
        run(n, mean, variance, output.fitted.data(), output.residuals.data());
        return output;
    }
};

}

#endif
