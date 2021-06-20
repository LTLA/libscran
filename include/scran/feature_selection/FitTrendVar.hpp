#ifndef FIT_TREND_VAR_H
#define FIT_TREND_VAR_H

#include <algorithm>
#include <vector>
#include "WeightedLowess/WeightedLowess.hpp"

namespace scran {

class FitTrendVar {
public:
    FitTrendVar& set_span(double s) {
        smoother.set_span(s);
        return *this;
    }
    
    FitTrendVar& set_span() {
        smoother.set_span();
        return *this;
    }

public:
    void run(size_t n, const double* mean, const double* variance, double* fitted, double* residuals) {
        xbuffer.resize(n);
        std::copy(mean, mean + n, xbuffer.begin());

        // Using the same quarter-root transform that limma::voom uses.
        ybuffer.resize(n);
        std::copy(variance, variance + n, ybuffer.begin());
        for (auto& Y : ybuffer) {
            Y = std::pow(Y, 0.25);
        }

        rbuffer.resize(n);
        smoother.run_in_place(n, xbuffer.data(), ybuffer.data(), NULL, fitted, residuals, rbuffer.data());

        for (size_t i = 0; i < n; ++i) {
            double val = fitted[i];
            fitted[i] = val * val * val * val;
            residuals[i] = variance[i] - fitted[i];
        }
        return;
    }

public:
    struct Results {
        Results(size_t n) : fitted(n), residuals(n) {}
        std::vector<double> fitted, residuals;
    };

    Results run(size_t n, const double* mean, const double* variance) {
        Results output(n);
        run(n, mean, variance, output.fitted.data(), output.residuals.data());
        return output;
    }

private:
    WeightedLowess::WeightedLowess smoother;
    std::vector<double> xbuffer, ybuffer, rbuffer;
};

}

#endif
