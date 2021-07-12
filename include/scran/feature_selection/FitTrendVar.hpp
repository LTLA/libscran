#ifndef SCRAN_FIT_TREND_VAR_H
#define SCRAN_FIT_TREND_VAR_H

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

    FitTrendVar& set_minimum_mean(double m = 0.1) {
        min_mean = m;
        return *this;
    }

    FitTrendVar& set_filter(bool f = true) {
        filter = f;
        return *this;
    }

    FitTrendVar& set_transform(bool t = true) {
        transform = t;
        return *this;
    }

public:
    static double quad(double x) {
        return x*x*x*x;
    }

    void run(size_t n, const double* mean, const double* variance, double* fitted, double* residuals) {
        xbuffer.resize(n);
        ybuffer.resize(n);

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

        fbuffer.resize(counter);
        rbuffer.resize(counter);
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
    double min_mean = 0.1;
    bool filter = true;
    bool transform = true;

    WeightedLowess::WeightedLowess smoother;
    std::vector<double> xbuffer, ybuffer, rbuffer, fbuffer;
};

}

#endif
