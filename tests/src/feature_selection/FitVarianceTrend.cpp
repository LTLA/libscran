#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/feature_selection/FitVarianceTrend.hpp"

#include "../utils/compare_almost_equal.h"

TEST(FitVarianceTrendTest, Basic) {
    std::vector<double> x { 0.86, 0.88, 0.98, 0.05, 0.69, 0.18, 0.2, 0.87, 0.8, 0.22, 0.54, 0.96, 0.45, 0.42, 0.53, 0.3, 0.84, 0.11, 0.74, 0.85 };
    std::vector<double> y (x);
    for (auto& y0 : y) { y0 *= 2; } // just to add some variety.

    scran::FitVarianceTrend ftv;
    auto output = ftv.set_transform(false).run(x.size(), x.data(), y.data());
    compare_almost_equal(output.fitted, y); // should be an exact fit for a straight line.

    // Again, with the transformation, no filtering.
    y = x;
    for (auto& y0 : y) { y0 *= y0 * y0 * y0; } 
    output = ftv.set_transform(true).set_filter(false).run(x.size(), x.data(), y.data());
    compare_almost_equal(output.fitted, y);
}

TEST(FitVarianceTrendTest, Extrapolation) {
    std::vector<double> x { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<double> y { 0, 0, 0, 0, 0, 6, 7, 8, 9, 10 };
    
    scran::FitVarianceTrend ftv;
    auto output = ftv.set_transform(false).set_minimum_mean(5.5).run(x.size(), x.data(), y.data());
    compare_almost_equal(output.fitted, x); // should be y = x, extrapolation to the filtered elements.
}

static std::vector<double> ftv_x { 
    0.38, 0.15, 0.59, 0.36, 0.83, 0.72, 0.56, 0.92, 0.72, 0.1, 0.2, 0.42, 0.12, 0.02, 
    0.75, 0.26, 0.99, 0.66, 0.73, 0.81, 0.45, 0.89, 0.37, 0.6, 0.85, 0.55, 0.33, 0.34, 
    0.56, 0.02, 0.33, 0.21, 0.18, 0.73, 0.67, 0.97, 0.13, 0.05, 0.66, 0.71, 0.44, 0.35, 
    0.34, 0.82, 0.54, 0.78, 0.21, 0.88, 0.01, 0.81
};

static std::vector<double> ftv_y { 
    0.78, 4.33, 0.72, 2.79, 2.16, 1.26, 2.23, 1.92, 0.59, 0.81, 1.68, 0.27, 1.28, 0.66, 
    0.36, 8.05, 0.69, 0.72, 0.91, 3.4, 0.39, 1.22, 0.75, 1.3, 0.57, 1, 1.12, 0.53, 0.87, 
    1.83, 0.81, 0.56, 0.81, 1.46, 0.34, 0.46, 0.54, 1.88, 1.55, 0.3, 0.44, 1.37, 1.4, 0.49, 
    1.32, 2.06, 0.4, 2.27, 1.52, 0.83
};

TEST(FitVarianceTrendTest, Residuals) {
    scran::FitVarianceTrend ftv;
    const auto& x = ftv_x;
    const auto& y = ftv_y;

    // Just repeating the residual calculation here, after transformation.
    auto output = ftv.run(x.size(), x.data(), y.data());
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ref[i] = y[i] - output.fitted[i];
    }
    EXPECT_EQ(output.residuals, ref);

    // And again, without transformation.
    output = ftv.set_transform(false).run(x.size(), x.data(), y.data());
    for (size_t i = 0; i < x.size(); ++i) {
        ref[i] = y[i] - output.fitted[i];
    }
    EXPECT_EQ(output.residuals, ref);
}

TEST(FitVarianceTrendTest, FixedMode) {
    scran::FitVarianceTrend ftv;

    const auto& x = ftv_x;
    const auto& y = ftv_y;
    auto output = ftv.run(x.size(), x.data(), y.data());

    ftv.set_use_fixed_width(true);
    ftv.set_minimum_window_count(10);
    ftv.set_fixed_width(0.2);
    auto foutput = ftv.run(x.size(), x.data(), y.data());

    EXPECT_NE(output.residuals, foutput.residuals);

    // They eventually converge when both all window widths are at their maximum;
    // either because of a large span, or because we need to get a minimum number of counts. 
    ftv.set_minimum_window_count();
    ftv.set_fixed_width();
    foutput = ftv.run(x.size(), x.data(), y.data());

    ftv.set_use_fixed_width();
    ftv.set_span(1);
    output = ftv.run(x.size(), x.data(), y.data());

    EXPECT_EQ(output.residuals, foutput.residuals);
}
