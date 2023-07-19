#include <gtest/gtest.h>
#include "macros.h"
#include "scran/utils/average_vectors.hpp"
#include <vector>

static int full_of_nans(const std::vector<double>& vec) {
    int nan_count = 0;
    for (auto x : vec) {
        nan_count += std::isnan(x);
    }
    return nan_count;
}

TEST(AverageVectors, Simple) {
    std::vector<std::vector<double> > stuff {
        std::vector<double>{1, 2, 3, 4, 5},
        std::vector<double>{2, 1, 2, 6, 8},
        std::vector<double>{3, 7, 3, 3, 2}
    };

    auto out = scran::average_vectors(5, std::vector<double*>{stuff[0].data(), stuff[1].data(), stuff[2].data()});
    std::vector<double> ref {2, 10.0/3, 8.0/3, 13/3.0, 5.0};
    EXPECT_EQ(out, ref);

    // Optimization when there's just one, or none.
    auto out_opt = scran::average_vectors(5, std::vector<double*>{stuff[0].data()});
    EXPECT_EQ(out_opt, stuff[0]);

    auto out_opt2 = scran::average_vectors(5, std::vector<double*>{});
    EXPECT_EQ(full_of_nans(out_opt2), out_opt2.size());
}

TEST(AverageVectors, Missings) {
    double n = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double> > stuff {
        std::vector<double>{1, 2, n, 4, n},
        std::vector<double>{2, n, 2, n, n},
        std::vector<double>{3, 7, 3, n, n}
    };

    auto out = scran::average_vectors(5, std::vector<double*>{stuff[0].data(), stuff[1].data(), stuff[2].data()});
    std::vector<double> ref {2, 9.0/2, 5.0/2, 4.0};
    EXPECT_EQ(std::vector<double>(out.begin(), out.begin() + 4), ref);
    EXPECT_TRUE(std::isnan(out[4]));
}

TEST(AverageVectors, Weighted) {
    std::vector<std::vector<double> > stuff {
        std::vector<double>{1, 2, 3, 4, 5},
        std::vector<double>{2, 1, 2, 6, 8},
        std::vector<double>{3, 7, 3, 3, 2}
    };

    // Using simple binary weights for testing purposes.
    std::vector<double> weights { 1, 0, 0 };
    std::vector<double*> ptrs{stuff[0].data(), stuff[1].data(), stuff[2].data()};
    auto out = scran::average_vectors_weighted(5, ptrs, weights.data());
    EXPECT_EQ(out, stuff[0]);

    std::vector<double> weights1 { 1, 0, 1 };
    auto out1 = scran::average_vectors_weighted(5, ptrs, weights1.data());
    auto ref1 = scran::average_vectors(5, std::vector<double*>{stuff[0].data(), stuff[2].data()});
    EXPECT_EQ(out1, ref1);

    std::vector<double> weights2 { 1, 1, 1 };
    auto out2 = scran::average_vectors_weighted(5, ptrs, weights2.data());
    auto ref2 = scran::average_vectors(5, ptrs);
    EXPECT_EQ(out2, ref2);

    // Now using some more complex weights.
    std::vector<double> weights3 { 0.5, 2, 1.5 };
    auto out3 = scran::average_vectors_weighted(5, ptrs, weights3.data());
    std::vector<double> ref3{ 2.250, 3.375, 2.500, 4.625, 5.375 };
    EXPECT_EQ(out3, ref3);

    // Optimizations.
    auto out_opt = scran::average_vectors_weighted(5, std::vector<double*>{stuff[0].data()}, weights1.data());
    EXPECT_EQ(out_opt, stuff[0]);

    auto out_opt2 = scran::average_vectors_weighted(5, std::vector<double*>{}, weights1.data());
    EXPECT_EQ(full_of_nans(out_opt2), out_opt2.size());

    double empty_weight = 0;
    auto out_opt3 = scran::average_vectors_weighted(5, std::vector<double*>{ stuff[1].data() }, &empty_weight);
    EXPECT_EQ(full_of_nans(out_opt3), out_opt3.size());

    std::vector<double> empty_weights(3);
    auto out_opt4 = scran::average_vectors_weighted(5, std::vector<double*>{stuff[0].data(), stuff[1].data(), stuff[2].data()}, empty_weights.data());
    EXPECT_EQ(full_of_nans(out_opt4), out_opt4.size());
}
