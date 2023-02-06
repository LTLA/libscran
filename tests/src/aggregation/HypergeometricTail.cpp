#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "scran/aggregation/HypergeometricTail.hpp"

TEST(HypergeometricTail, Basic) {
    scran::HypergeometricTail hyper;

    // Checking special cases (remember that upper tail is the default).
    EXPECT_EQ(hyper.run(0, 20, 30, 20), 1);
    EXPECT_EQ(hyper.run(20, 20, 30, 20), 0);

    // Checking for consistency with R. We use a fairly generous tolerance
    // as our stirling approximation is not as accurate as R's.

    {
        // > phyper(3, 55, 101, 23, lower.tail=FALSE)
        compare_almost_equal(0.9887246, hyper.run(3, 55, 101, 23), /* tol=*/ 0.01);

        // > phyper(5, 20, 30, 20, lower.tail=FALSE)
        compare_almost_equal(0.9307521, hyper.run(5, 20, 30, 20), /* tol=*/ 0.01);

        // > phyper(10, 21, 14, 18, lower.tail=FALSE)
        compare_almost_equal(0.5815154, hyper.run(10, 21, 14, 18), /* tol=*/ 0.05);

        // > phyper(20, 33, 8, 25, lower.tail=FALSE)
        compare_almost_equal(0.3743729, hyper.run(20, 33, 8, 25), /* tol=*/ 0.01);
    }

    // Repeating with the lower tail.
    hyper.set_upper_tail(false);
    EXPECT_EQ(hyper.run(0, 20, 30, 20), 0);
    EXPECT_EQ(hyper.run(20, 20, 30, 20), 1);

    {
        // > phyper(3, 55, 101, 23)
        compare_almost_equal(0.01127538, hyper.run(3, 55, 101, 23), /* tol=*/ 0.01);

        // > phyper(5, 20, 30, 20)
        compare_almost_equal(0.06924787, hyper.run(5, 20, 30, 20), /* tol=*/ 0.01);

        // > phyper(10, 21, 14, 18)
        compare_almost_equal(0.4184846, hyper.run(10, 21, 14, 18), /* tol=*/ 0.05);

        // > phyper(20, 33, 8, 25)
        compare_almost_equal(0.6256271, hyper.run(20, 33, 8, 25), /* tol=*/ 0.01);
    }
}

TEST(HypergeometricTail, Logged) {
    scran::HypergeometricTail hyper;
    auto lhyper = hyper;
    lhyper.set_log(true);

    EXPECT_EQ(lhyper.run(0, 20, 30, 20), 0);
    EXPECT_TRUE(std::isinf(lhyper.run(20, 20, 30, 20)));
    compare_almost_equal(std::log(hyper.run(3, 55, 101, 23)), lhyper.run(3, 55, 101, 23));
    compare_almost_equal(std::log(hyper.run(8, 19, 31, 14)), lhyper.run(8, 19, 31, 14));

    hyper.set_upper_tail(false);
    lhyper.set_upper_tail(false);

    EXPECT_TRUE(std::isinf(lhyper.run(0, 20, 30, 20)));
    EXPECT_EQ(lhyper.run(20, 20, 30, 20), 0);
    compare_almost_equal(std::log(hyper.run(3, 55, 101, 23)), lhyper.run(3, 55, 101, 23));
    compare_almost_equal(std::log(hyper.run(8, 19, 31, 14)), lhyper.run(8, 19, 31, 14));
}

TEST(HypergeometricTail, Cached) {
    int num_whites = 23;
    int num_blacks = 19;
    int num_draws = 13;

    scran::HypergeometricTail hyper;
    auto cache = hyper.new_cache();

    // Initializing.
    EXPECT_EQ(hyper.run(2, num_whites, num_blacks, num_draws), hyper.run(2, num_whites, num_blacks, num_draws, cache));
    EXPECT_EQ(cache.num_white, num_whites); 
    EXPECT_EQ(cache.cumulative.size(), 3);
    auto last_cum = cache.cumulative.size();
    auto last_scale = cache.scale;

    EXPECT_EQ(hyper.run(5, num_whites, num_blacks, num_draws), hyper.run(5, num_whites, num_blacks, num_draws, cache));
    EXPECT_TRUE(cache.cumulative.size() > last_cum);
    EXPECT_EQ(cache.scale, last_scale);
    last_cum = cache.cumulative.size();

    EXPECT_EQ(hyper.run(8, num_whites, num_blacks, num_draws), hyper.run(8, num_whites, num_blacks, num_draws, cache));
    EXPECT_EQ(cache.num_white, num_blacks); // tails are flipped, so we swap.
    EXPECT_NE(cache.scale, last_scale);
    last_scale = cache.scale;
    last_cum = cache.cumulative.size();

    EXPECT_EQ(hyper.run(12, num_whites, num_blacks, num_draws), hyper.run(12, num_whites, num_blacks, num_draws, cache));
    EXPECT_EQ(cache.scale, last_scale);

    // Cache breaking by going in decreasing order.
    EXPECT_EQ(hyper.run(7, num_whites, num_blacks, num_draws), hyper.run(7, num_whites, num_blacks, num_draws, cache));

    // Cache breaking by changing the constants.
    EXPECT_EQ(hyper.run(8, num_whites + 5, num_blacks, num_draws), hyper.run(8, num_whites + 5, num_blacks, num_draws, cache));
}

