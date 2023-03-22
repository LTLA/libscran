#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "scran/feature_set_enrichment/HypergeometricTail.hpp"

TEST(HypergeometricTail, Basic) {
    scran::HypergeometricTail hyper;

    // Checking for consistency with R. We use a fairly generous tolerance
    // as our stirling approximation is not as accurate as R's. Note that
    // the upper tail calculations involve subtracting 1 from the phyper call,
    // because R doesn't include the mass of 'x' in the upper tail.
    {
        // > phyper(3, 55, 101, 23, lower.tail=FALSE)
        compare_almost_equal(0.9887246, hyper.run(4, 55, 101, 23), /* tol=*/ 0.01);

        // > phyper(5, 20, 30, 20, lower.tail=FALSE)
        compare_almost_equal(0.9307521, hyper.run(6, 20, 30, 20), /* tol=*/ 0.01);

        // > phyper(10, 21, 14, 18, lower.tail=FALSE)
        compare_almost_equal(0.5815154, hyper.run(11, 21, 14, 18), /* tol=*/ 0.05);

        // > phyper(20, 33, 8, 25, lower.tail=FALSE)
        compare_almost_equal(0.3743729, hyper.run(21, 33, 8, 25), /* tol=*/ 0.01);

        // Check for correct behavior when num_black, num_white < num_drawn.
        {
            // > phyper(3, 5, 8, 10, lower.tail=FALSE)
            compare_almost_equal(0.6853147, hyper.run(4, 5, 8, 10), /* tol=*/ 0.05);

            // > phyper(13, 15, 18, 20, lower.tail=FALSE)
            compare_almost_equal(0.000500776, hyper.run(14, 15, 18, 20), /* tol=*/ 0.05);
        }

        // Check for correct boundary case behavior when num_black == num_drawn or num_white == num_drawn.
        {
            // > phyper(20, 33, 25, 25, lower.tail=FALSE)
            compare_almost_equal(0.0002843443, hyper.run(21, 33, 25, 25), /* tol=*/ 0.01);

            // > phyper(20, 33, 25, 33, lower.tail=FALSE)
            compare_almost_equal(0.1780022, hyper.run(21, 33, 25, 33), /* tol=*/ 0.01);

            // > phyper(20, 25, 25, 25, lower.tail=FALSE)
            compare_almost_equal(1.308459e-06, hyper.run(21, 25, 25, 25), /* tol=*/ 0.01);
        }
    }

    {
        hyper.set_upper_tail(false);

        // > phyper(3, 55, 101, 23)
        compare_almost_equal(0.01127538, hyper.run(3, 55, 101, 23), /* tol=*/ 0.01);

        // > phyper(5, 20, 30, 20)
        compare_almost_equal(0.06924787, hyper.run(5, 20, 30, 20), /* tol=*/ 0.01);

        // > phyper(10, 21, 14, 18)
        compare_almost_equal(0.4184846, hyper.run(10, 21, 14, 18), /* tol=*/ 0.05);

        // > phyper(20, 33, 8, 25)
        compare_almost_equal(0.6256271, hyper.run(20, 33, 8, 25), /* tol=*/ 0.01);

        // > phyper(20, 33, 25, 25)
        compare_almost_equal(0.9997157, hyper.run(20, 33, 25, 25), /* tol=*/ 0.01);
    }
}

TEST(HypergeometricTail, Logged) {
    scran::HypergeometricTail hyper;
    auto lhyper = hyper;
    lhyper.set_log(true);

    compare_almost_equal(std::log(hyper.run(3, 55, 101, 23)), lhyper.run(3, 55, 101, 23));
    compare_almost_equal(std::log(hyper.run(8, 19, 31, 14)), lhyper.run(8, 19, 31, 14));

    hyper.set_upper_tail(false);
    lhyper.set_upper_tail(false);

    compare_almost_equal(std::log(hyper.run(3, 55, 101, 23)), lhyper.run(3, 55, 101, 23));
    compare_almost_equal(std::log(hyper.run(8, 19, 31, 14)), lhyper.run(8, 19, 31, 14));
}

TEST(HypergeometricTail, EdgeCases) {
    {
        scran::HypergeometricTail hyper;

        // Checking special cases (remember that upper tail is the default).
        EXPECT_EQ(hyper.run(0, 20, 30, 20), 1);
        EXPECT_EQ(hyper.run(21, 20, 30, 20), 0);

        // However, upper tail requests do not hit a special case when num_drawn = drawn_white,
        // as the probability mass at drawn_white is computed here.
        compare_almost_equal(2.11066e-14, hyper.run(20, 20, 30, 20), /* tol=*/ 0.01);

        // The number of drawn white balls must be at least 11 here, because
        // there just aren't enough black balls; thus the probability of drawing
        // more balls is 100%. 
        EXPECT_EQ(hyper.run(10, 20, 9, 20), 1);
        EXPECT_EQ(hyper.run(11, 20, 9, 20), 1);
        EXPECT_TRUE(hyper.run(12, 20, 9, 20) < 1);

        // Repeating with the lower tail.
        hyper.set_upper_tail(false);
        EXPECT_EQ(hyper.run(0, 20, 30, 20), 0);
        EXPECT_EQ(hyper.run(20, 20, 30, 20), 1);
        EXPECT_EQ(hyper.run(10, 20, 9, 20), 0);
    }

    {
        scran::HypergeometricTail hyper;
        hyper.set_log(true);

        EXPECT_EQ(hyper.run(0, 20, 30, 20), 0);
        EXPECT_TRUE(std::isinf(hyper.run(21, 20, 30, 20)));

        hyper.set_upper_tail(false);
        EXPECT_TRUE(std::isinf(hyper.run(0, 20, 30, 20)));
        EXPECT_EQ(hyper.run(20, 20, 30, 20), 0);
    }
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
    EXPECT_TRUE(cache.cumulative.size() > 0);
    auto last_cum = cache.cumulative.size();
    auto last_scale = cache.scale;

    EXPECT_EQ(hyper.run(5, num_whites, num_blacks, num_draws), hyper.run(5, num_whites, num_blacks, num_draws, cache));
    EXPECT_TRUE(cache.cumulative.size() > last_cum);
    EXPECT_EQ(cache.scale, last_scale);
    last_cum = cache.cumulative.size();

    EXPECT_EQ(hyper.run(9, num_whites, num_blacks, num_draws), hyper.run(9, num_whites, num_blacks, num_draws, cache));
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

    // Checking that the cached method gives the same results when num_draws is greater than both num_blacks and num_whites.
    {
        auto cache = hyper.new_cache();
        int num_whites = 15;
        int num_blacks = 9;
        int num_draws = 20;

        EXPECT_EQ(hyper.run(11, num_whites, num_blacks, num_draws), hyper.run(11, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(12, num_whites, num_blacks, num_draws), hyper.run(12, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(15, num_whites, num_blacks, num_draws), hyper.run(15, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(10, num_whites, num_blacks, num_draws), hyper.run(10, num_whites, num_blacks, num_draws, cache));
    }

    // Checking that the cached method gives the same results when num_draws is equal to both num_blacks and num_whites.
    {
        auto cache = hyper.new_cache();
        int num_whites = 15;
        int num_blacks = 15;
        int num_draws = 15;

        EXPECT_EQ(hyper.run(11, num_whites, num_blacks, num_draws), hyper.run(11, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(12, num_whites, num_blacks, num_draws), hyper.run(12, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(15, num_whites, num_blacks, num_draws), hyper.run(15, num_whites, num_blacks, num_draws, cache));
        EXPECT_EQ(hyper.run(10, num_whites, num_blacks, num_draws), hyper.run(10, num_whites, num_blacks, num_draws, cache));
    }
}

