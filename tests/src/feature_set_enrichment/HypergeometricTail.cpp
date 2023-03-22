#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../utils/compare_almost_equal.h"
#include "scran/feature_set_enrichment/HypergeometricTail.hpp"

TEST(HypergeometricTail, LogFactorial) {
    EXPECT_EQ(0, scran::HypergeometricTail::lfactorial(0));

    std::vector<double> exact { 1 };
    for (int i = 1; i <= 12; ++i) {
        exact.push_back(exact.back() * i);
        EXPECT_EQ(std::log(exact.back()), scran::HypergeometricTail::lfactorial(i));
    }

    double sofar = std::log(exact.back());
    int counter = exact.size() - 1;
    for (int i = 100; i <= 1000; i += 50) {
        while (counter < i) {
            ++counter;
            sofar += std::log(counter);
        }
        compare_almost_equal(sofar, scran::HypergeometricTail::lfactorial(i));
    }
}

TEST(HypergeometricTail, Basic) {
    scran::HypergeometricTail hyper;

    // Checking for consistency with R. We use a fairly generous tolerance
    // as our stirling approximation is not as accurate as R's. Note that
    // the upper tail calculations involve subtracting 1 from the phyper call,
    // because R doesn't include the mass of 'x' in the upper tail.
    {
        // > phyper(3, 55, 101, 23, lower.tail=FALSE)
        compare_almost_equal(0.9887246, hyper.run(4, 55, 101, 23), /* tol=*/ 0.001);

        // > phyper(5, 20, 30, 20, lower.tail=FALSE)
        compare_almost_equal(0.9307521, hyper.run(6, 20, 30, 20), /* tol=*/ 0.001);

        // > phyper(10, 21, 14, 18, lower.tail=FALSE)
        compare_almost_equal(0.5815154, hyper.run(11, 21, 14, 18), /* tol=*/ 0.001);

        // > phyper(20, 33, 8, 25, lower.tail=FALSE)
        compare_almost_equal(0.3743729, hyper.run(21, 33, 8, 25), /* tol=*/ 0.001);

        // Check for correct behavior when num_black, num_white < num_drawn.
        {
            // > phyper(3, 5, 8, 10, lower.tail=FALSE)
            compare_almost_equal(0.6853147, hyper.run(4, 5, 8, 10), /* tol=*/ 0.001);

            // > phyper(13, 15, 18, 20, lower.tail=FALSE)
            compare_almost_equal(0.000500776, hyper.run(14, 15, 18, 20), /* tol=*/ 0.001);
        }

        // Check for correct boundary case behavior when num_black == num_drawn or num_white == num_drawn.
        {
            // > phyper(20, 33, 25, 25, lower.tail=FALSE)
            compare_almost_equal(0.0002843443, hyper.run(21, 33, 25, 25), /* tol=*/ 0.001);

            // > phyper(20, 33, 25, 33, lower.tail=FALSE)
            compare_almost_equal(0.1780022, hyper.run(21, 33, 25, 33), /* tol=*/ 0.001);

            // > phyper(20, 25, 25, 25, lower.tail=FALSE)
            compare_almost_equal(1.308459e-06, hyper.run(21, 25, 25, 25), /* tol=*/ 0.001);
        }
    }

    {
        hyper.set_upper_tail(false);

        // > phyper(3, 55, 101, 23)
        compare_almost_equal(0.01127538, hyper.run(3, 55, 101, 23), /* tol=*/ 0.001);

        // > phyper(5, 20, 30, 20)
        compare_almost_equal(0.06924787, hyper.run(5, 20, 30, 20), /* tol=*/ 0.001);

        // > phyper(10, 21, 14, 18)
        compare_almost_equal(0.4184846, hyper.run(10, 21, 14, 18), /* tol=*/ 0.001);

        // > phyper(20, 33, 8, 25)
        compare_almost_equal(0.6256271, hyper.run(20, 33, 8, 25), /* tol=*/ 0.001);

        // > phyper(20, 33, 25, 25)
        compare_almost_equal(0.9997157, hyper.run(20, 33, 25, 25), /* tol=*/ 0.001);
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
