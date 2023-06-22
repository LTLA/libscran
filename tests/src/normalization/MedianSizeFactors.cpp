#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/tatami.hpp"
#include "scran/normalization/MedianSizeFactors.hpp"

#include <cmath>
#include <random>

class MedianSizeFactorsTester : public ::testing::Test {
protected:
    void initialize(std::vector<double>& contents, std::vector<double>& multiplier, size_t seed) {
        std::mt19937_64 rng(seed);
        for (size_t r = 0; r < NR; ++r) {
            contents[r] = rng() % 100;
        }

        multiplier[0] = 1;

        auto cIt = contents.begin() + NR;
        for (size_t c = 1; c < NC; ++c, cIt += NR) {
            double& mult = multiplier[c];
            mult = static_cast<double>(rng() % 100 + 1)/10; // force it to be positive with the +1.
            for (size_t r = 0; r < NR; ++r) {
                *(cIt + r) = contents[r] * mult;
            }
        }
    }

protected:
    size_t NR = 100, NC = 10;
};

TEST_F(MedianSizeFactorsTester, EqualToLibSize) {
    std::vector<double> contents(NR * NC);
    std::vector<double> multiplier(NC);
    initialize(contents, multiplier, 421);

    std::unique_ptr<tatami::NumericMatrix> mat(new tatami::DenseColumnMatrix<double>(NR, NC, contents));
    scran::MedianSizeFactors med;
    auto res = med.run_with_mean(mat.get());

    // All factors should be proportional to the library size if all profiles are scaled versions of each other.
    double ratio;
    int failures = 0;
    for (size_t i = 0; i < NC; ++i) {
        double r = multiplier[i] / res.factors[i];
        if (i == 0) {
            ratio = r;
        } else {
            failures += (std::abs(1 - r/ratio) > 0.00000001);
        }
    }
    EXPECT_TRUE(failures == 0);
}

TEST_F(MedianSizeFactorsTester, Magnitude) {
    std::vector<double> contents(NR * NC);
    std::vector<double> multiplier(NC);
    initialize(contents, multiplier, 422);

    std::unique_ptr<tatami::NumericMatrix> mat(new tatami::DenseColumnMatrix<double>(NR, NC, contents));

    // Without centering, we should just get the multipliers back, given that
    // we're normalizing against the first profile (which has a multiplier of 1).
    scran::MedianSizeFactors med;
    med.set_center(false);
    auto res = med.run(mat.get(), contents.data());

    int failures = 0;
    for (size_t i = 0; i < NC; ++i) {
        double r = multiplier[i] / res.factors[i];
        failures += (std::abs(1 - r) > 0.00000001);
    }
    EXPECT_TRUE(failures == 0);

    // Plus centering, the mean should now be 1.
    med.set_center(true);
    auto res2 = med.run(mat.get(), contents.data());
    double mean = std::accumulate(res2.factors.begin(), res2.factors.end(), 0.0) / res2.factors.size();
    EXPECT_FLOAT_EQ(mean, 1);
}

TEST_F(MedianSizeFactorsTester, PriorCount) {
    std::vector<double> contents(NR * NC);
    std::vector<double> multiplier(NC);
    initialize(contents, multiplier, 423);

    std::unique_ptr<tatami::NumericMatrix> mat(new tatami::DenseColumnMatrix<double>(NR, NC, contents));
    scran::MedianSizeFactors med;
    med.set_center(false);

    // Prior count of zero should just give back the multipliers.
    {
        med.set_prior_count(0);
        auto res = med.run(mat.get(), contents.data());

        int failures = 0;
        for (size_t i = 0; i < NC; ++i) {
            double r = multiplier[i] / res.factors[i];
            failures += (std::abs(1 - r) > 0.00000001);
        }
        EXPECT_TRUE(failures == 0);
    }

    // Very high prior count should still give back the multipliers,
    // given that the only difference is the library size.
    {
        med.set_prior_count(10000);
        auto res = med.run(mat.get(), contents.data());

        int failures = 0;
        for (size_t i = 0; i < NC; ++i) {
            double r = multiplier[i] / res.factors[i];
            failures += (std::abs(1 - r) > 0.00000001);
        }
        EXPECT_TRUE(failures == 0);
    }
}

TEST_F(MedianSizeFactorsTester, Actual) {
    std::vector<double> contents(NR * NC);
    std::vector<double> multiplier(NC);

    std::mt19937_64 rng(1000);
    auto cIt = contents.begin();
    for (size_t c = 0; c < NC; ++c) {
        double& mult = multiplier[c];
        mult = static_cast<double>(rng() % 100)/10;
        for (size_t r = 0; r < NR; ++r, ++cIt) {
            *cIt = mult * (r * 100 + rng() % 100);
        }
    }

    std::unique_ptr<tatami::NumericMatrix> mat(new tatami::DenseColumnMatrix<double>(NR, NC, contents));

    scran::MedianSizeFactors med;
    med.set_center(false);
    auto res = med.run(mat.get(), contents.data());
    EXPECT_FLOAT_EQ(res.factors[0], 1);

    // Expect decent similarity, but not identity, to the library sizes.
    {
        int failures = 0;
        bool all_equal = true;
        for (size_t i = 0; i < NC; ++i) {
            double expected = multiplier[i] / multiplier[0];

            // Should be close, but not equal.
            failures += (std::abs(1 - expected / res.factors[i]) > 0.01);
            if (std::abs(1 - expected / res.factors[i]) > 0.00000001) {
                all_equal = false;
            }
        }
        EXPECT_TRUE(failures == 0);
        EXPECT_FALSE(all_equal);
    }

    // Expect different results.
    {
        auto med2 = med;
        med2.set_prior_count(1000000);
        auto res2 = med2.run(mat.get(), contents.data());
        EXPECT_FLOAT_EQ(res2.factors[0], 1);

        int failures = 0;
        for (size_t i = 0; i < NC; ++i) {
            double ratio = res2.factors[i] / res.factors[i];
            failures += (std::abs(1 - ratio) > 0.0001);
        }
        EXPECT_TRUE(failures > 0);
    }

    // Same results in parallel.
    {
        auto med2 = med;
        med2.set_num_threads(3);
        auto res2 = med2.run(mat.get(), contents.data());
        EXPECT_EQ(res2.factors, res.factors);
    }
}

TEST_F(MedianSizeFactorsTester, CompositionBias) {
    std::vector<double> contents(NR * NC);
    std::vector<double> multiplier(NC);
    initialize(contents, multiplier, 423);

    // Introduce some huge DE to make some composition bias to every other sample.
    // We add it to a different gene in each sample to make life a bit interesting.
    for (size_t i = 1; i < NC; ++i) {
        contents[i * NR + (i - 1) % NR] = 100000 * i;
    }

    std::unique_ptr<tatami::NumericMatrix> mat(new tatami::DenseColumnMatrix<double>(NR, NC, contents));

    scran::MedianSizeFactors med;
    med.set_center(false).set_prior_count(0); // for a simpler comparison of equality.
    auto res = med.run(mat.get(), contents.data());

    // We expect to get back the multipliers as the DE is ignored.
    int failures = 0;
    for (size_t i = 0; i < NC; ++i) {
        double r = multiplier[i] / res.factors[i];
        failures += (std::abs(1 - r) > 0.00000001);
    }
    EXPECT_TRUE(failures == 0);
}

TEST_F(MedianSizeFactorsTester, ZeroHandling) {
    std::vector<double> zcontents(NR * NC);
    std::vector<double> fcontents((NR + 1) * NC);

    std::mt19937_64 rng(1001);
    {
        auto cIt = fcontents.begin();
        auto zIt = zcontents.begin();
        for (size_t c = 0; c < NC; ++c) {
            double mult = static_cast<double>(rng() % 100 + 1)/10; // +1 to force non-zero multipliers.
            for (size_t r = 0; r < NR; ++r, ++cIt, ++zIt) {
                *cIt = mult * (rng() % 100) + 1; // everything is non-zero here, to avoid accidental undefined values...
                *zIt = *cIt;
            }
            ++cIt; // except for the last row, which is all-zero.
        }
    }
    
    // Leaving the last reference entry zero.
    // Also making the first reference entry zero.
    // Otherwise everything in between is non-zero.
    std::vector<double> ref(NR + 1);
    for (size_t r = 1; r < NR; ++r) {
        ref[r] = rng() % 100 + 1; 
    }

    std::unique_ptr<tatami::NumericMatrix> zmat(new tatami::DenseColumnMatrix<double>(NR, NC, zcontents));
    std::unique_ptr<tatami::NumericMatrix> fmat(new tatami::DenseColumnMatrix<double>(NR + 1, NC, fcontents));
   
    // Factors should be the same as if the last all-zero
    // row was never there, as indefinite values are just ignored.
    scran::MedianSizeFactors med;
    auto zres = med.run(zmat.get(), ref.data());
    auto fres = med.run(fmat.get(), ref.data());
    EXPECT_EQ(zres.factors, fres.factors);

    // Mimicking the infinity generation by replacing them with very big things.
    {
        ref[0] = 1;
        for (size_t c = 0; c < NC; ++c) {
            fcontents[c * (NR + 1)] = 10000000;
        }

        std::unique_ptr<tatami::NumericMatrix> fmat2(new tatami::DenseColumnMatrix<double>(NR + 1, NC, fcontents));

        scran::MedianSizeFactors med;
        med.set_prior_count(0); // need to set this, otherwise the relative shrinkage changes.

        auto fres = med.run(fmat.get(), ref.data());
        auto fres2 = med.run(fmat2.get(), ref.data());
        EXPECT_EQ(fres.factors, fres2.factors);
    }
}

