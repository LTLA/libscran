#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "scran/normalization/SanitizeSizeFactors.hpp"

#include <cmath>

TEST(SanitizeSizeFactorsTest, Error) {
    std::vector<double> sf { 1, 2, 3, 4, 5 };

    auto copy = sf;
    scran::SanitizeSizeFactors san;
    san.run(copy.size(), copy.data());
    EXPECT_EQ(copy, sf);

    sf[0] = 0;
    EXPECT_ANY_THROW({
        try {
            san.run(sf.size(), sf.data());
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("zero") != std::string::npos);
            throw;
        }
    });

    sf[0] = -1;
    EXPECT_ANY_THROW({
        try {
            san.run(sf.size(), sf.data());
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("negative") != std::string::npos);
            throw;
        }
    });

    sf[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_ANY_THROW({
        try {
            san.run(sf.size(), sf.data());
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("NaN") != std::string::npos);
            throw;
        }
    });

    sf[0] = std::numeric_limits<double>::infinity();
    EXPECT_ANY_THROW({
        try {
            san.run(sf.size(), sf.data());
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("infinite") != std::string::npos);
            throw;
        }
    });
}

TEST(SanitizeSizeFactorsTest, Ignored) {
    std::vector<double> sf { 0, -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() };

    auto copy = sf;
    scran::SanitizeSizeFactors san;
    san.set_handle_non_positive(scran::SanitizeSizeFactors::HandlerAction::IGNORE);
    san.set_handle_non_finite(scran::SanitizeSizeFactors::HandlerAction::IGNORE);
    san.run(copy.size(), copy.data());

    EXPECT_TRUE(std::isnan(copy[2]));
    copy[2] = 0;
    sf[2] = 0;
    EXPECT_EQ(copy, sf);
}

TEST(SanitizeSizeFactorsTest, SmallestReplacement) {
    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.1, 0.2, 0.3, 0.4, 0.5 };
        san.set_handle_zero(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);

        auto copy = sf;
        copy[1] = 0;
        san.run(copy.size(), copy.data());

        EXPECT_EQ(copy[1], 0.1);
        auto ref = sf;
        ref[1] = 0.1;
        EXPECT_EQ(copy, ref);
    }

    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.5, 0.2, 0.3, 0.4, 0.01 };
        san.set_handle_negative(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);

        auto copy = sf;
        copy[2] = -1;
        san.run(copy.size(), copy.data());

        EXPECT_EQ(copy[2], 0.01);
        auto ref = sf;
        ref[2] = 0.01;
        EXPECT_EQ(copy, ref);
    }

    // Works when the smallest value is already computed.
    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.2, 0.5, 0.04, 0.1, 0.3 };
        san.set_handle_non_positive(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);

        auto copy = sf;
        copy[0] = 0;
        copy[4] = -1;
        san.run(copy.size(), copy.data());

        EXPECT_EQ(copy[0], 0.04);
        EXPECT_EQ(copy[4], 0.04);
        auto ref = sf;
        ref[0] = 0.04;
        ref[4] = 0.04;
        EXPECT_EQ(copy, ref);
    }

    // Still works in the presence of non-finite values.
    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.2, std::numeric_limits<double>::quiet_NaN(), 0.5, 0.03, 0.4, std::numeric_limits<double>::infinity(), 0.3 };
        san.set_handle_non_positive(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);
        san.set_handle_non_finite(scran::SanitizeSizeFactors::HandlerAction::IGNORE);

        auto copy = sf;
        copy[0] = 0;
        copy.back() = -1;
        san.run(copy.size(), copy.data());

        EXPECT_EQ(copy[0], 0.03);
        EXPECT_EQ(copy.back(), 0.03);
    }
}

TEST(SanitizeSizeFactorsTest, LargestReplacement) {
    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.1, 0.2, 0.3, 0.4, 0.5 };
        san.set_handle_infinite(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);

        auto copy = sf;
        copy[1] = std::numeric_limits<double>::infinity();
        san.run(copy.size(), copy.data());

        EXPECT_EQ(copy[1], 0.5);
        auto ref = sf;
        ref[1] = 0.5;
        EXPECT_EQ(copy, ref);
    }

    // Still works in the presence of missing values.
    {
        scran::SanitizeSizeFactors san;
        std::vector<double> sf { 0.1, 0.2, 0.3, 0.4, 0.5, std::numeric_limits<double>::quiet_NaN() };
        san.set_handle_infinite(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);
        san.set_handle_nan(scran::SanitizeSizeFactors::HandlerAction::IGNORE);

        auto copy = sf;
        copy[2] = std::numeric_limits<double>::infinity();
        san.run(copy.size(), copy.data());
        EXPECT_EQ(copy[2], 0.5);
    }
}

TEST(SanitizeSizeFactorsTest, Missings) {
    scran::SanitizeSizeFactors san;
    std::vector<double> sf { 0.1, 0.2, 0.3, 0.4, 0.5, std::numeric_limits<double>::quiet_NaN() };
    san.set_handle_nan(scran::SanitizeSizeFactors::HandlerAction::SANITIZE);

    auto copy = sf;
    san.run(copy.size(), copy.data());
    EXPECT_EQ(copy.back(), 1);
    auto ref = sf;
    ref.back() = 1;
    EXPECT_EQ(copy, ref);
}

