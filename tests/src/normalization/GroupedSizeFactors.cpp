#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/data.h"
#include "../utils/compare_vectors.h"

#include "tatami/tatami.hpp"
#include "scran/normalization/GroupedSizeFactors.hpp"

#include <cmath>
#include <random>

TEST(GroupedSizeFactors, LibSize) {
    size_t NR = 100, NC = 10;

    // Just returns library size factors when there's one group.
    std::vector<double> contents(NR * NC);
    {
        std::mt19937_64 rng(1000);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            double mult = static_cast<double>(rng() % 100 + 1)/10; // true scaling factor.
            for (size_t r = 0; r < NR; ++r, ++cIt) {
                *cIt = mult * (r * 100 + rng() % 100); // increasing gene abundance.
            }
        }
    }
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    std::vector<int> groups(NC);
    auto res = grouper.run(&mat, groups.data());

    auto sums = tatami::column_sums(&mat);
    scran::CenterSizeFactors center;
    center.run(sums.size(), sums.data());

    int diffs = 0;
    for (size_t c = 0; c < NC; ++c) {
        diffs += (std::abs(1 - sums[c] / res.factors[c]) > 0.00000001);
    }
    EXPECT_TRUE(diffs == 0);
}

TEST(GroupedSizeFactors, ByGroupLibSize) {
    size_t NR = 1000, NC = 100;
    size_t ngroups = 3;

    std::vector<double> contents(NR * NC);
    std::vector<int> groups(NC);
    {
        std::mt19937_64 rng(1001);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            groups[c] = c % ngroups;
            int scaling = groups[c] + 1; // group-specific scaling.
            for (size_t r = 0; r < NR; ++r, ++cIt) {
                *cIt = scaling * (r * 100 + rng() % 100); // increasing gene abundance.
            }
        }
    }
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    auto res = grouper.run(&mat, groups.data());

    // Averages should match the groups.
    std::vector<double> averages(ngroups);
    std::vector<int> frequency(ngroups);
    for (size_t i = 0; i < res.factors.size(); ++i) {
        averages[groups[i]] += res.factors[i];
        ++frequency[groups[i]];
    }

    for (size_t j = 0; j < ngroups; ++j) {
        averages[j] /= frequency[j];
        if (j) {
            double ratio = averages[j] / averages[0];
            EXPECT_FALSE(std::abs(1 - ratio / (j + 1)) > 0.001);
        }
    }
}

TEST(GroupedSizeFactors, ByGroupComposition) {
    size_t NR = 1000, NC = 100;
    size_t ngroups = 3;

    std::vector<double> contents(NR * NC);
    std::vector<int> groups(NC);
    {
        std::mt19937_64 rng(1002);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            groups[c] = c % ngroups;
            for (size_t r = 0; r < NR; ++r, ++cIt) {
                if (groups[c] == 1 && r == 0) { // injecting strong DE into each group.
                    *cIt = 10000;
                } else if (groups[c] == 2 && r == 1) {
                    *cIt = 1000000;
                } else {
                    *cIt = r + rng() % 10;
                }
            }
        }
    }
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    auto res = grouper.run(&mat, groups.data());

    // Averages should be close to 1, once the composition bias is ignored.
    std::vector<double> averages(ngroups);
    std::vector<int> frequency(ngroups);
    for (size_t i = 0; i < res.factors.size(); ++i) {
        averages[groups[i]] += res.factors[i];
        ++frequency[groups[i]];
    }

    for (size_t j = 0; j < ngroups; ++j) {
        averages[j] /= frequency[j];
        if (j) {
            double ratio = averages[j] / averages[0];
            EXPECT_FALSE(std::abs(1 - ratio) > 0.001);
        }
    }

    // Same results in parallel.
    {
        scran::GroupedSizeFactors grouper2;
        grouper2.set_num_threads(3);
        auto res2 = grouper2.run(&mat, groups.data());
        EXPECT_EQ(res.factors, res2.factors);
    }
}

TEST(GroupedSizeFactors, Reference) {
    size_t NR = 100, NC = 10;
    size_t ngroups = 3;

    std::vector<double> contents(NR * NC);
    std::vector<int> groups(NC);
    {
        std::mt19937_64 rng(1004);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            groups[c] = c % ngroups;
            for (size_t r = 0; r < NR; ++r, ++cIt) {
                *cIt = (r * 100 + rng() % 100); // increasing gene abundance.
            }
        }
    }
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    // Make sure we get different results if we change the reference.
    scran::GroupedSizeFactors grouper;
    auto res0 = grouper.run(&mat, groups.data(), 0);
    auto res1 = grouper.run(&mat, groups.data(), 1);

    bool is_diff = false;
    for (size_t j = 0; j < ngroups; ++j) {
        if (std::abs(1 - res0.factors[j] / res1.factors[j]) > 0.0001) {
            is_diff = true;
        }
    }
    EXPECT_TRUE(is_diff);
}

TEST(GroupedSizeFactors, ZeroHandler) {
    size_t NR = 100, NC = 10;
    size_t ngroups = 3;

    std::vector<double> contents(NR * NC);
    std::vector<int> groups(NC);
    {
        std::mt19937_64 rng(1005);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            groups[c] = c % ngroups;
            if (groups[c] == 2) {
                for (size_t r = 0; r < NR; ++r, ++cIt) {
                    *cIt = (r * 100 + rng() % 100); // increasing gene abundance.
                }
            } else {
                *cIt = rng() % 200; // guarantee positive library sizes. 
                cIt += NR;
            }
        }
    }

    auto copy = contents;
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    grouper.set_prior_count(0); // no protection from size factors of zero in the pseudo-cells.
    EXPECT_ANY_THROW({
        try {
            grouper.run(&mat, groups.data(), 2);
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("positive") != std::string::npos);
            throw;
        }
    });

    grouper.set_handle_zeros(true);
    auto res0 = grouper.run(&mat, groups.data(), 2);
    int any_zero = 0;
    for (auto x : res0.factors) {
        any_zero += (x == 0);
    }
    EXPECT_TRUE(any_zero == 0);
}

TEST(GroupedSizeFactors, InfiniteHandler) {
    size_t NR = 100, NC = 10;
    size_t ngroups = 3;

    std::vector<double> contents(NR * NC);
    std::vector<int> groups(NC);
    {
        std::mt19937_64 rng(1005);
        auto cIt = contents.begin();
        for (size_t c = 0; c < NC; ++c) {
            groups[c] = c % ngroups;
            if (groups[c] == 2) {
                for (size_t r = 0; r < NR; ++r, ++cIt) {
                    *cIt = (r * 100 + rng() % 100); // increasing gene abundance.
                }
            } else {
                *cIt = rng() % 200; // guarantee positive library sizes. 
                cIt += NR;
            }
        }
    }

    auto copy = contents;
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    EXPECT_ANY_THROW({
        try {
            grouper.run(&mat, groups.data(), 0);
        } catch(std::exception& e) {
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("finite") != std::string::npos);
            throw;
        }
    });

    grouper.set_handle_non_finite(true);
    auto res0 = grouper.run(&mat, groups.data(), 0);
    int all_finite = 0;
    for (auto x : res0.factors) {
        all_finite += std::isfinite(x);
    }
    EXPECT_TRUE(all_finite == mat.ncol());
}

TEST(GroupedSizeFactors, Empty) { // handles all-zero matrices gracefully.
    size_t NR = 100, NC = 10;

    std::vector<double> contents(NR * NC);
    tatami::DenseColumnMatrix<double> mat(NR, NC, contents);

    scran::GroupedSizeFactors grouper;
    grouper.set_handle_non_finite(true).set_handle_zeros(true);

    std::vector<int> groups(NC);
    auto res0 = grouper.run(&mat, groups.data());
    EXPECT_EQ(res0.factors, std::vector<double>(NC)); 
}

