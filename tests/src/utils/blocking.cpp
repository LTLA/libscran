#include <gtest/gtest.h>
#include "macros.h"
#include "scran/utils/blocking.hpp"
#include <vector>

TEST(Blocking, CountIds) {
    {
        std::vector<int> x { 0, 5, 2, 1, 3 };
        EXPECT_EQ(scran::count_ids(x.size(), x.data()), 6);
    }

    // Correctly casts the type before increment.
    {
        std::vector<unsigned char> x { 255, 0, 5, 2, 3};
        EXPECT_EQ(scran::count_ids(x.size(), x.data()), 256);
    }

    EXPECT_EQ(scran::count_ids(0, static_cast<int*>(NULL)), 0);
}

TEST(Blocking, TabulateIds) {
    {
        std::vector<int> x { 0, 5, 5, 4, 2, 1, 1, 1, 3 };
        std::vector<int> expected { 1, 3, 1, 1, 1, 2 };
        EXPECT_EQ(scran::tabulate_ids(x.size(), x.data()), expected);
    }

    {
        std::vector<int> x { 1, 3, 2, 1, 3, 3, 5 };
        std::vector<int> expected { 0, 2, 1, 3, 0, 1 };
        EXPECT_EQ(scran::tabulate_ids(x.size(), x.data(), true), expected);

        EXPECT_ANY_THROW({
            try {
            scran::tabulate_ids(x.size(), x.data());
            } catch (std::exception& e) {
                EXPECT_TRUE(std::string(e.what()).find("no empty blocks") != std::string::npos);
                throw e;
            }
        });
    }
}

TEST(Blocking, VariableBlockWeight) {
    EXPECT_EQ(0, scran::variable_block_weight(0, {0, 0}));
    EXPECT_EQ(1, scran::variable_block_weight(1, {0, 0}));

    EXPECT_EQ(0, scran::variable_block_weight(0, {5, 10}));
    EXPECT_EQ(0.2, scran::variable_block_weight(6, {5, 10}));
    EXPECT_EQ(1, scran::variable_block_weight(10, {5, 10}));
    EXPECT_EQ(1, scran::variable_block_weight(15, {5, 10}));
}

TEST(Blocking, BlockWeightVector) {
    std::vector<int> sizes { 0, 10, 100, 1000 };

    {
        std::vector<double> expected { 0, 1, 1, 1 };
        EXPECT_EQ(expected, scran::compute_block_weights(sizes, scran::WeightPolicy::EQUAL, {}));
    }

    {
        std::vector<double> expected(sizes.begin(), sizes.end());
        EXPECT_EQ(expected, scran::compute_block_weights(sizes, scran::WeightPolicy::NONE, {}));
    }

    {
        std::vector<double> expected;
        for (auto s : sizes) {
            expected.push_back(scran::variable_block_weight(s, {20, 200}));
        }
        EXPECT_EQ(expected, scran::compute_block_weights(sizes, scran::WeightPolicy::VARIABLE, {20, 200}));
    }
}
