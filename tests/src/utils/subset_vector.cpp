#include <gtest/gtest.h>
#include "macros.h"
#include "scran/utils/subset_vector.hpp"
#include <vector>

TEST(SubsetVector, Simple) {
    std::vector<double> input{1, 2, 3, 4, 5};
    std::vector<int> subset{0, 1, 0, 1, 0};

    std::vector<double> retained{2, 4};
    EXPECT_EQ(retained, scran::subset_vector<true>(input, subset.data()));

    std::vector<double> discarded{1, 3, 5};
    EXPECT_EQ(discarded, scran::subset_vector<false>(input, subset.data()));
}
