#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "../data/Simulator.hpp"

#include "tatami/tatami.hpp"
#include "scran/normalization/quick_grouped_size_factors.hpp"

#include <cmath>
#include <random>

class quick_grouped_size_factors_Test : public ::testing::Test {
protected:
    std::shared_ptr<tatami::Matrix<double, int> > ptr;

    void SetUp() {
        size_t nr = 500, nc = 200;

        Simulator sim;
        sim.lower = 1;
        sim.upper = 10;
        auto vec = sim.vector(nr * nc);
        ptr.reset(new tatami::DenseRowMatrix<double, int>(nr, nc, std::move(vec)));
    }
};

TEST_F(quick_grouped_size_factors_Test, Simple) {
    // Just check that the damn thing runs and gives somewhat sensible output.
    auto out = scran::quick_grouped_size_factors::run(ptr.get());
    EXPECT_EQ(out.size(), 200);

    bool is_positive = true;
    for (auto x : out) {
        if (x <= 0) {
            is_positive = false;
        }
    }
    EXPECT_TRUE(is_positive);

    // Same results, but for maxing out coverage.
    std::vector<double> out2(200);
    scran::quick_grouped_size_factors::run(ptr.get(), out2.data());
    EXPECT_EQ(out, out2);
}
