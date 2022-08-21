#include <gtest/gtest.h>
#include "../utils/macros.h"

#include "scran/aggregation/DownsampleByNeighbors.hpp"

#include <set>
#include <random>
#include <vector>

class DownsampleByNeighborsTest : public ::testing::Test {
protected:
    std::vector<double> data;

    void fill(int ndim, int nobs) {
        data.resize(ndim * nobs); 
        std::mt19937_64 rng(91872631);
        std::normal_distribution ndist;
        for (auto& d : data) {
            d = ndist(rng);
        }
    }

    void fill_with_crap(int ndim, size_t i, double crap) {
        auto ptr = data.data() + ndim * i;
        std::fill(ptr, ptr + ndim, crap);
    }
};

TEST_F(DownsampleByNeighborsTest, Sanity) {
    int ndim = 5;
    int nobs = 101;

    fill(ndim, nobs);
    fill_with_crap(ndim, 7, 10.1);
    fill_with_crap(ndim, 13, -5.2);

    scran::DownsampleByNeighbors down;
    std::vector<int> assigned(nobs, -1);
    auto res = down.run(ndim, nobs, data.data(), assigned.data());
    EXPECT_TRUE(res.size() < 50); // should see some downsampling.

    // Check that all 'assigned' were filled with something.
    for (auto a : assigned) {
        EXPECT_TRUE(a >= 0);
    }

    // Ensure that our special points are also collected.
    std::set<int> collected(res.begin(), res.end());
    EXPECT_TRUE(collected.find(7) != collected.end());
    EXPECT_TRUE(collected.find(13) != collected.end());

    // Same results in parallel.
    down.set_num_threads(3);
    auto pres = down.run(ndim, nobs, data.data(), static_cast<int*>(NULL));
    EXPECT_EQ(pres, res);
}

TEST_F(DownsampleByNeighborsTest, Approximate) {
    int ndim = 8;
    int nobs = 1001;

    fill(ndim, nobs);
    fill_with_crap(ndim, 17, 10.1);
    fill_with_crap(ndim, 235, -5.2);

    scran::DownsampleByNeighbors down;
    down.set_approximate(true).set_num_neighbors(50);
    auto res = down.run(ndim, nobs, data.data(), static_cast<int*>(NULL));
    EXPECT_TRUE(res.size() < 200);

    std::set<int> collected(res.begin(), res.end());
    EXPECT_TRUE(collected.find(17) != collected.end());
    EXPECT_TRUE(collected.find(235) != collected.end());
}

TEST_F(DownsampleByNeighborsTest, Reference) {
    int ndim = 8;
    int nobs = 1001;
    fill(ndim, nobs);
    knncolle::VpTreeEuclidean<int, double> index(ndim, nobs, data.data());

    scran::DownsampleByNeighbors down;
    std::vector<int> assigned(nobs, -1);
    auto res = down.run(ndim, nobs, data.data(), assigned.data());
    auto num_neighbors = scran::DownsampleByNeighbors::Defaults::num_neighbors;

    // Reference calculation.
    std::vector<std::pair<std::pair<int, double>, int> > ordered, temp, temp2;
    ordered.reserve(nobs);
    std::vector<std::vector<std::pair<int, double> > > neighbors(nobs);
    for (size_t n = 0; n < nobs; ++n) {
        neighbors[n] = index.find_nearest_neighbors(n, num_neighbors);
        ordered.emplace_back(std::make_pair(0, neighbors[n].back().second), static_cast<int>(n));
    }

    std::vector<int> chosen;
    std::vector<char> covered(nobs);
    std::vector<int> ref_assigned(nobs);

    for (int k = 0; k <= num_neighbors; ++k) {
        std::sort(ordered.begin(), ordered.end());
        temp.clear();

        for (const auto& current : ordered) {
            if (current.first.first > k) {
                temp.push_back(current);
                continue;
            }

            auto index = current.second;
            if (covered[index]) {
                continue;
            }

            const auto& curneighbors = neighbors[index];
            int updated_num = 0;
            for (auto x : curneighbors) {
                updated_num += covered[x.first];
            }

            if (updated_num > k) {
                temp.push_back(current);
                continue;
            }

            chosen.push_back(index);
            covered[index] = 1;
            ref_assigned[index] = index;
            for (auto x : curneighbors) {
                if (!covered[x.first]) {
                    covered[x.first] = 1;
                    ref_assigned[x.first] = index;
                }
            }
        }

        temp2.clear();
        for (auto current : temp) {
            if (!covered[current.second]) {
                const auto& curneighbors = neighbors[current.second];
                int updated_num = 0;
                for (auto x : curneighbors) {
                    updated_num += covered[x.first];
                }
                current.first.first = updated_num;
                temp2.push_back(current);
            }
        }
        ordered.swap(temp2);
    }

    std::sort(chosen.begin(), chosen.end());

    EXPECT_EQ(res, chosen);
    EXPECT_EQ(assigned, ref_assigned);
    EXPECT_TRUE(res.size() <= 200);
}
