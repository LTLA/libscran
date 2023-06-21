#ifndef SCRANTEST_SIMULATOR_HPP
#define SCRANTEST_SIMULATOR_HPP

#include <vector>
#include <random>
#include "tatami/tatami.hpp"

struct Simulator {
    double lower = -10;
    double upper = 10;
    double density = 0.1;
    size_t seed = 1234567890;

    template<typename T = double>
    std::vector<T> vector(size_t length) const {
        std::mt19937_64 rng(seed);
        std::vector<T> values(length);
        std::uniform_real_distribution<> unif(lower, upper);

        if (density < 1) {
            std::uniform_real_distribution<> nonzero(0.0, 1.0);
            for (auto& v : values) {
                if (nonzero(rng) < density) {
                    v = unif(rng);
                }
            }
        } else {
            for (auto& v : values) {
                v = unif(rng);
            }
        }

        return values;
    }

    template<typename T = double, typename IDX = int> 
    tatami::DenseRowMatrix<T, IDX> matrix(size_t nr, size_t nc) const {
        return tatami::DenseRowMatrix<T, IDX>(nr, nc, vector(nr * nc));
    }
};

#endif
