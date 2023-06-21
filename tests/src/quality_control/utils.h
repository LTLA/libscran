#ifndef TEST_QUALITY_CONTROL_UTILS_H
#define TEST_QUALITY_CONTROL_UTILS_H

#include <vector>
#include <cstddef>
#include "tatami/tatami.hpp"

namespace quality_control {

inline std::vector<int> to_filter (size_t nr, const std::vector<size_t>& indices) {
    std::vector<int> keep_s(nr);
    for (auto i : indices) { keep_s[i] = 1; }
    return keep_s;        
}

inline std::vector<int> compute_num_detected(const tatami::NumericMatrix* matrix) {
    auto NC = matrix->ncol();
    std::vector<int> copy(NC);
    auto ext = matrix->dense_column();
    for (size_t c = 0; c < NC; ++c) {
        auto row = ext->fetch(c);
        for (auto& r : row) {
            copy[c] += (r != 0);
        }
    }
    return copy;
}

inline std::vector<int> create_blocks(size_t n, size_t b) {
    std::vector<int> block(n);
    for (size_t i = 0; i < n; ++i) { 
        block[i] = i % b;
    }
    return block;
}

}

#endif
