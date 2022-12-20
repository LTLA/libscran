#ifndef TEST_QUALITY_CONTROL_UTILS_H
#define TEST_QUALITY_CONTROL_UTILS_H

namespace quality_control {

static std::vector<int> to_filter (size_t nr, const std::vector<size_t>& indices) {
    std::vector<int> keep_s(nr);
    for (auto i : indices) { keep_s[i] = 1; }
    return keep_s;        
}

static std::vector<int> compute_num_detected(const tatami::NumericMatrix* matrix) {
    auto NC = matrix->ncol();
    std::vector<int> copy(NC);
    for (size_t c = 0; c < NC; ++c) {
        auto row = matrix->column(c);
        for (auto& r : row) {
            copy[c] += (r != 0);
        }
    }
    return copy;
}

}

#endif
