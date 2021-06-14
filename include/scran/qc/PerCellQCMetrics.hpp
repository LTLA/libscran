#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/typed_matrix.hpp"
#include "../utils/vector_to_pointers.hpp"

namespace scran {

template<typename X=uint8_t>
class PerCellQCMetrics {
public:
    PerCellQCMetrics() {}

    PerCellQCMetrics(const std::vector<X*>& s) : subsets(s.begin(), s.end()) {}

    PerCellQCMetrics(std::vector<const X*> s) : subsets(std::move(s)) {}

public:
    PerCellQCMetrics& set_subsets(std::vector<const X*> s) {
        subsets = s;
        return *this;
    }

public:
    struct Results {
        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_proportions(nsubsets, std::vector<double>(ncells)) {}
        std::vector<double> sums;
        std::vector<int> detected;
        std::vector<std::vector<double> > subset_proportions;
    };

public:
    template<typename T, typename IDX>
    Results run(const tatami::typed_matrix<T, IDX>* mat) {
        Results output(mat->ncol(), subsets.size());
        run(mat, output.sums.data(), output.detected.data(), vector_to_pointers(output.subset_proportions));
        return output;
    }

    template<typename T, typename IDX, typename S, typename D, typename P>
    void run(const tatami::typed_matrix<T, IDX>* mat, S* sums, D* detected, std::vector<P*> subset_proportions) {
        size_t nr = mat->nrow(), nc = mat->ncol();

        if (mat->prefer_rows()) {
            std::vector<T> buffer(nc);
            std::fill(sums, sums + nc, static_cast<S>(0));
            std::fill(detected, detected + nc, static_cast<D>(0));
            for (size_t s = 0; s < subsets.size(); ++s) {
                std::fill(subset_proportions[s], subset_proportions[s] + nc, static_cast<P>(0));
            }
            auto wrk = mat->new_workspace(false);

            if (mat->sparse()) {
                std::vector<IDX> ibuffer(nc);
                for (size_t r = 0; r < nr; ++r) {
                    auto range = mat->sparse_row(r, buffer.data(), ibuffer.data(), wrk.get());

                    for (size_t i = 0; i < range.number; ++i) {
                        sums[range.index[i]] += range.value[i];
                        detected[range.index[i]] += static_cast<D>(range.value[i] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            for (size_t i = 0; i < range.number; ++i) {
                                subset_proportions[s][range.index[i]] += range.value[i];
                            }
                        }
                    }
                }
                
            } else {
                for (size_t r = 0; r < nr; ++r) {
                    auto ptr = mat->row(r, buffer.data(), wrk.get());

                    for (size_t c = 0; c < nc; ++c) {
                        sums[c] += ptr[c];
                        detected[c] += static_cast<D>(ptr[c] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            auto& sub = subset_proportions[s];
                            for (size_t c = 0; c < nc; ++c) {
                                sub[c] += ptr[c];
                            }
                        }
                    }
                }
            }

        } else {
            std::vector<T> buffer(nr);
            auto wrk = mat->new_workspace(true);

            if (mat->sparse()) {
                std::vector<IDX> ibuffer(nr);
                for (size_t c = 0; c < nc; ++c) {
                    auto range = mat->sparse_column(c, buffer.data(), ibuffer.data(), wrk.get());

                    sums[c] = 0;
                    detected[c] = 0;
                    for (size_t r = 0; r < range.number; ++r) {
                        sums[c] += range.value[r];
                        detected[c] += static_cast<D>(range.value[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions[s][c] = 0;
                        for (size_t i = 0; i < range.number; ++i) {
                            prop += sub[range.index[i]] * range.value[i];
                        }
                    }
                }

            } else {
                for (size_t c = 0; c < nc; ++c) {
                    auto ptr = mat->column(c, buffer.data(), wrk.get());

                    sums[c] = 0;
                    detected[c] = 0;
                    for (size_t r = 0; r < nr; ++r) {
                        sums[c] += ptr[r];
                        detected[c] += static_cast<D>(ptr[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions[s][c] = 0;
                        for (size_t r = 0; r < nr; ++r) {
                            prop += sub[r] * ptr[r];
                        }
                    }
                }
            }
        }

        for (size_t s = 0; s < subsets.size(); ++s) {
            auto& prop = subset_proportions[s];
            for (size_t c = 0; c < nc; ++c) {
                if (sums[c]!=0) {
                    prop[c] /= sums[c];
                } else {
                    prop[c] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        return;
    }

private:
    std::vector<const X*> subsets;
};

}

#endif
