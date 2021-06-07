#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "tatami/base/typed_matrix.hpp"
#include "../utils/vector_to_pointers.hpp"

namespace scran {

template<typename SUB>
class PerCellQCMetrics {
public:
    PerCellQCMetrics() {}

    const std::vector<double>& get_sums() const {
        return sums;
    }

    const std::vector<int>& get_detected() const {
        return detected;
    }

    const std::vector<std::vector<double> >& get_subset_proportions() const {
        return subset_proportions;
    }

    PerCellQCMetrics& set_subsets(std::vector<const SUB*> s) {
        subsets = s;
        return *this;
    }

public:
    template<typename T, typename IDX, typename S=T, typename N=int, typename P=double>
    void compute_metrics(const tatami::typed_matrix<T, IDX>* mat, S* sums_out, N* detected_out, std::vector<P*>& subset_proportions_out) {
        size_t nr = mat->nrow(), nc = mat->ncol();

        if (mat->prefer_rows()) {
            std::vector<T> buffer(nc);
            std::fill(sums_out, sums_out + nc, static_cast<T>(0));
            std::fill(detected_out, detected_out + nc, static_cast<N>(0));
            for (size_t s = 0; s < subsets.size(); ++s) {
                std::fill(subset_proportions_out[s], subset_proportions_out[s] + nc, static_cast<P>(0));
            }
            auto wrk = mat->new_workspace(false);

            if (mat->sparse()) {
                std::vector<IDX> ibuffer(nc);
                for (size_t r = 0; r < nr; ++r) {
                    auto range = mat->sparse_row(r, buffer.data(), ibuffer.data(), wrk.get());

                    for (size_t i = 0; i < range.number; ++i) {
                        sums_out[range.index[i]] += range.value[i];
                        detected_out[range.index[i]] += static_cast<N>(range.value[i] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            for (size_t i = 0; i < range.number; ++i) {
                                subset_proportions_out[s][range.index[i]] += range.value[i];
                            }
                        }
                    }
                }
                
            } else {
                for (size_t r = 0; r < nr; ++r) {
                    auto ptr = mat->row(r, buffer.data(), wrk.get());

                    for (size_t c = 0; c < nc; ++c) {
                        sums_out[c] += ptr[c];
                        detected_out[c] += static_cast<N>(ptr[c] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        if (subsets[s][r]) {
                            auto& sub = subset_proportions_out[s];
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

                    sums_out[c] = 0;
                    detected_out[c] = 0;
                    for (size_t r = 0; r < range.number; ++r) {
                        sums_out[c] += range.value[r];
                        detected_out[c] += static_cast<N>(range.value[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions_out[s][c] = 0;
                        for (size_t i = 0; i < range.number; ++i) {
                            prop += sub[range.index[i]] * range.value[i];
                        }
                    }
                }

            } else {
                for (size_t c = 0; c < nc; ++c) {
                    auto ptr = mat->column(c, buffer.data(), wrk.get());

                    sums_out[c] = 0;
                    detected_out[c] = 0;
                    for (size_t r = 0; r < nr; ++r) {
                        sums_out[c] += ptr[r];
                        detected_out[c] += static_cast<N>(ptr[r] > 0);
                    }

                    for (size_t s = 0; s < subsets.size(); ++s) {
                        const auto& sub = subsets[s];
                        auto& prop = subset_proportions_out[s][c] = 0;
                        for (size_t r = 0; r < nr; ++r) {
                            prop += sub[r] * ptr[r];
                        }
                    }
                }
            }
        }

        for (size_t s = 0; s < subsets.size(); ++s) {
            auto& prop = subset_proportions_out[s];
            for (size_t c = 0; c < nc; ++c) {
                if (sums_out[c]!=0) {
                    prop[c] /= sums_out[c];
                } else {
                    prop[c] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        return;
    }

    template<typename T, typename IDX>
    void compute_metrics(const tatami::typed_matrix<T, IDX>* mat) {
        sums.resize(mat->ncol());
        detected.resize(mat->ncol());
        subset_proportions.resize(subsets.size());
        for (size_t s = 0; s < subsets.size(); ++s) {
            subset_proportions[s].resize(mat->ncol());
        }

        std::vector<double*> out_subset_proportions = vector_to_pointers(subset_proportions);
        compute_metrics(mat, sums.data(), detected.data(), out_subset_proportions);
        return;
    }

private:
    std::vector<const SUB*> subsets;
    std::vector<double> sums;
    std::vector<int> detected;
    std::vector<std::vector<double> > subset_proportions;
};

}

#endif
