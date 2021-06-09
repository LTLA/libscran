#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/typed_matrix.hpp"
#include "../utils/vector_to_pointers.hpp"

namespace scran {

template<typename X=uint8_t, typename S=double, typename D=int, typename P=double>
class PerCellQCMetrics {
public:
    PerCellQCMetrics() {}

    PerCellQCMetrics& set_sums(S* p) {
        sums = p;
        return *this;
    }

    PerCellQCMetrics& set_detected(D* p) {
        detected = p;
        return *this;
    }

    PerCellQCMetrics& set_subset_proportions(std::vector<P*> ptrs) {
        subset_proportions = ptrs;
        return *this;
    }

    PerCellQCMetrics& set_subsets(std::vector<const X*> s) {
        subsets = s;
        return *this;
    }

public:
    const S* get_sums() const {
        if (sums == NULL) {
            return internal_sums.data();
        } else {
            return sums;
        }
    }

    const D* get_detected() const {
        if (detected == NULL) {
            return internal_detected.data();
        } else {
            return detected;
        }
    }

    std::vector<const P*> get_subset_proportions() const {
        if (subset_proportions.size()) {
            return std::vector<const P*>(subset_proportions.begin(), subset_proportions.end());
        } else {
            std::vector<const P*> output;
            for (const auto& x : internal_subset_proportions) {
                output.push_back(x.data());
            }
            return output;
        }
    }

public:
    template<typename T, typename IDX>
    void run(const tatami::typed_matrix<T, IDX>* mat) {
        // Setting up the memory structures in a smart way
        // that works when users don't provide it.
        S* sums_out = sums;
        if (sums == NULL) {
            internal_sums.resize(mat->ncol());
            sums_out = internal_sums.data();
        }

        D* detected_out = detected;
        if (detected == NULL) {
            internal_detected.resize(mat->ncol());
            detected_out = internal_detected.data();
        }

        if (subset_proportions.size() != 0 && subset_proportions.size() != subsets.size()) {
            throw std::runtime_error("mismatching number of input/outputs for subset proportions");
        }
        internal_subset_proportions.resize(subsets.size());

        std::vector<double*> subset_proportions_out(subsets.size());
        for (size_t s = 0; s < subsets.size(); ++s) {
            if (subset_proportions.size()) {
                subset_proportions_out[s] = subset_proportions[s];
            } else {
                internal_subset_proportions[s].resize(mat->ncol());
                subset_proportions_out[s] = internal_subset_proportions[s].data();
            }
        }

        // Actually running the function's logic.
        internal_run(mat, sums_out, detected_out, subset_proportions_out);
        return;
    }

private:
    template<typename T, typename IDX>
    void internal_run(const tatami::typed_matrix<T, IDX>* mat, S* sums_out, D* detected_out, std::vector<P*>& subset_proportions_out) {
        size_t nr = mat->nrow(), nc = mat->ncol();

        if (mat->prefer_rows()) {
            std::vector<T> buffer(nc);
            std::fill(sums_out, sums_out + nc, static_cast<S>(0));
            std::fill(detected_out, detected_out + nc, static_cast<D>(0));
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
                        detected_out[range.index[i]] += static_cast<D>(range.value[i] > 0);
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
                        detected_out[c] += static_cast<D>(ptr[c] > 0);
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
                        detected_out[c] += static_cast<D>(range.value[r] > 0);
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
                        detected_out[c] += static_cast<D>(ptr[r] > 0);
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

private:
    S* sums = NULL;
    D* detected = NULL;
    std::vector<P*> subset_proportions;

    std::vector<const X*> subsets;
    std::vector<S> internal_sums;
    std::vector<D> internal_detected;
    std::vector<std::vector<P> > internal_subset_proportions;
};

}

#endif
