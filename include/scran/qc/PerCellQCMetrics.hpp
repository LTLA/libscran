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
        stored_sums = p;
        return *this;
    }

    PerCellQCMetrics& set_detected(D* p) {
        stored_detected = p;
        return *this;
    }

    PerCellQCMetrics& set_subset_proportions(std::vector<P*> ptrs) {
        stored_subset_proportions = ptrs;
        return *this;
    }

    PerCellQCMetrics& set_subsets(std::vector<const X*> s) {
        subsets = s;
        return *this;
    }

public:
    struct Results {
        Results(size_t n, PerCellQCMetrics<X, S, D, P>& parent) : ncells(n) {
            // Setting up the memory structures in a smart way
            // that works when users don't provide it.
            sums = parent.stored_sums;
            if (sums == NULL) {
                internal_sums.resize(n);
                sums = internal_sums.data();
            }

            detected = parent.stored_detected;
            if (detected == NULL) {
                internal_detected.resize(n);
                detected = internal_detected.data();
            }

            size_t nsubsets = parent.subsets.size();
            if (parent.stored_subset_proportions.size()) {
                if (parent.stored_subset_proportions.size() != nsubsets) {
                    throw std::runtime_error("mismatching number of input/outputs for subset proportions");
                }
                subset_proportions = parent.stored_subset_proportions;
            } else {
                internal_subset_proportions.resize(nsubsets);
                subset_proportions.resize(nsubsets);
                for (size_t s = 0; s < nsubsets; ++s) {
                    internal_subset_proportions[s].resize(n);
                    subset_proportions[s] = internal_subset_proportions[s].data();
                }
            }

            return;
        }

        size_t ncells;
        S* sums = NULL;
        D* detected = NULL;
        std::vector<P*> subset_proportions;
    private:
        std::vector<S> internal_sums;
        std::vector<D> internal_detected;
        std::vector<std::vector<P> > internal_subset_proportions;
    };

public:
    template<typename T, typename IDX>
    Results run(const tatami::typed_matrix<T, IDX>* mat) {
        size_t nr = mat->nrow(), nc = mat->ncol();
        Results output(nc, *this);

        S* sums_out = output.sums;
        D* detected_out = output.detected;
        auto& subset_proportions_out = output.subset_proportions;

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

        return output;
    }

private:
    S* stored_sums = NULL;
    D* stored_detected = NULL;
    std::vector<P*> stored_subset_proportions;
    std::vector<const X*> subsets;
};

}

#endif
