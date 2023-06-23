#ifndef SCRAN_MATRIX_CALCULATOR_HPP
#define SCRAN_MATRIX_CALCULATOR_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>

#include "../utils/vector_to_pointers.hpp"
#include "../feature_selection/blocked_variances.hpp"
#include "auc.hpp"
#include "tatami/tatami.hpp"

namespace scran {

namespace differential_analysis {

/****************************************************************************
 * The MatrixCalculator class handles the Factory construction and the apply()
 * call, given an Overlord object that describes how to process the statistics
 * in a 'with-AUC' and 'without-AUC' scenario. The overlord should implement:
 *
 * - `needs_auc()`, indicating whether AUCs should be computed.
 * - `obtain_auc_buffer()`, returning a pointer to an array to store the AUCs.
 *
 * Check out PairwiseMatrix and ScoreMarkers for concrete examples.
 ****************************************************************************/

class MatrixCalculator {
public:
    MatrixCalculator(int nt, double t) : num_threads(nt), threshold(t) {}

private:
    int num_threads;
    double threshold;

public:
    struct State {
        State(size_t n) : means(n), variances(n), detected(n) {}
        std::vector<double> means, variances, detected;
        std::vector<int> level_size;
    };

public:
    template<class Matrix, typename G, class Overlord>
    State run(const Matrix* p, const G* group, int ngroups, Overlord& overlord) const {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0, end = p->ncol(); i < end; ++i) {
            ++(group_size[group[i]]);
        }
        return core(p, group, std::move(group_size), group, ngroups, static_cast<const int*>(NULL), 1, overlord);
    }

    template<class Matrix, typename G, typename B, class Overlord>
    State run_blocked(const Matrix* p, const G* group, int ngroups, const B* block, int nblocks, Overlord& overlord) const {
        if (block == NULL) {
            return run(p, group, ngroups, overlord);
        }

        size_t ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);
        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = group[i] * nblocks + block[i];
            ++(combo_size[combos[i]]);
        }

        return core(p, combos.data(), std::move(combo_size), group, ngroups, block, nblocks, overlord);
    }

private:
    struct AucBundle {
        AucBundle(size_t ngroups, size_t nblocks, const std::vector<int>& level_size) :
            paired(nblocks),
            num_zeros(nblocks, std::vector<int>(ngroups)),
            totals(nblocks, std::vector<int>(ngroups)),
            auc_buffer(ngroups * ngroups),
            denominator(ngroups, std::vector<double>(ngroups))
        {
            auto lsIt = level_size.begin();
            for (size_t g = 0; g < ngroups; ++g) {
                for (size_t b = 0; b < nblocks; ++b, ++lsIt) {
                    totals[b][g] = *lsIt;
                }
            }

            for (size_t b = 0; b < nblocks; ++b) {
                for (int g1 = 0; g1 < ngroups; ++g1) {
                    for (int g2 = 0; g2 < ngroups; ++g2) {
                        denominator[g1][g2] += totals[b][g1] * totals[b][g2];
                    }
                }
            }
        }

        std::vector<PairedStore> paired;
        std::vector<std::vector<int> > num_zeros;
        std::vector<std::vector<int> > totals;
        std::vector<double> auc_buffer;
        std::vector<std::vector<double> > denominator;
    };

    struct DummyBundle {
        DummyBundle(size_t, size_t, const std::vector<int>&) {}
    };

    template<bool sparse_, bool auc_, typename Data_, typename Index_, typename Level_, typename Group_, typename Block_, class State_, class Overlord_>
    void by_row(const tatami::Matrix<Data_, Index_>* p, const Level_* level, const std::vector<int>& level_size, const Group_* group, size_t ngroups, const Block_* block, size_t nblocks, State_& state, Overlord_& overlord) const {
        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            auto NC = p->ncol();
            std::vector<Data_> vbuffer(NC);
            typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(NC);
            auto ext = tatami::consecutive_extractor<true, sparse_>(p, start, length);

            // AUC-only object.
            typename std::conditional<auc_, AucBundle, DummyBundle>::type auc_info(ngroups, nblocks, level_size);

            size_t nlevels = level_size.size();
            auto offset = nlevels * start;
            for (Index_ r = start, end = r + length; r < end; ++r, offset += nlevels) {
                auto mptr = state.means.data() + offset;
                auto vptr = state.variances.data() + offset;
                auto dptr = state.detected.data() + offset;

                if constexpr(!sparse_) {
                    auto ptr = ext->fetch(r, vbuffer.data());
                    feature_selection::blocked_variance_with_mean<true>(ptr, NC, level, nlevels, level_size.data(), mptr, vptr);

                    std::fill(dptr, dptr + nlevels, 0);
                    for (size_t j = 0; j < NC; ++j) {
                        dptr[level[j]] += (ptr[j] != 0);
                    }

                    if constexpr(auc_) {
                        for (auto& z : auc_info.num_zeros) {
                            std::fill(z.begin(), z.end(), 0);
                        }
                        for (auto& p : auc_info.paired) {
                            p.clear();
                        }

                        for (Index_ c = 0; c < NC; ++c) {
                            auto b = block[c];
                            auto g = group[c];
                            if (ptr[c]) {
                                auc_info.paired[b].emplace_back(ptr[c], g);
                            } else {
                                ++(auc_info.num_zeros[b][g]);
                            }
                        }

                        auto store = overlord.prepare_auc_buffer(r, ngroups);
                        process_auc_for_rows(ngroups, nblocks, threshold, auc_info, store);
                    }

                } else {
                    auto range = ext->fetch(r, vbuffer.data(), ibuffer.data());
                    feature_selection::blocked_variance_with_mean<true>(range, level, nlevels, level_size.data(), mptr, vptr, dptr);

                    if constexpr(auc_) {
                        auto nzIt = auc_info.num_zeros.begin();
                        for (const auto& t : auc_info.totals) {
                            std::copy(t.begin(), t.end(), nzIt->begin());
                            ++nzIt;
                        }
                        for (auto& p : auc_info.paired) {
                            p.clear();
                        }

                        for (size_t j = 0; j < range.number; ++j) {
                            if (range.value[j]) {
                                size_t c = range.index[j];
                                auto b = block[c];
                                auto g = group[c];
                                auc_info.paired[b].emplace_back(range.value[j], g);
                                --(auc_info.num_zeros[b][g]);
                            }
                        }

                        auto store = overlord.prepare_auc_buffer(r, ngroups);
                        process_auc_for_rows(ngroups, nblocks, threshold, auc_info, store);
                    }
                }
            }
        }, p->nrow(), num_threads);
    }

    static void process_auc_for_rows(size_t ngroups, size_t nblocks, double threshold, AucBundle& bundle, double* output) {
        std::fill(output, output + ngroups * ngroups, 0);
        auto& auc_buffer = bundle.auc_buffer;

        for (size_t b = 0; b < nblocks; ++b) {
            auto& pr = bundle.paired[b];
            auto& nz = bundle.num_zeros[b];
            const auto& tt = bundle.totals[b];

            std::fill(auc_buffer.begin(), auc_buffer.end(), 0);
            if (threshold) {
                compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), threshold, false);
            } else {
                compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), false);
            }

            // Adding to the blocks.
            for (size_t g = 0, end = auc_buffer.size(); g < end; ++g) {
                output[g] += auc_buffer[g];
            }
        }

        for (size_t g1 = 0; g1 < ngroups; ++g1) {
            for (size_t g2 = 0; g2 < ngroups; ++g2) {
                auto& current = output[g1 * ngroups + g2];
                if (bundle.denominator[g1][g2]) {
                    current /= bundle.denominator[g1][g2];
                } else {
                    current = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }

private:
    template<bool sparse_, typename Data_, typename Index_, typename Level_, class State_>
    void by_column(const tatami::Matrix<Data_, Index_>* p, const Level_* level, const std::vector<int>& level_size, State_& state) const {
        size_t nlevels = level_size.size();

        tatami::parallelize([&](size_t, Index_ start, Index_ length) -> void {
            auto NC = p->ncol();
            std::vector<Data_> vbuffer(length);
            typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(length);
            auto ext = tatami::consecutive_extractor<false, sparse_>(p, 0, NC, start, length);

            std::vector<std::vector<double> > tmp_means(nlevels), tmp_vars(nlevels), tmp_detected(nlevels);
            for (size_t l = 0; l < nlevels; ++ l) {
                tmp_means[l].resize(length);
                tmp_vars[l].resize(length);
                tmp_detected[l].resize(length);
            }
            std::vector<int> tmp_counts(nlevels);

            for (Index_ c = 0; c < NC; ++c) {
                auto b = level[c];
                auto mptr = tmp_means[b].data();
                auto vptr = tmp_vars[b].data();
                auto dptr = tmp_detected[b].data();
            
                if constexpr(!sparse_) {
                    auto ptr = ext->fetch(c, vbuffer.data());
                    tatami::stats::variances::compute_running(ptr, length, mptr, vptr, tmp_counts[b]);
                    for (Index_ j = 0; j < length; ++j, ++dptr) {
                        *dptr += (ptr[j] != 0);
                    }
                } else {
                    auto range = ext->fetch(c, vbuffer.data(), ibuffer.data());
                    tatami::stats::variances::compute_running(range, mptr, vptr, dptr, tmp_counts[b], /* skip_zeros = */ true, /* subtract = */ start);
                }
            }

            for (size_t l = 0; l < nlevels; ++l) {
                if constexpr(!sparse_) {
                    tatami::stats::variances::finish_running(length, tmp_means[l].data(), tmp_vars[l].data(), tmp_counts[l]);
                } else {
                    tatami::stats::variances::finish_running(length, tmp_means[l].data(), tmp_vars[l].data(), tmp_detected[l].data(), tmp_counts[l]);
                }
            }

            transpose_for_column(tmp_means, state.means.data(), start, length);
            transpose_for_column(tmp_vars, state.variances.data(), start, length);
            transpose_for_column(tmp_detected, state.detected.data(), start, length);
        }, p->nrow(), num_threads);
    }

    template<typename Index_>
    static void transpose_for_column(const std::vector<std::vector<double> >& source, double* sink, Index_ start, Index_ length) {
        size_t nlevels = source.size();
        for (Index_ r = 0; r < length; ++r) {
            auto output = sink + (r + start) * nlevels;
            for (size_t l = 0; l < nlevels; ++l, ++output) {
                *output = source[l][r];
            }
        }
    }

private:
    template<typename Data_, typename Index_, typename Level_, typename Group_, typename Block_, class Overlord_>
    State core(const tatami::Matrix<Data_, Index_>* p, const Level_* level, std::vector<int> level_size, const Group_* group, size_t ngroups, const Block_* block, size_t nblocks, Overlord_& overlord) const {
        auto ngenes = p->nrow();
        size_t nlevels = level_size.size();
        State state(static_cast<size_t>(ngenes) * nlevels);

        if (!overlord.needs_auc()) {
            if (p->prefer_rows()) {
                if (p->sparse()) {
                    by_row<true, false>(p, level, level_size, group, ngroups, block, nblocks, state, overlord);
                } else {
                    by_row<false, false>(p, level, level_size, group, ngroups, block, nblocks, state, overlord);
                }
            } else {
                if (p->sparse()) {
                    by_column<true>(p, level, level_size, state);
                } else {
                    by_column<false>(p, level, level_size, state);
                }
            }
        } else {
            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<Block_> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            if (p->sparse()) {
                by_row<true, true>(p, level, level_size, group, ngroups, block, nblocks, state, overlord);
            } else {
                by_row<false, true>(p, level, level_size, group, ngroups, block, nblocks, state, overlord);
            }
        }

        // Dividing through to get the actual detected proportions.
        // Don't bother parallelizing this, given how simple it is.
        auto dptr = state.detected.data();
        for (Index_ gene = 0; gene < ngenes; ++gene) {
            size_t in_offset = gene * nlevels;
            for (size_t l = 0; l < nlevels; ++l, ++dptr) {
                if (level_size[l]) {
                    *dptr /= level_size[l];
                } else {
                    *dptr = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        // Moving it in for output's sake.
        state.level_size = std::move(level_size);
        return state;
    }
};

}
/**
 * @endcond
 */

}

#endif
