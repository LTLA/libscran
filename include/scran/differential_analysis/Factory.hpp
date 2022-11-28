#ifndef SCRAN_BIDIMENSIONAL_FACTORY_HPP
#define SCRAN_BIDIMENSIONAL_FACTORY_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>

#include "../feature_selection/blocked_variances.hpp"
#include "cohens_d.hpp"
#include "auc.hpp"
#include "lfc.hpp"
#include "delta_detected.hpp"

namespace scran {

namespace differential_analysis {

/****************************************************************************
 * The 'simple' class of factories computes per-group statistics that are
 * ultimately used to compute pairwise effect sizes. It does not need to
 * compute the pairwise effects directly because we're not using the AUC yet.
 ****************************************************************************/

template<typename Stat, typename Level>
struct SimpleBundle {
    SimpleBundle(size_t nr, size_t nc, const Level* l, const std::vector<int>* ls, Stat* m, Stat* v, Stat* d) :
        NR(nr), NC(nc), levels(l), level_size_ptr(ls), means(m), variances(v), detected(d) {}

    size_t NR, NC;
    const Level* levels;
    const std::vector<int>* level_size_ptr;
    Stat* means, *detected, *variances;
};

struct SimpleByRow { 
    SimpleByRow(SimpleBundle<Stat, Level> d) : details(std::move(d)) {}

public:
    template<typename T>
    void compute(size_t i, const T* ptr) {
        auto nlevels = details.level_size_ptr->size();
        auto offset = nlevels * i;

        feature_selection::blocked_variance_with_mean<true>(
            ptr, 
            details.NC, 
            details.levels, 
            nlevels,
            details.level_size_ptr->data(), 
            details.means + offset, 
            details.variances + offset
        );

        auto tmp_nzeros = details.detected + nlevels * i;
        std::fill(tmp_nzeros, tmp_nzeros + nlevels, 0);
        for (size_t j = 0; j < details.NC; ++j) {
            tmp_nzeros[details.levels[j]] += (ptr[j] > 0);
        }
    }

    template<class SparseRange>
    void compute(size_t i, const SparseRange& range) {
        auto nlevels = details.level_size_ptr->size();
        auto offset = nlevels * i;

        feature_selection::blocked_variance_with_mean<true>(
            range, 
            details.levels, 
            nlevels,
            details.level_size_ptr->data(), 
            details.means + offset, 
            details.variances + offset, 
            details.detected + offset
        );
    }

public:
    SimpleBundle<Stat, Level> details;
};

template<typename Stat, typename Level>
class SimpleBidimensionalFactory {
public:
    SimpleBidimensionalFactory(size_t nr, size_t nc, const Level* l, const std::vector<int>* ls, Stat* m, Stat* v, Stat* d) : 
        details(nr, nc, l, ls, m, v, d) {}

private:
    SimpleBundle<Stat, Level> details;

public:
    SimpleByRow dense_direct() const {
        return SimpleByRow(details);
    }

    SimpleByRow sparse_direct() const {
        return SimpleByRow(details);
    };

public:
    struct ByCol { 
        ByCol(size_t size, SimpleBundle<Stat, Level> d) : 
            details(std::move(d)), 
            tmp_means(details.level_size_ptr->size(), std::vector<Stat>(size)),
            tmp_vars(details.level_size_ptr->size(), std::vector<Stat>(size)),
            tmp_detected(details.level_size_ptr->size(), std::vector<Stat>(size)),
            tmp_counts(details.level_size_ptr->size())
        {}

    protected:
        SimpleBundle<Stat, Level> details;
        std::vector<std::vector<Stat> > tmp_means, tmp_vars, tmp_detected;
        std::vector<int> tmp_counts;
        size_t counter = 0;
    };

public:
    struct DenseByCol : public ByCol {
        DenseByCol(size_t s, size_t e, SimpleBundle<Stat, Level> d) : ByCol(e - s, std::move(d)), start_row(s), thread_rows(e - s) {}

        template<typename T>
        void add(const T* ptr) {
            auto b = this->details.levels[this->counter];
            tatami::stats::variances::compute_running(ptr, thread_rows, this->tmp_means[b].data(), this->tmp_vars[b].data(), this->tmp_counts[b]);

            auto ndetected = this->tmp_detected[b].data();
            for (size_t j = 0; j < thread_rows; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++(this->counter);
        }

        void finish() {
            auto nlevels = this->details.level_size_ptr->size();
            for (size_t b = 0; b < nlevels; ++b) {
                tatami::stats::variances::finish_running(thread_rows, this->tmp_means[b].data(), this->tmp_vars[b].data(), this->tmp_counts[b]);
            }

            transpose(this->tmp_means, this->details.means);
            transpose(this->tmp_vars, this->details.variances);
            transpose(this->tmp_detected, this->details.detected);
        }

    protected:
        size_t start_row, thread_rows;

        void transpose(const std::vector<std::vector<Stat> >& source, Stat* sink) {
            size_t nlevels = source.size();
            for (size_t r = 0; r < thread_rows; ++r) {
                auto output = sink + (r + start_row) * nlevels;
                for (size_t b = 0; b < nlevels; ++b, ++output) {
                    *output = source[b][r];
                }
            }
        }
    };

    DenseByCol dense_running() const {
        return DenseByCol(0, this->details.NR, this->details);
    }

    DenseByCol dense_running(size_t start, size_t end) const {
        return DenseByCol(start, end, this->details);
    }

public:
    struct SparseByCol : public ByCol { 
        SparseByCol(size_t s, size_t e, size_t nr, SimpleBundle<Stat, Level> d) : ByCol(nr, std::move(d)), start_row(s), thread_rows(e - s) {} 

        template<class SparseRange>
        void add(const SparseRange& range) {
            auto b = this->details.levels[this->counter];
            
            // TODO: add a offset to tatami so that we can subtract an offset from the range.
            tatami::stats::variances::compute_running(range, this->tmp_means[b].data(), this->tmp_vars[b].data(), this->tmp_detected[b].data(), this->tmp_counts[b]);

            ++(this->counter);
        }

        void finish() {
            auto nlevels = this->details.level_size_ptr->size();
            for (size_t b = 0; b < nlevels; ++b) {
                tatami::stats::variances::finish_running(
                    thread_rows,
                    this->tmp_means[b].data() + start_row, 
                    this->tmp_vars[b].data() + start_row, 
                    this->tmp_detected[b].data() + start_row,
                    this->tmp_counts[b]
                );
            }

            transpose(this->tmp_means, this->details.means);
            transpose(this->tmp_vars, this->details.variances);
            transpose(this->tmp_detected, this->details.detected);
        }

    protected:
        size_t start_row, thread_rows;

        void transpose(const std::vector<std::vector<Stat> >& source, Stat* sink) {
            size_t nlevels = source.size();
            for (size_t r = 0; r < thread_rows; ++r) {
                auto g = r + start_row;
                auto output = sink + g * nlevels;
                for (size_t b = 0; b < nlevels; ++b, ++output) {
                    *output = source[b][g];
                }
            }
        }
    };

    SparseByCol sparse_running() {
        return SparseByCol(0, this->details.NR, this->details.NR, this->details);
    }

    SparseByCol sparse_running(size_t start, size_t end) {
        // Just making the temporary vectors with all rows, so that we don't 
        // have to worry about subtracting the indices when doing sparse iteration.
        return SparseByCol(start, end, this->details.NR, this->details);
    }
};

/****************************************************************************
 * The 'complex' class of factories performs pairwise comparisons directly for
 * the effect sizes that cannot be easily derived from simple per-group
 * statistics. This currently includes the AUC.
 ****************************************************************************/

template<typename Group, typename Block, class Overlord>
struct ComplexPerRowBundle {
    ComplexPerRowBundle(const Group* g, int ng, const Block* b, int nb, double t, Overlord* store) : group(g), block(b), ngroups(ng), nblocks(nb), threshold(t), overlord(ova) {}
    const Group* group;
    const Block* block;
    int ngroups, nblocks;
    double threshold;
    Overlord* overlord;
};

template<typename Stat, typename Level, typename Group, typename Block> 
struct ComplexPerRowFactory {
public:
    ComplexPerRowFactory(size_t nr, size_t nc, const Level* l, const std::vector<int>* ls, Stat* m, Stat* v, Stat* d, const Group* g, int ng, const Block* b, int nb, double t, Overlord* ova) :
        simple_details(nr, nc, l, ls, m, v, d), extra_details(g, ng, b, nb, t, ova) {}

private:
    SimpleBundle<Stat, Level> simple_details;
    ComplexPerRowBundle<Group, Block, Overlord> extra_details;

public:
    template<class Worker>
    struct ByRow {
        ByRow(SimpleBundle<Stat, Level> d1, ComplexPerRowBundle<Stat, Group, Block> d2, Worker w) : 
            simple_handler(std::move(d1)),
            extra_details(std::move(d2)),
            worker(std::move(w)),
            paired(extra_details.nblocks), 
            num_zeros(extra_details.nblocks, std::vector<int>(extra_details.ngroups)), 
            totals(extra_details.nblocks, std::vector<int>(extra_details.ngroups)), 
            auc_buffer1(extra_details.ngroups * extra_details.ngroups), 
            auc_buffer2(extra_details.ngroups * extra_details.ngroups), 
            denominator(extra_details.ngroups, std::vector<double>(extra_details.ngroups))
        {
            const auto& ngroups = extra_details.ngroups;
            const auto& nblocks = extra_details.nblocks;

            auto lsIt = this->details.level_size_ptr->begin();
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

            return;
        }

    private:
        SimpleByRow<Stat, Level> simple_handler;
        ComplexPerRowBundle<Stat, Group, Block> extra_details;
        Worker worker;

        // AUC handlers.
        std::vector<PairedStore> paired;
        std::vector<std::vector<int> > num_zeros;
        std::vector<std::vector<int> > totals;
        std::vector<double> auc_buffer1, auc_buffer2;
        std::vector<std::vector<double> > denominator;

        void process_auc(size_t i) {
            const auto& ngroups = extra_details.ngroups;
            auto output = extra_details.auc + i * ngroups * ngroups;
            std::fill(auc_buffer2.begin(), auc_buffer2.end(), 0);

            for (size_t b = 0; b < extra_details.nblocks; ++b) {
                auto& pr = paired[b];
                auto& nz = num_zeros[b];
                const auto& tt = totals[b];

                std::fill(auc_buffer1.begin(), auc_buffer1.end(), 0);

                if (extra_details.threshold) {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer1.data(), extra_details.threshold, false);
                } else {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer1.data(), false);
                }

                // Adding to the blocks.
                for (size_t g = 0, end = auc_buffer2.size(); g < end; ++g) {
                    auc_buffer2[g] += auc_buffer1[g];
                }
            }

            for (size_t g1 = 0; g1 < ngroups; ++g1) {
                for (size_t g2 = 0; g2 < ngroups; ++g2) {
                    auto& current = auc_buffer2[g1 * ngroups + g2];
                    if (denominator[g1][g2]) {
                        current /= denominator[g1][g2];
                    } else {
                        current = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }

    public:
        template<typename T>
        void compute(size_t i, const T* ptr) {
            for (auto& z : num_zeros) {
                std::fill(z.begin(), z.end(), 0);
            }
            for (auto& p : paired) {
                p.clear();
            }

            for (size_t c = 0; c < simple_handler.details.NC; ++c) {
                auto b = extra_details.block[c];
                auto g = extra_details.group[c];
                if (ptr[c]) {
                    paired[b].push_back(std::make_pair(ptr[c], g));
                } else {
                    ++(num_zeros[b][g]);
                }
            }

            process_auc(i);
            worker.process(i, extra_details.ngroups, auc_buffer2.data());

            simpler.compute(i, ptr);
            return;
        }

        template<class SparseRange>
        void compute(size_t i, const SparseRange& range) {
            for (size_t b = 0; b < extra_details.nblocks; ++b) {
                std::copy(totals[b].begin(), totals[b].end(), num_zeros[b].begin());
            }
            for (auto& p : paired) {
                p.clear();
            }

            for (size_t j = 0; j < range.number; ++j) {
                if (range.value[j]) {
                    size_t c = range.index[j];
                    auto b = extra_details.block[c];
                    auto g = extra_details.group[c];
                    paired[b].push_back(std::make_pair(range.value[j], g));
                    --(num_zeros[b][g]);
                }
            }

            process_auc(i);
            worker.process(i, extra_details.ngroups, auc_buffer2.data());

            simpler.compute(i, range);
            return;
        }
    };

public:
    ByRow dense_direct() {
        return ByRow(details, overlord->complex_worker());
    }

    ByRow sparse_direct() {
        return ByRow(details, overlord->complex_worker());
    }
};

}

}

#endif
