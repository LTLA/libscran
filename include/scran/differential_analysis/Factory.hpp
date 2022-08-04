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

/******* Common functions for per-row filling ********/

namespace per_row {

template<class Bundle, typename Data, typename Stat>
void fill_tmp_nzeros(const Bundle& details, const Data* ptr, std::vector<Stat>& tmp_nzeros) {
    std::fill(tmp_nzeros.begin(), tmp_nzeros.end(), 0);
    for (size_t j = 0; j < details.NC; ++j) {
        tmp_nzeros[details.levels[j]] += (ptr[j] > 0);
    }
}

template<class Bundle, typename Stat>
void transfer_common_stats(size_t row, const std::vector<Stat>& tmp_means, const std::vector<Stat>& tmp_nzeros, const std::vector<Stat>& tmp_vars, Bundle& details) {
    const auto& level_size = *(details.level_size_ptr);

    // Transferring the computed means.
    for (size_t l = 0; l < level_size.size(); ++l) {
        details.means[l][row] = tmp_means[l];
    }

    // Computing and transferring the proportion detected.
    for (size_t l = 0; l < level_size.size(); ++l) {
        auto& ndetected = details.detected[l][row];
        if (level_size[l]) {
            ndetected = tmp_nzeros[l] / level_size[l];
        } else {
            ndetected = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Computing the various effect sizes.
    size_t offset = row * details.ngroups * details.ngroups;
    if (details.cohen) {
        compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), level_size, details.ngroups, details.nblocks, details.cohen + offset, details.threshold);
    }
    if (details.lfc) {
        compute_pairwise_lfc(tmp_means.data(), level_size, details.ngroups, details.nblocks, details.lfc + offset);
    }
    if (details.delta_detected) {
        compute_pairwise_delta_detected(tmp_nzeros.data(), level_size, details.ngroups, details.nblocks, details.delta_detected + offset);
    }
}

}

/******* Factory with running support, when AUCs are not desired ********/

template<typename Stat, typename Level>
struct BidimensionalBundle {
    BidimensionalBundle(size_t nr, size_t nc, std::vector<Stat*> m, std::vector<Stat*> d, Stat* cohen_, Stat* lfc_, Stat* delta_detected_, const Level* l, const std::vector<int>* ls, int ng, int nb, double t) : 
        NR(nr), NC(nc), means(std::move(m)), detected(std::move(d)), cohen(cohen_), lfc(lfc_), delta_detected(delta_detected_), levels(l), level_size_ptr(ls), ngroups(ng), nblocks(nb), threshold(t) {}

    size_t NR, NC;

    std::vector<Stat*> means, detected;
    Stat *cohen;
    Stat *lfc;
    Stat *delta_detected;

    const Level* levels;
    const std::vector<int>* level_size_ptr;

    int ngroups, nblocks;
    double threshold;
};

template<typename Stat, typename Level> 
struct BidimensionalFactory {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Stat*> m, std::vector<Stat*> d, Stat* cohen, Stat* lfc, Stat* delta_detected, const Level* l, const std::vector<int>* ls, int ng, int nb, double t) : 
        details(nr, nc, std::move(m), std::move(d), cohen, lfc, delta_detected, l, ls, ng, nb, t) {}

protected:
    BidimensionalBundle<Stat, Level> details;

public:
    struct ByRow { 
        ByRow(BidimensionalBundle<Stat, Level> d) :
            details(std::move(d)),
            tmp_means(details.level_size_ptr->size()), 
            tmp_vars(details.level_size_ptr->size()), 
            tmp_nzeros(details.level_size_ptr->size())
        {}

    private:
        BidimensionalBundle<Stat, Level> details;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros;

    public:
        template<typename T>
        void compute(size_t i, const T* ptr) {
            feature_selection::blocked_variance_with_mean<true>(ptr, details.NC, details.levels, *(details.level_size_ptr), tmp_means, tmp_vars);
            per_row::fill_tmp_nzeros(details, ptr, tmp_nzeros);
            per_row::transfer_common_stats(i, tmp_means, tmp_nzeros, tmp_vars, details);
        }

        template<class SparseRange>
        void compute(size_t i, const SparseRange& range) {
            feature_selection::blocked_variance_with_mean<true>(range, details.levels, *(details.level_size_ptr), tmp_means, tmp_vars, tmp_nzeros);
            per_row::transfer_common_stats(i, tmp_means, tmp_nzeros, tmp_vars, details);
        }
    };

    ByRow dense_direct() const {
        return ByRow(details);
    }

    ByRow sparse_direct() const {
        return ByRow(details);
    }

public:
    struct ByCol { 
        ByCol(BidimensionalBundle<Stat, Level> d, size_t num) : 
            details(std::move(d)),
            tmp_vars(details.level_size_ptr->size(), std::vector<double>(num)), 
            counts(details.level_size_ptr->size())
        {}

        void finalize_by_cols(size_t start, size_t end) {
            const auto& level_size = *(details.level_size_ptr);
            size_t shift = (details.ngroups) * (details.ngroups);
            size_t offset = shift * start;

            auto dd_ptr = details.delta_detected;
            if (dd_ptr) {
                std::vector<int> tmp_detected(level_size.size());
                dd_ptr += offset;
                for (size_t i = start; i < end; ++i) {
                    for (size_t l = 0; l < tmp_detected.size(); ++l) {
                        tmp_detected[l] = details.detected[l][i];
                    }
                    compute_pairwise_delta_detected(tmp_detected.data(), level_size, details.ngroups, details.nblocks, dd_ptr);
                    dd_ptr += shift;
                } 
            }

            // Dividing to obtain the proportion of detected cells per group.
            for (size_t b = 0; b < level_size.size(); ++b) {
                auto ptr = details.detected[b];
                if (level_size[b]) {
                    for (size_t r = start; r < end; ++r) {
                        ptr[r] /= level_size[b];
                    }
                } else {
                    std::fill(ptr + start, ptr + end, std::numeric_limits<double>::quiet_NaN());
                }
            }

            // We transfer values to a temporary buffer for cache efficiency upon
            // repeated accesses in pairwise calculations.
            auto cohen_ptr = details.cohen;
            auto lfc_ptr = details.lfc;
            if (cohen_ptr || lfc_ptr) {
                std::vector<double> tmp_means(level_size.size()), tmp_vars_single(level_size.size());
                if (cohen_ptr) {
                    cohen_ptr += offset;
                }
                if (lfc_ptr) {
                    lfc_ptr += offset;
                }

                for (size_t i = start; i < end; ++i) {
                    for (size_t l = 0; l < tmp_means.size(); ++l) {
                        tmp_means[l] = details.means[l][i];
                    }
                    if (cohen_ptr) {
                        for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                            tmp_vars_single[l] = tmp_vars[l][i];
                        }
                        compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), level_size, details.ngroups, details.nblocks, cohen_ptr, details.threshold);
                        cohen_ptr += shift;
                    }
                    if (lfc_ptr) {
                        compute_pairwise_lfc(tmp_means.data(), level_size, details.ngroups, details.nblocks, lfc_ptr);
                        lfc_ptr += shift;
                    }
                }
            }
        }
    protected:
        BidimensionalBundle<Stat, Level> details;
        std::vector<std::vector<double> > tmp_vars;
        std::vector<int> counts;
        size_t counter = 0;
    };

    struct DenseByCol : public ByCol {
        DenseByCol(size_t n, BidimensionalBundle<Stat, Level> d) : ByCol(std::move(d), n), num(n) {}

        template<typename T>
        void add(const T* ptr) {
            auto b = this->details.levels[this->counter];
            tatami::stats::variances::compute_running(ptr, num, this->details.means[b], this->tmp_vars[b].data(), this->counts[b]);

            auto ndetected = this->details.detected[b];
            for (size_t j = 0; j < num; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->details.level_size_ptr->size(); ++b) {
                tatami::stats::variances::finish_running(num, this->details.means[b], this->tmp_vars[b].data(), this->counts[b]);
            }
            this->finalize_by_cols(0, num);
        }
    private:
        size_t num;
    };

    DenseByCol dense_running() const {
        return DenseByCol(details.NR, details);
    }

    DenseByCol dense_running(size_t start, size_t end) const {
        auto copy = this->details;

        // Shifting all the pointers to pretend that we have a smaller number
        // of rows, so that we can reduce the size of the temporary vectors. 
        for (auto& m : copy.means) {
            m += start;
        }
        for (auto& d : copy.detected) {
            d += start;
        }

        size_t shift = copy.ngroups * copy.ngroups * start;
        copy.cohen += shift;
        copy.lfc += shift;
        copy.delta_detected += shift;

        return DenseByCol(end - start, std::move(copy));
    }

    struct SparseByCol : public ByCol { 
        SparseByCol(size_t nr, size_t s, size_t e, BidimensionalBundle<Stat, Level> d) : ByCol(std::move(d), nr), start(s), end(e) {}

        template<class SparseRange>
        void add(const SparseRange& range) {
            auto b = this->details.levels[this->counter];
            
            // TODO: add a offset to tatami so that we can subtract an offset from the range.
            tatami::stats::variances::compute_running(range, this->details.means[b], this->tmp_vars[b].data(), this->details.detected[b], this->counts[b]);

            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->details.level_size_ptr->size(); ++b) {
                tatami::stats::variances::finish_running(
                    end - start, 
                    this->details.means[b] + start, 
                    this->tmp_vars[b].data() + start, 
                    this->details.detected[b] + start, 
                    this->counts[b]
                );
            }
            this->finalize_by_cols(start, end);
        }
    private:
        size_t start, end;
    };

    SparseByCol sparse_running() {
        return SparseByCol(details.NR, 0, details.NR, details);
    }

    SparseByCol sparse_running(size_t start, size_t end) {
        // Just making the temporary vectors with all rows, so that we don't 
        // have to worry about subtracting the indices when doing sparse iteration.
        return SparseByCol(details.NR, start, end, details);
    }
};

/******* Per-row factory when the AUC is desired ********/

template<typename Stat, typename Level, typename Group, typename Block>
struct PerRowBundle {
    PerRowBundle(
        size_t nr,
        size_t nc,
        std::vector<Stat*> m,
        std::vector<Stat*> d,
        Stat* cohen_,
        Stat* auc_,
        Stat* lfc_,
        Stat* delta_detected_,
        const Level* l,
        const std::vector<int>* ls,
        const Group* g,
        int ng,
        const Block* b,
        int nb,
        double t
    ) : 
        NR(nr), 
        NC(nc), 
        means(std::move(m)), 
        detected(std::move(d)), 
        cohen(cohen_), 
        auc(auc_), 
        lfc(lfc_), 
        delta_detected(delta_detected_), 
        levels(l), 
        level_size_ptr(ls), 
        group(g),
        block(b),
        ngroups(ng), 
        nblocks(nb), 
        threshold(t) 
    {}

    size_t NR, NC;

    std::vector<Stat*> means, detected;
    Stat *cohen;
    Stat *auc;
    Stat *lfc;
    Stat *delta_detected;

    const Level* levels;
    const std::vector<int>* level_size_ptr;

    const Group* group;
    const Block* block;
    int ngroups, nblocks;

    double threshold;
};

template<typename Stat, typename Level, typename Group, typename Block> 
struct PerRowFactory {
public:
    PerRowFactory(
        size_t nr,
        size_t nc,
        std::vector<Stat*> m,
        std::vector<Stat*> d,
        Stat* cohen,
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected,
        const Level* l,
        const std::vector<int>* ls,
        const Group* g,
        int ng,
        const Block* b,
        int nb,
        double t
    ) : details(nr, nc, std::move(m), std::move(d), cohen, auc, lfc, delta_detected, l, ls, g, ng, b, nb, t) {}

protected:
    PerRowBundle<Stat, Level, Group, Block> details;

public:
    struct ByRow {
        ByRow(PerRowBundle<Stat, Level, Group, Block> d) : 
            details(std::move(d)), 

            tmp_means(details.level_size_ptr->size()), 
            tmp_vars(details.level_size_ptr->size()), 
            tmp_nzeros(details.level_size_ptr->size()), 

            paired(details.nblocks), 
            num_zeros(details.nblocks, std::vector<int>(details.ngroups)), 
            totals(details.nblocks, std::vector<int>(details.ngroups)), 
            auc_buffer(details.ngroups * details.ngroups), 
            denominator(details.ngroups, std::vector<double>(details.ngroups))
        {
            const auto& ngroups = details.ngroups;
            const auto& nblocks = details.nblocks;

            auto lsIt = details.level_size_ptr->begin();
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
        PerRowBundle<Stat, Level, Group, Block> details;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros;

        // AUC handlers.
        std::vector<PairedStore> paired;
        std::vector<std::vector<int> > num_zeros;
        std::vector<std::vector<int> > totals;
        std::vector<double> auc_buffer;
        std::vector<std::vector<double> > denominator;

        void process_auc(size_t i) {
            const auto& ngroups = details.ngroups;
            auto output = details.auc + i * ngroups * ngroups;

            for (size_t b = 0; b < details.nblocks; ++b) {
                auto& pr = paired[b];
                auto& nz = num_zeros[b];
                const auto& tt = totals[b];

                std::fill(auc_buffer.begin(), auc_buffer.end(), 0);

                if (details.threshold) {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), details.threshold, false);
                } else {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), false);
                }

                // Adding to the blocks.
                for (size_t g1 = 0; g1 < ngroups; ++g1) {
                    for (size_t g2 = 0; g2 < ngroups; ++g2) {
                        output[g1 * ngroups + g2] += auc_buffer[g1 * ngroups + g2];
                    }
                }
            }

            for (size_t g1 = 0; g1 < ngroups; ++g1) {
                for (size_t g2 = 0; g2 < ngroups; ++g2) {
                    if (denominator[g1][g2]) {
                        output[g1 * ngroups + g2] /= denominator[g1][g2];
                    } else {
                        output[g1 * ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
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

            for (size_t c = 0; c < details.NC; ++c) {
                auto b = details.block[c];
                auto g = details.group[c];
                if (ptr[c]) {
                    paired[b].push_back(std::make_pair(ptr[c], g));
                } else {
                    ++(num_zeros[b][g]);
                }
            }

            process_auc(i);

            // And also computing everything else.
            feature_selection::blocked_variance_with_mean<true>(ptr, details.NC, details.levels, *(details.level_size_ptr), tmp_means, tmp_vars);
            per_row::fill_tmp_nzeros(details, ptr, tmp_nzeros);
            per_row::transfer_common_stats(i, tmp_means, tmp_nzeros, tmp_vars, details);
            return;
        }

        template<class SparseRange>
        void compute(size_t i, const SparseRange& range) {
            for (size_t b = 0; b < details.nblocks; ++b) {
                std::copy(totals[b].begin(), totals[b].end(), num_zeros[b].begin());
            }
            for (auto& p : paired) {
                p.clear();
            }

            for (size_t j = 0; j < range.number; ++j) {
                if (range.value[j]) {
                    size_t c = range.index[j];
                    auto b = details.block[c];
                    auto g = details.group[c];
                    paired[b].push_back(std::make_pair(range.value[j], g));
                    --(num_zeros[b][g]);
                }
            }

            process_auc(i);

            // And also computing everything else.
            feature_selection::blocked_variance_with_mean<true>(range, details.levels, *(details.level_size_ptr), tmp_means, tmp_vars, tmp_nzeros);
            per_row::transfer_common_stats(i, tmp_means, tmp_nzeros, tmp_vars, details);
            return;
        }
    };

    ByRow dense_direct() {
        return ByRow(details);
    }

    ByRow sparse_direct() {
        return ByRow(details);
    }
};

}

}

#endif
