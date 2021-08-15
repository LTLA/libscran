#ifndef SCRAN_BIDIMENSIONAL_FACTORY_HPP
#define SCRAN_BIDIMENSIONAL_FACTORY_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "../feature_selection/blocked_variances.hpp"
#include "cohens_d.hpp"
#include "auc.hpp"

namespace scran {

namespace differential_analysis {

/******* Base class for factories and workers ********/

template<typename Effect, typename Level, typename Stat>
struct Base {
    Base(std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
        means(m), detected(d), effects(e), levels(l), level_size(ls), ngroups(ng), nblocks(nb), threshold(t) {}

    Effect* cohen() {
        return effects[0];
    }

    Effect* auc() {
        return effects[1];
    }
    
    std::vector<Stat*>& means, detected;
    std::vector<Effect*>& effects;

    const Level* levels;
    const std::vector<int>& level_size;

    int ngroups, nblocks;
    double threshold;
};

/******* Factory with running support, when AUCs are not desired ********/

template<typename Effect, typename Level, typename Stat> 
struct BidimensionalFactory {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
        NR(nr), NC(nc), details(m, d, e, l, ls, ng, nb, t) {}

    static constexpr bool supports_sparse = true;
    static constexpr bool supports_running = true;

protected:
    size_t NR, NC;
    Base<Effect, Level, Stat> details;

public:
    struct ByRow { 
        ByRow(size_t nr, Base<Effect, Level, Stat>& d) : NR(nr), 
            tmp_means(d.level_size.size()), tmp_vars(d.level_size.size()), tmp_nzeros(d.level_size.size()), 
            buffer(d.ngroups * d.ngroups),
            details(d) {}

        void transfer(size_t i) {
            // Transferring the computed means.
            for (size_t l = 0; l < details.level_size.size(); ++l) {
                details.means[l][i] = tmp_means[l];
            }

            // Computing and transferring the proportion detected.
            for (size_t l = 0; l < details.level_size.size(); ++l) {
                auto& ndetected = details.detected[l][i];
                if (details.level_size[l]) {
                    ndetected = tmp_nzeros[l] / details.level_size[l];
                } else {
                    ndetected = std::numeric_limits<double>::quiet_NaN();
                }
            }

            if (details.cohen()) {
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), details.level_size, details.ngroups, details.nblocks, 
                    details.cohen() + i * details.ngroups * details.ngroups, details.threshold);
            }
        }

    protected:
        size_t NR;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros, buffer;
    public:
        Base<Effect, Level, Stat> details;
    };

public:
    struct DenseByRow : public ByRow {
        DenseByRow(size_t nr, size_t nc, Base<Effect, Level, Stat>& d) : NC(nc), ByRow(nr, d) {}

        template<typename T>
        void compute(size_t i, const T* ptr, T* buffer) {
            feature_selection::blocked_variance_with_mean<true>(ptr, NC, this->details.levels, this->details.level_size, this->tmp_means, this->tmp_vars);

            std::fill(this->tmp_nzeros.begin(), this->tmp_nzeros.end(), 0);
            for (size_t j = 0; j < NC; ++j) {
                this->tmp_nzeros[this->details.levels[j]] += (ptr[j] > 0);
            }

            this->transfer(i);
        }
    private:
        size_t NC;
    };

    DenseByRow dense_direct() {
        return DenseByRow(NR, NC, details);
    }

public:
    struct SparseByRow : public ByRow {
        SparseByRow(size_t nr, Base<Effect, Level, Stat>& d) : ByRow(nr, d) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            feature_selection::blocked_variance_with_mean<true>(range, this->details.levels, this->details.level_size, this->tmp_means, this->tmp_vars, this->tmp_nzeros);
            this->transfer(i);
        }
    };

    SparseByRow sparse_direct() {
        return SparseByRow(NR, details);
    }

private:
    struct ByCol {
        ByCol(size_t nr, Base<Effect, Level, Stat>& d) : NR(nr), tmp_vars(nr * d.level_size.size()), counts(d.level_size.size()), details(d) {}

        void finalize () { 
            // Dividing to obtain the proportion of detected cells per group.
            for (size_t b = 0; b < details.level_size.size(); ++b) {
                auto start = details.detected[b];
                if (details.level_size[b]) {
                    for (size_t r = 0; r < NR; ++r) {
                        start[r] /= details.level_size[b];
                    }
                } else {
                    std::fill(start, start + NR, std::numeric_limits<double>::quiet_NaN());
                }
            }

            // Finalizing Cohen's d. We transfer values to a temporary buffer
            // for cache efficiency upon repeated accesses in pairwise calculations.
            if (details.cohen()) {
                std::vector<double> tmp_means(details.level_size.size()), tmp_vars_single(details.level_size.size());
                auto estart = details.cohen();
                int shift = (details.ngroups) * (details.ngroups);
                for (size_t i = 0; i < NR; ++i, estart += shift) {
                    for (size_t l = 0; l < tmp_means.size(); ++l) {
                        tmp_means[l] = details.means[l][i];
                    }
                    for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                        tmp_vars_single[l] = tmp_vars[i + l * NR];
                    }
                    compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), details.level_size, details.ngroups, details.nblocks, estart, details.threshold);
                }
            }
        }
    protected:
        size_t NR;
        std::vector<double> tmp_vars;
        std::vector<int> counts;
        int counter = 0;
    public:
        Base<Effect, Level, Stat> details;
    };

public:
    struct DenseByCol : public ByCol {
        DenseByCol(size_t nr, Base<Effect, Level, Stat>& d) : ByCol(nr, d) {}

        template<typename T>
        void add(const T* ptr, T* buffer) {
            auto b = this->details.levels[this->counter];
            tatami::stats::variances::compute_running(ptr, this->NR, this->details.means[b], this->tmp_vars.data() + b * this->NR, this->counts[b]);

            auto ndetected = this->details.detected[b];
            for (size_t j = 0; j < this->NR; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->details.level_size.size(); ++b) {
                tatami::stats::variances::finish_running(this->NR, this->details.means[b], this->tmp_vars.data() + b * this->NR, this->counts[b]);
            }
            this->finalize();
        }
    };

    DenseByCol dense_running() {
        return DenseByCol(this->NR, this->details);
    }

public:
    struct SparseByCol : public ByCol {
        SparseByCol(size_t nr, Base<Effect, Level, Stat>& d) : ByCol(nr, d) {}

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            auto b = (this->details.levels)[this->counter];
            tatami::stats::variances::compute_running(range, this->details.means[b], this->tmp_vars.data() + b * this->NR, this->details.detected[b], this->counts[b]);
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->details.level_size.size(); ++b) {
                auto offset = b * this->NR; 
                tatami::stats::variances::finish_running(this->NR, this->details.means[b], this->tmp_vars.data() + b * this->NR, this->details.detected[b], this->counts[b]);
            }
            this->finalize();
        }
    };

    SparseByCol sparse_running() {
        return SparseByCol(this->NR, this->details);
    }
};

/******* Per-row factory when the AUC is desired ********/

template<typename Effect, typename Level, typename Stat, typename Group, typename Block> 
struct PerRowFactory {
public:
    PerRowFactory(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, 
        const Group* g, int ng, const Block* b, int nb, double t) : 
        group(g), block(b), factory(nr, nc, m, d, e, l, ls, ng, nb, t) {}

    static constexpr bool supports_sparse = true;
    static constexpr bool supports_running = false;
private:
    BidimensionalFactory<Effect, Level, Stat> factory;
    const Group* group;
    const Block* block;

public:
    template<class Component>
    struct ByRow {
        ByRow(const Group* g, const Block* b, Component c) : 
            component(std::move(c)), paired(component.details.nblocks), 
            num_zeros(component.details.nblocks, std::vector<int>(component.details.ngroups)), 
            totals(component.details.nblocks, std::vector<int>(component.details.ngroups)), 
            auc_buffer(component.details.ngroups * component.details.ngroups), 
            denominator(component.details.nblocks, std::vector<double>(component.details.ngroups * component.details.ngroups)),
            group(g), block(b)
        {
            const auto& ngroups = component.details.ngroups;
            const auto& nblocks = component.details.nblocks;

            auto lsIt = component.details.level_size.begin();
            for (size_t g = 0; g < ngroups; ++g) {
                for (size_t b = 0; b < nblocks; ++b, ++lsIt) {
                    totals[b][g] = *lsIt;
                }
            }

            for (size_t b = 0; b < nblocks; ++b) {
                for (int g1 = 0; g1 < ngroups; ++g1) {
                    for (int g2 = 0; g2 < ngroups; ++g1) {
                        denominator[b][g1 * ngroups + g2] += totals[b][g1] * totals[b][g2];
                    }
                }
            }

            return;
        }

        Component component;
        std::vector<PairedStore> paired;
        std::vector<std::vector<int> > num_zeros;
        std::vector<std::vector<int> > totals;
        std::vector<double> auc_buffer;
        std::vector<std::vector<double> > denominator;
        const Group* group;
        const Block* block;

        void process() {
            const auto& ngroups = component.details.ngroups;
            auto output = component.details.auc();
            for (size_t b = 0; b < component.details.nblocks; ++b) {
                auto& pr = paired[b];
                auto& nz = num_zeros[b];
                const auto& tt = totals[b];

                std::fill(auc_buffer.begin(), auc_buffer.end(), 0);

                if (component.details.threshold) {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), component.details.threshold, false);
                } else {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), false);
                }

                // Adding to the blocks.
                for (size_t g1 = 0; g1 < ngroups; ++g1) {
                    for (size_t g2 = 0; g2 < ngroups; ++g1) {
                        output[g1 * ngroups + g2] += auc_buffer[g1 * ngroups + g2];
                    }
                }
            }

            for (size_t g1 = 0; g1 < ngroups; ++g1) {
                for (size_t g2 = 0; g2 < ngroups; ++g1) {
                    if (denominator[g1][g2]) {
                        output[g1 * ngroups + g2] /= denominator[g1][g2];
                    } else {
                        output[g1 * ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }
    };

public:
    template<class Component>
    struct DenseByRow : public ByRow<Component>  {
        DenseByRow(const Group* g, const Block* b, Component c) : ByRow<Component>(g, b, std::move(c)) {}

        template<typename T>
        void compute(size_t i, const T* ptr, T* buffer) {
            for (auto& z : this->num_zeros) {
                std::fill(z.begin(), z.end(), 0);
            }
            for (auto& p : this->paired) {
                p.clear();
            }

            for (size_t c = 0; c < NC; ++c) {
                auto b = this->block[c];
                auto g = this->group[c];
                if (ptr[c]) {
                    this->paired[b].push_back(std::make_pair(ptr[c], g));
                } else {
                    ++(this->num_zeros[b][g]);
                }
            }

            this->process();
            this->component.template compute(i, ptr, buffer);
            return;
        }
    private:
        size_t NC;
    };

    auto dense_direct() {
        return DenseByRow<decltype(factory.dense_direct())>(group, block, factory.dense_direct());
    }

public:
    template<class Component>
    struct SparseByRow : public ByRow<Component>  {
        SparseByRow(const Group* g, const Block* b, Component c) : ByRow<Component>(g, b, std::move(c)) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            for (size_t b = 0; b < this->component.details.nblocks; ++b) {
                std::copy(this->totals[b].begin(), this->totals[b].end(), this->num_zeros[b].begin());
            }
            for (auto& p : this->paired) {
                p.clear();
            }

            for (size_t j = 0; j < range.number; ++j) {
                if (range.value[j]) {
                    size_t c = range.index[j];
                    auto b = this->block[c];
                    auto g = this->group[c];
                    this->paired[b].push_back(std::make_pair(range.value[j], g));
                    --(this->num_zeros[b][g]);
                }
            }

            this->process();
            this->component.template compute(i, range, xbuffer, ibuffer);
            return;
        }
    };

    auto sparse_direct() {
        return SparseByRow<decltype(factory.sparse_direct())>(group, block, factory.sparse_direct());
    }
};

}

}

#endif
