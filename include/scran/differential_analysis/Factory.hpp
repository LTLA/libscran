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

/******* Base base class for both factories ********/

template<typename Effect, typename Level, typename Stat> 
struct BaseFactory : public Base<Effect, Level, Stat> {
public:
    BaseFactory(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
        NR(nr), NC(nc), Base<Effect, Level, Stat>(m, d, e, l, ls, ng, nb, t) {}

    static constexpr bool supports_sparse = true;
protected:
    size_t NR, NC;

public:
    struct ByRow : public Base<Effect, Level, Stat> {
        ByRow(size_t nr, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            NR(nr), tmp_means(ls.size()), tmp_vars(ls.size()), tmp_nzeros(ls.size()), buffer(ng * ng),
            Base<Effect, Level, Stat>(m, d, e, l, ls, ng, nb, t) {}

        void transfer(size_t i) {
            // Transferring the computed means.
            for (size_t l = 0; l < this->level_size.size(); ++l) {
                this->means[l][i] = tmp_means[l];
            }

            // Computing and transferring the proportion detected.
            for (size_t l = 0; l < this->level_size.size(); ++l) {
                auto& ndetected = this->detected[l][i];
                if (this->level_size[l]) {
                    ndetected = tmp_nzeros[l] / this->level_size[l];
                } else {
                    ndetected = std::numeric_limits<double>::quiet_NaN();
                }
            }

            if (this->cohen()) {
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), this->level_size, this->ngroups, this->nblocks, 
                    this->cohen() + i * this->ngroups * this->ngroups, this->threshold);
            }
        }
    protected:
        size_t NR;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros, buffer;
    };

public:
    struct DenseByRow : public ByRow {
        DenseByRow(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            NC(nc), ByRow(nr, m, d, e, l, ls, ng, nb, t) {}

        template<typename T>
        void compute(size_t i, const T* ptr, T* buffer) {
            feature_selection::blocked_variance_with_mean<true>(ptr, NC, this->levels, this->level_size, this->tmp_means, this->tmp_vars);

            std::fill(this->tmp_nzeros.begin(), this->tmp_nzeros.end(), 0);
            for (size_t j = 0; j < NC; ++j) {
                this->tmp_nzeros[this->levels[j]] += (ptr[j] > 0);
            }

            this->transfer(i);
        }
    private:
        size_t NC;
    };

    DenseByRow dense_direct() {
        return DenseByRow(NR, NC, this->means, this->detected, this->effects, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }

public:
    struct SparseByRow : public ByRow {
        SparseByRow(size_t nr, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            ByRow(nr, m, d, e, l, ls, ng, nb, t) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            feature_selection::blocked_variance_with_mean<true>(range, this->levels, this->level_size, this->tmp_means, this->tmp_vars, this->tmp_nzeros);
            this->transfer(i);
        }
    };

    SparseByRow sparse_direct() {
        return SparseByRow(NR, this->means, this->detected, this->effects, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }
};

/******* Factory with running support, when AUCs are not desired ********/

template<typename Effect, typename Level, typename Stat> 
struct BidimensionalFactory : public BaseFactory<Effect, Level, Stat> {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
        BaseFactory<Effect, Level, Stat>(nr, nc, m, d, e, l, ls, ng, nb, t) {}

    static constexpr bool supports_running = true;

private:
    struct ByCol : public Base<Effect, Level, Stat> {
        ByCol(size_t nr, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            NR(nr), tmp_vars(nr * ls.size()), counts(ls.size()), Base<Effect, Level, Stat>(m, d, e, l, ls, ng, nb, t) {} 

        void finalize () { 
            // Dividing to obtain the proportion of detected cells per group.
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                auto start = this->detected[b];
                if (this->level_size[b]) {
                    for (size_t r = 0; r < NR; ++r) {
                        start[r] /= this->level_size[b];
                    }
                } else {
                    std::fill(start, start + NR, std::numeric_limits<double>::quiet_NaN());
                }
            }

            // Finalizing Cohen's d. We transfer values to a temporary buffer
            // for cache efficiency upon repeated accesses in pairwise calculations.
            if (this->cohen()) {
                std::vector<double> tmp_means(this->level_size.size()), tmp_vars_single(this->level_size.size());
                auto estart = this->cohen();
                int shift = (this->ngroups) * (this->ngroups);
                for (size_t i = 0; i < NR; ++i, estart += shift) {
                    for (size_t l = 0; l < tmp_means.size(); ++l) {
                        tmp_means[l] = this->means[l][i];
                    }
                    for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                        tmp_vars_single[l] = tmp_vars[i + l * NR];
                    }
                    compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), this->level_size, this->ngroups, this->nblocks, estart, this->threshold);
                }
            }
        }
    protected:
        size_t NR;
        std::vector<double> tmp_vars;
        std::vector<int> counts;
        int counter = 0;
    };

public:
    struct DenseByCol : public ByCol {
        DenseByCol(size_t nr, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            ByCol(nr, m, d, e, l, ls, ng, nb, t) {}

        template<typename T>
        void add(const T* ptr, T* buffer) {
            auto b = this->levels[this->counter];
            tatami::stats::variances::compute_running(ptr, this->NR, this->means[b], this->tmp_vars.data() + b * this->NR, this->counts[b]);

            auto ndetected = this->detected[b];
            for (size_t j = 0; j < this->NR; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                tatami::stats::variances::finish_running(this->NR, this->means[b], this->tmp_vars.data() + b * this->NR, this->counts[b]);
            }
            this->finalize();
        }
    };

    DenseByCol dense_running() {
        return DenseByCol(this->NR, this->means, this->detected, this->effects, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }

public:
    struct SparseByCol : public ByCol {
        SparseByCol(size_t nr, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, int ng, int nb, double t) : 
            ByCol(nr, m, d, e, l, ls, ng, nb, t) {}

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            auto b = this->levels[this->counter];
            tatami::stats::variances::compute_running(range, this->means[b], this->tmp_vars.data() + b * this->NR, this->detected[b], this->counts[b]);
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                auto offset = b * this->NR; 
                tatami::stats::variances::finish_running(this->NR, this->means[b], this->tmp_vars.data() + b * this->NR, this->detected[b], this->counts[b]);
            }
            this->finalize();
        }
    };

    SparseByCol sparse_running() {
        return SparseByCol(this->NR, this->means, this->detected, this->effects, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }
};

/******* Per-row factory when the AUC is desired ********/

template<typename Effect, typename Level, typename Stat, typename Group, typename Block> 
struct PerRowFactory : public BaseFactory<Effect, Level, Stat> {
public:
    PerRowFactory(size_t nr, size_t nc, std::vector<Stat*>& m, std::vector<Stat*>& d, std::vector<Effect*>& e, const Level* l, const std::vector<int>& ls, 
        const Group* g, int ng, const Block* b, int nb, double t) : 
        group(g), block(b), BaseFactory<Effect, Level, Stat>(nr, nc, m, d, e, l, ls, ng, nb, t) {}

    static constexpr bool supports_running = false;
private:
    const Group* group;
    const Block* block;

public:
    template<class Component>
    struct ByRow {
        ByRow(const Group* g, const Block* b, Component c) : 
            component(std::move(c)), paired(component.nblocks), 
            num_zeros(component.nblocks, std::vector<int>(component.ngroups)), 
            totals(component.nblocks, std::vector<int>(component.ngroups)), 
            auc_buffer(component.ngroups * component.ngroups), 
            denominator(component.nblocks, std::vector<double>(component.ngroups * component.ngroups)),
            group(g), block(b)
        {
            auto lsIt = component.level_size.begin();
            for (size_t g = 0; g < component.ngroups; ++g) {
                for (size_t b = 0; b < component.nblocks; ++b, ++lsIt) {
                    totals[b][g] = *lsIt;
                }
            }

            for (size_t b = 0; b < component.nblocks; ++b) {
                for (int g1 = 0; g1 < component.ngroups; ++g1) {
                    for (int g2 = 0; g2 < component.ngroups; ++g1) {
                        denominator[b][g1 * component.ngroups + g2] += totals[b][g1] * totals[b][g2];
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
            auto output = component.auc();
            for (size_t b = 0; b < component.nblocks; ++b) {
                auto& pr = paired[b];
                auto& nz = num_zeros[b];
                const auto& tt = totals[b];

                std::fill(auc_buffer.begin(), auc_buffer.end(), 0);

                if (component.threshold) {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), component.threshold, false);
                } else {
                    compute_pairwise_auc(pr, nz, tt, auc_buffer.data(), false);
                }

                // Adding to the blocks.
                for (size_t g1 = 0; g1 < component.ngroups; ++g1) {
                    for (size_t g2 = 0; g2 < component.ngroups; ++g1) {
                        output[g1 * component.ngroups + g2] += auc_buffer[g1 * component.ngroups + g2];
                    }
                }
            }

            for (size_t g1 = 0; g1 < component.ngroups; ++g1) {
                for (size_t g2 = 0; g2 < component.ngroups; ++g1) {
                    if (denominator[g1][g2]) {
                        output[g1 * component.ngroups + g2] /= denominator[g1][g2];
                    } else {
                        output[g1 * component.ngroups + g2] = std::numeric_limits<double>::quiet_NaN();
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
        return DenseByRow(group, block, BaseFactory<Effect, Level, Stat>::dense_direct());
    }

public:
    template<class Component>
    struct SparseByRow : public ByRow<Component>  {
        SparseByRow(const Group* g, const Block* b, Component c) : ByRow<Component>(g, b, std::move(c)) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            for (size_t b = 0; b < this->component.nblocks; ++b) {
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
        return SparseByRow(group, block, BaseFactory<Effect, Level, Stat>::sparse_direct());
    }
};

}

}

#endif
