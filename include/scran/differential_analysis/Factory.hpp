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

template<typename Stat, typename Level>
struct Base {
    Base(std::vector<Stat*> m, std::vector<Stat*> d, std::vector<Stat*> e, const Level* l, const std::vector<int>* ls, int ng, int nb, double t) : 
        means(std::move(m)), detected(std::move(d)), effects(std::move(e)), levels(l), level_size_ptr(ls), ngroups(ng), nblocks(nb), threshold(t) {}

    Stat* cohen() {
        return effects[0];
    }

    Stat* auc() {
        return effects[1];
    }
    
    std::vector<Stat*> means, detected;
    std::vector<Stat*> effects;

    const Level* levels;
    const std::vector<int>* level_size_ptr;

    int ngroups, nblocks;
    double threshold;
};

/******* Factory with running support, when AUCs are not desired ********/

template<typename Stat, typename Level> 
struct BidimensionalFactory {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Stat*> m, std::vector<Stat*> d, std::vector<Stat*> e, const Level* l, const std::vector<int>* ls, int ng, int nb, double t) : 
        NR(nr), NC(nc), details(std::move(m), std::move(d), std::move(e), l, ls, ng, nb, t) {}

protected:
    size_t NR, NC;
    Base<Stat, Level> details;

public:
    struct ByRow { 
        ByRow(Base<Stat, Level> d) :
            tmp_means(d.level_size_ptr->size()), 
            tmp_vars(d.level_size_ptr->size()), 
            tmp_nzeros(d.level_size_ptr->size()), 
            buffer(d.ngroups * d.ngroups),
            details(std::move(d)) {}

        void transfer(size_t i) {
            const auto& level_size = *(details.level_size_ptr);

            // Transferring the computed means.
            for (size_t l = 0; l < level_size.size(); ++l) {
                details.means[l][i] = tmp_means[l];
            }

            // Computing and transferring the proportion detected.
            for (size_t l = 0; l < level_size.size(); ++l) {
                auto& ndetected = details.detected[l][i];
                if (level_size[l]) {
                    ndetected = tmp_nzeros[l] / level_size[l];
                } else {
                    ndetected = std::numeric_limits<double>::quiet_NaN();
                }
            }

            if (details.cohen()) {
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), level_size, details.ngroups, details.nblocks, 
                    details.cohen() + i * details.ngroups * details.ngroups, details.threshold);
            }
        }

    protected:
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros, buffer;
    public:
        Base<Stat, Level> details;
    };

public:
    struct DenseByRow : public ByRow {
        DenseByRow(size_t nc, Base<Stat, Level> d) : NC(nc), ByRow(std::move(d)) {}

        template<typename T>
        void compute(size_t i, const T* ptr) {
            feature_selection::blocked_variance_with_mean<true>(ptr, NC, this->details.levels, *(this->details.level_size_ptr), this->tmp_means, this->tmp_vars);

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
        return DenseByRow(NC, details);
    }

public:
    struct SparseByRow : public ByRow {
        SparseByRow(Base<Stat, Level> d) : ByRow(std::move(d)) {}

        template<class SparseRange>
        void compute(size_t i, SparseRange&& range) {
            feature_selection::blocked_variance_with_mean<true>(range, this->details.levels, *(this->details.level_size_ptr), this->tmp_means, this->tmp_vars, this->tmp_nzeros);
            this->transfer(i);
        }
    };

    SparseByRow sparse_direct() {
        return SparseByRow(details);
    }

private:
    static void finalize_by_cols(size_t start, size_t end, const std::vector<std::vector<double> >& tmp_vars, Base<Stat, Level>& details) {
        const auto& level_size = *(details.level_size_ptr);

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

        // Finalizing Cohen's d. We transfer values to a temporary buffer
        // for cache efficiency upon repeated accesses in pairwise calculations.
        if (details.cohen()) {
            std::vector<double> tmp_means(level_size.size()), tmp_vars_single(level_size.size());
            int shift = (details.ngroups) * (details.ngroups);
            auto estart = details.cohen() + shift * start;

            for (size_t i = start; i < end; ++i, estart += shift) {
                for (size_t l = 0; l < tmp_means.size(); ++l) {
                    tmp_means[l] = details.means[l][i];
                }
                for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                    tmp_vars_single[l] = tmp_vars[l][i];
                }
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), level_size, details.ngroups, details.nblocks, estart, details.threshold);
            }
        }
    };

public:
    struct DenseByCol {
        DenseByCol(size_t start, size_t end, Base<Stat, Level> d) : 
            num(end - start), 
            tmp_vars(d.level_size_ptr->size(), std::vector<double>(num)), 
            counts(d.level_size_ptr->size()),
            details(std::move(d)) {}

        template<typename T>
        void add(const T* ptr) {
            auto b = details.levels[counter];
            tatami::stats::variances::compute_running(ptr, num, details.means[b], tmp_vars[b].data(), counts[b]);

            auto ndetected = details.detected[b];
            for (size_t j = 0; j < num; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++counter;
        }

        void finish() {
            for (size_t b = 0; b < details.level_size_ptr->size(); ++b) {
                tatami::stats::variances::finish_running(num, details.means[b], tmp_vars[b].data(), counts[b]);
            }
            finalize_by_cols(0, num, tmp_vars, details);
        }
    private:
        size_t counter = 0, num;
        std::vector<std::vector<double> > tmp_vars;
        std::vector<int> counts;
        Base<Stat, Level> details;
    };

    DenseByCol dense_running() {
        return DenseByCol(0, this->NR, this->details);
    }

    DenseByCol dense_running(size_t start, size_t end) {
        auto copy = this->details;

        for (auto& m : copy.means) {
            m += start;
        }
        for (auto& d : copy.detected) {
            d += start;
        }

        size_t shift = copy.ngroups * copy.ngroups * start;
        for (auto& e : copy.effects) {
            e += shift;
        }

        return DenseByCol(start, end, std::move(copy));
    }

public:
    struct SparseByCol { 
        SparseByCol(size_t nr, size_t s, size_t e, Base<Stat, Level> d) : 
            start(s), end(e), 
            tmp_vars(d.level_size_ptr->size(), std::vector<double>(nr)), 
            counts(d.level_size_ptr->size()),
            details(std::move(d)) {}

        template<class SparseRange>
        void add(SparseRange&& range) {
            auto b = details.levels[counter];
            tatami::stats::variances::compute_running(range, details.means[b], tmp_vars[b].data(), details.detected[b], counts[b]);
            ++counter;
        }

        void finish() {
            for (size_t b = 0; b < details.level_size_ptr->size(); ++b) {
                tatami::stats::variances::finish_running(end - start, 
                    details.means[b] + start, 
                    tmp_vars[b].data() + start, 
                    details.detected[b] + start, 
                    counts[b]);
            }
            finalize_by_cols(start, end, tmp_vars, details);
        }
    private:
        size_t start, end, counter = 0;
        std::vector<std::vector<double> > tmp_vars;
        std::vector<int> counts;
        Base<Stat, Level> details;
    };

    SparseByCol sparse_running() {
        return SparseByCol(NR, 0, NR, this->details);
    }

    SparseByCol sparse_running(size_t start, size_t end) {
        return SparseByCol(NR, start, end, this->details);
    }
};

/******* Per-row factory when the AUC is desired ********/

template<typename Stat, typename Level, typename Group, typename Block> 
struct PerRowFactory {
public:
    PerRowFactory(size_t nr, size_t nc, std::vector<Stat*> m, std::vector<Stat*> d, std::vector<Stat*> e, const Level* l, const std::vector<int>* ls, 
        const Group* g, int ng, const Block* b, int nb, double t) : 
        NC(nc), group(g), block(b), 
        factory(nr, nc, std::move(m), std::move(d), std::move(e), l, ls, ng, nb, t) {}

    static constexpr bool supports_sparse = true;
    static constexpr bool supports_running = false;
private:
    size_t NC;
    const Group* group;
    const Block* block;
    BidimensionalFactory<Stat, Level> factory;

public:
    template<class Component>
    struct ByRow {
        ByRow(const Group* g, const Block* b, Component c) : 
            component(std::move(c)), 
            paired(component.details.nblocks), 
            num_zeros(component.details.nblocks, std::vector<int>(component.details.ngroups)), 
            totals(component.details.nblocks, std::vector<int>(component.details.ngroups)), 
            auc_buffer(component.details.ngroups * component.details.ngroups), 
            denominator(component.details.ngroups, std::vector<double>(component.details.ngroups)),
            group(g), block(b)
        {
            const auto& ngroups = component.details.ngroups;
            const auto& nblocks = component.details.nblocks;

            auto lsIt = component.details.level_size_ptr->begin();
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

        Component component;
        std::vector<PairedStore> paired;
        std::vector<std::vector<int> > num_zeros;
        std::vector<std::vector<int> > totals;
        std::vector<double> auc_buffer;
        std::vector<std::vector<double> > denominator;
        const Group* group;
        const Block* block;

        void process(size_t i) {
            const auto& ngroups = component.details.ngroups;
            auto output = component.details.auc() + i * ngroups * ngroups;

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
    };

public:
    template<class Component>
    struct DenseByRow : public ByRow<Component>  {
        DenseByRow(size_t nc, const Group* g, const Block* b, Component c) : NC(nc), ByRow<Component>(g, b, std::move(c)) {}

        template<typename T>
        void compute(size_t i, const T* ptr) {
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

            this->process(i);
            this->component.template compute(i, ptr);
            return;
        }
    private:
        size_t NC;
    };

    auto dense_direct() {
        return DenseByRow<decltype(factory.dense_direct())>(NC, group, block, factory.dense_direct());
    }

public:
    template<class Component>
    struct SparseByRow : public ByRow<Component>  {
        SparseByRow(const Group* g, const Block* b, Component c) : ByRow<Component>(g, b, std::move(c)) {}

        template<class SparseRange>
        void compute(size_t i, SparseRange&& range) {
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

            this->process(i);
            this->component.template compute(i, range);
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
