#ifndef SCRAN_BIDIMENSIONAL_FACTORY_HPP
#define SCRAN_BIDIMENSIONAL_FACTORY_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "cohens_d.hpp"
#include "summarize_comparisons.hpp"
#include "../feature_selection/blocked_variances.hpp"

namespace scran {

namespace differential_analysis {

template<typename Effect, typename Level, typename Stat>
struct Common {
    Common(std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
        pairwise_effects(effects), cluster_stats(clust), levels(l), level_size(ls), ngroups(ng), nblocks(nb), threshold(t) {}

    std::vector<Effect*>& pairwise_effects;
    std::vector<Stat*>& cluster_stats;
    const Level* levels;
    std::vector<int>& level_size;
    int ngroups, nblocks;
    double threshold;
};

template<typename Effect, typename Level, typename Stat> 
struct BidimensionalFactory : public Common<Effect, Level, Stat> {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
        NR(nr), NC(nc), Common<Effect, Level, Stat>(effects, clust, l, ls, ng, nb, t) {}

    static constexpr bool supports_running = true;
    static constexpr bool supports_sparse = true;
private:
    size_t NR, NC;

public:
    struct ByRow : public Common<Effect, Level, Stat> {
        ByRow(size_t nr, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            NR(nr), tmp_means(ls.size()), tmp_vars(ls.size()), tmp_nzeros(ls.size()), buffer(ng * ng),
            Common<Effect, Level, Stat>(effects, clust, l, ls, ng, nb, t) {}

        void transfer(size_t i) {

            // Transferring the computed means.
            auto means = this->cluster_stats[0] + i;
            for (size_t l = 0; l < this->level_size.size(); ++l, means += NR) {
                *means = tmp_means[l];
            }

            // Computing and transferring the proportion detected.
            auto ndetected = this->cluster_stats[1] + i;
            for (size_t l = 0; l < this->level_size.size(); ++l, ndetected += NR) {
                if (this->level_size[l]) {
                    *ndetected = tmp_nzeros[l] / this->level_size[l];
                } else {
                    *ndetected = std::numeric_limits<double>::quiet_NaN();
                }
            }

            compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), this->level_size, this->ngroups, this->nblocks, 
                this->pairwise_effects[0] + i * this->ngroups * this->ngroups, this->threshold);
        }
    protected:
        size_t NR;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros, buffer;
    };

public:
    struct DenseByRow : public ByRow {
        DenseByRow(size_t nr, size_t nc, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            NC(nc), ByRow(nr, effects, clust, l, ls, ng, nb, t) {}

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
        return DenseByRow(NR, NC, this->pairwise_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }

public:
    struct SparseByRow : public ByRow {
        SparseByRow(size_t nr, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            ByRow(nr, effects, clust, l, ls, ng, nb, t) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            feature_selection::blocked_variance_with_mean<true>(range, this->levels, this->level_size, this->tmp_means, this->tmp_vars, this->tmp_nzeros);
            this->transfer(i);
        }
    };

    SparseByRow sparse_direct() {
        return SparseByRow(NR, this->pairwise_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }

private:
    struct ByCol : public Common<Effect, Level, Stat> {
        ByCol(size_t nr, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            NR(nr), tmp_vars(nr * ls.size()), counts(ls.size()), Common<Effect, Level, Stat>(effects, clust, l, ls, ng, nb, t) {} 

        void finalize () { 
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                auto start = this->cluster_stats[1] + b * NR;
                if (this->level_size[b]) {
                    for (size_t r = 0; r < NR; ++r) {
                        start[r] /= this->level_size[b];
                    }
                } else {
                    std::fill(start, start + NR, std::numeric_limits<double>::quiet_NaN());
                }
            }

            // Finalizing Cohen's d.
            std::vector<double> tmp_means(this->level_size.size()), tmp_vars_single(this->level_size.size());
            auto estart = this->pairwise_effects[0];
            int shift = (this->ngroups) * (this->ngroups);
            for (size_t i = 0; i < NR; ++i, estart += shift) {
                for (size_t l = 0; l < tmp_means.size(); ++l) {
                    tmp_means[l] = this->cluster_stats[0][i + l * NR];
                }
                for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                    tmp_vars_single[l] = tmp_vars[i + l * NR];
                }
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), this->level_size, this->ngroups, this->nblocks, estart, this->threshold);
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
        DenseByCol(size_t nr, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            ByCol(nr, effects, clust, l, ls, ng, nb, t) {}

        template<typename T>
        void add(const T* ptr, T* buffer) {
            auto b = this->levels[this->counter];
            auto offset = b * this->NR;
            tatami::stats::variances::compute_running(ptr, this->NR, this->cluster_stats[0] + offset, this->tmp_vars.data() + offset, this->counts[b]);

            auto ndetected = this->cluster_stats[1] + offset;
            for (size_t j = 0; j < this->NR; ++j, ++ndetected) {
                *ndetected += (ptr[j] > 0);
            }
 
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                auto offset = b * this->NR; 
                tatami::stats::variances::finish_running(this->NR, this->cluster_stats[0] + offset, this->tmp_vars.data() + offset, this->counts[b]);
            }
            this->finalize();
        }
    };

    DenseByCol dense_running() {
        return DenseByCol(NR, this->pairwise_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }

public:
    struct SparseByCol : public ByCol {
        SparseByCol(size_t nr, std::vector<Effect*>& effects, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb, double t) : 
            ByCol(nr, effects, clust, l, ls, ng, nb, t) {}

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            auto b = this->levels[this->counter];
            auto offset = b * this->NR;
            tatami::stats::variances::compute_running(range, this->cluster_stats[0] + offset, this->tmp_vars.data() + offset, this->cluster_stats[1] + offset, this->counts[b]);
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->level_size.size(); ++b) {
                auto offset = b * this->NR; 
                tatami::stats::variances::finish_running(this->NR, this->cluster_stats[0] + offset, this->tmp_vars.data() + offset, this->cluster_stats[1] + offset, this->counts[b]);
            }
            this->finalize();
        }
    };

    SparseByCol sparse_running() {
        return SparseByCol(NR, this->pairwise_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks, this->threshold);
    }
};

}

}

#endif
