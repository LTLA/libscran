#ifndef SCRAN_BIDIMENSIONAL_FACTORY_HPP
#define SCRAN_BIDIMENSIONAL_FACTORY_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "../feature_selection/blocked_variances.hpp"

namespace scran {

namespace differential_analysis {

template<typename Effect, typename Level, typename Stat>
struct Common {
    Common(std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
        comparison_effects(comps), cluster_stats(clust), levels(l), level_size(ls), ngroups(ng), nblocks(nb) {}

    std::vector<Effect*>& comparison_effects;
    std::vector<Stat*>& cluster_stats;
    const Level* levels;
    std::vector<int>& level_size;
    int ngroups, nblocks;
};

template<typename Effect, typename Level, typename Stat> 
struct BidimensionalFactory : public Common<Effect, Level, Stat> {
public:
    BidimensionalFactory(size_t nr, size_t nc, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
        NR(nr), NC(nc), Common<Effect, Level, Stat>(comps, clust, l, ls, ng, nb) {}

    static constexpr bool supports_running = true;
    static constexpr bool supports_sparse = true;
private:
    size_t NR, NC;

private:
    struct ByRow : public Common<Effect, Level, Stat> {
        ByRow(size_t nr, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            NR(nr), tmp_means(ls.size()), tmp_vars(ls.size()), tmp_nzeros(ls.size()), buffer(ng * ng), weightsum(ng * ng),
            Common<Effect, Level, Stat>(comps, clust, l, ls, ng, nb) {}

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
                    *ndetected = tmp_nzeros[l] / level_size[l];
                } else {
                    *ndetected = std::numeric_limits<double>::quiet_NaN();
                }
            }

            // Actually computing the effect sizes.
            compute_pairwise_cohens_d(tmp_means.data(), tmp_vars.data(), this->levels, this->level_size, this->ngroups, this->nblocks, buffer.data(), weightsum.data(), 1);
            summarize_comparisons(this->ngroups, buffer.data(), i, NR, this->comparison_effects);
        }
    protected:
        size_t NR;
        std::vector<double> tmp_means, tmp_vars, tmp_nzeros, buffer, weightsum;
    };

public:
    struct DenseByRow : public ByRow<Effect, Level, Stat> {
        DenseByRow(size_t nc, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            NC(nc), ByRow<Effect, Level, Stat>(comps, clust, l, ls, ng, nb) {}

        template<typename T>
        void compute(size_t i, const T* ptr, T* buffer) {
            feature_selection::blocked_variance_with_mean<blocked>(ptr, NC, this->levels, this->level_size, tmp_means, tmp_vars);

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
        return DenseByRow(NC, this->comparison_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks);
    }

public:
    struct SparseByRow : public ByRow<Effect, Level, Stat> {
        SparseByRow(std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            ByRow<Effect, Level, Stat>(comps, clust, l, ls, ng, nb) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            feature_selection::blocked_variance_with_mean<blocked>(range, this->levels, this->level_size, this->tmp_means, this->tmp_vars, this->tmp_nzeros);
            this->transfer(i);
        }
    };

    SparseByRow sparse_direct() {
        return SparseByRow(this->comparison_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks);
    }

private:
    struct ByCol : public Common<Effect, Level, Stat> {
        ByCol(size_t nr, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            NR(nr), tmp_vars(nr * ls.size()), counts(ls.size()), Common<Effect, Level, Stat>(comps, clust, l, ls, ng, nb) {} 

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

            std::vector<double> buffer(ng * ng), weightsum(ng * ng), tmp_means(this->level_size.size()), tmp_vars_single(this->level_size.size());
            for (size_t i = 0; i < NR; ++i) {
                for (size_t l = 0; l < tmp_means.size(); ++l) {
                    tmp_means[l] = this->cluster_stats[0][i + l * NR];
                }
                for (size_t l = 0; l < tmp_vars_single.size(); ++l) {
                    tmp_vars_single[l] = tmp_vars[i + l * NR];
                }
                compute_pairwise_cohens_d(tmp_means.data(), tmp_vars_single.data(), this->levels, this->level_size, this->ngroups, this->nblocks, buffer.data(), weightsum.data());
                summarize_comparisons(this->ngroups, buffer.data(), i, NR, this->comparison_effects);
            }
        }
    protected:
        size_t NR;
        std::vector<double> tmp_vars;
        std::vector<int> counts;
        int counter = 0;
    };

public:
    struct DenseByCol : public ByCol<Effect, Level, Stat> {
        DenseByCol(size_t nr, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            ByCol<Effect, Level, Stat>(nr, comps, clust, l, ls, ng, nb) {}

        template<typename T>
        void add(const T* ptr, T* buffer) {
            auto b = this->levels[this->counter];
            auto offset = b * this->NR;
            tatami::stats::variances::compute_running(ptr, this->NR, this->cluster_stats[0] + offset, this->tmp_vars.data() + offset, this->counts[b]);

            auto ndetected = this->cluster_stats[1] + offset;
            for (size_t j = 0; j < NR; ++j, ++ndetected) {
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
        return DenseByCol(NR, this->comparison_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks);
    }

public:
    struct SparseByCol : public ByCol<Effect, Level, Stat> {
        SparseByCol(size_t nr, std::vector<Effect*>& comps, std::vector<Stat*>& clust, const Level* l, std::vector<int>& ls, int ng, int nb) : 
            ByCol<Effect, Level, Stat>(nr, comps, clust, l, ls, ng, nb) {}

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            auto b = levels[this->counter];
            auto offset = b * NR;
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
        return SparseByCol(NR, this->comparison_effects, this->cluster_stats, this->levels, this->level_size, this->ngroups, this->nblocks);
    }
};

}

}

#endif
