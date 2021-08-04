#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "BidimensionalFactory.hpp"
#include "tatami/stats/apply.hpp"

/**
 * @file ScoreMarkers.hpp
 *
 * @brief Compute marker scores for each gene in each group of cells.
 */

namespace scran {

/**
 * @brief Score each gene as a candidate marker for each group of cells.
 *
 * Markers are identified by pairwise comparisons between groups of cells (e.g., clusters, cell types).
 * Given `n` groups, each group has `n - 1` comparisons and thus `n - 1` effect sizes.
 * For each group, scores for each gene are obtained by taking summary statistics - e.g., the minimum, median, mean - of the effect sizes across all of that group's comparisons.
 * Users can then sort by any of these summaries to obtain a ranking of potential marker genes for each group.
 * The choice of effect size and summary statistic determines the characteristics of the resulting marker set.
 *
 * @section Effect sizes
 * Cohen's d is the standardized log-fold change between two groups.
 * This is defined as the difference in the mean log-expression for each group scaled by the average standard deviation across the two groups.
 * (Technically, we should use the pooled variance; however, this introduces some unpleasant asymmetry depending on the variance of the larger group, so we take a simple average instead.)
 * Cohen's d is analogous to the t-statistic in a two-sample t-test and avoids spuriously large effect sizes from comparisons between highly variable groups.
 * We can also interpret Cohen's d as the number of standard deviations between the two group means.
 *
 * The definitions of these effect sizes are generalized when testing against a non-zero log-fold change threshold:
 *
 * - Cohen's d is redefined as the standardized difference between the observed log-fold change and the specified threshold, analogous to the TREAT method from **limma**.
 * Large positive values are only obtained when the observed log-fold change is significantly greater than the threshold.
 * 
 * @section Summary statistics
 * The choice of summary statistic dictates the interpretation of the ranking.
 * Given a group X:
 * 
 * - A large mean effect size indicates that the gene is upregulated in X compared to the average of the other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is a good general-purpose summary statistic for ranking, usually by decreasing size to obtain upregulated markers in X.
 * - A large median effect size indicates that the gene is upregulated in X compared to most (>50\%) other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is also good for ranking, with the advantage of being more robust to outlier effects (but also the disadvantage of being less sensitive to strong effects in a minority of comparisons).
 * - A large minimum effect size indicates that the gene is upregulated in X compared to all other groups.
 *   A small value indicates that the gene is downregulated in X compared to at least one other group.
 *   This is the most stringent summary as markers will only have extreme values if they are _uniquely_ up/downregulated in X compared to every other group.
 *   However, it may not be effective if X is closely related to any of the groups.
 * - The "minimum rank" (a.k.a. min-rank) is defined by ranking genes based on decreasing effect size _within_ each comparison, and then taking the smallest rank _across_ comparisons.
 *   A minimum rank of 1 means that the gene is the top upregulated gene in at least one comparison to another group.
 *   More generally, a minimum rank of T indicates that the gene is the T-th upregulated gene in at least one comparison. 
 *   Applying a threshold on the minimum rank is useful for obtaining a set of genes that, in combination, are guaranteed to distinguish X from every other group.
 *
 * The exact definition of "large" and "small" depends on the choice of effect size.
 * For Cohen's d, the value must be positive to be considered "large", and negative to be considered "small".
 * 
 * Additionally, these interpretations change slightly when a log-fold change threshold is supplied.
 * For a positive threshold, only large values can be unambiguously interpreted as upregulation; small values do not have any guaranteed meaning.
 * Conversely, for a negative threshold, only small values can be unambiguously interpreted as downregulation.
 *
 * @section Other statistics
 * We report the mean log-expression of all cells in each cluster, as well as the proportion of cells with detectable expression in each cluster.
 */
class ScoreMarkers {
private:
    double threshold = 0;

public:
    ScoreMarkers& set_threshold(double t = 0) {
        threshold = t;
        return *this;
    }

public:
    template<class MAT, typename G, typename Stat>
    void run(const MAT* p, const G* group, Stat* means, Stat* detected, std::vector<Stat*> cohen) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        std::vector<Stat*> auc;
        run_internal(p, group, ngroups, means, detected, cohen, auc);
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked(const MAT* p, const G* group, const B* block, Stat* means, Stat* detected, std::vector<Stat*> cohen) {
        if (block == NULL) {
            run(p, group, std::move(cohen), means, detected);
            return;
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        std::vector<Stat*> auc;
        run_blocked_internal(p, group, block, ngroups, means, detected, cohen, auc);
        return;
    }

private:
    template<class MAT, typename G, typename Stat>
    void run_internal(const MAT* p, const G* group, int ngroups, Stat* means, Stat* detected, std::vector<Stat*>& cohen, std::vector<Stat*>& auc) {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, ngroups, 1, means, detected, cohen, auc);
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked_internal(const MAT* p, const G* group, const B* block, int ngroups, Stat* means, Stat* detected, std::vector<Stat*>& cohen, std::vector<Stat*>& auc) {
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        int ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);

        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = block[i] * ngroups + group[i];
            ++(combo_size[combos[i]]);
        }
        core(p, combos.data(), combo_size, ngroups, nblocks, means, detected, cohen, auc);
    }

    template<class MAT, typename G, class Gs, typename Stat>
    void core(const MAT* p, const G* level, const Gs& level_size, int ngroups, int nblocks, Stat* means, Stat* detected, std::vector<Stat*>& cohen, std::vector<Stat*>& auc) {
        std::vector<Stat*> stats { means, detected };
        std::vector<double> tmp_means, tmp_detected;
        if (nblocks > 1) {
            tmp_means.resize(nblocks * ngroups * p->nrow());
            tmp_detected.resize(nblocks * ngroups * p->nrow());
            stats[0] = tmp_means.data();
            stats[1] = tmp_detected.data();
        }

        const bool do_cohen = !cohen.empty();
        std::vector<Stat> cohens_d(do_cohen ? p->nrow() * ngroups * ngroups : 0);
        const bool do_wilcox = !auc.empty();
        std::vector<Stat> wilcox_auc(do_wilcox ? p->nrow() * ngroups * ngroups : 0);

        std::vector<Stat*> pairwise_effects(2, NULL);
        if (do_cohen) {
            pairwise_effects[0] = cohens_d.data();
        }
        if (do_wilcox) {
            pairwise_effects[1] = wilcox_auc.data();
        }

        if (!do_wilcox) {
            differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), pairwise_effects, stats, level, level_size, ngroups, nblocks, threshold);
            tatami::apply<0>(p, fact);
        } else {
        }

        // Computing weighted means across blocks.
        if (nblocks > 1) {
            std::fill(means, means + ngroups * p->nrow(), 0);
            std::fill(detected, detected + ngroups * p->nrow(), 0);

            for (int g = 0; g < ngroups; ++g) {
                double total = 0;
                size_t offset_dest = g * p->nrow();

                for (int b = 0; b < nblocks; ++b) {
                    double mult = level_size[b * ngroups + g];
                    if (mult) {
                        auto means_dest = means + offset_dest;
                        auto detected_dest = detected + offset_dest;

                        size_t offset_src = (b * ngroups + g) * p->nrow();
                        auto means_src = tmp_means.data() + offset_src;
                        auto detected_src = tmp_detected.data() + offset_src;

                        for (size_t r = 0; r < p->nrow(); ++r, ++means_src, ++detected_src, ++detected_dest, ++means_dest) {
                            *detected_dest += mult * (*detected_src);
                            *means_dest += mult * (*means_src);
                        }

                        total += mult;
                    }
                }

                auto means_dest = means + offset_dest;
                auto detected_dest = detected + offset_dest;
                if (total) {
                    for (size_t r = 0; r < p->nrow(); ++r) {
                        means_dest[r] /= total;
                        detected_dest[r] /= total;
                    }
                } else {
                    std::fill(means_dest, means_dest + p->nrow(), std::numeric_limits<double>::quiet_NaN());
                    std::fill(detected_dest, detected_dest + p->nrow(), std::numeric_limits<double>::quiet_NaN());
                }
            }
        }

        std::vector<std::pair<double, size_t> > buffer(p->nrow());
        if (do_cohen) {
            if (cohen[3]) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, cohens_d.data(), cohen[3], buffer);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, cohens_d.data(), cohen); // non-const w.r.t. cohens_d, so done after min-rank calculations.
        }
        if (do_wilcox) {
            if (auc[3]) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, wilcox_auc.data(), auc[3], buffer);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, wilcox_auc.data(), auc); // non-const w.r.t. wilcox_auc, so done after min-rank calculations.
        }
        return;
    }

public:
    struct Results {
        Results(size_t ngenes, int ngroups, int neffects) : 
            effects(neffects, std::vector<std::vector<double> >(4, std::vector<double>(ngenes * ngroups))), 
            means(ngenes * ngroups), detected(ngenes * ngroups) {}

        std::vector<std::vector<std::vector<double> > > effects;
        std::vector<double> means;
        std::vector<double> detected;
    };

    template<class MAT, typename G> 
    Results run(const MAT* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results res(p->nrow(), ngroups, 1); 

        std::vector<double*> cohen_ptrs;
        for (size_t o = 0; o < 4; ++o) {
            cohen_ptrs.push_back(res.effects[0][o].data());
        }

        std::vector<double*> auc_ptrs;
        run_internal(p, group, ngroups, res.means.data(), res.detected.data(), cohen_ptrs, auc_ptrs);
        return res;
    }

    template<class MAT, typename G, typename B> 
    Results run_blocked(const MAT* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results res(p->nrow(), ngroups, 1); 

        std::vector<double*> cohen_ptrs;
        for (size_t o = 0; o < 4; ++o) {
            cohen_ptrs.push_back(res.effects[0][o].data());
        }

        std::vector<double*> auc_ptrs;
        run_blocked_internal(p, group, block, ngroups, res.means.data(), res.detected.data(), cohen_ptrs, auc_ptrs);

        return res;
    }
};

}

#endif
