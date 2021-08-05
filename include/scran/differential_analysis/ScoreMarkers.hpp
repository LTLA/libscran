#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "BidimensionalFactory.hpp"
#include "tatami/stats/apply.hpp"
#include "../utils/vector_to_pointers.hpp"

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
    void run(const MAT* p, const G* group, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        std::vector<std::vector<Stat*> > cohen) 
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        decltype(cohen) auc;
        run_internal(p, group, ngroups, means, detected, cohen, auc);
    }        

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked(const MAT* p, const G* group, const B* block, 
        std::vector<std::vector<Stat*> > means, 
        std::vector<std::vector<Stat*> > detected, 
        std::vector<std::vector<Stat*> > cohen)
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        decltype(cohen) auc;
        if (block == NULL) {
            run_internal(p, group, ngroups, means[0], detected[0], cohen, auc);
            return;
        }
    
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        run_blocked_internal(p, group, block, ngroups, nblocks, means, detected, cohen, auc);
        return;
    }

private:
    template<class MAT, typename G, typename Stat>
    void run_internal(const MAT* p, const G* group, int ngroups, 
        std::vector<Stat*>& means, 
        std::vector<Stat*>& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc) 
    {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, ngroups, 1, means, detected, cohen, auc);
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked_internal(const MAT* p, const G* group, const B* block, int ngroups, int nblocks,
        std::vector<std::vector<Stat*> >& means, 
        std::vector<std::vector<Stat*> >& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc) 
    {
        int ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);

        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = block[i] * ngroups + group[i];
            ++(combo_size[combos[i]]);
        }

        std::vector<Stat*> means2(ncombos), detected2(ncombos);
        auto mIt = means2.begin(), dIt = detected2.begin();
        for (int b = 0; b < nblocks; ++b) {
            for (int g = 0; g < ngroups; ++g, ++mIt, ++dIt) {
                *mIt = means[b][g];
                *dIt = detected[b][g];
            }
        }

        core(p, combos.data(), combo_size, ngroups, nblocks, means2, detected2, cohen, auc);
    }

    template<class MAT, typename G, class Gs, typename Stat>
    void core(const MAT* p, const G* level, const Gs& level_size, int ngroups, int nblocks, 
        std::vector<Stat*>& means, 
        std::vector<Stat*>& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc)
    {
        const bool do_cohen = !cohen.empty();
        std::vector<Stat> cohens_d(do_cohen ? p->nrow() * ngroups * ngroups : 0);
        Stat* cohens_ptr = do_cohen ? cohens_d.data() : NULL;

        const bool do_wilcox = !auc.empty();
        std::vector<Stat> wilcox_auc(do_wilcox ? p->nrow() * ngroups * ngroups : 0);

        if (!do_wilcox) {
            differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), means, detected, cohens_ptr, level, level_size, ngroups, nblocks, threshold);
            tatami::apply<0>(p, fact);
        } else {
        }

        std::vector<std::pair<double, size_t> > buffer(p->nrow());
        if (do_cohen) {
            if (cohen[4].size()) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, cohens_d.data(), cohen[4], buffer);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, cohens_d.data(), cohen); // non-const w.r.t. cohens_d, so done after min-rank calculations.
        }
        if (do_wilcox) {
            if (auc[4].size()) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, wilcox_auc.data(), auc[4], buffer);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, wilcox_auc.data(), auc); // non-const w.r.t. wilcox_auc, so done after min-rank calculations.
        }

        return;
    }

public:
    struct Results {
        Results(size_t ngenes, int ngroups, int nblocks) : 
            cohen(differential_analysis::n_summaries, std::vector<std::vector<double> >(ngroups, std::vector<double>(ngenes))), 
            means(nblocks, std::vector<std::vector<double> >(ngroups, std::vector<double>(ngenes))),
            detected(means)
        {}

        std::vector<std::vector<std::vector<double> > > cohen;
        std::vector<std::vector<std::vector<double> > > auc;
        std::vector<std::vector<std::vector<double> > > means;
        std::vector<std::vector<std::vector<double> > > detected;
    };

    template<class MAT, typename G> 
    Results run(const MAT* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results res(p->nrow(), ngroups, 1); 

        auto mean_ptrs = vector_to_pointers(res.means[0]);
        auto detect_ptrs = vector_to_pointers(res.detected[0]);

        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        decltype(cohen_ptrs) auc_ptrs;
        run_internal(p, group, ngroups, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs);
        return res;
    }

    template<class MAT, typename G, typename B> 
    Results run_blocked(const MAT* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        Results res(p->nrow(), ngroups, nblocks); 

        auto mean_ptrs = vector_to_pointers2(res.means);
        auto detect_ptrs = vector_to_pointers2(res.detected);
        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        decltype(cohen_ptrs) auc_ptrs;
        run_blocked_internal(p, group, block, ngroups, nblocks, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs);

        return res;
    }

private:
    std::vector<std::vector<double*> > vector_to_pointers2(std::vector<std::vector<std::vector<double> > >& input) {
        std::vector<std::vector<double*> > ptrs;
        for (auto& current : input) {
            ptrs.push_back(vector_to_pointers(current));
        }
        return ptrs;
    }
};

}

#endif
