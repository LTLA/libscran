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
 * For each group, scores for each gene are obtained by taking summary statistics - namely, the minimum, median, mean and maximum - of the effect sizes across all of that group's comparisons.
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
 * The definitions of these effect sizes are generalized when testing against a non-zero log-fold change threshold.
 * Cohen's d is redefined as the standardized difference between the observed log-fold change and the specified threshold, analogous to the TREAT method from **limma**.
 * Large positive values are only obtained when the observed log-fold change is significantly greater than the threshold.
 * 
 * @section Summary statistics
 * The choice of summary statistic dictates the interpretation of the ranking:
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
 *   This can be used to obtain markers that are uniquely up/downregulated in X.
 * - A small minimum rank indicates that the gene is one of the top upregulated genes in at least one comparison to another group.
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

    template<typename Stat>
    static void shift_for_min_rank(size_t ngenes, std::vector<Stat*>& output) {
        constexpr int minrank_col = 3;
        for (auto& o : output) {
            o += minrank_col * ngenes;
        }
    }

public:
    ScoreMarkers& set_threshold(double t = 0) {
        threshold = t;
        return *this;
    }

public:
    template<class MAT, typename G, typename Stat>
    void run(const MAT* p, const G* group, std::vector<Stat*> output, Stat* means, Stat* detected) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        run_internal(p, group, ngroups, std::move(output), means, detected);
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked(const MAT* p, const G* group, const B* block, std::vector<Stat*> output, Stat* means, Stat* detected) {
        if (block == NULL) {
            run(p, group, std::move(output), means, detected);
            return;
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        run_blocked_internal(p, group, block, ngroups, nblocks, std::move(output), means, detected);
        return;
    }

private:
    template<class MAT, typename G, typename Stat>
    void run_internal(const MAT* p, const G* group, int ngroups, std::vector<Stat*> output, Stat* means, Stat* detected) {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }

        std::vector<Stat*> stats { means, detected };
        std::vector<Stat> cohens_d(p->nrow() * ngroups * ngroups);
        std::vector<Stat*> pairwise_effects{ cohens_d.data() };

        differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), pairwise_effects, stats, group, group_size, ngroups, 1, threshold);
        tatami::apply<0>(p, fact);

        std::vector<std::pair<double, size_t> > buffer(p->nrow());

        // Finishing up Cohen's d.
        {
            differential_analysis::compute_min_rank(p->nrow(), ngroups, cohens_d.data(), output[3], buffer);
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, cohens_d.data(), output); // non-const w.r.t. cohens_d, so done after min-rank calculations.
        }
        return;
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked_internal(const MAT* p, const G* group, const B* block, int ngroups, int nblocks, std::vector<Stat*> output, Stat* means, Stat* detected) {
        int ncombos = ngroups * nblocks;
        std::vector<int> combos(p->ncol());
        std::vector<int> combo_size(ncombos);

        for (size_t i = 0; i < combos.size(); ++i) {
            combos[i] = block[i] * ngroups + group[i];
            ++(combo_size[combos[i]]);
        }

        std::vector<Stat*> stats { means, detected };
        std::vector<Stat> cohens_d(p->nrow() * ngroups * ngroups);
        std::vector<Stat*> pairwise_effects{ cohens_d.data() };
        differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), cohens_d, stats, combos.data(), combo_size, ngroups, nblocks, threshold);
        tatami::apply<0>(p, fact);

        std::vector<std::pair<double, size_t> > buffer(p->nrow());

        // Finishing up Cohen's d.
        {
            differential_analysis::compute_min_rank(p->nrow(), ngroups, cohens_d.data(), output[3], buffer);
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, cohens_d.data(), output); // non-const w.r.t. cohens_d, so done after min-rank calculations.
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

        run(p, group, std::move(cohen_ptrs), res.means.data(), res.detected.data());
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

        std::vector<double*> cohen_ptrs;
        for (size_t o = 0; o < ngroups; ++o) {
            cohen_ptrs.push_back(res.effects[0][o].data());
        }
        run_blocked_internal(p, group, block, ngroups, nblocks, std::move(cohen_ptrs), res.means.data(), res.detected.data());

        return res;
    }
};

}

#endif
