#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "Factory.hpp"
#include "summarize_comparisons.hpp"

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
 * Markers are identified by differential expression analyses between pairs of groups of cells (e.g., clusters, cell types).
 * Given `n` groups, each group is involved in `n - 1` pairwise comparisons and thus has `n - 1` effect sizes.
 * For each group, we compute summary statistics - e.g., the minimum, median, mean - of the effect sizes across all of that group's comparisons.
 * Users can then sort by any of these summaries to obtain a ranking of potential marker genes for each group.
 * The choice of effect size and summary statistic determines the characteristics of the resulting marker set.
 *
 * @section Effect sizes
 * Cohen's d is the standardized log-fold change between two groups.
 * This is defined as the difference in the mean log-expression for each group scaled by the average standard deviation across the two groups.
 * (Technically, we should use the pooled variance; however, this introduces some unpleasant asymmetry depending on the variance of the larger group, so we take a simple average instead.)
 * A positive value indicates that the gene is upregulated in the first gene compared to the second.
 * Cohen's d is analogous to the t-statistic in a two-sample t-test and avoids spuriously large effect sizes from comparisons between highly variable groups.
 * We can also interpret Cohen's d as the number of standard deviations between the two group means.
 *
 * The area under the curve (AUC) can be interpreted as the probability that a randomly chosen observation in one group is greater than a randomly chosen observation in the other group. 
 * Values greater than 0.5 indicate that a gene is upregulated in the first group.
 * The AUC is closely related to the U-statistic used in the Wilcoxon rank sum test. 
 * The key difference between the AUC and Cohen's d is that the former is less sensitive to the variance within each group, e.g.,
 * if two distributions exhibit no overlap, the AUC is the same regardless of the variance of each distribution. 
 * This may or may not be desirable as it improves robustness to outliers but reduces the information available to obtain a highly resolved ranking. 
 *
 * @section With a log-fold change threshold:
 * Setting a log-fold change threshold can be helpful as it prioritizes genes with large shifts in expression instead of those with low variances.
 * Currently, only positive thresholds are supported - this focuses on genes upregulated in the first group compared to the second.
 * The effect size definitions are generalized when testing against a non-zero log-fold change threshold.
 *
 * Cohen's d is redefined as the standardized difference between the observed log-fold change and the specified threshold, analogous to the TREAT method from **limma**.
 * Large positive values are only obtained when the observed log-fold change is significantly greater than the threshold.
 * For example, if we had a threshold of 2 and we obtained a Cohen's d of 3, this means that the observed log-fold change was 3 standard deviations above 2.
 * Importantly, a negative Cohen's d cannot be intepreted as downregulation, as the log-fold change may still be positive but less than the threshold.
 * 
 * The AUC generalized to the probability of obtaining a random observation in one group that is greater than a random observation plus the threshold in the other group.
 * For example, if we had a threshold of 2 and we obtained an AUC of 0.8, this means that - 80% of the time - 
 * the random observation from the first group would be greater than a random observation from the second group by 2 or more.
 * Again, AUCs below 0.5 cannot be interpreted as downregulation, as it may be caused by a positive log-fold change that is less than the threshold.
 * 
 * @section Summary statistics
 * The choice of summary statistic dictates the interpretation of the ranking.
 * Given a group X:
 * 
 * - A large mean effect size indicates that the gene is upregulated in X compared to the average of the other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is a good general-purpose summary statistic for ranking, usually by decreasing size to obtain upregulated markers in X.
 * - A large median effect size indicates that the gene is upregulated in X compared to most (>50%) other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is also a good general-purpose summary with the advantage of being more robust to outlier effects 
 *   (but also the disadvantage of being less sensitive to strong effects in a minority of comparisons).
 * - A large minimum effect size indicates that the gene is upregulated in X compared to all other groups.
 *   A small value indicates that the gene is downregulated in X compared to at least one other group.
 *   For upregulation, this is the most stringent summary as markers will only have extreme values if they are _uniquely_ upregulated in X compared to every other group.
 *   However, it may not be effective if X is closely related to any of the groups.
 * - A large maximum effect size indicates that the gene is upregulated in X compared to at least one other group.
 *   A small value indicates that the gene is downregulated in X compared to all other groups.
 *   For downregulation, this is the most stringent summary as markers will only have extreme values if they are _uniquely_ downregulated in X compared to every other group.
 *   However, it may not be effective if X is closely related to any of the groups.
 * - The "minimum rank" (a.k.a. min-rank) is defined by ranking genes based on decreasing effect size _within_ each comparison, and then taking the smallest rank _across_ comparisons.
 *   A minimum rank of 1 means that the gene is the top upregulated gene in at least one comparison to another group.
 *   More generally, a minimum rank of T indicates that the gene is the T-th upregulated gene in at least one comparison. 
 *   Applying a threshold on the minimum rank is useful for obtaining a set of genes that, in combination, are guaranteed to distinguish X from every other group.
 *
 * The exact definition of "large" and "small" depends on the choice of effect size.
 * For Cohen's d, the value must be positive to be considered "large", and negative to be considered "small".
 * For the AUC, a value greater than 0.5 is considered "large" and less than 0.5 is considered "small".
 * Note that this interpretation is contingent on the log-fold change threshold - for positive thresholds, small values cannot be unambiguously interpreted as downregulation.
 *
 * @section Blocked comparisons
 * In the presence of multiple batches, we can block on the batch of origin for each cell.
 * Comparisons are only performed between the groups of cells in the same batch (also called "blocking level" below).
 * The batch-specific effect sizes are then combined into a single aggregate value for calculation of summary statistics.
 * This strategy avoids most problems related to batch effects as we never directly compare across different blocking levels.
 *
 * Specifically, for each gene and each pair of groups, we obtain one effect size per blocking level.
 * We consolidate these into a single statistic by computing the weighted mean across levels.
 * The weight for each level is defined as the product of the sizes of the two groups;
 * this favors contribution from levels with more cells in both groups, where the effect size is presumably more reliable.
 * (Obviously, levels with no cells in either group will not contribute anything to the weighted mean.)
 *
 * If two groups never co-occur in the same blocking level, no effect size will be computed.
 * We do not attempt to reconcile batch effects in a partially confounded scenario.
 * As such, this particular pair of groups will not contribute to the calculation of the summary statistics for either group.
 *
 * @section Other statistics
 * We report the mean log-expression of all cells in each group, as well as the proportion of cells with detectable expression in each group.
 * These statistics are useful for quickly interpreting the differences in expression driving the effect size summaries.
 *
 * If blocking is involved, we compute the mean and proportion for each group in each separate blocking level.
 * This is helpful for detecting differences in expression between batches.
 * They can also be combined into a single statistic for each group by using the `average_vectors()` or `average_vectors_weighted()` functions.
 */
class ScoreMarkers {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_threshold()` for details.
         */
        static constexpr double threshold = 0;

        /**
         * See `set_compute_cohen()` for details.
         */
        static constexpr bool compute_cohen = true;

        /**
         * See `set_compute_auc()` for details.
         */
        static constexpr bool compute_auc = true;

        /**
         * See `set_summary_min()` for details.
         */
        static constexpr bool summary_min = true;

        /**
         * See `set_summary_mean()` for details.
         */
        static constexpr bool summary_mean = true;

        /**
         * See `set_summary_median()` for details.
         */
        static constexpr bool summary_median = true;

        /**
         * See `set_summary_max()` for details.
         */
        static constexpr bool summary_max = true;

        /**
         * See `set_summary_rank()` for details.
         */
        static constexpr bool summary_rank = true;
    };
private:
    double threshold = Defaults::threshold;
    bool do_cohen = Defaults::compute_cohen;
    bool do_auc = Defaults::compute_auc;
    bool use_min = Defaults::summary_min;
    bool use_mean = Defaults::summary_mean;
    bool use_median = Defaults::summary_median;
    bool use_max = Defaults::summary_max;
    bool use_rank = Defaults::summary_rank;

public:
    /**
     * @param t Threshold on the log-fold change.
     * This should be non-negative.
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_threshold(double t = Defaults::threshold) {
        threshold = t;
        return *this;
    }

    /**
     * @param c Whether to compute Cohen's d.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_cohen(bool c = true) {
        do_cohen = c;
        return *this;
    }

    /**
     * @param a Whether to compute AUCs.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_auc(bool a = true) {
        do_auc = a;
        return *this;
    }

    /**
     * @param s Whether to compute the minimum summary statistic.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_min(bool s = true) {
        use_min = s;
        return *this;
    }

    /**
     * @param s Whether to compute the minimum summary statistic.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_mean(bool s = true) {
        use_mean = s;
        return *this;
    }

    /**
     * @param s Whether to compute the median summary statistic.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_median(bool s = true) {
        use_median = s;
        return *this;
    }

    /**
     * @param s Whether to compute the maximum summary statistic.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_max(bool s = true) {
        use_max = s;
        return *this;
    }

    /**
     * @param s Whether to compute the minimum rank summary statistic.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_rank(bool s = true) {
        use_rank = s;
        return *this;
    }

public:
    /**
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam Stat Floating-point type to store the statistics.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * These should be 0-based and consecutive.
     * @param[out] means Pointers to arrays of length equal to the number of rows in `p`,
     * used to store the mean expression of each group.
     * @param[out] detected Pointers to arrays of length equal to the number of rows in `p`,
     * used to store the proportion of detected expression in each group.
     * @param[out] cohen Vector of vector of pointers to arrays of length equal to the number of rows in `p`.
     * Each inner vector corresponds to a summary statistic for Cohen's d, ordered as in `differential_analysis::summary`.
     * Each pointer corresponds to a group and is filled with the relevant summary statistic for that group.
     * @param[out] auc Vector of vector of pointers to arrays of length equal to the number of rows in `p`,
     * Each inner vector corresponds to a summary statistic for the AUC, ordered as in `differential_analysis::summary`.
     * Each pointer corresponds to a group and is filled with the relevant summary statistic for that group.
     * 
     * If `cohen` is of length 0, Cohen's d is not computed.
     * Similarly, if `auc` is of length 0, AUC is not computed.
     * (`set_compute_cohen()` and `set_compute_auc()` have no effect here.)
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     *
     * @return `means`, `detected`, `cohen` and `auc` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename Stat>
    void run(const Matrix* p, const G* group, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        std::vector<std::vector<Stat*> > cohen, 
        std::vector<std::vector<Stat*> > auc) 
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        run_internal(p, group, ngroups, means, detected, cohen, auc);
    }        

    /**
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam B Integer type for the block assignments.
     * @tparam Stat Floating-point type to store the statistics.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * These should be 0-based and consecutive.
     * @param[in] block Pointer to an array of length equal to the number of columns in `p`, containing the blocking factor.
     * Levels should be 0-based and consecutive.
     * @param[out] means Vector of vectors of pointers to arrays of length equal to the number of rows in `p`.
     * Each inner vector corresponds to a group and each pointer therein contains the mean expression in a blocking level.
     * @param[out] detected Pointers to arrays of length equal to the number of rows in `p`.
     * Each inner vector corresponds to a group and each pointer therein contains the proportion of detected expression in a blocking level.
     * @param[out] cohen Vector of vector of pointers to arrays of length equal to the number of rows in `p`.
     * Each inner vector corresponds to a summary statistic for Cohen's d, ordered as in `differential_analysis::summary`.
     * Each pointer corresponds to a group and is filled with the relevant summary statistic for that group.
     * @param[out] auc Vector of vector of pointers to arrays of length equal to the number of rows in `p`,
     * Each inner vector corresponds to a summary statistic for the AUC, ordered as in `differential_analysis::summary`.
     * Each pointer corresponds to a group and is filled with the relevant summary statistic for that group.
     * 
     * If `cohen` is of length 0, Cohen's d is not computed.
     * Similarly, if `auc` is of length 0, AUC is not computed.
     * (`set_compute_cohen()` and `set_compute_auc()` have no effect here.)
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     *
     * @return `means`, `detected`, `cohen` and `auc` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename B, typename Stat>
    void run_blocked(const Matrix* p, const G* group, const B* block, 
        std::vector<std::vector<Stat*> > means, 
        std::vector<std::vector<Stat*> > detected, 
        std::vector<std::vector<Stat*> > cohen,
        std::vector<std::vector<Stat*> > auc) 
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
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
        core(p, group, group_size, group, ngroups, static_cast<const int*>(NULL), 1, means, detected, cohen, auc);
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
            combos[i] = group[i] * nblocks + block[i];
            ++(combo_size[combos[i]]);
        }

        std::vector<Stat*> means2(ncombos), detected2(ncombos);
        auto mIt = means2.begin(), dIt = detected2.begin();
        for (int g = 0; g < ngroups; ++g) {
            for (int b = 0; b < nblocks; ++b, ++mIt, ++dIt) {
                *mIt = means[g][b];
                *dIt = detected[g][b];
            }
        }

        core(p, combos.data(), combo_size, group, ngroups, block, nblocks, means2, detected2, cohen, auc);
    }

    template<class MAT, typename L, class Ls, typename G, typename B, typename Stat>
    void core(const MAT* p, 
        const L* level, const Ls& level_size, 
        const G* group, int ngroups, 
        const B* block, int nblocks, 
        std::vector<Stat*>& means, 
        std::vector<Stat*>& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc)
    {
        const bool do_cd = !cohen.empty();
        std::vector<Stat> cohens_d(do_cd ? p->nrow() * ngroups * ngroups : 0);
        Stat* cohens_ptr = do_cd ? cohens_d.data() : NULL;

        const bool do_wilcox = !auc.empty();
        std::vector<Stat> wilcox_auc(do_wilcox ? p->nrow() * ngroups * ngroups : 0);

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::ScoreMarkers", "Performing pairwise comparisons between groups of cells");
#endif
        if (!do_wilcox) {
            std::vector<Stat*> effects{cohens_ptr};
            differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), means, detected, effects, level, &level_size, ngroups, nblocks, threshold);
            tatami::apply<0>(p, fact);
        } else {
            std::vector<Stat*> effects{cohens_ptr, wilcox_auc.data()};

            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<B> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            differential_analysis::PerRowFactory fact(p->nrow(), p->ncol(), means, detected, effects, level, &level_size, group, ngroups, block, nblocks, threshold);
            tatami::apply<0>(p, fact);
        }

        if (do_cd) {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::ScoreMarkers", "Summarizing Cohen's D across comparisons");
#endif
            auto& min_rank_cohen = cohen[scran::differential_analysis::MIN_RANK];
            if (min_rank_cohen.size()) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, cohens_d.data(), min_rank_cohen);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, cohens_d.data(), cohen); // non-const w.r.t. cohens_d, so done after min-rank calculations.
        }
        if (do_wilcox) {
#ifdef SCRAN_LOGGER
            SCRAN_LOGGER("scran::ScoreMarkers", "Summarizing the AUC across comparisons");
#endif
            auto& min_rank_auc = auc[scran::differential_analysis::MIN_RANK];
            if (min_rank_auc.size()) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, wilcox_auc.data(), min_rank_auc);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, wilcox_auc.data(), auc); // non-const w.r.t. wilcox_auc, so done after min-rank calculations.
        }

        return;
    }

public:
    /** 
     * @tparam Stat Floating-point type to store the statistics.
     * @brief Marker effect size summaries and other statistics.
     */
    template<typename Stat>
    struct Results {
        /**
         * @cond
         */
        Results(size_t ngenes, int ngroups, int nblocks, bool do_cohen, bool do_auc, bool do_min, bool do_mean, bool do_median, bool do_max, bool do_rank) { 
            auto fill_inner = [&](int N, auto& type) {
                type.reserve(N);
                for (int n = 0; n < N; ++n) {
                    type.emplace_back(ngenes);
                }
            };
            
            means.resize(ngroups);
            detected.resize(ngroups);
            for (int g = 0; g < ngroups; ++g) {
                fill_inner(nblocks, means[g]);
                fill_inner(nblocks, detected[g]);
            }

            auto fill_effect = [&](auto& effect) {
                effect.resize(differential_analysis::n_summaries);
                if (do_min) {
                    fill_inner(ngroups, effect[differential_analysis::MIN]);
                }
                if (do_mean) {
                    fill_inner(ngroups, effect[differential_analysis::MEAN]);
                }
                if (do_median) {
                    fill_inner(ngroups, effect[differential_analysis::MEDIAN]);
                }
                if (do_max) {
                    fill_inner(ngroups, effect[differential_analysis::MAX]);
                }
                if (do_rank) {
                    fill_inner(ngroups, effect[differential_analysis::MIN_RANK]);
                }
                return;
            };

            if (do_cohen) {
                fill_effect(cohen);
            }
            if (do_auc) {
                fill_effect(auc);
            }
            return;
        }
        /**
         * @endcond
         */

        typedef std::vector<Stat> Vector;
        std::vector<std::vector<Vector> > cohen;
        std::vector<std::vector<Vector> > auc;
        std::vector<std::vector<Vector> > means;
        std::vector<std::vector<Vector> > detected;
    };

    template<typename Stat = double, class MAT, typename G> 
    Results<Stat> run(const MAT* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, 1, do_cohen, do_auc, use_min, use_mean, use_median, use_max, use_rank); 

        auto mean_ptrs = vector_to_pointers3(res.means);
        auto detect_ptrs = vector_to_pointers3(res.detected);

        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        auto auc_ptrs = vector_to_pointers2(res.auc);
        run_internal(p, group, ngroups, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs);
        return res;
    }

    template<typename Stat = double, class MAT, typename G, typename B> 
    Results<Stat> run_blocked(const MAT* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, nblocks, do_cohen, do_auc, use_min, use_mean, use_median, use_max, use_rank); 

        auto mean_ptrs = vector_to_pointers2(res.means);
        auto detect_ptrs = vector_to_pointers2(res.detected);
        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        auto auc_ptrs = vector_to_pointers2(res.auc);
        run_blocked_internal(p, group, block, ngroups, nblocks, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs);

        return res;
    }

private:
    template<typename Stat>
    std::vector<std::vector<Stat*> > vector_to_pointers2(std::vector<std::vector<std::vector<Stat> > >& input) {
        std::vector<std::vector<Stat*> > ptrs;
        for (auto& current : input) {
            ptrs.push_back(vector_to_pointers(current));
        }
        return ptrs;
    }

    template<typename Stat>
    std::vector<Stat*> vector_to_pointers3(std::vector<std::vector<std::vector<Stat> > >& input) {
        std::vector<Stat*> ptrs;
        for (auto& current : input) {
            ptrs.push_back(current[0].data()); // first vector from each element.
        }
        return ptrs;
    }
};

}

#endif
