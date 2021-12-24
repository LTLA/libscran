#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "Factory.hpp"
#include "summarize_comparisons.hpp"

#include "tatami/stats/apply.hpp"
#include "../utils/vector_to_pointers.hpp"

#include <array>

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
 * @section effect-sizes Effect sizes
 * The log-fold change (LFC) is the difference in the mean log-expression between groups.
 * This is fairly straightforward to interpret - as log-fold change of +1 corresponds to a two-fold upregulation in the first group compared to the second.
 * For this interpretation, we assume that the input matrix contains log-transformed normalized expression values.
 *
 * The delta-detected is the difference in the proportion of cells with detected expression between groups.
 * This lies between 1 and -1, with the extremes occurring when a gene is silent in one group and detected in all cells of the other group.
 * For this interpretation, we assume that the input matrix contains non-negative expression values, where a value of zero corresponds to lack of detectable expression.
 *
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
 * @section lfc-threshold With a log-fold change threshold
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
 * @section summary Summary statistics
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
 * For Cohen's d, LFC and delta-detected, the value must be positive to be considered "large", and negative to be considered "small".
 * For the AUC, a value greater than 0.5 is considered "large" and less than 0.5 is considered "small".
 * Note that this interpretation is contingent on the log-fold change threshold - for positive thresholds, small effects cannot be unambiguously interpreted as downregulation.
 *
 * @section blocked Blocked comparisons
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
 * @section other Other statistics
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
     * Array type indicating whether each summary statistic should be computed.
     * Each entry corresponds to a summary statistic enumerated in `differential_analysis::summary`.
     */
    typedef std::array<bool, differential_analysis::n_summaries> ComputeSummaries;

    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * Specify that all summary statistics should be computed for a particular effect size.
         */
        static constexpr ComputeSummaries compute_all_summaries() {
            ComputeSummaries output { 0 };
            for (int i = 0; i < differential_analysis::n_summaries; ++i) {
                output[i] = true;
            }
            return output;
        }

        /**
         * Specify that no summary statistics should be computed for a particular effect size.
         */
        static constexpr ComputeSummaries compute_no_summaries() {
            ComputeSummaries output { 0 };
            for (int i = 0; i < differential_analysis::n_summaries; ++i) {
                output[i] = false;
            }
            return output;
        }

        /**
         * See `set_threshold()` for details.
         */
        static constexpr double threshold = 0;
    };
private:
    double threshold = Defaults::threshold;

    ComputeSummaries do_cohen = Defaults::compute_all_summaries();
    ComputeSummaries do_auc = Defaults::compute_all_summaries();
    ComputeSummaries do_lfc = Defaults::compute_all_summaries();
    ComputeSummaries do_delta_detected = Defaults::compute_all_summaries();

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
     * @param c Which summary statistics to compute for Cohen's d.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_cohen(ComputeSummaries s = Defaults::compute_all_summaries()) {
        do_cohen = s;
        return *this;
    }

    /**
     * @param s Whether to compute Cohen's d at all.
     *
     * This is an alias for `set_compute_cohen()` where `c = true` is equivalent to `s = Defaults::compute_all_summaries()`
     * and `c = false` is equivalent to `s = Defaults::compute_no_summaries()`.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_cohen(bool c) {
        std::fill(do_cohen.begin(), do_cohen.end(), c);
        return *this;
    }

    /**
     * @param s A summary statistic of interest.
     * @param c Whether to compute the summary statistic `s` for Cohen's d.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_cohen(differential_analysis::summary s, bool c) {
        do_cohen[s] = c;
        return *this;
    }

    /**
     * @param c Which summary statistics to compute for the AUC.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_auc(ComputeSummaries s = Defaults::compute_all_summaries()) {
        do_auc = s;
        return *this;
    }

    /**
     * @param s Whether to compute the AUC at all.
     *
     * This is an alias for `set_compute_auc()` where `c = true` is equivalent to `s = Defaults::compute_all_summaries()`
     * and `c = false` is equivalent to `s = Defaults::compute_no_summaries()`.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_auc(bool c) {
        std::fill(do_auc.begin(), do_auc.end(), c);
        return *this;
    }

    /**
     * @param s A summary statistic of interest.
     * @param c Whether to compute the summary statistic `s` for the AUC.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_auc(differential_analysis::summary s, bool c) {
        do_auc[s] = c;
        return *this;
    }

    /**
     * @param c Which summary statistics to compute for the LFC.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_lfc(ComputeSummaries s = Defaults::compute_all_summaries()) {
        do_lfc = s;
        return *this;
    }

    /**
     * @param s Whether to compute the LFC at all.
     *
     * This is an alias for `set_compute_lfc()` where `c = true` is equivalent to `s = Defaults::compute_all_summaries()`
     * and `c = false` is equivalent to `s = Defaults::compute_no_summaries()`.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_lfc(bool c) {
        std::fill(do_lfc.begin(), do_lfc.end(), c);
        return *this;
    }

    /**
     * @param s A summary statistic of interest.
     * @param c Whether to compute the summary statistic `s` for the LFC.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_lfc(differential_analysis::summary s, bool c) {
        do_lfc[s] = c;
        return *this;
    }

    /**
     * @param c Which summary statistics to compute for the delta detected.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_delta_detected(ComputeSummaries s = Defaults::compute_all_summaries()) {
        do_delta_detected = s;
        return *this;
    }

    /**
     * @param s Whether to compute the delta detected at all.
     *
     * This is an alias for `set_compute_delta_detected()` where `c = true` is equivalent to `s = Defaults::compute_all_summaries()`
     * and `c = false` is equivalent to `s = Defaults::compute_no_summaries()`.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_delta_detected(bool c) {
        std::fill(do_delta_detected.begin(), do_delta_detected.end(), c);
        return *this;
    }

    /**
     * @param s A summary statistic of interest.
     * @param c Whether to compute the summary statistic `s` for the delta detected.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_compute_delta_detected(differential_analysis::summary s, bool c) {
        do_delta_detected[s] = c;
        return *this;
    }

private:
    void set_everyone(differential_analysis::summary s, bool c) {
        do_cohen[s] = c;
        do_auc[s] = c;
        do_lfc[s] = c;
        do_delta_detected[s] = c;
        return;
    }

public:
    /**
     * @param s Whether to compute the minimum summary statistic for any effect size.
     *
     * This overrides any previous settings for the minimum from the effect-size-specific setters, e.g., `set_compute_cohen()`.
     * However, it can also be overridden by later calls to those setters.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_min(bool s) {
        set_everyone(differential_analysis::MIN, s);
        return *this;
    }

    /**
     * @param s Whether to compute the mean summary statistic for any effect size.
     *
     * This overrides any previous settings for the mean from the effect-size-specific setters, e.g., `set_compute_cohen()`.
     * However, it can also be overridden by later calls to those setters.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_mean(bool s) {
        set_everyone(differential_analysis::MEAN, s);
        return *this;
    }

    /**
     * @param s Whether to compute the median summary statistic for any effect size.
     *
     * This overrides any previous settings for the median from the effect-size-specific setters, e.g., `set_compute_cohen()`.
     * However, it can also be overridden by later calls to those setters.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_median(bool s) {
        set_everyone(differential_analysis::MEDIAN, s);
        return *this;
    }

    /**
     * @param s Whether to compute the maximum summary statistic for any effect size.
     *
     * This overrides any previous settings for the maximum from the effect-size-specific setters, e.g., `set_compute_cohen()`.
     * However, it can also be overridden by later calls to those setters.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_max(bool s) {
        set_everyone(differential_analysis::MAX, s);
        return *this;
    }

    /**
     * @param s Whether to compute the minimum rank summary statistic for any effect size.
     *
     * This overrides any previous settings for the minimum rank from the effect-size-specific setters, e.g., `set_compute_cohen()`.
     * However, it can also be overridden by later calls to those setters.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_summary_rank(bool s) {
        set_everyone(differential_analysis::MIN_RANK, s);
        return *this;
    }

public:
    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups.
     *
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
     * @param[out] auc Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the AUC.
     * @param[out] lfc Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the log-fold change instead of Cohen's d.
     * @param[out] delta_detected Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the delta in the detected proportions.
     * 
     * If `cohen` is of length 0, Cohen's d is not computed.
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     * The same applies to `auc`, `lfc` and `delta_detected`.
     * (`set_compute_cohen()` and related functions have no effect here.)
     *
     * @return `means`, `detected`, `cohen` and `auc` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename Stat>
    void run(const Matrix* p, const G* group, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        std::vector<std::vector<Stat*> > cohen, 
        std::vector<std::vector<Stat*> > auc,
        std::vector<std::vector<Stat*> > lfc,
        std::vector<std::vector<Stat*> > delta_detected) 
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        run_internal(p, group, ngroups, means, detected, cohen, auc, lfc, delta_detected);
    }        

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups in multiple blocks.
     *
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
     * @param[out] auc Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the AUC.
     * @param[out] lfc Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the log-fold change instead of Cohen's d.
     * @param[out] delta_detected Vector of vector of pointers as described for `cohen`, but instead storing summary statistics for the delta in the detected proportions.
     * 
     * If `cohen` is of length 0, Cohen's d is not computed.
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     * The same applies to `auc`, `lfc` and `delta_detected`.
     * (`set_compute_cohen()` and related functions have no effect here.)
     *
     * @return `means`, `detected`, `cohen` and `auc` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename B, typename Stat>
    void run_blocked(const Matrix* p, const G* group, const B* block, 
        std::vector<std::vector<Stat*> > means, 
        std::vector<std::vector<Stat*> > detected, 
        std::vector<std::vector<Stat*> > cohen,
        std::vector<std::vector<Stat*> > auc,
        std::vector<std::vector<Stat*> > lfc,
        std::vector<std::vector<Stat*> > delta_detected) 
    {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        if (block == NULL) {
            std::vector<Stat*> means2(ngroups), detected2(ngroups);
            for (size_t g = 0; g < ngroups; ++g) {
                means2[g] = means[g][0];
                detected2[g] = detected[g][0];
            }
            run_internal(p, group, ngroups, means2, detected2, cohen, auc);
            return;
        }
    
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        run_blocked_internal(p, group, block, ngroups, nblocks, means, detected, cohen, auc, lfc, delta_detected);
        return;
    }

private:
    template<class MAT, typename G, typename Stat>
    void run_internal(const MAT* p, const G* group, int ngroups, 
        std::vector<Stat*>& means, 
        std::vector<Stat*>& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc,
        std::vector<std::vector<Stat*> >& lfc,
        std::vector<std::vector<Stat*> >& delta_detected) 
    {
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, group, ngroups, static_cast<const int*>(NULL), 1, means, detected, cohen, auc, lfc, delta_detected);
    }

    template<class MAT, typename G, typename B, typename Stat>
    void run_blocked_internal(const MAT* p, const G* group, const B* block, int ngroups, int nblocks,
        std::vector<std::vector<Stat*> >& means, 
        std::vector<std::vector<Stat*> >& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc,
        std::vector<std::vector<Stat*> >& lfc,
        std::vector<std::vector<Stat*> >& delta_detected) 
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

        core(p, combos.data(), combo_size, group, ngroups, block, nblocks, means2, detected2, cohen, auc, lfc, delta_detected);
    }

    template<class MAT, typename L, class Ls, typename G, typename B, typename Stat>
    void core(const MAT* p, 
        const L* level, const Ls& level_size, 
        const G* group, int ngroups, 
        const B* block, int nblocks, 
        std::vector<Stat*>& means, 
        std::vector<Stat*>& detected, 
        std::vector<std::vector<Stat*> >& cohen, 
        std::vector<std::vector<Stat*> >& auc,
        std::vector<std::vector<Stat*> >& lfc,
        std::vector<std::vector<Stat*> >& delta_detected) 
    {
        size_t buffer_size = p->nrow() * ngroups * ngroups;

        const bool do_cohen = !cohen.empty();
        std::vector<Stat> cohen_buffer(do_cohen ?  buffer_size : 0);
        Stat* cohen_ptr = do_cohen ? cohen_buffer.data() : NULL;

        const bool do_auc = !auc.empty();
        std::vector<Stat> auc_buffer(do_auc ? buffer_size : 0);
        Stat* auc_ptr = do_auc ? auc_buffer.data() : NULL;

        const bool do_lfc = !lfc.empty();
        std::vector<Stat> lfc_buffer(do_lfc ? buffer_size : 0);
        Stat* lfc_ptr = do_lfc ? lfc_buffer.data() : NULL;

        const bool do_delta = !delta_detected.empty();
        std::vector<Stat> delta_buffer(do_delta ? buffer_size : 0);
        Stat* delta_ptr = do_delta ? delta_buffer.data() : NULL;

        std::vector<Stat*> effects { cohen_ptr, auc_ptr, lfc_ptr, delta_ptr };

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::ScoreMarkers", "Performing pairwise comparisons between groups of cells");
#endif
        if (!do_auc) {
            differential_analysis::BidimensionalFactory fact(p->nrow(), p->ncol(), means, detected, effects, level, &level_size, ngroups, nblocks, threshold);
            tatami::apply<0>(p, fact);

        } else {
            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<B> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            differential_analysis::PerRowFactory fact(p->nrow(), p->ncol(), means, detected, effects, level, &level_size, group, ngroups, block, nblocks, threshold);
            tatami::apply<0>(p, fact);
        }

        auto summarize = [&](Stat* ptr, std::vector<std::vector<Stat*> >& output) -> void {
            auto& min_rank = output[scran::differential_analysis::MIN_RANK];
            if (min_rank.size()) {
                differential_analysis::compute_min_rank(p->nrow(), ngroups, ptr, min_rank);
            }
            differential_analysis::summarize_comparisons(p->nrow(), ngroups, ptr, output); // non-const w.r.t. ptr's values, so this is done after min-rank calculations.
        };

        if (do_cohen) {
            summarize(cohen_ptr, cohen);
        }
        if (do_auc) {
            summarize(auc_ptr, auc);
        }
        if (do_lfc) {
            summarize(lfc_ptr, lfc);
        }
        if (do_delta) {
            summarize(delta_ptr, delta_detected);
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
        Results(
            size_t ngenes, 
            int ngroups, 
            int nblocks, 
            const ComputeSummaries& do_cohen, 
            const ComputeSummaries& do_auc, 
            const ComputeSummaries& do_lfc, 
            const ComputeSummaries& do_delta_detected)
        { 
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

            auto fill_effect = [&](const ComputeSummaries& do_this, auto& effect) {
                bool has_any = false;
                for (size_t i = 0; i < do_this.size(); ++i) {
                    if (do_this[i]) {
                        has_any = true;
                        break;
                    }
                }

                if (has_any) {
                    effect.resize(differential_analysis::n_summaries);
                    if (do_this[differential_analysis::MIN]) {
                        fill_inner(ngroups, effect[differential_analysis::MIN]);
                    }
                    if (do_this[differential_analysis::MEAN]) {
                        fill_inner(ngroups, effect[differential_analysis::MEAN]);
                    }
                    if (do_this[differential_analysis::MEDIAN]) {
                        fill_inner(ngroups, effect[differential_analysis::MEDIAN]);
                    }
                    if (do_this[differential_analysis::MAX]) {
                        fill_inner(ngroups, effect[differential_analysis::MAX]);
                    }
                    if (do_this[differential_analysis::MIN_RANK]) {
                        fill_inner(ngroups, effect[differential_analysis::MIN_RANK]);
                    }
                }
                return;
            };

            fill_effect(do_cohen, cohen);
            fill_effect(do_auc, auc);
            fill_effect(do_lfc, lfc);
            fill_effect(do_delta_detected, delta_detected);
            return;
        }
        /**
         * @endcond
         */

        /**
         * Summary statistics for Cohen's d.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > cohen;

        /**
         * Summary statistics for the AUC.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > auc;

        /**
         * Summary statistics for the log-fold change.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > lfc;

        /**
         * Summary statistics for the delta in the detected proportions.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > delta_detected;

        /**
         * Mean expression in each group.
         * Elements of the outer vector corresponds to the different groups;
         * elements of the middle vector correspond to the different blocking levels (this is of length 1 for `run()`);
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > means;

        /**
         * Proportion of detected expression in each group.
         * Elements of the outer vector corresponds to the different groups;
         * elements of the middle vector correspond to the different blocking levels (this is of length 1 for `run()`);
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat> > > detected;
    };

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups. 
     *
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam Stat Floating-point type to store the statistics.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * These should be 0-based and consecutive.
     *
     * @return A `Results` object containing the summary statistics and the other per-group statistics.
     * Whether particular statistics are computed depends on the configuration from `set_compute_cohen()` and related setters.
     */
    template<typename Stat = double, class MAT, typename G> 
    Results<Stat> run(const MAT* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, 1, do_cohen, do_auc, do_lfc, do_delta_detected); 

        auto mean_ptrs = vector_to_pointers3(res.means);
        auto detect_ptrs = vector_to_pointers3(res.detected);

        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        auto auc_ptrs = vector_to_pointers2(res.auc);
        auto lfc_ptrs = vector_to_pointers2(res.lfc);
        auto delta_ptrs = vector_to_pointers2(res.delta_detected);

        run_internal(p, group, ngroups, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs, lfc_ptrs, delta_ptrs);
        return res;
    }

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups in multiple blocks.
     *
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
     *
     * @return A `Results` object containing the summary statistics and the other per-group statistics.
     * Whether particular statistics are computed depends on the configuration from `set_compute_cohen()` and related setters.
     */
    template<typename Stat = double, class MAT, typename G, typename B> 
    Results<Stat> run_blocked(const MAT* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }
    
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, nblocks, do_cohen, do_auc, do_lfc, do_delta_detected); 

        auto mean_ptrs = vector_to_pointers2(res.means);
        auto detect_ptrs = vector_to_pointers2(res.detected);

        auto cohen_ptrs = vector_to_pointers2(res.cohen);
        auto auc_ptrs = vector_to_pointers2(res.auc);
        auto lfc_ptrs = vector_to_pointers2(res.lfc);
        auto delta_ptrs = vector_to_pointers2(res.delta_detected);

        run_blocked_internal(p, group, block, ngroups, nblocks, mean_ptrs, detect_ptrs, cohen_ptrs, auc_ptrs, lfc_ptrs, delta_ptrs);
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
