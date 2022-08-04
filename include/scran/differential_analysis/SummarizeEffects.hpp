#ifndef SCRAN_SUMMARIZE_EFFECTS_HPP
#define SCRAN_SUMMARIZE_EFFECTS_HPP

#include "../utils/macros.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "summarize_comparisons.hpp"

#include <vector>

/**
 * @file SummarizeEffects.hpp
 *
 * @brief Summarize the effect sizes from pairwise comparisons.
 */

namespace scran {

/**
 * @brief Summarize pairwise effects into summary statistics per group.
 *
 * This class computes the statistics that are used for marker detection in `ScoreMarkers`.
 * Briefly, given `n` groups, each group is involved in `n - 1` pairwise comparisons and thus has `n - 1` effect sizes, as computed by `PairwiseEffects`.
 * For each group, we compute summary statistics - e.g., the minimum, median, mean - of the effect sizes across all of that group's comparisons.
 * Users can then sort by any of these summaries to obtain a ranking of potential marker genes for each group.
 * 
 * The choice of summary statistic dictates the interpretation of the ranking.
 * Given a group X:
 * 
 * - A large mean effect size indicates that the gene is upregulated in X compared to the average of the other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is a good general-purpose summary statistic for ranking, usually by decreasing size to obtain upregulated markers in X.
 * - A large median effect size indicates that the gene is upregulated in X compared to most (>50%) other groups.
 *   A small value indicates that the gene is downregulated in X instead.
 *   This is also a good general-purpose summary, with the advantage of being more robust to outlier effects compared to the mean.
 *   However, it also has the disadvantage of being less sensitive to strong effects in a minority of comparisons.
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
 * The exact definition of "large" and "small" depends on the choice of effect size from `PairwiseEffects`.
 * For Cohen's d, LFC and delta-detected, the value must be positive to be considered "large", and negative to be considered "small".
 * For the AUC, a value greater than 0.5 is considered "large" and less than 0.5 is considered "small".
 *
 * The interpretation above is also contingent on the log-fold change threshold used in `PairwiseEffects`.
 * For positive thresholds, small effects cannot be unambiguously interpreted as downregulation, as the effect is already adjusted to account for the threshold.
 * As a result, only large effects can be interpreted as evidence for upregulation.
 *
 * `NaN` effect sizes are allowed, e.g., if two groups do not exist in the same block for a blocked analysis in `PairwiseEffects`.
 * This class will ignore `NaN` values when computing each summary.
 * If all effects are `NaN` for a particular group, the summary statistic will also be `NaN`.
 *
 * All choices of summary statistics are enumerated by `differential_analysis::summary`.
 */
class SummarizeEffects {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_compute_min()`.
         */
        static constexpr bool compute_min = true;

        /**
         * See `set_compute_mean()`.
         */
        static constexpr bool compute_mean = true;

        /**
         * See `set_compute_median()`.
         */
        static constexpr bool compute_median = true;

        /**
         * See `set_compute_max()`.
         */
        static constexpr bool compute_max = true;

        /**
         * See `set_compute_min_rank()`.
         */
        static constexpr bool compute_min_rank = true;
    };

private:
    int num_threads = Defaults::num_threads;
    bool compute_min = Defaults::compute_min;
    bool compute_mean = Defaults::compute_mean;
    bool compute_median = Defaults::compute_median;
    bool compute_max = Defaults::compute_max;
    bool compute_min_rank = Defaults::compute_min_rank;

public:
    /**
     * @param n Number of threads to use. 
     * @return A reference to this `SummarizeEffects` object.
     */
    SummarizeEffects& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

    /**
     * @param c Whether to report the minimum of the pairwise effects.
     * @return A reference to this `SummarizeEffects` object.
     *
     * This has no effect on the `run()` overload that accepts a `summaries` vector.
     * For this method, the minimum is calculated if `summaries[differential_analysis::MIN]` is of non-zero length.
     */
    SummarizeEffects& set_compute_min(bool c = Defaults::compute_min) {
        compute_min = c;
        return *this;
    }

    /**
     * @param c Whether to report the mean of the pairwise effects.
     * @return A reference to this `SummarizeEffects` object.
     * 
     * This has no effect on the `run()` overload that accepts a `summaries` vector.
     * For this method, the minimum is calculated if `summaries[differential_analysis::MEAN]` is of non-zero length.
     */
    SummarizeEffects& set_compute_mean(bool c = Defaults::compute_mean) {
        compute_mean = c;
        return *this;
    }

    /**
     * @param c Whether to report the median of the pairwise effects.
     * @return A reference to this `SummarizeEffects` object.
     *
     * This has no effect on the `run()` overload that accepts a `summaries` vector.
     * For this method, the minimum is calculated if `summaries[differential_analysis::MEDIAN]` is of non-zero length.
     */
    SummarizeEffects& set_compute_median(bool c = Defaults::compute_median) {
        compute_median = c;
        return *this;
    }

    /**
     * @param c Whether to report the maximum of the pairwise effects.
     * @return A reference to this `SummarizeEffects` object.
     *
     * This has no effect on the `run()` overload that accepts a `summaries` vector.
     * For this method, the minimum is calculated if `summaries[differential_analysis::MAX]` is of non-zero length.
     */
    SummarizeEffects& set_compute_max(bool c = Defaults::compute_max) {
        compute_max = c;
        return *this;
    }

    /**
     * @param c Whether to report the min-rank of the pairwise effects.
     * @return A reference to this `SummarizeEffects` object.
     * 
     * This has no effect on the `run()` overload that accepts a `summaries` vector.
     * For this method, the minimum is calculated if `summaries[differential_analysis::MIN_RANK]` is of non-zero length.
     */
    SummarizeEffects& set_compute_min_rank(bool c = Defaults::compute_min_rank) {
        compute_min_rank = c;
        return *this;
    }

public:
    /**
     * Summarize the effect sizes for the pairwise comparisons to obtain a set of summary statistics for each gene in each group.
     *
     * If `summaries` is of length 0, no summaries are computed.
     * If any of the inner vectors of `summaries` are of length 0, the corresponding summary statistic is not computed.
     *
     * @tparam Stat Floating point type for the statistics.
     *
     * @param ngenes Number of genes.
     * @param ngroups Number of groups.
     * @param[in] effects Pointer to a 3-dimensional array containing the pairwise statistics, see `PairwiseEffects::Results` for details.
     * @param[out] summaries Vector of vector of pointers to arrays of length equal to the number of genes.
     * The vector should be of length equal to `differential_analysis::n_summaries` (see `differential_analysis::summary`).
     * Each inner vector corresponds to a summary statistic - i.e., minimum, mean, median, maximum and min-rank - and should be of length equal to the number of groups.
     * Each pointer corresponds to a group, and points to an array that is used to store the associated summary statistic across all genes for that group.
     */
    template<typename Stat>
    void run(size_t ngenes, size_t ngroups, const Stat* effects, std::vector<std::vector<Stat*> > summaries) const {
        if (summaries.empty()) {
            return;
        }

        auto& min_rank = summaries[differential_analysis::MIN_RANK];
        if (min_rank.size()) {
            differential_analysis::compute_min_rank(ngenes, ngroups, effects, min_rank, num_threads);
        }

        differential_analysis::summarize_comparisons(ngenes, ngroups, effects, summaries, num_threads); 
    }

    /**
     * Summarize the effect sizes for the pairwise comparisons to obtain a set of summary statistics for each gene in each group.
     *
     * If `summaries` is of length 0, no summaries are computed.
     * If any of the inner vectors of `summaries` are of length 0, the corresponding summary statistic is not computed.
     *
     * @tparam Stat Floating point type for the statistics.
     *
     * @param ngenes Number of genes.
     * @param ngroups Number of groups.
     * @param[in] effects Pointer to a 3-dimensional array containing the pairwise statistics, see `PairwiseEffects::Results` for details.
     *
     * @return A vector of vectors of vectors containing summary effects for each gene in each group.
     * The vector is of length equal to `differential_analysis::n_summaries` (see `differential_analysis::summary`).
     * Each inner vector corresponds to a summary statistic - i.e., minimum, mean, median, maximum and min-rank - and is of length equal to the number of groups.
     * Each pointer corresponds to a group, and points to an array containing the associated summary statistic across all genes for that group.
     */
    template<typename Stat>
    std::vector<std::vector<std::vector<Stat> > > run(size_t ngenes, size_t ngroups, const Stat* effects) const {
        std::vector<std::vector<std::vector<Stat> > > output(differential_analysis::n_summaries);

        auto inflate = [&](auto& o) -> void {
            o.resize(ngroups);
            for (auto& o2 : o) {
                o2.resize(ngenes);
            }
        };

        if (compute_min) {
            inflate(output[differential_analysis::MIN]);
        }
        if (compute_mean) {
            inflate(output[differential_analysis::MEAN]);
        }
        if (compute_median) {
            inflate(output[differential_analysis::MEDIAN]);
        }
        if (compute_max) {
            inflate(output[differential_analysis::MAX]);
        }
        if (compute_min_rank) {
            inflate(output[differential_analysis::MIN_RANK]);
        }

        run(ngenes, ngroups, effects, vector_to_pointers(output));
        return output;
    }
};

}

#endif
