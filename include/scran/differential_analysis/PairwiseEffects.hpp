#ifndef SCRAN_PAIRWISE_EFFECTS_HPP
#define SCRAN_PAIRWISE_EFFECTS_HPP

#include "../utils/macros.hpp"

#include "Factory.hpp"
#include "../utils/vector_to_pointers.hpp"

#include "tatami/stats/apply.hpp"

/**
 * @file PairwiseEffects.hpp
 *
 * @brief Compute pairwise effect sizes between groups of cells.
 */

namespace scran {

/**
 * @cond
 */
template<typename Stat>
std::vector<Stat*> vector_to_pointers3(std::vector<std::vector<std::vector<Stat> > >& input) {
    std::vector<Stat*> ptrs;
    for (auto& current : input) {
        ptrs.push_back(current.front().data()); // first vector from each element.
    }
    return ptrs;
}
/**
 * @endcond
 */

/**
 * @brief Compute pairwise effect size between groups of cells.
 *
 * This class computes the effect sizes for the pairwise comparisons used in `ScoreMarkers`, prior to any ranking of marker genes.
 * It may be desirable to call this function directly if the pairwise effects themselves are of interest, rather than per-group summaries.
 *
 * @section effect-sizes Choice of effect sizes
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
 * @section blocked Blocked comparisons
 * In the presence of multiple batches, we can block on the batch of origin for each cell.
 * Comparisons are only performed between the groups of cells in the same batch (also called "blocking level" below).
 * The batch-specific effect sizes are then combined into a single aggregate value for output.
 * This strategy avoids most problems related to batch effects as we never directly compare across different blocking levels.
 *
 * Specifically, for each gene and each pair of groups, we obtain one effect size per blocking level.
 * We consolidate these into a single statistic by computing the weighted mean across levels.
 * The weight for each level is defined as the product of the sizes of the two groups;
 * this favors contribution from levels with more cells in both groups, where the effect size is presumably more reliable.
 * (Obviously, levels with no cells in either group will not contribute anything to the weighted mean.)
 *
 * If two groups never co-occur in the same blocking level, no effect size will be computed and a `NaN` is reported in the output.
 * We do not attempt to reconcile batch effects in a partially confounded scenario.
 *
 * @section other Other statistics
 * We report the mean log-expression of all cells in each group, as well as the proportion of cells with detectable expression in each group.
 * These statistics are useful for quickly interpreting the differences in expression driving the effect size summaries.
 *
 * If blocking is involved, we compute the mean and proportion for each group in each separate blocking level.
 * This is helpful for detecting differences in expression between batches.
 * They can also be combined into a single statistic for each group by using the `average_vectors()` or `average_vectors_weighted()` functions.
 */
class PairwiseEffects {
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
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_compute_cohen()`.
         */
        static constexpr bool compute_cohen = true;

        /**
         * See `set_compute_auc()`.
         */
        static constexpr bool compute_auc = true;

        /**
         * See `set_compute_lfc()`.
         */
        static constexpr bool compute_lfc = true;

        /**
         * See `set_compute_delta_detected()`.
         */
        static constexpr bool compute_delta_detected = true;
    };

private:
    double threshold = Defaults::threshold;
    int num_threads = Defaults::num_threads;
    bool do_cohen = Defaults::compute_cohen;
    bool do_auc = Defaults::compute_auc;
    bool do_lfc = Defaults::compute_lfc;
    bool do_delta_detected = Defaults::compute_delta_detected;

public:
    /**
     * @param t Threshold on the log-fold change.
     * This should be non-negative.
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_threshold(double t = Defaults::threshold) {
        threshold = t;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

    /**
     * @param c Whether to compute Cohen's d. 
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_compute_cohen(bool c = Defaults::compute_cohen) {
        do_cohen = c;
        return *this;
    }

    /**
     * @param c Whether to compute the AUC.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_compute_auc(bool c = Defaults::compute_auc) {
        do_auc = c;
        return *this;
    }

    /**
     * @param c Whether to compute the log-fold change.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_compute_lfc(bool c = Defaults::compute_lfc) {
        do_lfc = c;
        return *this;
    }

    /**
     * @param c Whether to compute the delta-detected.
     *
     * This only has an effect for `run()` methods that return `Results`.
     * Otherwise, we make this decision based on the validity of the input pointers. 
     *
     * @return A reference to this `PairwiseEffects` object.
     */
    PairwiseEffects& set_compute_delta_detected(bool c = Defaults::compute_delta_detected) {
        do_delta_detected = c;
        return *this;
    }

public:
    /**
     * Compute effect sizes for pairwise comparisons between groups.
     * On output, `means`, `detected`, `cohen`, `auc`, `lfc` and `delta_detected` are filled with their corresponding statistics. 
     *
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam Stat Floating-point type to store the statistics.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[out] means Vector of length equal to the number of groups.
     * Each element corresponds to a group and is a pointer to an array of length equal to the number of rows in `p`.
     * This is used to store the mean expression of each group across all genes.
     * @param[out] detected Vector of length equal to the number of groups,
     * Each element corresponds to a group and is a pointer to an array of length equal to the number of rows in `p`.
     * This is used to store the proportion of detected expression in each group.
     * @param[out] cohen Pointer to an array of length equal to $GN^2$, where `N` is the number of groups and `G` is the number of genes (see `Results` for details).
     * This is filled with the Cohen's d for the pairwise comparisons between groups across all genes.
     * @param[out] auc Pointer to an array as described for `cohen`, but instead storing the AUC.
     * @param[out] lfc Pointer to an array as described for `cohen`, but instead storing the log-fold change. 
     * @param[out] delta_detected Pointer to an array as described for `cohen`, but instead the delta in the detected proportions.
     */
    template<class Matrix, typename G, typename Stat>
    void run(
        const Matrix* p, 
        const G* group, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        Stat* cohen,
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        size_t ngroups = means.size();
        std::vector<int> group_size(ngroups);
        for (size_t i = 0; i < p->ncol(); ++i) {
            ++(group_size[group[i]]);
        }
        core(p, group, group_size, group, ngroups, static_cast<const int*>(NULL), 1, std::move(means), std::move(detected), cohen, auc, lfc, delta_detected);
    }

    /**
     * Compute effect sizes for pairwise comparisons between groups, accounting for any blocking factor in the dataset.
     * On output, `means`, `detected`, `cohen`, `auc`, `lfc` and `delta_detected` are filled with their corresponding statistics. 
     *
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam B Integer type for the block assignments.
     * @tparam Stat Floating-point type to store the statistics.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[in] block Pointer to an array of length equal to the number of columns in `p`, containing the blocking factor.
     * Block identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[out] means Vector of length equal to the number of groups.
     * Each element corresponds to a group and is another vector of length equal to the number of blocks.
     * Each entry of the inner vector is a pointer to an array of length equal to the number of rows in `p`,
     * which is used to store the mean expression of each group across all genes.
     * @param[out] detected Vector of length equal to the number of groups.
     * Each element corresponds to a group and is another vector of length equal to the number of blocks.
     * Each entry of the inner vector is a pointer to an array of length equal to the number of rows in `p`,
     * which is used to store the proportion of detected expression in each group.
     * @param[out] cohen Pointer to an array of length equal to $GN^2$, where `N` is the number of groups and `G` is the number of genes (see `Results` for details).
     * This is filled with the Cohen's d for the pairwise comparisons between groups across all genes.
     * @param[out] auc Pointer to an array as described for `cohen`, but instead storing the AUC.
     * @param[out] lfc Pointer to an array as described for `cohen`, but instead storing the log-fold change. 
     * @param[out] delta_detected Pointer to an array as described for `cohen`, but instead the delta in the detected proportions.
     */
    template<class Matrix, typename G, typename B, typename Stat>
    void run_blocked(
        const Matrix* p, 
        const G* group, 
        const B* block, 
        std::vector<std::vector<Stat*> > means, 
        std::vector<std::vector<Stat*> > detected, 
        Stat* cohen,
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        if (block == NULL) {
            run(p, group, fetch_first(means), fetch_first(detected), cohen, auc, lfc, delta_detected);
            return;
        }

        size_t ngroups = means.size();
        size_t nblocks = (ngroups ? means[0].size() : 0); // if means.size() == 0, then there are no groups, so there are no blocks, either.
        size_t ncombos = ngroups * nblocks;
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

        core(p, combos.data(), combo_size, group, ngroups, block, nblocks, std::move(means2), std::move(detected2), cohen, auc, lfc, delta_detected);
    }

private:
    template<class Matrix, typename L, class Ls, typename G, typename B, typename Stat>
    void core(const Matrix* p, 
        const L* level, 
        const Ls& level_size, 
        const G* group, 
        size_t ngroups, 
        const B* block, 
        size_t nblocks, 
        std::vector<Stat*> means, 
        std::vector<Stat*> detected, 
        Stat* cohen, 
        Stat* auc,
        Stat* lfc,
        Stat* delta_detected) 
    const {
        size_t buffer_size = p->nrow() * ngroups * ngroups;

        if (auc == NULL) {
            differential_analysis::BidimensionalFactory fact(
                p->nrow(), 
                p->ncol(), 
                std::move(means), 
                std::move(detected), 
                cohen,
                lfc,
                delta_detected,
                level, 
                &level_size, 
                ngroups, 
                nblocks, 
                threshold
            );
            tatami::apply<0>(p, fact, num_threads);

        } else {
            // Need to remake this, as there's no guarantee that 'blocks' exists.
            std::vector<B> tmp_blocks;
            if (!block) {
                tmp_blocks.resize(p->ncol());
                block = tmp_blocks.data();
            }

            differential_analysis::PerRowFactory fact(
                p->nrow(), 
                p->ncol(), 
                std::move(means), 
                std::move(detected), 
                cohen,
                auc,
                lfc,
                delta_detected,
                level, 
                &level_size, 
                group, 
                ngroups, 
                block, 
                nblocks, 
                threshold
            );

            tatami::apply<0>(p, fact, num_threads);
        }
    }

public:
    /**
     * @brief Pairwise effect size results.
     *
     * @tparam Stat Floating-point type to store the statistics.
     *
     * For any given effect size, the pairwise statistics are stored in a 3-dimensional array.
     * The first dimension is the fastest changing, is of length equal to the number of groups, and represents the first group.
     * The second dimension is the next fastest changing, is also of length equal to the number of groups, and represents the second group.
     * The third dimension is the slowest changing, is of length equal to the number of genes, and represents the gene.
     * Thus, an entry $(i, j, k)$ represents the effect size of gene $k$ for group $i$ against group $j$.
     */
    template<typename Stat>
    struct Results {
        /**
         * @cond
         */
        Results(size_t ngenes, size_t ngroups, bool do_cohen, bool do_auc, bool do_lfc, bool do_delta) {
            size_t nelements = ngenes * ngroups * ngroups;
            if (do_cohen) {
                cohen.resize(nelements);
            }
            if (do_auc) {
                auc.resize(nelements);
            }
            if (do_lfc) {
                lfc.resize(nelements);
            }
            if (do_delta) {
                delta_detected.resize(nelements);
            }
        }
        /**
         * @endcond
         */

        /**
         * Vector of pairwise Cohen's d, to be interpreted as a 3-dimensional array.
         */
        std::vector<Stat> cohen;

        /**
         * Vector of pairwise AUCs, to be interpreted as a 3-dimensional array.
         */
        std::vector<Stat> auc;

        /**
         * Vector of pairwise log-fold changes, to be interpreted as a 3-dimensional array.
         */
        std::vector<Stat> lfc;

        /**
         * Vector of pairwise delta-detected, to be interpreted as a 3-dimensional array.
         */
        std::vector<Stat> delta_detected;
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
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[out] means Vector of length equal to the number of groups.
     * Each element corresponds to a group and is a pointer to an array of length equal to the number of rows in `p`.
     * This is used to store the mean expression of each group across all genes.
     * @param[out] detected Vector of length equal to the number of groups,
     * Each element corresponds to a group and is a pointer to an array of length equal to the number of rows in `p`.
     * This is used to store the proportion of detected expression in each group.
     *
     * @return A `Results` object is returned containing the pairwise effects.
     * `means` and `detected` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename Stat>
    Results<Stat> run(const Matrix* p, const G* group, std::vector<Stat*> means, std::vector<Stat*> detected) const {
        auto ngroups = means.size();
        Results<Stat> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group, 
            std::move(means), 
            std::move(detected), 
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
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
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[in] block Pointer to an array of length equal to the number of columns in `p`, containing the blocking factor.
     * Block identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[out] means Vector of length equal to the number of groups.
     * Each element corresponds to a group and is another vector of length equal to the number of blocks.
     * Each entry of the inner vector is a pointer to an array of length equal to the number of rows in `p`,
     * which is used to store the mean expression of each group across all genes.
     * @param[out] detected Vector of length equal to the number of groups.
     * Each element corresponds to a group and is another vector of length equal to the number of blocks.
     * Each entry of the inner vector is a pointer to an array of length equal to the number of rows in `p`,
     * which is used to store the proportion of detected expression in each group.
     *
     * @return A `Results` object is returned containing the pairwise effects.
     * `means` and `detected` are filled with their corresponding statistics on output.
     */
    template<class Matrix, typename G, typename B, typename Stat>
    Results<Stat> run_blocked(const Matrix* p, const G* group, const B* block, std::vector<std::vector<Stat*> > means, std::vector<std::vector<Stat*> > detected) const {
        if (block == NULL) {
            return run(p, group, fetch_first(means), fetch_first(detected));
        }

        auto ngroups = means.size();
        Results<Stat> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run_blocked(
            p,
            group,
            block,
            std::move(means),
            std::move(detected),
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return res;
    }

public:
    /**
     * @brief Pairwise effect size results, with per-group means.
     *
     * @tparam Stat Floating-point type to store the statistics.
     *
     * See `Results` for more details on how to interpret the 3-dimensional effect size arrays.
     */
    template<typename Stat>
    struct ResultsWithMeans : public Results<Stat> {
        /**
         * @cond
         */
        ResultsWithMeans(size_t ngenes, size_t ngroups, size_t nblocks, bool do_cohen, bool do_auc, bool do_lfc, bool do_delta) :
            Results<Stat>(ngenes, ngroups, do_cohen, do_auc, do_lfc, do_delta), means(ngroups), detected(ngroups) 
        {
            for (size_t g = 0; g < ngroups; ++g) {
                means[g].reserve(nblocks);
                detected[g].reserve(nblocks);
                for (size_t b = 0; b < nblocks; ++b) {
                    means[g].emplace_back(ngenes);
                    detected[g].emplace_back(ngenes);
                }
            }
        }
        /**
         * @endcond
         */

        /**
         * Each element of `means` corresponds to a group and is itself a vector of length equal to the number of blocks.
         * For group `i`, each element of `means[i]` corresponds to a block and is itself a vector of length equal to the number of genes.
         * For block `j`, each element of `means[i][j]` corresponds to a gene and contains the mean expression of that gene in block `j` for group `i`.
         */
        std::vector<std::vector<std::vector<Stat> > > means;

        /**
         * Each element of `detected` corresponds to a group and is itself a vector of length equal to the number of blocks.
         * For group `i`, each element of `detected[i]` corresponds to a block and is itself a vector of length equal to the number of genes.
         * For block `j`, each element of `detected[i][j]` corresponds to a gene and contains the proportion of detected expression of that gene in block `j` for group `i`.
         */
        std::vector<std::vector<std::vector<Stat> > > detected;
    };

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups.
     *
     * @tparam Stat Floating-point type to store the statistics.
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     *
     * @return A `ResultsWithMeans` object is returned containing the pairwise effects, plus the mean expression and detected proportion in each group.
     */
    template<typename Stat = double, class Matrix, typename G>
    ResultsWithMeans<Stat> run(const Matrix* p, const G* group) {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        ResultsWithMeans<Stat> res(p->nrow(), ngroups, 1, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group, 
            vector_to_pointers3(res.means), 
            vector_to_pointers3(res.detected), 
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return res; 
    }

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups in multiple blocks.
     *
     * @tparam Stat Floating-point type to store the statistics.
     * @tparam Matrix A **tatami** matrix class, usually a `NumericMatrix`.
     * @tparam G Integer type for the group assignments.
     * @tparam B Integer type for the block assignments.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * Group identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     * @param[in] block Pointer to an array of length equal to the number of columns in `p`, containing the blocking factor.
     * Block identifiers should be 0-based and should contain all integers in $[0, N)$ where $N$ is the number of unique groups.
     *
     * @return A `ResultsWithMeans` object is returned containing the pairwise effects, plus the mean expression and detected proportion in each group and block.
     */
    template<typename Stat = double, class Matrix, typename G, typename B> 
    ResultsWithMeans<Stat> run_blocked(const Matrix* p, const G* group, const B* block) {
        if (block == NULL) {
            return run(p, group);
        }

        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        ResultsWithMeans<Stat> res(p->nrow(), ngroups, nblocks, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run_blocked(
            p,
            group,
            block,
            vector_to_pointers(res.means),
            vector_to_pointers(res.detected),
            harvest_pointer(res.cohen, do_cohen),
            harvest_pointer(res.auc, do_auc),
            harvest_pointer(res.lfc, do_lfc),
            harvest_pointer(res.delta_detected, do_delta_detected)
        );
        return res;
    }

private:
    template<typename Ptr>
    static std::vector<Ptr> fetch_first(const std::vector<std::vector<Ptr> >& input) {
        std::vector<Ptr> output;
        output.reserve(input.size());
        for (const auto& i : input) {
            output.push_back(i.front());
        }
        return output;
    }

    template<typename Stat> 
    static Stat* harvest_pointer(std::vector<Stat>& source, bool use) {
        if (use) {
            return source.data();
        } else {
            return static_cast<Stat*>(NULL);
        }
    }
};

}

#endif