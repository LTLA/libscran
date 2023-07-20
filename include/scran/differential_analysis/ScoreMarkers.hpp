#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "../utils/macros.hpp"

#include "MatrixCalculator.hpp"
#include "cohens_d.hpp"
#include "simple_diff.hpp"
#include "summarize_comparisons.hpp"

#include "../utils/vector_to_pointers.hpp"
#include "../utils/average_vectors.hpp"
#include "../utils/blocking.hpp"

#include "tatami/tatami.hpp"

#include <array>
#include <map>
#include <vector>

/**
 * @file ScoreMarkers.hpp
 *
 * @brief Compute marker scores for each gene in each group of cells.
 */

namespace scran {

/**
 * @cond
 */
namespace differential_analysis {

enum class CacheAction : unsigned char { SKIP, COMPUTE, CACHE };

// This cache tries to store as many of the reverse effects as possible before
// it starts evicting. Evictions are based on the principle that it is better
// to store effects that will be re-used quickly, thus freeing up the cache for
// future stores. The 'speed' of reusability of each cache entry depends on the
// first group in the comparison corresponding to each cached effect size; the
// smaller the first group, the sooner it will be reached when iterating across
// groups in the ScoreMarkers function.
//
// So, the policy is to evict cache entries when the identity of the first
// group in the cached entry is larger than the identity of the first group for
// the incoming entry. Given that, if the cache is full, we have to throw away
// one of these effects anyway, I'd prefer to hold onto the one we're using
// soon, because at least it'll be freed up rapidly.
template<typename Stat_>
struct EffectsCacher {
    EffectsCacher(size_t nge, int ngr, int cs) :
        ngenes(nge),
        ngroups(ngr),
        cache_size(cs),
        actions(ngroups),
        staging_cache(ngroups)
    {
        vector_pool.reserve(cache_size);
    }

public:
    size_t ngenes;
    int ngroups;
    int cache_size;

    std::vector<CacheAction> actions;

    // 'staging_cache' contains the set of cached effects in the other
    // direction, i.e., all other groups compared to the current group. This is
    // only used to avoid repeated look-ups in 'cached' while filling the
    // effect size vectors; they will ultimately be transferred to cached after
    // the processing for the current group is complete.
    std::vector<std::vector<Stat_> > staging_cache;

    // 'vector_pool' allows us to recycle effect size vectors to avoid reallocations.
    std::vector<std::vector<Stat_> > vector_pool;

    // 'cached' contains the cached effect size vectors from previous groups. Note
    // that the use of a map is deliberate as we need the sorting.
    std::map<std::pair<int, int>, std::vector<Stat_> > cached;

public:
    void clear() {
        cached.clear();
    }

public:
    void configure(int group, std::vector<double>& full_set) {
        // During calculation of effects, the current group (i.e., 'group') is
        // the first group in the comparison and 'other' is the second group.
        // However, remember that we want to cache the reverse effects, so in
        // the cached entry, 'group' is second and 'other' is first.
        for (int other = 0; other < ngroups; ++other) {
            if (other == group) {
                actions[other] = CacheAction::SKIP;
                continue;
            }

            if (cache_size == 0) {
                actions[other] = CacheAction::COMPUTE;
                continue;
            }

            // If 'other' is later than 'group', it's a candidate to be cached,
            // as it will be reused when the current group becomes 'other'.
            if (other > group) {
                actions[other] = CacheAction::CACHE;
                continue;
            }

            // Need to recompute cache entries that were previously evicted. We
            // do so if the cache is empty or the first group of the first cached
            // entry has a higher index than the current group (and thus the 
            // desired comparison's effects cannot possibly exist in the cache).
            if (cached.empty()) { 
                actions[other] = CacheAction::COMPUTE;
                continue;
            }

            const auto& front = cached.begin()->first;
            if (front.first > group || front.second > other) { 
                // Technically, the second clause should be (front.first == group && front.second > other).
                // However, less-thans should be impossible as they should have been used up during processing
                // of previous 'group' values. Thus, equality is already implied if the first condition fails.
                actions[other] = CacheAction::COMPUTE;
                continue;
            }

            // If we got past the previous clause, this implies that the first cache entry
            // contains the effect sizes for the desired comparison (i.e., 'group - other').
            // We thus transfer the cached vector to the full_set.
            actions[other] = CacheAction::SKIP;
            auto& x = cached.begin()->second;
            for (size_t i = 0, end = x.size(); i < end; ++i) {
                full_set[other + i * ngroups] = x[i];
            }

            vector_pool.emplace_back(std::move(x)); // recycle memory to avoid heap reallocations.
            cached.erase(cached.begin());
        }

        // Refining our choice of cacheable entries by doing a dummy run and
        // seeing whether eviction actually happens. If it doesn't, we won't
        // bother caching, because that would be a waste of memory accesses.
        for (int other = 0; other < ngroups; ++other) {
            if (actions[other] != CacheAction::CACHE) {
                continue;
            }

            std::pair<int, int> key(other, group);
            if (cached.size() < static_cast<size_t>(cache_size)) {
                cached[key] = std::vector<Stat_>();
                prepare_staging_cache(other);
                continue;
            }

            // Looking at the last cache entry. If the first group of this
            // entry is larger than the first group of the incoming entry, we
            // evict it, as the incoming entry has faster reusability.
            auto it = cached.end();
            --it;
            if ((it->first).first > other) {
                auto& evicted = it->second;
                if (evicted.size()) {
                    vector_pool.emplace_back(std::move(evicted));
                }
                cached.erase(it);
                prepare_staging_cache(other);
                cached[key] = std::vector<Stat_>();
            } else {
                // Otherwise, if we're not going to do any evictions, we
                // indicate that we shouldn't even bother computing the
                // reverse, because we're won't cache the incoming entry.
                actions[other] = CacheAction::COMPUTE;
            }
        }
    }

    void transfer(int group) {
        for (int other = 0; other < ngroups; ++other) {
            if (actions[other] != CacheAction::CACHE) {
                continue;
            }

            // All the to-be-staged logic is already implemented in 'configure()'.
            auto& staged = staging_cache[other];
            std::pair<int, int> key(other, group);
            cached[key] = std::move(staged);
        }
    }

private:
    void prepare_staging_cache(int other) {
        if (vector_pool.empty()) {
            staging_cache[other].resize(ngenes);
        } else {
            // Recycling existing heap allocations.
            staging_cache[other] = std::move(vector_pool.back());
            vector_pool.pop_back();
        }
    }
};

}
/**
 * @endcond
 */

/**
 * @brief Score each gene as a candidate marker for each group of cells.
 *
 * Markers are identified by differential expression analyses between pairs of groups of cells (e.g., clusters, cell types).
 * Given `n` groups, each group is involved in `n - 1` pairwise comparisons and thus has `n - 1` effect sizes.
 * For each group, we compute summary statistics - e.g., median, mean - of the effect sizes across all of that group's comparisons.
 * Users can then sort by any of these summaries to obtain a ranking of potential marker genes for each group.
 *
 * The choice of effect size and summary statistic determines the characteristics of the resulting marker set.
 * For the effect sizes: we compute Cohen's d, the area under the curve (AUC), the log-fold change and the delta-detected,
 * which are described in more detail in the documentation for `PairwiseEffects`,
 * For the summary statistics: we compute the minimum, mean, median, maximum and min-rank of the effect sizes across each group's pairwise comparisons,
 * which are described in `SummarizeEffects`.
 *
 * If the dataset contains blocking factors such as batch or sample, we compute the effect size within each level of the blocking factor.
 * This avoids interference from batch effects or sample-to-sample variation.
 * Users can also adjust the effect size to account for a minimum log-fold change threshold,
 * in order to focus on markers with larger changes in expression.
 * See `PairwiseEffects` for more details. 
 *
 * As a courtesy, we also compute the mean expression a
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

        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;

        /** 
         * See `set_cache_size()` for details.
         */
        static constexpr int cache_size = 100;

        /**
         * See `set_block_weight_policy()` for more details.
         */
        static constexpr WeightPolicy block_weight_policy = WeightPolicy::VARIABLE;

        /**
         * See `set_variable_block_weight_parameters()` for more details.
         */
        static constexpr VariableBlockWeightParameters variable_block_weight_parameters = VariableBlockWeightParameters();
    };

private:
    double threshold = Defaults::threshold;
    WeightPolicy block_weight_policy = Defaults::block_weight_policy;
    VariableBlockWeightParameters variable_block_weight_parameters = Defaults::variable_block_weight_parameters;

    ComputeSummaries do_cohen = Defaults::compute_all_summaries();
    ComputeSummaries do_auc = Defaults::compute_all_summaries();
    ComputeSummaries do_lfc = Defaults::compute_all_summaries();
    ComputeSummaries do_delta_detected = Defaults::compute_all_summaries();

    int cache_size = Defaults::cache_size;
    int nthreads = Defaults::num_threads;

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
     * @param n Number of threads to use. 
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param c Size of the cache, in terms of the number of pairwise comparisons.
     * Larger values speed up the comparisons at the cost of higher memory consumption.
     *
     * @return A reference to this `ScoreMarkers` object.
     */
    ScoreMarkers& set_cache_size(int c = Defaults::cache_size) {
        cache_size = c;
        return *this;
    }

    /**
     * @param w Policy to use for weighting blocks when computing average statistics/effect sizes across blocks.
     * 
     * @return A reference to this `ScoreMarkers` instance.
     */
    ScoreMarkers& set_block_weight_policy(WeightPolicy w = Defaults::block_weight_policy) {
        block_weight_policy = w;
        return *this;
    }

    /**
     * @param v Parameters for the variable block weights, see `variable_block_weight()` for more details.
     * Only used when the block weight policy is set to `WeightPolicy::VARIABLE`.
     * 
     * @return A reference to this `ScoreMarkers` instance.
     */
    ScoreMarkers& set_variable_block_weight_parameters(VariableBlockWeightParameters v = Defaults::variable_block_weight_parameters) {
        variable_block_weight_parameters = v;
        return *this;
    }

public:
    /**
     * @param s Which summary statistics to compute for Cohen's d.
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
     * @param c Whether to compute Cohen's d at all.
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
     * @param s Which summary statistics to compute for the AUC.
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
     * @param c Whether to compute the AUC at all.
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
     * @param s Which summary statistics to compute for the LFC.
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
     * @param c Whether to compute the LFC at all.
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
     * @param s Which summary statistics to compute for the delta detected.
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
     * @param c Whether to compute the delta detected at all.
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
    ScoreMarkers& set_summary_min_rank(bool s) {
        set_everyone(differential_analysis::MIN_RANK, s);
        return *this;
    }

public:
    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups.
     * On completion, `means`, `detected`, `cohen`, `auc`, `lfc` and `delta_detected` are filled with their corresponding statistics.
     *
     * If `cohen` is of length 0, Cohen's d is not computed.
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     * The same applies to `auc`, `lfc` and `delta_detected`.
     * (`set_compute_cohen()` and related functions have no effect here.)
     *
     * @tparam Data_ Matrix data type.
     * @tparam Index_ Matrix index type.
     * @tparam Group_ Integer type for the group assignments.
     * @tparam Stat_ Floating-point type to store the statistics.
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
     */
    template<typename Value_, typename Index_, typename Group_, typename Stat_>
    void run(const tatami::Matrix<Value_, Index_>* p, const Group_* group, 
        std::vector<Stat_*> means, 
        std::vector<Stat_*> detected, 
        std::vector<std::vector<Stat_*> > cohen, 
        std::vector<std::vector<Stat_*> > auc,
        std::vector<std::vector<Stat_*> > lfc,
        std::vector<std::vector<Stat_*> > delta_detected) 
    const {
        differential_analysis::MatrixCalculator runner(nthreads, threshold, block_weight_policy, variable_block_weight_parameters);

        size_t ngenes = p->nrow();
        size_t ngroups = means.size();
        Overlord<Stat_> overlord(ngenes, ngroups, auc.empty());
        auto state = runner.run(p, group, ngroups, overlord);

        process_simple_effects(ngenes, ngroups, 1, state, means, detected, cohen, lfc, delta_detected);
        summarize_auc(ngenes, ngroups, state, auc, overlord.auc_buffer);
    }

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups in multiple blocks.
     * On completion, `means`, `detected`, `cohen`, `auc`, `lfc` and `delta_detected` are filled with their corresponding statistics.
     *
     * If `cohen` is of length 0, Cohen's d is not computed.
     * If any of the inner vectors are of length 0, the corresponding summary statistic is not computed.
     * The same applies to `auc`, `lfc` and `delta_detected`.
     * (`set_compute_cohen()` and related functions have no effect here.)
     *
     * @tparam Data_ Matrix data type.
     * @tparam Index_ Matrix index type.
     * @tparam Group_ Integer type for the group assignments.
     * @tparam Block_ Integer type for the block assignments.
     * @tparam Stat_ Floating-point type to store the statistics.
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
     */
    template<typename Value_, typename Index_, typename Group_, typename Block_, typename Stat_>
    void run_blocked(const tatami::Matrix<Value_, Index_>* p, const Group_* group, const Block_* block, 
        std::vector<Stat_*> means, 
        std::vector<Stat_*> detected, 
        std::vector<std::vector<Stat_*> > cohen,
        std::vector<std::vector<Stat_*> > auc,
        std::vector<std::vector<Stat_*> > lfc,
        std::vector<std::vector<Stat_*> > delta_detected) 
    const {
        differential_analysis::MatrixCalculator runner(nthreads, threshold, block_weight_policy, variable_block_weight_parameters);

        size_t ngenes = p->nrow();
        size_t ngroups = means.size();
        size_t nblocks = count_ids(p->ncol(), block);
        Overlord<Stat_> overlord(ngenes, ngroups, auc.empty());
        auto state = runner.run_blocked(p, group, ngroups, block, nblocks, overlord);

        int ncombos = ngroups * nblocks;
        std::vector<std::vector<Stat_> > means_store(ncombos), detected_store(ncombos);
        std::vector<Stat_*> means2(ncombos), detected2(ncombos);
        for (int c = 0; c < ncombos; ++c) {
            means_store[c].resize(ngenes);
            detected_store[c].resize(ngenes);
            means2[c] = means_store[c].data();
            detected2[c] = detected_store[c].data();
        }

        process_simple_effects(ngenes, ngroups, nblocks, state, means2, detected2, cohen, lfc, delta_detected);
        summarize_auc(ngenes, ngroups, state, auc, overlord.auc_buffer);

        // Averaging the remaining statistics.
        std::vector<double> weights(nblocks);
        std::vector<Stat_*> mstats(nblocks), dstats(nblocks);

        for (int gr = 0; gr < ngroups; ++gr) {
            for (int b = 0; b < nblocks; ++b) {
                size_t offset = gr * static_cast<size_t>(nblocks) + b;
                weights[b] = state.level_weight[offset];
                mstats[b] = means2[offset];
                dstats[b] = detected2[offset];
            }

            average_vectors_weighted(ngenes, mstats, weights.data(), means[gr]);
            average_vectors_weighted(ngenes, dstats, weights.data(), detected[gr]);
        }
    }

private:
    template<typename Stat_>
    class Overlord {
    public:
        Overlord(size_t nr, size_t ng, bool skip_auc) : skipped(skip_auc), auc_buffer(skip_auc ? 0 : nr * ng * ng) {}

        bool needs_auc() const {
            return !skipped;
        }

        bool skipped;
        std::vector<Stat_> auc_buffer;

        Stat_* prepare_auc_buffer(size_t gene, size_t ngroups) { 
            return auc_buffer.data() + gene * ngroups * ngroups;
        }
    };

    template<typename Stat_>
    void process_simple_effects(
        size_t ngenes, // using size_t consistently here, to eliminate integer overflow bugs when computing products.
        size_t ngroups,
        size_t nblocks,
        const differential_analysis::MatrixCalculator::State& state, 
        std::vector<Stat_*>& means,
        std::vector<Stat_*>& detected,
        std::vector<std::vector<Stat_*> >& cohen,
        std::vector<std::vector<Stat_*> >& lfc,
        std::vector<std::vector<Stat_*> >& delta_detected) 
    const {
        const auto& level_weight = state.level_weight;
        auto nlevels = level_weight.size();
        const auto* tmp_means = state.means.data();
        const auto* tmp_variances = state.variances.data();
        const auto* tmp_detected = state.detected.data();

        // Transferring the block-wise statistics over.
        {
            auto my_means = tmp_means;
            auto my_detected = tmp_detected;
            for (size_t gene = 0; gene < ngenes; ++gene) {
                for (size_t l = 0; l < nlevels; ++l, ++my_means, ++my_detected) {
                    means[l][gene] = *my_means;
                    detected[l][gene] = *my_detected;
                }
            }
        }

        // Looping over each group and computing the various summaries. We do this on
        // a per-group basis to avoid having to store the full group-by-group matrix of
        // effect sizes that we would otherwise need as input to SummarizeEffects.
        differential_analysis::EffectsCacher<Stat_> cache(ngenes, ngroups, cache_size);
        std::vector<double> full_set(ngroups * ngenes);

        if (cohen.size()) {
            cache.clear();
            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group, full_set);

                tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
                    size_t in_offset = nlevels * start;
                    auto my_means = tmp_means + in_offset;
                    auto my_variances = tmp_variances + in_offset;

                    const auto& actions = cache.actions;
                    auto& staging_cache = cache.staging_cache;

                    auto cohen_ptr = full_set.data() + start * ngroups;
                    std::vector<double> effect_buffer(ngroups);

                    for (size_t gene = start, end = start + length; gene < end; ++gene, my_means += nlevels, my_variances += nlevels, cohen_ptr += ngroups) {
                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == differential_analysis::CacheAction::SKIP) {
                                continue;
                            }

                            if (actions[other] == differential_analysis::CacheAction::COMPUTE) {
                                cohen_ptr[other] = differential_analysis::compute_pairwise_cohens_d<false>(group, other, my_means, my_variances, level_weight, ngroups, nblocks, threshold);
                                continue;
                            }

                            auto tmp = differential_analysis::compute_pairwise_cohens_d<true>(group, other, my_means, my_variances, level_weight, ngroups, nblocks, threshold);
                            cohen_ptr[other] = tmp.first;
                            staging_cache[other][gene] = tmp.second;
                        }

                        differential_analysis::summarize_comparisons(ngroups, cohen_ptr, group, gene, cohen, effect_buffer);
                    }
                }, ngenes, nthreads);

                if (cohen[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), cohen[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }

        if (lfc.size()) {
            cache.clear();
            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group, full_set);

                tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
                    auto my_means = tmp_means + nlevels * start;

                    const auto& actions = cache.actions;
                    auto& staging_cache = cache.staging_cache;

                    auto lfc_ptr = full_set.data() + start * ngroups;
                    std::vector<double> effect_buffer(ngroups);

                    for (size_t gene = start, end = start + length; gene < end; ++gene, my_means += nlevels, lfc_ptr += ngroups) {
                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == differential_analysis::CacheAction::SKIP) {
                                continue;
                            }

                            auto val = differential_analysis::compute_pairwise_simple_diff(group, other, my_means, level_weight, ngroups, nblocks);
                            lfc_ptr[other] = val;
                            if (actions[other] == differential_analysis::CacheAction::CACHE) {
                                staging_cache[other][gene] = -val;
                            } 
                        }

                        differential_analysis::summarize_comparisons(ngroups, lfc_ptr, group, gene, lfc, effect_buffer);
                    }
                }, ngenes, nthreads);

                if (lfc[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), lfc[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }

        if (delta_detected.size()) {
            cache.clear();
            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group, full_set);

                tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
                    auto my_detected = tmp_detected + nlevels * start;

                    const auto& actions = cache.actions;
                    auto& staging_cache = cache.staging_cache;

                    auto delta_detected_ptr = full_set.data() + start * ngroups;
                    std::vector<double> effect_buffer(ngroups);

                    for (size_t gene = start, end = start + length; gene < end; ++gene, my_detected += nlevels, delta_detected_ptr += ngroups) {
                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == differential_analysis::CacheAction::SKIP) {
                                continue;
                            }

                            auto val = differential_analysis::compute_pairwise_simple_diff(group, other, my_detected, level_weight, ngroups, nblocks);
                            delta_detected_ptr[other] = val;
                            if (actions[other] == differential_analysis::CacheAction::CACHE) {
                                staging_cache[other][gene] = -val;
                            } 
                        }

                        differential_analysis::summarize_comparisons(ngroups, delta_detected_ptr, group, gene, delta_detected, effect_buffer);
                    }
                }, ngenes, nthreads);

                if (delta_detected[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), delta_detected[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }
    }

    template<typename Stat_>
    void summarize_auc(
        size_t ngenes, 
        size_t ngroups,
        const differential_analysis::MatrixCalculator::State& state, 
        std::vector<std::vector<Stat_*> >& auc,
        std::vector<Stat_>& auc_buffer) 
    const {
        // If we need the min-rank AUCs, there's no choice but to hold everything in memory.
        if (auc.size()) {
            differential_analysis::summarize_comparisons(ngenes, ngroups, auc_buffer.data(), auc, nthreads);
            if (auc[differential_analysis::summary::MIN_RANK].size()) {
                differential_analysis::compute_min_rank(ngenes, ngroups, auc_buffer.data(), auc[differential_analysis::summary::MIN_RANK], nthreads);
            }
        }
    }

public:
    /** 
     * @brief Results of the marker scoring.
     * 
     * @tparam Stat_ Floating-point type to store the statistics.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ScoreMarkers::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    template<typename Stat_>
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(
            size_t ngenes, 
            int ngroups, 
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

            fill_inner(ngroups, means);
            fill_inner(ngroups, detected);

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
        std::vector<std::vector<std::vector<Stat_> > > cohen;

        /**
         * Summary statistics for the AUC.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat_> > > auc;

        /**
         * Summary statistics for the log-fold change.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat_> > > lfc;

        /**
         * Summary statistics for the delta in the detected proportions.
         * Elements of the outer vector correspond to the different summary statistics (see `differential_analysis::summary`);
         * elements of the middle vector correspond to the different groups;
         * and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<std::vector<Stat_> > > delta_detected;

        /**
         * Mean expression in each group.
         * Elements of the outer vector corresponds to the different groups, and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<Stat_> > means;

        /**
         * Proportion of detected expression in each group.
         * Elements of the outer vector corresponds to the different groups, and elements of the inner vector correspond to individual genes.
         */
        std::vector<std::vector<Stat_> > detected;
    };

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups. 
     *
     * @tparam Stat_ Floating-point type to store the statistics.
     * @tparam Data_ Matrix data type.
     * @tparam Index_ Matrix index type.
     * @tparam Group_ Integer type for the group assignments.
     *
     * @param p Pointer to a **tatami** matrix instance.
     * @param[in] group Pointer to an array of length equal to the number of columns in `p`, containing the group assignments.
     * These should be 0-based and consecutive.
     *
     * @return A `Results` object containing the summary statistics and the other per-group statistics.
     * Whether particular statistics are computed depends on the configuration from `set_compute_cohen()` and related setters.
     */
    template<typename Stat_ = double, typename Value_, typename Index_, typename Group_>
    Results<Stat_> run(const tatami::Matrix<Value_, Index_>* p, const Group_* group) const {
        auto ngroups = count_ids(p->ncol(), group);
        Results<Stat_> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group,
            vector_to_pointers(res.means),
            vector_to_pointers(res.detected),
            vector_to_pointers(res.cohen),
            vector_to_pointers(res.auc),
            vector_to_pointers(res.lfc),
            vector_to_pointers(res.delta_detected)
        );
        return res;
    }

    /**
     * Score potential marker genes by computing summary statistics across pairwise comparisons between groups in multiple blocks.
     *
     * @tparam Stat_ Floating-point type to store the statistics.
     * @tparam Data_ Matrix data type.
     * @tparam Index_ Matrix index type.
     * @tparam Group_ Integer type for the group assignments.
     * @tparam Block_ Integer type for the block assignments.
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
    template<typename Stat_ = double, typename Value_, typename Index_, typename Group_, typename Block_>
    Results<Stat_> run_blocked(const tatami::Matrix<Value_, Index_>* p, const Group_* group, const Block_* block) const {
        if (block == NULL) {
            return run(p, group);
        }

        auto ngroups = count_ids(p->ncol(), group);
        Results<Stat_> res(p->nrow(), ngroups, do_cohen, do_auc, do_lfc, do_delta_detected); 

        run_blocked(
            p, 
            group,
            block,
            vector_to_pointers(res.means),
            vector_to_pointers(res.detected),
            vector_to_pointers(res.cohen),
            vector_to_pointers(res.auc),
            vector_to_pointers(res.lfc),
            vector_to_pointers(res.delta_detected)
        );
        return res;
    }
};

}

#endif
