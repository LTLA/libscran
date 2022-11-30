#ifndef SCRAN_SCORE_MARKERS_HPP
#define SCRAN_SCORE_MARKERS_HPP

#include "../utils/macros.hpp"

#include "Factory.hpp"
#include "cohens_d.hpp"
#include "simple_diff.hpp"
#include "summarize_comparisons.hpp"
#include "../utils/vector_to_pointers.hpp"

#include "tatami/stats/apply.hpp"

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

// This cache tries to store as many of the reverse effects as possible before
// it starts evicting. Evictions are based on the principle that it is better
// to store effects that will be re-used quickly, thus freeing up the cache for
// future stores. The 'speed' of reusability of each cache entry depends on the
// first group in the comparison corresponding to each cached effect size; the
// smaller the first group, the sooner it will be reached when iterating across
// groups in the ScoreMarkers function.
//
// So, the policy is to evict cache entries when the identity of the first
// group in the cached entry is larger than the corresponding value for the
// incoming entry. Given that, if the cache is full, we have to throw away
// one of these effects anyway, I'd prefer to hold onto the one we're using
// soon, because at least it'll be freed up rapidly.
template<typename Stat>
struct EffectsCacher {
    EffectsCacher(size_t ngenes_, int ngroups_, int cache_size_) :
        ngenes(ngenes_),
        ngroups(ngroups_),
        cache_size(cache_size_),
        full_set(ngroups * ngenes),
        actions(ngroups),
        staging_cache(ngroups)
    {
        vector_pool.reserve(cache_size);
    }

public:
    size_t ngenes;
    int ngroups;
    int cache_size;

    std::vector<double> full_set;
    std::vector<unsigned char> actions;
    std::vector<std::vector<Stat> > staging_cache;
    std::vector<std::vector<Stat> > vector_pool;

    std::map<std::pair<int, int>, std::vector<Stat> > cached;

public:
    void clear() {
        cached.clear();
    }

public:
    void configure(int group) {
        for (int other = 0; other < ngroups; ++other) {
            if (other == group) {
                actions[other] = 0;
                continue;
            }

            if (cache_size == 0) {
                actions[other] = 1;
                continue;
            }

            // Impossible to be cached by a previous run, so we can skip a 
            // look-up - however, it is a candidate to _be_ cached, possibly
            // evicting other existing cache entries.
            if (other > group) {
                actions[other] = 2;
                continue;
            }

            // Need to recompute cache entries that were previously evicted.
            if (cached.empty()) { 
                actions[other] = 1;
                continue;
            }

            const auto& front = cached.begin()->first;
            if (front.first > group || front.second > other) { // less-thans should be impossible.
                actions[other] = 1;
                continue;
            }

            // Transferring the cached vector to the full_set.
            actions[other] = 0;
            auto& x = cached.begin()->second;
            for (size_t i = 0, end = x.size(); i < end; ++i) {
                full_set[other + i * ngroups] = x[i];
            }

            vector_pool.emplace_back(std::move(x)); // recycle memory to avoid heap reallocations.
            cached.erase(cached.begin());
        }

        // Refining our choice by doing a dummy run and seeing
        // whether eviction actually happens. If it doesn't, we won't
        // bother caching, beause that would be a waste of memory accesses.
        for (int other = 0; other < ngroups; ++other) {
            if (actions[other] != 2) {
                continue;
            }

            std::pair<int, int> key(other, group);
            if (cached.size() < static_cast<size_t>(cache_size)) {
                cached[key] = std::vector<Stat>();
                prepare_staging_cache(other);
                continue;
            }

            auto it = cached.end();
            --it;
            if ((it->first).first > other) {
                // Evict the existing entry. This is safe to do so as the
                // evicted entry won't be used in this round anyway.
                auto& evicted = it->second;
                if (evicted.size()) {
                    vector_pool.emplace_back(std::move(evicted));
                }
                cached.erase(it);
                prepare_staging_cache(other);
                cached[key] = std::vector<Stat>();
            } else {
                // No evictions, so we don't even bother computing the reverse.
                actions[other] = 1;
            }
        }
    }

    void transfer(int group) {
        for (int other = 0; other < ngroups; ++other) {
            if (actions[other] != 2) {
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
    };
private:
    double threshold = Defaults::threshold;

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

public:
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
    ScoreMarkers& set_summary_min_rank(bool s) {
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
    const {
        differential_analysis::EffectsCalculator runner(nthreads, threshold);

        int ngroups = means.size();
        Overlord<Stat> overlord(p->nrow(), ngroups, auc.empty());
        auto state = runner.run(p, group, ngroups, overlord);

        process_simple_effects(state, means, detected, cohen, lfc, delta_detected);
        summarize_auc(state, auc, overlord.auc_buffer);
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
    const {
        differential_analysis::EffectsCalculator runner(nthreads, threshold);

        int ngroups = means.size();
        int nblocks = (ngroups ? means[0].size() : 0); // no blocks = no groups.
        Overlord<Stat> overlord(p->nrow(), ngroups, auc.empty());
        auto state = runner.run_blocked(p, group, ngroups, block, nblocks, overlord);

        int ncombos = ngroups * nblocks;
        std::vector<Stat*> means2(ncombos), detected2(ncombos);
        auto mIt = means2.begin(), dIt = detected2.begin();
        for (int g = 0; g < ngroups; ++g) {
            for (int b = 0; b < nblocks; ++b, ++mIt, ++dIt) {
                *mIt = means[g][b];
                *dIt = detected[g][b];
            }
        }

        process_simple_effects(state, means2, detected2, cohen, lfc, delta_detected);
        summarize_auc(state, auc, overlord.auc_buffer);
    }

private:
    template<typename Stat>
    class Overlord {
    public:
        Overlord(size_t nr, int ng, bool skip_auc) : skipped(skip_auc), auc_buffer(skip_auc ? 0 : nr * ng * ng) {}

        bool needs_auc() const {
            return !skipped;
        }

        bool skipped;
        std::vector<Stat> auc_buffer;

        struct ComplexWorker {
            ComplexWorker(Stat* auc_) : auc(auc_) {}
            Stat* auc;

            Stat* prepare_auc_buffer(size_t gene, int ngroups) { 
                return auc + gene * ngroups * ngroups;
            }

            // No-op, because it's copied straight into the auc_buffer.
            void consume_auc_buffer(size_t, int, const double*) {}
        };

        ComplexWorker complex_worker() {
            return ComplexWorker(this->auc_buffer.data());
        }
    };

    template<typename Stat>
    void process_simple_effects(
        const differential_analysis::EffectsCalculator::State& state, 
        std::vector<Stat*>& means,
        std::vector<Stat*>& detected,
        std::vector<std::vector<Stat*> >& cohen,
        std::vector<std::vector<Stat*> >& lfc,
        std::vector<std::vector<Stat*> >& delta_detected) 
    const {
        auto ngenes = state.ngenes;
        const auto& level_size = state.level_size;
        auto nlevels = level_size.size();
        auto ngroups = state.ngroups;
        auto nblocks = state.nblocks;
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
        differential_analysis::EffectsCacher<Stat> cache(ngenes, ngroups, cache_size);
        auto& full_set = cache.full_set;
        auto& actions = cache.actions;
        auto& staging_cache = cache.staging_cache;

        if (cohen.size()) {
            cache.clear();

            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp parallel num_threads(nthreads)
                {
                    std::vector<double> effect_buffer(ngroups);
                    #pragma omp for
                    for (size_t gene = 0; gene < ngenes; ++gene) {
#else
                SCRAN_CUSTOM_PARALLEL(ngenes, [&](size_t start, size_t end) -> void {
                    std::vector<double> effect_buffer(ngroups);
                    for (size_t gene = start; gene < end; ++gene) {
#endif

                        size_t in_offset = nlevels * gene;
                        auto my_means = tmp_means + in_offset;
                        auto my_variances = tmp_variances + in_offset;

                        auto cohen_ptr = full_set.data() + gene * ngroups;
                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == 0) {
                                continue;
                            }

                            if (actions[other] == 1) {
                                cohen_ptr[other] = differential_analysis::compute_pairwise_cohens_d<false>(group, other, my_means, my_variances, level_size, ngroups, nblocks, threshold);
                                continue;
                            }

                            auto tmp = differential_analysis::compute_pairwise_cohens_d<true>(group, other, my_means, my_variances, level_size, ngroups, nblocks, threshold);
                            cohen_ptr[other] = tmp.first;
                            staging_cache[other][gene] = tmp.second;
                        }

                        differential_analysis::summarize_comparisons(ngroups, cohen_ptr, group, gene, cohen, effect_buffer);

#ifndef SCRAN_CUSTOM_PARALLEL
                    }
                }
#else
                    }
                }, nthreads);
#endif

                if (cohen[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), cohen[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }

        if (lfc.size()) {
            cache.clear();

            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp parallel num_threads(nthreads)
                {
                    std::vector<double> effect_buffer(ngroups);
                    #pragma omp for
                    for (size_t gene = 0; gene < ngenes; ++gene) {
#else
                SCRAN_CUSTOM_PARALLEL(ngenes, [&](size_t start, size_t end) -> void {
                    std::vector<double> effect_buffer(ngroups);
                    for (size_t gene = start; gene < end; ++gene) {
#endif

                        auto my_means = tmp_means + nlevels * gene;
                        auto lfc_ptr = full_set.data() + gene * ngroups;
                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == 0) {
                                continue;
                            }

                            auto val = differential_analysis::compute_pairwise_simple_diff(group, other, my_means, level_size, ngroups, nblocks);
                            lfc_ptr[other] = val;
                            if (actions[other] == 2) {
                                staging_cache[other][gene] = -val;
                            } 
                        }

                        differential_analysis::summarize_comparisons(ngroups, lfc_ptr, group, gene, lfc, effect_buffer);

#ifndef SCRAN_CUSTOM_PARALLEL
                    }
                }
#else
                    }
                }, nthreads);
#endif

                if (lfc[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), lfc[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }

        if (delta_detected.size()) {
            cache.clear();

            for (int group = 0; group < ngroups; ++group) {
                cache.configure(group);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp parallel num_threads(nthreads)
                {
                    std::vector<double> effect_buffer(ngroups);
                    #pragma omp for
                    for (size_t gene = 0; gene < ngenes; ++gene) {
#else
                SCRAN_CUSTOM_PARALLEL(ngenes, [&](size_t start, size_t end) -> void {
                    std::vector<double> effect_buffer(ngroups);
                    for (size_t gene = start; gene < end; ++gene) {
#endif

                        auto my_detected = tmp_detected + gene * nlevels;
                        auto delta_detected_ptr = full_set.data() + gene * ngroups;

                        for (int other = 0; other < ngroups; ++other) {
                            if (actions[other] == 0) {
                                continue;
                            }

                            auto val = differential_analysis::compute_pairwise_simple_diff(group, other, my_detected, level_size, ngroups, nblocks);
                            delta_detected_ptr[other] = val;
                            if (actions[other] == 2) {
                                staging_cache[other][gene] = -val;
                            } 
                        }

                        differential_analysis::summarize_comparisons(ngroups, delta_detected_ptr, group, gene, delta_detected, effect_buffer);

#ifndef SCRAN_CUSTOM_PARALLEL
                    }
                }
#else
                    }
                }, nthreads);
#endif

                if (delta_detected[differential_analysis::summary::MIN_RANK].size()) {
                    differential_analysis::compute_min_rank(ngenes, ngroups, group, full_set.data(), delta_detected[differential_analysis::summary::MIN_RANK][group], nthreads);
                }

                cache.transfer(group);
            }
        }
    }

    template<typename Stat>
    void summarize_auc(
        const differential_analysis::EffectsCalculator::State& state, 
        std::vector<std::vector<Stat*> >& auc,
        std::vector<Stat>& auc_buffer) 
    const {
        // If we need the min-rank AUCs, there's no choice but to hold everything in memory.
        if (auc.size()) {
            auto ngenes = state.ngenes;
            auto ngroups = state.ngroups;
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
     * @tparam Stat Floating-point type to store the statistics.
     * @brief Marker effect size summaries and other statistics.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ScoreMarkers::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    template<typename Stat>
    struct Results {
        /**
         * @cond
         */
        Results() {}

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
    Results<Stat> run(const MAT* p, const G* group) const {
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, 1, do_cohen, do_auc, do_lfc, do_delta_detected); 
        run(
            p, 
            group,
            vector_to_front_pointers(res.means),
            vector_to_front_pointers(res.detected),
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
    Results<Stat> run_blocked(const MAT* p, const G* group, const B* block) const {
        if (block == NULL) {
            return run(p, group);
        }
 
        auto ngroups = *std::max_element(group, group + p->ncol()) + 1;
        auto nblocks = *std::max_element(block, block + p->ncol()) + 1;
        Results<Stat> res(p->nrow(), ngroups, nblocks, do_cohen, do_auc, do_lfc, do_delta_detected); 

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
