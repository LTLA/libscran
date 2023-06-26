#ifndef SCRAN_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_SUMMARIZE_COMPARISONS_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

/**
 * @file summarize_comparisons.hpp
 *
 * @brief Utilities for effect summarization.
 */

namespace scran {

/**
 * @brief Utilities for differential analysis.
 */
namespace differential_analysis {

/**
 * Choice of summary statistic for the pairwise effects of a given group, see `SummarizeEffects` for details.
 * `n_summaries` is used to denote the number of summaries. 
 */
enum summary {
    MIN,
    MEAN,
    MEDIAN,
    MAX,
    MIN_RANK,
    n_summaries 
};

/**
 * @cond
 */
template<class IT>
double median (IT start, size_t n) {
    int halfway = n / 2;
    std::nth_element(start, start + halfway, start + n);

    if (n % 2 == 1) {
        return start[halfway];
    }

    double med = start[halfway];
    std::nth_element(start, start + halfway - 1, start + halfway);
    return (med + start[halfway - 1])/ 2;
}

template<typename Stat>
void summarize_comparisons(int ngroups, const Stat* effects, int group, size_t gene, std::vector<std::vector<Stat*> >& output, std::vector<Stat>& buffer) {
    auto ebegin = buffer.data();
    auto elast = ebegin;	

    // Ignoring the self comparison and pruning out NaNs.
    {
        auto eptr = effects;
        for (int r = 0; r < ngroups; ++r, ++eptr) {
            if (r == group || std::isnan(*eptr)) {
                continue;
            }
            *elast = *eptr;
            ++elast;
        }
    }

    int ncomps = elast - ebegin;
    if (ncomps == 0) {
        for (size_t i = 0; i < MIN_RANK; ++i) {
            if (output[i].size()) {
                output[i][group][gene] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    } else if (ncomps == 1) {
        for (size_t i = 0; i < MIN_RANK; ++i) { 
            if (output[i].size()) {
                output[i][group][gene] = *ebegin;
            }
        }
    } else {
        if (output[MIN].size()) {
            output[MIN][group][gene] = *std::min_element(ebegin, elast);
        }
        if (output[MEAN].size()) {
            output[MEAN][group][gene] = std::accumulate(ebegin, elast, 0.0) / ncomps; 
        }
        if (output[MEDIAN].size()) {
            output[MEDIAN][group][gene] = median(ebegin, ncomps); 
        }
        if (output[MAX].size()) {
            output[MAX][group][gene] = *std::max_element(ebegin, elast);
        }
    }
}

template<typename Stat>
void summarize_comparisons(size_t ngenes, int ngroups, const Stat* effects, std::vector<std::vector<Stat*> >& output, int threads) {
#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> effect_buffer(ngroups);
        #pragma omp for
        for (size_t gene = 0; gene < ngenes; ++gene) {
#else
    SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
        std::vector<double> effect_buffer(ngroups);
        for (size_t gene = start, end = start + length; gene < end; ++gene) {
#endif

            auto base = effects + gene * ngroups * ngroups;
            for (int l = 0; l < ngroups; ++l) {
                summarize_comparisons(ngroups, base + l * ngroups, l, gene, output, effect_buffer);
            }

#ifndef SCRAN_CUSTOM_PARALLEL 
        }
	}
#else
        }
    }, ngenes, threads);
#endif

    return;
}

template<typename Stat>
size_t fill_and_sort_rank_buffer(const Stat* effects, size_t stride, std::vector<std::pair<Stat, int> >& buffer) {
    auto bIt = buffer.begin();
    for (size_t i = 0, end = buffer.size(); i < end; ++i, effects += stride) {
        if (!std::isnan(*effects)) {
            bIt->first = -*effects; // negative to sort by decreasing value.
            bIt->second = i;
            ++bIt;
        }
    }
    std::sort(buffer.begin(), bIt);
    return bIt - buffer.begin();
}

template<typename Stat, typename Rank>
void compute_min_rank_internal(size_t use, const std::vector<std::pair<Stat, int> >& buffer, Rank* output) {
    Rank counter = 1;
    for (size_t i = 0; i < use; ++i) {
        auto& current = output[buffer[i].second];
        if (counter < current) {
            current = counter;
        }
        ++counter;
    }
}

template<typename Stat>
void compute_min_rank(size_t ngenes, size_t ngroups, int group, const Stat* effects, Stat* output, int threads) {
    // Assign groups to threads, minus the 'group' itself.
    std::vector<std::vector<int> > assignments(threads);
    {
        size_t per_thread = std::ceil(static_cast<double>(ngroups - 1) / threads);
        auto cur_thread = assignments.begin();
        for (size_t counter = 0; counter < ngroups; ++counter) {
            if (counter == group) {
                continue;
            }
            if (cur_thread->size() >= per_thread) {
                ++cur_thread;
            }
            cur_thread->push_back(counter);
        }
    }

    std::vector<std::vector<int> > stores(threads, std::vector<int>(ngenes, ngenes + 1));

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        #pragma omp for
        for (int t = 0; t < threads; ++t) {
            for (auto g : assignments[t]) {
#else
    SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        for (size_t t = start, end = start + length; t < end; ++t) {  // should be a no-op loop, but we do this just in case.
            for (auto g : assignments[t]) {
#endif

                auto used = fill_and_sort_rank_buffer(effects + g, ngroups, buffer);
                compute_min_rank_internal(used, buffer, stores[t].data());

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
    }
#else
            }
        }
    }, threads, threads);
#endif

    std::fill(output, output + ngenes, ngenes + 1); 
    for (int t = 0; t < threads; ++t) {
        auto copy = output;
        for (auto x : stores[t]) {
            if (x < *copy) {
                *copy = x;
            }
            ++copy;
        }
    }
}

template<typename Stat>
void compute_min_rank(size_t ngenes, size_t ngroups, const Stat* effects, std::vector<Stat*>& output, int threads) {
    size_t shift = ngroups * ngroups;

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        #pragma omp for
        for (size_t g = 0; g < ngroups; ++g) {
#else
    SCRAN_CUSTOM_PARALLEL([&](size_t, size_t start, size_t length) -> void {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        for (size_t g = start, end = start + length; g < end; ++g) { 
#endif

            auto target = output[g];
            std::fill(target, target + ngenes, ngenes + 1); 
            auto base = effects + g * ngroups;

            for (int g2 = 0; g2 < ngroups; ++g2) {
                if (g == g2) {
                    continue;
                }
                auto used = fill_and_sort_rank_buffer(base + g2, shift, buffer);
                compute_min_rank_internal(used, buffer, target);
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, ngroups, threads);
#endif
}

/**
 * @endcond
 */

}

}

#endif
