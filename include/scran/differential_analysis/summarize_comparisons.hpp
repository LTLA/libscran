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
void summarize_comparisons(size_t ngenes, int ngroups, const Stat* effects, std::vector<std::vector<Stat*> >& output, int threads) {
#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(threads)
    {
    std::vector<double> effect_buffer(ngroups);
    #pragma omp for
    for (size_t gene = 0; gene < ngenes; ++gene) {
#else
    SCRAN_CUSTOM_PARALLEL(ngenes, [&](size_t start, size_t end) -> void {
    std::vector<double> effect_buffer(ngroups);
    for (size_t gene = start; gene < end; ++gene) {
#endif

        auto base = effects + gene * ngroups * ngroups;
        for (int l = 0; l < ngroups; ++l) {
			auto ebegin = effect_buffer.data();
		    auto elast = ebegin;	

            // Ignoring the self comparison and pruning out NaNs.
            {
                auto eptr = base + l * ngroups;
                for (int r = 0; r < ngroups; ++r, ++eptr) {
                    if (r == l || std::isnan(*eptr)) {
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
                        output[i][l][gene] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            } else if (ncomps == 1) {
                for (size_t i = 0; i < MIN_RANK; ++i) { 
                    if (output[i].size()) {
                        output[i][l][gene] = *ebegin;
                    }
                }
            } else {
                if (output[MIN].size()) {
                    output[MIN][l][gene] = *std::min_element(ebegin, elast);
                }
                if (output[MEAN].size()) {
                    output[MEAN][l][gene] = std::accumulate(ebegin, elast, 0.0) / ncomps; 
                }
                if (output[MEDIAN].size()) {
                    output[MEDIAN][l][gene] = median(ebegin, ncomps); 
                }
                if (output[MAX].size()) {
                    output[MAX][l][gene] = *std::max_element(ebegin, elast);
                }
            }
        }
    }
#ifndef SCRAN_CUSTOM_PARALLEL            
	}
#else
    }, threads);
#endif

    return;
}

template<typename Stat>
void compute_min_rank(size_t ngenes, int ngroups, const Stat* effects, std::vector<Stat*>& output, int threads) {
    auto shift = ngroups * ngroups;

#ifndef SCRAN_CUSTOM_PARALLEL
    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        #pragma omp for
        for (int g = 0; g < ngroups; ++g) {
#else
    SCRAN_CUSTOM_PARALLEL(ngroups, [&](size_t start, size_t end) -> void {
        std::vector<std::pair<Stat, int> > buffer(ngenes);
        for (int g = start; g < end; ++g) {        
#endif
            auto target = output[g];
            std::fill(target, target + ngenes, ngenes + 1); 

            for (int g2 = 0; g2 < ngroups; ++g2) {
                if (g == g2) {
                    continue;
                }

                auto copy = effects + g * ngroups + g2;
                auto bIt = buffer.begin();
                for (size_t i = 0; i < ngenes; ++i, copy += shift) {
                    if (!std::isnan(*copy)) {
                        bIt->first = -*copy; // negative to sort by decreasing value.
                        bIt->second = i;
                        ++bIt;
                    }
                }
                std::sort(buffer.begin(), bIt);

                double counter = 1;
                for (auto bcopy = buffer.begin(); bcopy != bIt; ++bcopy) {
                    target[bcopy->second] = std::min(counter, target[bcopy->second]);
                    ++counter;
                }
            }
        }
#ifndef SCRAN_CUSTOM_PARALLEL
    }
#else
    }, threads);
#endif
}

/**
 * @endcond
 */

}

}

#endif
